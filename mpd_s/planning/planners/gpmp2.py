from typing import Any, Dict, Tuple

import torch

from mpd_s.planning.costs import (
    CostCollision,
    CostComposite,
    CostGoalPrior,
    CostGP,
    FactorCost,
)
from mpd_s.planning.planners.classical_planner import ClassicalPlanner
from mpd_s.torch_timer import TimerCUDA
from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase


def build_gpmp2_cost_composite(
    robot: RobotBase,
    env: EnvBase,
    n_support_points: int,
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    sigma_start: float,
    sigma_goal_prior: float,
    sigma_gp: float,
    sigma_collision: float,
    tensor_args: Dict[str, Any],
    use_extra_objects: bool = False,
) -> FactorCost:
    costs = []

    start_state = torch.cat(
        (start_pos, torch.zeros(start_pos.nelement(), **tensor_args))
    )
    cost_gp_prior = CostGP(
        robot=robot,
        n_support_points=n_support_points,
        start_state=start_state,
        sigma_start=sigma_start,
        sigma_gp=sigma_gp,
        tensor_args=tensor_args,
    )
    costs.append(cost_gp_prior)

    goal_state = torch.cat((goal_pos, torch.zeros_like(goal_pos)), dim=-1)
    cost_goal_prior = CostGoalPrior(
        robot=robot,
        n_support_points=n_support_points,
        goal_state=goal_state,
        sigma_goal_prior=sigma_goal_prior,
        tensor_args=tensor_args,
    )
    costs.append(cost_goal_prior)

    cost_collision = CostCollision(
        robot=robot,
        env=env,
        n_support_points=n_support_points,
        sigma_collision=sigma_collision,
        use_extra_objects=use_extra_objects,
        tensor_args=tensor_args,
    )
    costs.append(cost_collision)

    cost_composite = CostComposite(
        robot=robot,
        n_support_points=n_support_points,
        costs=costs,
        tensor_args=tensor_args,
    )
    return cost_composite


class GPMP2(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
        n_dim: int,
        n_support_points: int,
        dt: float,
        n_interpolate: int,
        sigma_start: float,
        sigma_gp: float,
        sigma_goal_prior: float,
        sigma_collision: float,
        step_size: float,
        delta: float,
        method: str,
        use_extra_objects: bool,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(
            name="GPMP2",
            env=env,
            robot=robot,
            use_extra_objects=use_extra_objects,
            tensor_args=tensor_args,
        )
        self.n_dim = n_dim
        self.dim = 2 * self.n_dim
        self.n_support_points = n_support_points
        self.N = self.dim * self.n_support_points
        self.dt = dt
        self.delta = delta
        self.method = method

        self.n_interpolate = n_interpolate
        self.sigma_start = sigma_start
        self.sigma_gp = sigma_gp
        self.sigma_goal_prior = sigma_goal_prior
        self.sigma_collision = sigma_collision
        self.step_size = step_size

    def _build_start_goal_cost(self, start_pos: torch.Tensor, goal_pos: torch.Tensor):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.start_state = torch.cat([start_pos, torch.zeros_like(start_pos)], dim=-1)
        self.goal_states = torch.cat([goal_pos, torch.zeros_like(goal_pos)], dim=-1)
        self.cost = build_gpmp2_cost_composite(
            robot=self.robot,
            env=self.env,
            n_support_points=self.n_support_points,
            start_pos=start_pos,
            goal_pos=goal_pos,
            use_extra_objects=self.use_extra_objects,
            sigma_start=self.sigma_start,
            sigma_gp=self.sigma_gp,
            sigma_collision=self.sigma_collision,
            sigma_goal_prior=self.sigma_goal_prior,
            tensor_args=self.tensor_args,
        )

    def optimize(
        self,
        trajectories: torch.Tensor,
        n_optimization_steps: int = 1,
        print_freq: int = None,
        debug: bool = False,
    ) -> torch.Tensor:
        self.start_pos = trajectories[0, 0, : self.n_dim]
        self.goal_pos = trajectories[0, -1, : self.n_dim]
        self._build_start_goal_cost(self.start_pos, self.goal_pos)
        self.n_optimization_steps = n_optimization_steps
        b, K = None, None
        with TimerCUDA() as t_opt:
            for step in range(1, n_optimization_steps + 1):
                trajectories, b, K = self._step(trajectories)
                if debug and print_freq and step % print_freq == 0:
                    self.print_info(step, t_opt.elapsed, self.get_costs(b, K))

            if debug:
                self.print_info(
                    n_optimization_steps, t_opt.elapsed, self.get_costs(b, K)
                )

        return trajectories

    def _step(self, trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A, b, K = self.cost.get_linear_system(
            trajectories=trajectories, n_interpolate=self.n_interpolate
        )
        trajectories = trajectories.detach()

        J_t_J, g = self._get_grad_terms(
            A,
            b,
            K,
            delta=self.delta,
        )

        d_theta = self.get_torch_solve(
            J_t_J,
            g,
            method=self.method,
        )

        d_theta = d_theta.view(*trajectories.shape)

        trajectories = trajectories + self.step_size * d_theta

        return trajectories, b, K

    def _get_grad_terms(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        K: torch.Tensor,
        delta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        I = torch.eye(self.N, self.N, device=self.tensor_args["device"], dtype=A.dtype)
        A_t_K = A.transpose(-2, -1) @ K
        A_t_A = A_t_K @ A

        diag_A_t_A = A_t_A.mean(0) * I
        J_t_J = A_t_A + delta * diag_A_t_A
        g = A_t_K @ b

        return J_t_J, g

    def get_torch_solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        if method == "inverse":
            res = torch.linalg.solve(A, b)
        elif method == "cholesky":
            l, _ = torch.linalg.cholesky_ex(A)
            res = torch.cholesky_solve(b, l)

        elif method == "lstq":
            # empirically slower
            res = torch.linalg.lstsq(A, b)[0]
        else:
            raise NotImplementedError

        return res

    def get_costs(self, errors: torch.Tensor, w_mat: torch.Tensor) -> torch.Tensor:
        if errors is None or w_mat is None:
            return None
        costs = errors.transpose(1, 2) @ w_mat.unsqueeze(0) @ errors
        return costs.reshape(-1)

    def print_info(self, step: int, t: float, costs: torch.Tensor) -> None:
        pad = len(str(self.n_optimization_steps))
        mean_cost = costs.mean().item() if costs is not None else float("nan")
        min_cost = costs.min().item() if costs is not None else float("nan")
        max_cost = costs.max().item() if costs is not None else float("nan")
        print(
            f"Step: {step:>{pad}}/{self.n_optimization_steps:>{pad}} "
            f"| Time: {t:.3f}s "
            f"| Cost (mean/min/max): {mean_cost:.3e}/{min_cost:.3e}/{max_cost:.3e}"
        )

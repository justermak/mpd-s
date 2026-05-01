from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

from mpd_s.planning.factors import FieldFactor, GPFactor, UnaryFactor
from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase


class Cost(ABC):
    def __init__(
        self, robot: RobotBase, n_support_points: int, tensor_args: Dict[str, Any]
    ) -> None:
        self.robot = robot
        self.dim = 2 * robot.n_dim
        self.n_support_points = n_support_points
        self.tensor_args = tensor_args

    @abstractmethod
    def __call__(self, trajectories: torch.Tensor, n_interpolate: int) -> torch.Tensor:
        pass


class CostObstacles(Cost):
    def __init__(
        self,
        robot: RobotBase,
        env: EnvBase,
        n_support_points: int,
        lambda_obstacles: float,
        use_extra_objects: bool,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args)
        self.env = env
        self.lambda_obstacles = lambda_obstacles
        self.use_extra_objects = use_extra_objects

    def __call__(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        trajectories_interpolated = self.robot.linearly_interpolate_trajectories(
            trajectories, n_interpolate=n_interpolate
        )
        cost = (
            self.robot.compute_cost(
                env=self.env,
                trajectories=trajectories_interpolated,
                on_extra=self.use_extra_objects,
            ).sum(-1)
            * self.lambda_obstacles
        )
        return cost


class CostJointVelocity(Cost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        lambda_velocity: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args)
        self.lambda_velocity = lambda_velocity

    def __call__(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        trajectories_vel = self.robot.get_velocity(
            trajectories=trajectories,
            mode="forward",
        )
        cost = 0.5 * (trajectories_vel**2).sum(-1).sum(-1) * self.lambda_velocity
        return cost


class CostJointAcceleration(Cost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        lambda_acceleration: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args)
        self.lambda_acceleration = lambda_acceleration

    def __call__(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        trajectories_acc = self.robot.get_acceleration(
            trajectories=trajectories,
            mode="forward",
        )
        cost = 0.5 * (trajectories_acc**2).sum(-1).sum(-1) * self.lambda_acceleration
        return cost


class CostJointJerk(Cost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        lambda_jerk: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args)
        self.lambda_jerk = lambda_jerk

    def __call__(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        trajectories_jerk = self.robot.get_jerk(
            trajectories=trajectories,
            mode="forward",
        )
        cost = 0.5 * (trajectories_jerk**2).sum(-1).sum(-1) * self.lambda_jerk
        return cost


class FactorCost(ABC):
    def __init__(
        self, robot: RobotBase, n_support_points: int, tensor_args: Dict[str, Any]
    ):
        self.robot = robot
        self.dim = 2 * robot.n_dim
        self.n_support_points = n_support_points
        self.tensor_args = tensor_args

    @abstractmethod
    def __call__(self, trajectories, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        pass


class CostComposite(FactorCost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        costs: List[FactorCost],
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.costs = costs

    def __call__(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        cost_total = 0
        for cost in self.costs:
            if cost is None:
                continue
            cost_total += (
                cost(trajectories=trajectories, n_interpolate=n_interpolate)
                if isinstance(cost, CostCollision)
                else cost(trajectories=trajectories)
            )

        return cost_total

    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        trajectories.requires_grad_(True)
        batch_size = trajectories.shape[0]
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost in self.costs:
            if cost is None:
                continue
            A, b, K = cost.get_linear_system(
                trajectories=trajectories, n_interpolate=n_interpolate
            )
            if A is None or b is None or K is None:
                continue
            optim_dim += A.shape[1]
            As.append(A.detach())
            bs.append(b.detach())
            Ks.append(K.detach())

        A = torch.cat(As, dim=1)
        b = torch.cat(bs, dim=1)
        K = torch.zeros(batch_size, optim_dim, optim_dim, **self.tensor_args)
        offset = 0
        for i in range(len(Ks)):
            dim = Ks[i].shape[1]
            K[:, offset : offset + dim, offset : offset + dim] = Ks[i]
            offset += dim
        return A, b, K


class CostCollision(FactorCost):
    def __init__(
        self,
        robot: RobotBase,
        env: EnvBase,
        n_support_points: int,
        sigma_collision: float,
        tensor_args: Dict[str, Any],
        use_extra_objects: bool = False,
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.env = env
        self.sigma_collision = sigma_collision
        self.use_extra_objects = use_extra_objects
        self.obst_factor = FieldFactor(
            n_dim=self.robot.n_dim,
            sigma=self.sigma_collision,
            use_extra_objects=self.use_extra_objects,
        )

    def __call__(self, trajectories: torch.Tensor, n_interpolate: int):
        cost = (
            self.obst_factor.get_error(
                trajectories=trajectories,
                env=self.env,
                robot=self.robot,
                n_interpolate=n_interpolate,
                calc_jacobian=False,
                return_full_error=True,
            ).sum(-1)
            / self.sigma_collision**2
        )

        return cost

    def get_linear_system(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        A, b, K = None, None, None
        batch_size = trajectories.shape[0]

        err_obst, H_obst = self.obst_factor.get_error(
            trajectories=trajectories,
            env=self.env,
            robot=self.robot,
            n_interpolate=n_interpolate,
            calc_jacobian=True,
            return_full_error=False,
        )

        A = torch.zeros(
            batch_size,
            self.n_support_points - 1,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )
        A[:, :, : H_obst.shape[-1]] = H_obst
        # shift each row by self.dim
        idxs = torch.arange(A.shape[-1], **self.tensor_args).repeat(A.shape[-2], 1)
        idxs = (
            idxs
            - torch.arange(
                self.dim,
                (idxs.shape[0] + 1) * self.dim,
                self.dim,
                **self.tensor_args,
            ).view(-1, 1)
        ) % idxs.shape[-1]
        idxs = idxs.to(torch.int64)
        A = torch.gather(A, -1, idxs.repeat(batch_size, 1, 1))

        # old code not vectorized
        # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L275

        b = err_obst.unsqueeze(-1)
        K = (
            torch.eye((self.n_support_points - 1), **self.tensor_args).repeat(
                batch_size, 1, 1
            )
            / self.sigma_collision**2
        )

        return A, b, K


class CostGPTrajectory(FactorCost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        sigma_gp: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)

        self.sigma_gp = sigma_gp

        self.gp_prior = GPFactor(
            dim=self.robot.n_dim,
            sigma=self.sigma_gp,
            dt=self.robot.dt,
            num_factors=self.n_support_points - 1,
            tensor_args=self.tensor_args,
        )

    def __call__(self, trajectories: torch.Tensor):
        if trajectories.shape[-1] == self.robot.n_dim:
            trajectories = torch.cat(
                [trajectories, torch.zeros_like(trajectories)], dim=-1
            )
        err_gp = self.gp_prior.get_error(trajectories, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs

    def get_linear_system(self, trajectories: torch.Tensor):
        pass


class CostGP(FactorCost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        start_state: torch.Tensor,
        sigma_start: float,
        sigma_gp: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.start_state = start_state
        self.sigma_start = sigma_start
        self.sigma_gp = sigma_gp

        self.start_prior = UnaryFactor(
            dim=self.dim,
            mean=self.start_state,
            tensor_args=self.tensor_args,
        )

        self.gp_prior = GPFactor(
            dim=self.robot.n_dim,
            sigma=self.sigma_gp,
            dt=self.robot.dt,
            num_factors=self.n_support_points - 1,
            tensor_args=self.tensor_args,
        )

    def __call__(self, trajectories: torch.Tensor):
        pass

    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        batch_size = trajectories.shape[0]
        A = torch.zeros(
            batch_size,
            self.dim * self.n_support_points,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )
        b = torch.zeros(
            batch_size, self.dim * self.n_support_points, 1, **self.tensor_args
        )
        K = torch.zeros(
            batch_size,
            self.dim * self.n_support_points,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )

        err_p, H_p = self.start_prior.get_error(trajectories[:, [0]])
        A[:, : self.dim, : self.dim] = H_p
        b[:, : self.dim] = err_p
        K[:, : self.dim, : self.dim] = (
            torch.eye(self.dim, **self.tensor_args) / self.sigma_start**2
        )

        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajectories)

        A[:, self.dim :, : -self.dim] = torch.block_diag(*H1_gp)
        A[:, self.dim :, self.dim :] += torch.block_diag(*H2_gp)
        b[:, self.dim :] = err_gp.flatten(1, 2)
        K[:, self.dim :, self.dim :] += torch.block_diag(*self.gp_prior.Q_inv)

        return A, b, K


class CostGoalPrior(FactorCost):
    def __init__(
        self,
        robot: RobotBase,
        n_support_points: int,
        goal_state: torch.Tensor,
        sigma_goal_prior: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.goal_state = goal_state
        self.sigma_goal_prior = sigma_goal_prior

        self.goal_prior = UnaryFactor(
            dim=self.dim,
            mean=self.goal_state,
            tensor_args=self.tensor_args,
        )

    def __call__(self, trajectories: torch.Tensor):
        pass

    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        A = torch.zeros(
            trajectories.shape[0],
            self.dim,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )

        err_g, H_g = self.goal_prior.get_error(trajectories[:, [-1]])
        A[:, :, -self.dim :] = H_g
        b = err_g
        K = torch.eye(self.dim, **self.tensor_args) / self.sigma_goal_prior**2

        return A, b, K

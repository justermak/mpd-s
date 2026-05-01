import os
import time
from typing import Any, Dict, List, Optional

import torch

from mpd_s.dataset.data_transform import NormalizerBase
from mpd_s.planning.costs import Cost
from mpd_s.planning.planners.classical_planner import ClassicalPlanner
from mpd_s.torch_timer import TimerCUDA
from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase
from mpd_s.visualizer import Visualizer


class GradientOptimization(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
        normalizer: NormalizerBase,
        n_support_points: int,
        n_control_points: Optional[int],
        costs: List[Cost],
        max_grad_norm: float,
        n_interpolate: int,
        tensor_args: Dict[str, Any],
        use_extra_objects: bool = False,
    ):
        super().__init__(
            name="GradientOptimization",
            env=env,
            robot=robot,
            use_extra_objects=use_extra_objects,
            tensor_args=tensor_args,
        )
        self.normalizer = normalizer
        self.n_support_points = n_support_points
        self.n_control_points = n_control_points
        self.costs = costs
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate

    def compute_gradient(
        self,
        x: torch.Tensor,
        return_cost: bool = False,
    ) -> torch.Tensor:
        trajectories_normalized = x.clone()

        with torch.enable_grad():
            trajectories_normalized.requires_grad_(True)
            trajectories_unnormalized = self.normalizer.unnormalize(
                trajectories_normalized
            )

            if self.n_control_points is not None:
                trajectories_pos = self.robot.get_trajectories_from_bsplines(
                    control_points=trajectories_unnormalized,
                    n_support_points=self.n_support_points,
                )
            else:
                trajectories_pos = self.robot.get_position(
                    trajectories=trajectories_unnormalized,
                )
            costs = [
                cost(
                    trajectories=trajectories_pos,
                    n_interpolate=self.n_interpolate,
                ).sum()
                for cost in self.costs
            ]
            
            cost = sum(costs)

            grad = torch.autograd.grad(cost, trajectories_normalized)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 1e-8, self.max_grad_norm + 1e-8) / (grad_norm + 1e-8)
                grad = scale_ratio * grad

            n_fixed = 2 if self.n_control_points is not None else 1
            grad[..., :n_fixed, :] = 0.0
            grad[..., -n_fixed:, :] = 0.0

        if return_cost:
            return grad, costs
        return grad

    def __call__(self, trajectories: torch.Tensor) -> torch.Tensor:
        return -self.compute_gradient(
            x=trajectories,
            return_cost=False,
        )

    def optimize(
        self,
        trajectories: torch.Tensor,
        n_optimization_steps: Optional[int],
        print_freq: Optional[int] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        x = trajectories.clone()
        if self.n_control_points is not None and x.shape[-2] == self.n_support_points:
            x = self.robot.get_trajectories_from_bsplines(
                control_points=x,
                n_support_points=self.n_support_points,
            )
        else:
            x = self.normalizer.normalize(x)
        cost = None
        if debug:
            self.print_info(0, 0.0, self.compute_gradient(x=x, return_cost=True)[1], x)
        with TimerCUDA() as t_opt:
            for i in range(1, n_optimization_steps + 1):
                if debug and print_freq and i % print_freq == 0:
                    self.print_info(i, t_opt.elapsed, cost, x)

                grad, cost = self.compute_gradient(x=x, return_cost=True)
                x = self.robot.gradient_step(x, -grad)

        return x

    def print_info(
        self, step: int, elapsed_time: float, costs: List[torch.Tensor], x: torch.Tensor
    ) -> None:
        costs_val = [cost.item() if cost is not None else float("nan") for cost in costs]
        print(f"Step {step} | Time: {elapsed_time:.3f} sec | Costs: {costs_val}")
        visualizer = Visualizer(self.env, self.robot, self.use_extra_objects)
        start_pos = x[0, 0, :]
        goal_pos = x[0, -1, :]
        ts = time.time()
        trajectories = x if self.n_control_points is None else self.robot.get_trajectories_from_bsplines(
            control_points=x, n_support_points=self.n_support_points
        )
        os.makedirs("imgs", exist_ok=True)
        visualizer.render_scene(trajectories=trajectories, start_pos=start_pos, goal_pos=goal_pos, save_path=f"imgs/{ts}_debug_step_{step}.png")

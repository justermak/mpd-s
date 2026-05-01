from typing import Any, Dict
import time
import os 

import torch

from mpd_s.planning.planners.classical_planner import ClassicalPlanner
from mpd_s.planning.planners.gpmp2 import GPMP2
from mpd_s.planning.planners.gradient_optimization import GradientOptimization
from mpd_s.planning.planners.rrt_connect import RRTConnect
from mpd_s.visualizer import Visualizer


class HybridPlanner(ClassicalPlanner):
    def __init__(
        self,
        sampling_based_planner: RRTConnect,
        optimization_based_planner: GPMP2 | GradientOptimization,
        smoothen: bool,
        create_straight_line_trajectories: bool,
        n_trajectories: int,
        n_support_points: int,
        n_control_points: int,
        tensor_args: Dict[str, Any],
    ):
        name = f"HybridPlanner ({sampling_based_planner.name if sampling_based_planner is not None else 'None'}+{optimization_based_planner.name if optimization_based_planner is not None else 'None'})"
        if sampling_based_planner is not None:
            super().__init__(
                name=name,
                env=sampling_based_planner.env,
                robot=sampling_based_planner.robot,
                use_extra_objects=sampling_based_planner.use_extra_objects,
                tensor_args=tensor_args,
            )
        elif optimization_based_planner is not None:
            super().__init__(
                name=name,
                env=optimization_based_planner.env,
                robot=optimization_based_planner.robot,
                use_extra_objects=optimization_based_planner.use_extra_objects,
                tensor_args=tensor_args,
            )
        else:
            raise ValueError(
                "At least one of sampling_based_planner or optimization_based_planner must be provided."
            )
        self.sampling_based_planner = sampling_based_planner
        self.optimization_based_planner = optimization_based_planner
        self.smoothen = smoothen
        self.n_support_points = n_support_points
        self.n_control_points = n_control_points
        self.n_trajectories = n_trajectories
        self.create_straight_line_trajectories = create_straight_line_trajectories
        if self.sampling_based_planner is not None:
            self.sampling_based_planner.n_trajectories = n_trajectories
        if self.optimization_based_planner is not None:
            self.optimization_based_planner.n_control_points = n_control_points

    def optimize(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        n_sampling_steps: int,
        n_optimization_steps: int,
        print_freq: int = 200,
        debug: bool = False,
    ) -> torch.Tensor:
        if self.sampling_based_planner is not None and n_sampling_steps is not None:
            trajectories = self.sampling_based_planner.optimize(
                start_pos=start_pos,
                goal_pos=goal_pos,
                n_sampling_steps=n_sampling_steps,
                print_freq=print_freq,
                debug=debug,
            )

        else:
            trajectories = []

        if self.smoothen:
            trajectories_smooth = [
                self.robot.get_trajectories_from_bsplines(
                    torch.cat(
                        [trajectory[:1, :], trajectory, trajectory[-1:, :]],
                        dim=0,
                    ),
                    n_support_points=self.n_support_points,
                )
                for trajectory in trajectories
            ]
            
        else:
            max_len = max([trajectory.shape[0] for trajectory in trajectories], default=0)
            trajectories_smooth = [
                torch.cat(
                    [
                        trajectory,
                        trajectory[-1:].repeat(
                            max_len - trajectory.shape[0], 1
                        ),
                    ],
                    dim=0,
                )
                for trajectory in trajectories
            ]

        initial_trajectories = torch.stack(trajectories_smooth) if len(trajectories_smooth) > 0 else torch.empty(
            (0, self.n_support_points, self.robot.n_dim), **self.tensor_args
        )
        
        if self.create_straight_line_trajectories:
            n_initial = initial_trajectories.shape[0]
            if n_initial < self.n_trajectories:
                straight_line_trajectory = (
                    self.robot.create_straight_line_trajectories(
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        n_support_points=self.n_support_points,
                    )
                )
                straight_line_trajectories = straight_line_trajectory.repeat(
                    self.n_trajectories - n_initial, 1, 1
                )
                initial_trajectories = torch.cat(
                    [initial_trajectories, straight_line_trajectories], dim=0
                )

        if (
            self.optimization_based_planner is not None
            and n_optimization_steps is not None
            and initial_trajectories.shape[0] > 0
        ):
            if self.optimization_based_planner.name == "GPMP2":
                initial_trajectories = torch.cat(
                    [
                        initial_trajectories,
                        self.robot.get_velocity(initial_trajectories, mode="avg"),
                    ],
                    dim=-1,
                )

            if self.optimization_based_planner.n_control_points is not None:
                initial_trajectories = self.robot.fit_bsplines_to_trajectories(
                    trajectories=initial_trajectories,
                    n_control_points=self.optimization_based_planner.n_control_points,
                )

            trajectories = self.optimization_based_planner.optimize(
                trajectories=initial_trajectories,
                n_optimization_steps=n_optimization_steps,
                print_freq=print_freq // 2,
                debug=debug,
            )

            if self.optimization_based_planner.n_control_points is not None:
                trajectories = self.robot.get_trajectories_from_bsplines(
                    control_points=trajectories,
                    n_support_points=self.n_support_points,
                )

        else:
            trajectories = initial_trajectories

        return trajectories

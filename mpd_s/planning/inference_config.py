from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from mpd_s.dataset.dataset import TrajectoryDataset
from mpd_s.model.generative_models import GenerativeModel
from mpd_s.planning.costs import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
    CostJointAcceleration,
    CostJointJerk,
    CostJointVelocity,
    CostObstacles,
)
from mpd_s.planning.planners.gpmp2 import GPMP2
from mpd_s.planning.planners.gradient_optimization import GradientOptimization
from mpd_s.planning.planners.hybrid_planner import HybridPlanner
from mpd_s.planning.planners.rrt_connect import RRTConnect


class ModelWrapperBase(ABC):
    def __init__(self, name: str, use_extra_objects: bool):
        self.name = name
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        pass


class GenerativeModelWrapper(ModelWrapperBase):
    def __init__(
        self,
        use_extra_objects: bool,
        model: GenerativeModel,
        model_name: str,
        guide: GradientOptimization,
        t_start_guide: float,
        n_guide_steps: int,
        additional_args: Dict[str, Any],
    ):
        super().__init__(name="Generative", use_extra_objects=use_extra_objects)
        self.model = model
        self.model_name = model_name
        self.guide = guide
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.additional_args = additional_args

    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        context = self.model.build_context(data)
        start_pos = data["start_pos"]
        goal_pos = data["goal_pos"]

        trajectories_iters_normalized = self.model.run_inference(
            n_samples=n_trajectories_per_task,
            context=context,
            guide=self.guide,
            n_guide_steps=self.n_guide_steps,
            t_start_guide=self.t_start_guide,
            debug=debug,
            **self.additional_args,
        )

        trajectories_iters = dataset.normalizer.unnormalize(
            trajectories_iters_normalized
        )

        if dataset.n_control_points is not None:
            trajectories_iters[..., :2, :] = start_pos.unsqueeze(0)
            trajectories_iters[..., -2:, :] = goal_pos.unsqueeze(0)
            trajectories_iters = dataset.robot.get_trajectories_from_bsplines(
                control_points=trajectories_iters,
                n_support_points=dataset.n_support_points,
            )
        else:
            trajectories_iters[..., 0, :] = start_pos.unsqueeze(0)
            trajectories_iters[..., -1, :] = goal_pos.unsqueeze(0)

        trajectories_final = trajectories_iters[-1]

        return trajectories_iters, trajectories_final


class MPDModelWrapper(ModelWrapperBase):
    def __init__(
        self,
        model: Any,
        guide: GradientOptimization,
        start_guide_steps_fraction: float,
        n_guide_steps: int,
        ddim: bool,
        use_extra_objects: bool,
    ):
        super().__init__(name="MPD", use_extra_objects=use_extra_objects)
        self.model = model
        self.guide = guide
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        hard_conditions = {
            0: torch.cat(
                [
                    data["start_pos_normalized"],
                    torch.zeros_like(data["start_pos_normalized"]),
                ],
                dim=-1,
            ),
            dataset.n_support_points - 1: torch.cat(
                [
                    data["goal_pos_normalized"],
                    torch.zeros_like(data["goal_pos_normalized"]),
                ],
                dim=-1,
            ),
        }

        trajectories_iters_normalized = self.model.run_inference(
            context=None,
            hard_conds=hard_conditions,
            n_samples=n_trajectories_per_task,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            guide=self.guide,
            n_guide_steps=self.n_guide_steps,
            horizon=dataset.n_support_points,
            return_chain=True,
            ddim=self.ddim,
        )

        trajectories_iters = dataset.normalizer.unnormalize(
            trajectories_iters_normalized
        )
        trajectories_final = trajectories_iters[-1]

        return trajectories_iters, trajectories_final


class ClassicalPlannerWrapper(ModelWrapperBase):
    def __init__(
        self,
        use_extra_objects: bool,
        planner: HybridPlanner,
        n_sampling_steps: int,
        n_optimization_steps: int,
    ):
        super().__init__(name="Classical", use_extra_objects=use_extra_objects)
        self.planner = planner
        self.n_sampling_steps = n_sampling_steps
        self.n_optimization_steps = n_optimization_steps

    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        start_pos = data["start_pos"]
        goal_pos = data["goal_pos"]

        trajectories = self.planner.optimize(
            start_pos=start_pos,
            goal_pos=goal_pos,
            n_sampling_steps=self.n_sampling_steps,
            n_optimization_steps=self.n_optimization_steps,
            debug=debug,
        )

        trajectories_iters = None
        trajectories_final = trajectories

        return trajectories_iters, trajectories_final


class ModelConfigBase(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def prepare(
        self,
        dataset: TrajectoryDataset,
        n_trajectories_per_task: int,
        tensor_args: Dict[str, Any],
    ) -> ModelWrapperBase:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class GenerativeModelConfig(ModelConfigBase):
    def __init__(
        self,
        model: GenerativeModel,
        model_name: str,
        t_start_guide: float,
        n_guide_steps: int,
        use_extra_objects: bool,
        lambda_obstacles: float,
        lambda_velocity: float,
        lambda_acceleration: float,
        lambda_jerk: float,
        max_grad_norm: float,
        n_interpolate: int,
        additional_args: Dict[str, Any],
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.model = model
        self.model_name = model_name
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.lambda_obstacles = lambda_obstacles
        self.lambda_velocity = lambda_velocity
        self.lambda_acceleration = lambda_acceleration
        self.lambda_jerk = lambda_jerk
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.additional_args = additional_args

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> GenerativeModelWrapper:
        collision_cost = None
        if self.lambda_obstacles is not None:
            collision_cost = CostObstacles(
                robot=dataset.robot,
                env=dataset.env,
                n_support_points=dataset.n_support_points,
                lambda_obstacles=self.lambda_obstacles,
                use_extra_objects=self.use_extra_objects,
                tensor_args=tensor_args,
            )

        velocity_cost = None
        if self.lambda_velocity is not None:
            velocity_cost = CostJointVelocity(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                lambda_velocity=self.lambda_velocity,
                tensor_args=tensor_args,
            )

        acceleration_cost = None
        if self.lambda_acceleration is not None:
            acceleration_cost = CostJointAcceleration(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                lambda_acceleration=self.lambda_acceleration,
                tensor_args=tensor_args,
            )

        jerk_cost = None
        if self.lambda_jerk is not None:
            jerk_cost = CostJointJerk(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                lambda_jerk=self.lambda_jerk,
                tensor_args=tensor_args,
            )

        costs = [
            cost
            for cost in [
                collision_cost,
                velocity_cost,
                acceleration_cost,
                jerk_cost,
            ]
            if cost is not None
        ]

        guide = GradientOptimization(
            env=dataset.env,
            robot=dataset.generating_robot,
            normalizer=dataset.normalizer,
            n_support_points=dataset.n_support_points,
            n_control_points=dataset.n_control_points,
            costs=costs,
            max_grad_norm=self.max_grad_norm,
            n_interpolate=self.n_interpolate,
            tensor_args=tensor_args,
            use_extra_objects=self.use_extra_objects,
        )

        return GenerativeModelWrapper(
            use_extra_objects=self.use_extra_objects,
            model=self.model,
            model_name=self.model_name,
            guide=guide,
            t_start_guide=self.t_start_guide,
            n_guide_steps=self.n_guide_steps,
            additional_args=self.additional_args,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_extra_objects": self.use_extra_objects,
            "lambda_obstacles": self.lambda_obstacles,
            "lambda_velocity": self.lambda_velocity,
            "lambda_acceleration": self.lambda_acceleration,
            "lambda_jerk": self.lambda_jerk,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "t_start_guide": self.t_start_guide,
            "n_guide_steps": self.n_guide_steps,
            "additional_args": self.additional_args,
        }


class MPDConfig(ModelConfigBase):
    def __init__(
        self,
        model: Any,
        use_extra_objects: bool,
        sigma_collision: float,
        sigma_gp: float,
        max_grad_norm: float,
        n_interpolate: int,
        start_guide_steps_fraction: float,
        n_guide_steps: int,
        ddim: bool,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.model = model
        self.sigma_collision = sigma_collision
        self.sigma_gp = sigma_gp
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> MPDModelWrapper:
        guide = None
        if self.n_guide_steps > 0:
            collision_costs = [
                CostCollision(
                    robot=dataset.robot,
                    env=dataset.env,
                    n_support_points=dataset.n_support_points,
                    sigma_collision=self.sigma_collision,
                    use_extra_objects=self.use_extra_objects,
                    tensor_args=tensor_args,
                )
            ]

            sharpness_costs = [
                CostGPTrajectory(
                    robot=dataset.robot,
                    n_support_points=dataset.n_support_points,
                    sigma_gp=self.sigma_gp,
                    tensor_args=tensor_args,
                )
            ]

            costs = collision_costs + sharpness_costs

            cost = CostComposite(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                costs=costs,
                tensor_args=tensor_args,
            )

            guide = GradientOptimization(
                env=dataset.env,
                robot=dataset.generating_robot,
                normalizer=dataset.normalizer,
                n_support_points=dataset.n_support_points,
                n_control_points=dataset.n_control_points,
                costs=[cost],
                max_grad_norm=self.max_grad_norm,
                n_interpolate=self.n_interpolate,
                tensor_args=tensor_args,
                use_extra_objects=self.use_extra_objects,
            )

        return MPDModelWrapper(
            model=self.model,
            guide=guide,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            n_guide_steps=self.n_guide_steps,
            ddim=self.ddim,
            use_extra_objects=self.use_extra_objects,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_extra_objects": self.use_extra_objects,
            "sigma_collision": self.sigma_collision,
            "sigma_gp": self.sigma_gp,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
        }


class ClassicalConfig(ModelConfigBase):
    def __init__(
        self,
        use_extra_objects: bool,
        dataset: TrajectoryDataset,
        sampling_based_planner_name: Optional[str],
        optimization_based_planner_name: Optional[str],
        n_sampling_steps: int,
        n_optimization_steps: int,
        smoothen: bool,
        create_straight_line_trajectories: bool,
        n_dim: int,
        rrt_connect_max_radius: float,
        rrt_connect_n_points: int,
        rrt_connect_n_samples: int,
        gpmp2_n_interpolate: int,
        gpmp2_sigma_start: float,
        gpmp2_sigma_goal_prior: float,
        gpmp2_sigma_gp: float,
        gpmp2_sigma_collision: float,
        gpmp2_step_size: float,
        gpmp2_delta: float,
        gpmp2_method: str,
        grad_lambda_obstacles: float,
        grad_lambda_velocity: float,
        grad_lambda_acceleration: float,
        grad_lambda_jerk: float,
        grad_max_grad_norm: float,
        grad_n_interpolate: int,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.dataset = dataset
        self.sampling_based_planner_name = sampling_based_planner_name
        self.optimization_based_planner_name = optimization_based_planner_name
        self.n_sampling_steps = n_sampling_steps
        self.n_optimization_steps = n_optimization_steps
        self.smoothen = smoothen
        self.create_straight_line_trajectories = create_straight_line_trajectories
        self.n_dim = n_dim
        self.rrt_connect_max_radius = rrt_connect_max_radius
        self.rrt_connect_n_points = rrt_connect_n_points
        self.rrt_connect_n_samples = rrt_connect_n_samples
        self.gpmp2_n_interpolate = gpmp2_n_interpolate
        self.gpmp2_sigma_start = gpmp2_sigma_start
        self.gpmp2_sigma_goal_prior = gpmp2_sigma_goal_prior
        self.gpmp2_sigma_gp = gpmp2_sigma_gp
        self.gpmp2_sigma_collision = gpmp2_sigma_collision
        self.gpmp2_step_size = gpmp2_step_size
        self.gpmp2_delta = gpmp2_delta
        self.gpmp2_method = gpmp2_method
        self.grad_lambda_obstacles = grad_lambda_obstacles
        self.grad_lambda_velocity = grad_lambda_velocity
        self.grad_lambda_acceleration = grad_lambda_acceleration
        self.grad_lambda_jerk = grad_lambda_jerk
        self.grad_max_grad_norm = grad_max_grad_norm
        self.grad_n_interpolate = grad_n_interpolate

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> ClassicalPlannerWrapper:
        sampling_based_planner = None
        if self.sampling_based_planner_name is not None:
            sampling_based_planner = RRTConnect(
                env=dataset.env,
                robot=dataset.generating_robot,
                n_trajectories=n_trajectories_per_task,
                max_radius=self.rrt_connect_max_radius,
                n_points=self.rrt_connect_n_points,
                n_samples=self.rrt_connect_n_samples,
                use_extra_objects=self.use_extra_objects,
                tensor_args=tensor_args,
            )

        optimization_based_planner = None
        if self.optimization_based_planner_name == "GPMP2":
            optimization_based_planner = GPMP2(
                env=dataset.env,
                robot=dataset.generating_robot,
                n_dim=dataset.generating_robot.n_dim,
                n_support_points=dataset.n_support_points,
                dt=dataset.generating_robot.dt,
                n_interpolate=self.gpmp2_n_interpolate,
                sigma_start=self.gpmp2_sigma_start,
                sigma_gp=self.gpmp2_sigma_gp,
                sigma_goal_prior=self.gpmp2_sigma_goal_prior,
                sigma_collision=self.gpmp2_sigma_collision,
                step_size=self.gpmp2_step_size,
                delta=self.gpmp2_delta,
                method=self.gpmp2_method,
                use_extra_objects=self.use_extra_objects,
                tensor_args=tensor_args,
            )
        elif self.optimization_based_planner_name == "GradientOptimization":
            collision_cost = None
            if self.grad_lambda_obstacles is not None:
                collision_cost = CostObstacles(
                    robot=dataset.generating_robot,
                    env=dataset.env,
                    n_support_points=dataset.n_support_points,
                    lambda_obstacles=self.grad_lambda_obstacles,
                    use_extra_objects=self.use_extra_objects,
                    tensor_args=tensor_args,
                )

            velocity_cost = None
            if self.grad_lambda_velocity is not None:
                velocity_cost = CostJointVelocity(
                    robot=dataset.generating_robot,
                    n_support_points=dataset.n_support_points,
                    lambda_velocity=self.grad_lambda_velocity,
                    tensor_args=tensor_args,
                )

            acceleration_cost = None
            if self.grad_lambda_acceleration is not None:
                acceleration_cost = CostJointAcceleration(
                    robot=dataset.generating_robot,
                    n_support_points=dataset.n_support_points,
                    lambda_acceleration=self.grad_lambda_acceleration,
                    tensor_args=tensor_args,
                )

            jerk_cost = None
            if self.grad_lambda_jerk is not None:
                jerk_cost = CostJointJerk(
                    robot=dataset.generating_robot,
                    n_support_points=dataset.n_support_points,
                    lambda_jerk=self.grad_lambda_jerk,
                    tensor_args=tensor_args,
                )

            costs = [
                cost
                for cost in [
                    collision_cost,
                    velocity_cost,
                    acceleration_cost,
                    jerk_cost,
                ]
                if cost is not None
            ]
            optimization_based_planner = GradientOptimization(
                env=dataset.env,
                robot=dataset.generating_robot,
                normalizer=dataset.normalizer,
                n_support_points=dataset.n_support_points,
                n_control_points=dataset.n_control_points,
                costs=costs,
                max_grad_norm=self.grad_max_grad_norm,
                n_interpolate=self.grad_n_interpolate,
                tensor_args=tensor_args,
                use_extra_objects=self.use_extra_objects,
            )

        planner = HybridPlanner(
            sampling_based_planner=sampling_based_planner,
            optimization_based_planner=optimization_based_planner,
            smoothen=self.smoothen,
            create_straight_line_trajectories=self.create_straight_line_trajectories,
            n_trajectories=n_trajectories_per_task,
            n_support_points=dataset.n_support_points,
            n_control_points=dataset.n_control_points,
            tensor_args=tensor_args,
        )

        wrapper = ClassicalPlannerWrapper(
            use_extra_objects=self.use_extra_objects,
            planner=planner,
            n_sampling_steps=self.n_sampling_steps,
            n_optimization_steps=self.n_optimization_steps,
        )

        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_sampling_steps": self.n_sampling_steps,
            "n_optimization_steps": self.n_optimization_steps,
            "use_extra_objects": self.use_extra_objects,
            "n_dim": self.n_dim,
            "rrt_connect_max_radius": self.rrt_connect_max_radius,
            "rrt_connect_n_points": self.rrt_connect_n_points,
            "rrt_connect_n_samples": self.rrt_connect_n_samples,
            "gpmp2_n_interpolate": self.gpmp2_n_interpolate,
            "gpmp2_sigma_start": self.gpmp2_sigma_start,
            "gpmp2_sigma_goal_prior": self.gpmp2_sigma_goal_prior,
            "gpmp2_sigma_gp": self.gpmp2_sigma_gp,
            "gpmp2_sigma_collision": self.gpmp2_sigma_collision,
            "gpmp2_step_size": self.gpmp2_step_size,
            "gpmp2_delta": self.gpmp2_delta,
            "gpmp2_method": self.gpmp2_method,
            "grad_lambda_obstacles": self.grad_lambda_obstacles,
            "grad_lambda_velocity": self.grad_lambda_velocity,
            "grad_lambda_acceleration": self.grad_lambda_acceleration,
            "grad_lambda_jerk": self.grad_lambda_jerk,
            "grad_max_grad_norm": self.grad_max_grad_norm,
            "grad_n_interpolate": self.grad_n_interpolate,
        }

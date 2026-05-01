from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from mpd_s.dataset.dataset import TrajectoryDataset
from mpd_s.model.generative_models import GenerativeModel
from mpd_s.planning.metrics import (
    compute_free_points,
    compute_free_trajectories,
    compute_ISJ,
    compute_path_length,
    compute_success_rate,
    compute_waypoints_stddev,
)
from mpd_s.planning.planners.gradient_optimization import GradientOptimization
from mpd_s.visualizer import Visualizer


def _log_trajectories_metrics(
    model: GenerativeModel,
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    context: torch.Tensor,
    dataset: TrajectoryDataset,
    planning_visualizer: Visualizer,
    tensorboard_writer: SummaryWriter,
    prefix: str,
    suffix: str,
    step: int,
    guide: GradientOptimization,
    inference_args: Dict[str, Any],
    use_extra_objects: bool,
) -> None:
    trajectories_normalized = model.run_inference(
        n_samples=20,
        context=context,
        guide=guide,
        **inference_args,
    )[-1]

    trajectories = dataset.normalizer.unnormalize(trajectories_normalized)

    if dataset.n_control_points is not None:
        trajectories[..., :2, :] = start_pos.unsqueeze(0)
        trajectories[..., -2:, :] = goal_pos.unsqueeze(0)
        trajectories = dataset.robot.get_trajectories_from_bsplines(
            control_points=trajectories,
            n_support_points=dataset.n_support_points,
        )
    else:
        trajectories[..., 0, :] = start_pos.unsqueeze(0)
        trajectories[..., -1, :] = goal_pos.unsqueeze(0)

    trajectories_collision, trajectories_free, trajectories_collision_mask = (
        dataset.robot.get_trajectories_collision_and_free(
            env=dataset.env, trajectories=trajectories, on_extra=use_extra_objects
        )
    )

    tensorboard_writer.add_scalar(
        f"{prefix}success_rate{suffix}",
        compute_success_rate(trajectories_free),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}free_trajectories{suffix}",
        compute_free_trajectories(trajectories_free, 20),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}free_points{suffix}",
        compute_free_points(trajectories_collision_mask),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}avg_path_length{suffix}",
        compute_path_length(trajectories_free, dataset.robot).mean().item(),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}avg_ISJ{suffix}",
        compute_ISJ(trajectories_free, dataset.robot).mean().item(),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}waypoints_stddev{suffix}",
        compute_waypoints_stddev(trajectories_free, dataset.robot),
        step,
    )

    fig, ax = planning_visualizer.render_scene(
        trajectories=trajectories,
        start_pos=trajectories[0, 0],
        goal_pos=trajectories[0, -1],
        save_path=None,
    )

    tensorboard_writer.add_figure(
        f"{prefix}trajectories_figure{suffix}",
        fig,
        step,
    )

    plt.close(fig)


def log(
    step: int,
    model: GenerativeModel,
    subset: Subset,
    prefix: str,
    tensorboard_writer: SummaryWriter,
    guide: GradientOptimization,
    guide_extra: GradientOptimization,
    inference_args: Dict[str, Any],
    train_losses: dict = None,
    val_losses: dict = None,
    debug: bool = False,
):
    model.eval()
    with torch.no_grad():
        if tensorboard_writer is not None:
            if train_losses is not None:
                for loss_name, loss_value in train_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}", loss_value, step
                    )

            if val_losses is not None:
                for loss_name, loss_value in val_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}",
                        loss_value,
                        step,
                    )

        dataset: TrajectoryDataset = subset.dataset
        trajectory_id = np.random.choice(subset.indices)
        data = dataset[trajectory_id]

        context = model.build_context(data)
        start_pos = data["start_pos"]
        goal_pos = data["goal_pos"]

        planning_visualizer = Visualizer(
            env=dataset.env, robot=dataset.robot, use_extra_objects=False
        )
        planning_visualizer_extra = Visualizer(
            env=dataset.env, robot=dataset.robot, use_extra_objects=True
        )

        if tensorboard_writer is not None:
            _log_trajectories_metrics(
                model=model,
                start_pos=start_pos,
                goal_pos=goal_pos,
                context=context,
                dataset=dataset,
                planning_visualizer=planning_visualizer,
                tensorboard_writer=tensorboard_writer,
                prefix=prefix,
                suffix="",
                step=step,
                guide=None,
                inference_args=inference_args,
                use_extra_objects=False,
            )

        if guide is not None and tensorboard_writer is not None:
            _log_trajectories_metrics(
                model=model,
                start_pos=start_pos,
                goal_pos=goal_pos,
                context=context,
                dataset=dataset,
                planning_visualizer=planning_visualizer,
                tensorboard_writer=tensorboard_writer,
                prefix=prefix,
                suffix="_guide",
                step=step,
                guide=guide,
                inference_args=inference_args,
                use_extra_objects=False,
            )

        if guide_extra is not None and tensorboard_writer is not None:
            _log_trajectories_metrics(
                model=model,
                start_pos=start_pos,
                goal_pos=goal_pos,
                context=context,
                dataset=dataset,
                planning_visualizer=planning_visualizer_extra,
                tensorboard_writer=tensorboard_writer,
                prefix=prefix,
                suffix="_guide_extra",
                step=step,
                guide=guide_extra,
                inference_args=inference_args,
                use_extra_objects=True,
            )

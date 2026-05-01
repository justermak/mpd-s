from typing import Tuple

import numpy as np
import torch
from scipy import stats

from mpd_s.universe.robot import RobotBase


def compute_success_rate(trajectories_free: torch.Tensor) -> float:
    assert trajectories_free.ndim == 3

    return float(trajectories_free.nelement() > 0)


def compute_free_trajectories(
    trajectories_free: torch.Tensor, n_trajectories_per_task: int
) -> float:
    assert trajectories_free.ndim == 3
    cnt_free = trajectories_free.shape[0]
    fraction = cnt_free / n_trajectories_per_task

    return fraction


def compute_free_points(trajectories_collision_mask: torch.Tensor) -> float:
    assert trajectories_collision_mask.ndim == 2
    fraction = (
        ~trajectories_collision_mask
    ).sum().item() / trajectories_collision_mask.numel()

    return fraction


def compute_path_length(trajectories: torch.Tensor, robot: RobotBase) -> torch.Tensor:
    assert trajectories.ndim == 3
    if trajectories.shape[0] == 0:
        return torch.tensor(0.0)
    trajectories_pos = robot.get_position(trajectories)
    path_length = torch.linalg.norm(torch.diff(trajectories_pos, dim=-2), dim=-1).sum(
        -1
    )

    return path_length


def compute_ISJ(trajectories: torch.Tensor, robot: RobotBase) -> torch.Tensor:
    assert trajectories.ndim == 3
    if trajectories.shape[0] == 0:
        return torch.tensor(0.0)
    trajectories_pos = robot.get_position(trajectories)
    trajectories_vel = torch.diff(trajectories_pos, dim=-2) / robot.dt
    trajectories_acc = torch.diff(trajectories_vel, dim=-2) / robot.dt
    trajectories_jerk = torch.diff(trajectories_acc, dim=-2) / robot.dt
    integrated_squared_jerk = (trajectories_jerk**2).sum(-1).sum(-1) * robot.dt

    return integrated_squared_jerk


def compute_waypoints_stddev(
    trajectories: torch.Tensor, robot: RobotBase
) -> float:
    assert trajectories.ndim == 3
    if trajectories.shape[0] < 3:
        return 0.0
    
    trajectories_pos = (
        robot.get_position(trajectories).permute(1, 0, 2)
        if robot is not None
        else trajectories.permute(1, 0, 2)
    )
    mean = torch.mean(trajectories_pos, dim=-2, keepdim=True)
    cov = (
        (trajectories_pos.unsqueeze(-2) - mean.unsqueeze(-2))
        * (trajectories_pos.unsqueeze(-1) - mean.unsqueeze(-1))
    ).sum(dim=-3) / (trajectories_pos.shape[-2] - 1)
    std = (cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] ** 2).mean() ** 0.5

    return std.item()


def bootstrap_confidence_interval(
    data: list, confidence_level: float = 0.95, n_resamples: int = 10000
) -> Tuple[float, float]:
    if data is None or len(data) <= 1:
        return None, None

    res = stats.bootstrap(
        (data,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
    )

    ci_lower = res.confidence_interval.low
    ci_upper = res.confidence_interval.high

    center = (ci_lower + ci_upper) / 2
    half_width = (ci_upper - ci_lower) / 2

    return center, half_width

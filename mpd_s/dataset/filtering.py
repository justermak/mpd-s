from typing import Set

import torch

from mpd_s.planning.metrics import compute_ISJ, compute_path_length
from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase


def filter_longest_trajectories(
    trajectories: torch.Tensor,
    robot: RobotBase,
    env: EnvBase,
    task_start_idxs: torch.Tensor,
    portion: float,
) -> Set[int]:
    n_tasks = len(task_start_idxs) - 1
    indices_to_exclude = set()

    for task_id in range(n_tasks):
        task_start = task_start_idxs[task_id].item()
        task_end = task_start_idxs[task_id + 1].item()
        task_size = task_end - task_start

        if task_size == 0:
            continue

        task_trajectories = trajectories[task_start:task_end]
        path_lengths = compute_path_length(task_trajectories, robot)
        n_to_filter = int(portion * task_size)

        if n_to_filter > 0:
            _, longest_indices = torch.topk(path_lengths, k=n_to_filter)
            global_indices = [task_start + idx.item() for idx in longest_indices]
            indices_to_exclude.update(global_indices)

    return indices_to_exclude


def filter_roughest_trajectories(
    trajectories: torch.Tensor,
    robot: RobotBase,
    env: EnvBase,
    task_start_idxs: torch.Tensor,
    portion: float,
) -> Set[int]:
    if portion <= 0:
        return set()

    n_tasks = len(task_start_idxs) - 1
    indices_to_exclude = set()

    for task_id in range(n_tasks):
        task_start = task_start_idxs[task_id].item()
        task_end = task_start_idxs[task_id + 1].item()
        task_size = task_end - task_start

        if task_size == 0:
            continue

        task_trajectories = trajectories[task_start:task_end]
        ISJs = compute_ISJ(task_trajectories, robot)
        n_to_filter = int(portion * task_size) // 2 * 2

        if n_to_filter > 0:
            _, roughest_indices = torch.topk(ISJs, k=n_to_filter)
            global_indices = [task_start + idx.item() for idx in roughest_indices]
            indices_to_exclude.update(global_indices)

    return indices_to_exclude


def filter_collision(
    trajectories: torch.Tensor,
    robot: RobotBase,
    env: EnvBase,
    task_start_idxs: torch.Tensor,
) -> Set[int]:
    n_tasks = len(task_start_idxs) - 1
    indices_to_exclude = set()

    for task_id in range(n_tasks):
        task_start = task_start_idxs[task_id].item()
        task_end = task_start_idxs[task_id + 1].item()
        task_size = task_end - task_start

        if task_size == 0:
            continue

        task_trajectories = trajectories[task_start:task_end]

        _, _, points_collision_mask = robot.get_trajectories_collision_and_free(
            env=env, trajectories=task_trajectories
        )

        trajectories_collision_mask = points_collision_mask.any(dim=-1)

        collision_indices = torch.nonzero(trajectories_collision_mask).squeeze(-1)

        if collision_indices.numel() > 0:
            global_indices = [task_start + idx.item() for idx in collision_indices]
            indices_to_exclude.update(global_indices)

    return indices_to_exclude


def get_filter_functions():
    return {
        "filter_longest_trajectories": filter_longest_trajectories,
        "filter_roughest_trajectories": filter_roughest_trajectories,
        "filter_collision": filter_collision,
    }

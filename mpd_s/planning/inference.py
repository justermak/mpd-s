import os
import pickle
from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from mpd_s.dataset.dataset import TrajectoryDataset
from mpd_s.planning.inference_config import ModelConfigBase, ModelWrapperBase
from mpd_s.planning.metrics import (
    bootstrap_confidence_interval,
    compute_free_points,
    compute_free_trajectories,
    compute_ISJ,
    compute_path_length,
    compute_success_rate,
    compute_waypoints_stddev,
)
from mpd_s.torch_timer import TimerCUDA
from mpd_s.visualizer import Visualizer


def run_inference_for_task(
    dataset: TrajectoryDataset,
    data: dict,
    n_trajectories_per_task: int,
    model_wrapper: ModelWrapperBase,
    return_full_data: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    robot = dataset.robot
    env = dataset.env

    with TimerCUDA() as timer_model_sampling:
        trajectories_iters, trajectories_final = model_wrapper.sample(
            dataset=dataset,
            data=data,
            n_trajectories_per_task=n_trajectories_per_task,
            debug=debug,
        )
    time = timer_model_sampling.elapsed

    if trajectories_final is None or trajectories_final.shape[0] == 0:
        stats = {
            "success_rate": 0.0,
            "avg_free_trajectories": None,
            "avg_free_points": None,
            "path_length_best": None,
            "avg_ISJ": None,
            "avg_path_length": None,
            "waypoints_stddev": None,
            "time": time,
        }
        return {"stats": stats}

    (
        trajectories_final_collision,
        trajectories_final_free,
        points_final_collision_mask,
    ) = robot.get_trajectories_collision_and_free(
        env=env,
        trajectories=trajectories_final,
        on_extra=model_wrapper.use_extra_objects,
    )

    trajectories_final = torch.cat(
        [trajectories_final_free, trajectories_final_collision], dim=0
    )  # move free trajectories to the front

    success_rate = compute_success_rate(trajectories_final_free)
    avg_free_trajectories = compute_free_trajectories(
        trajectories_final_free, n_trajectories_per_task
    )
    avg_free_points = compute_free_points(points_final_collision_mask)

    avg_path_length = None
    avg_ISJ = None
    waypoints_stddev = None
    path_length_best = None

    if trajectories_final_free.shape[0] > 0:
        path_lengths = compute_path_length(trajectories_final_free, robot)
        path_length_best = torch.min(path_lengths).item()
        avg_path_length = path_lengths.mean().item()
        avg_ISJ = compute_ISJ(trajectories_final_free, robot).mean().item()
        waypoints_stddev = compute_waypoints_stddev(trajectories_final_free, robot)

    stats = {
        "success_rate": success_rate,
        "avg_free_trajectories": avg_free_trajectories,
        "avg_free_points": avg_free_points,
        "path_length_best": path_length_best,
        "avg_path_length": avg_path_length,
        "avg_ISJ": avg_ISJ,
        "waypoints_stddev": waypoints_stddev,
        "time": time,
    }

    if not return_full_data:
        return {"stats": stats}

    start_pos = dataset.normalizer.unnormalize(data["start_pos_normalized"])

    goal_pos = dataset.normalizer.unnormalize(data["goal_pos_normalized"])

    best_traj_idx = None
    traj_final_best = None
    if trajectories_final_free.shape[0] > 0:
        best_traj_idx = torch.argmin(path_lengths).item()
        traj_final_best = trajectories_final_free[best_traj_idx]

    full_data = {
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "trajectories_iters": trajectories_iters,
        "trajectories_final": trajectories_final,
        "trajectories_final_collision": trajectories_final_collision,
        "trajectories_final_free": trajectories_final_free,
        "best_traj_idx": best_traj_idx,
        "traj_final_best": traj_final_best,
    }

    return {"stats": stats, "full_data": full_data}


def run_inference_on_dataset(
    subset: Subset,
    n_tasks: int,
    n_trajectories_per_task: int,
    model_wrapper: ModelWrapperBase,
    debug: bool = False,
) -> Dict[str, Any]:
    statistics = []
    full_data_sample = None
    dataset: TrajectoryDataset = subset.dataset

    return_full_data = True
    for i in tqdm(range(n_tasks), desc="Processing tasks"):
        while True:
            idx = np.random.choice(subset.indices)
            data = dataset[idx]

            if model_wrapper.name != "Classical":
                break

            start_pos = data["start_pos"]
            goal_pos = data["goal_pos"]
            points = torch.stack((start_pos, goal_pos))
            collision_mask = dataset.generating_robot.get_collision_mask(
                env=dataset.env, points=points, on_extra=model_wrapper.use_extra_objects
            )
            if not collision_mask.any():
                break

        task_results = run_inference_for_task(
            dataset=dataset,
            data=data,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            return_full_data=return_full_data,
            debug=debug,
        )

        statistics.append(task_results["stats"])
        if return_full_data and "full_data" in task_results:
            full_data_sample = task_results["full_data"]
            return_full_data = False

    df_statistics = pd.DataFrame(statistics)

    result = {"statistics": df_statistics}

    if full_data_sample is not None:
        result["sample"] = full_data_sample

    return result


def compute_stats(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    df = results.get("statistics")
    if df is None or df.empty:
        return None

    final_stats = {}

    metrics = [
        "time",
        "success_rate",
        "avg_free_trajectories",
        "avg_free_points",
        "path_length_best",
        "avg_path_length",
        "avg_ISJ",
        "waypoints_stddev",
    ]

    for col in metrics:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if data.empty:
            center, hw = None, None
        else:
            center, hw = bootstrap_confidence_interval(data.values)

        final_stats[f"{col}_center"] = float(center) if center is not None else None
        final_stats[f"{col}_hw"] = float(hw) if hw is not None else None

    return final_stats


def print_stats(results, n_tasks, n_trajectories_per_task, results_dir=None):
    rows = []
    splits = []

    for split_name in ["train", "val", "test"]:
        stats = results.get(f"{split_name}_stats")
        if stats is None:
            continue

        splits.append(split_name.upper())

        row = {
            "n_tasks": n_tasks,
            "n_trajectories_per_task": n_trajectories_per_task,
            "success_rate": f"{stats['success_rate_center'] * 100:.2f} ± {stats['success_rate_hw'] * 100:.2f}",
            "time": f"{stats['time_center']:.3f} ± {stats['time_hw']:.3f}",
        }

        if stats.get("avg_free_trajectories_center") is not None:
            row["avg_free_trajectories"] = (
                f"{stats['avg_free_trajectories_center'] * 100:.2f} ± {stats['avg_free_trajectories_hw']:.2f}"
            )

        if stats.get("avg_free_points_center") is not None:
            row["avg_free_points"] = (
                f"{stats['avg_free_points_center'] * 100:.2f} ± {stats['avg_free_points_hw']:.2f}"
            )

        if stats.get("path_length_best_center") is not None:
            row["path_length_best"] = (
                f"{stats['path_length_best_center']:.4f} ± {stats['path_length_best_hw']:.4f}"
            )

        if stats.get("avg_path_length_center") is not None:
            row["avg_path_length"] = (
                f"{stats['avg_path_length_center']:.4f} ± {stats['avg_path_length_hw']:.4f}"
            )

        if stats.get("avg_ISJ_center") is not None:
            row["avg_ISJ"] = (
                f"{stats['avg_ISJ_center']:.4f} ± {stats['avg_ISJ_hw']:.4f}"
            )

        if stats.get("waypoints_stddev_center") is not None:
            row["waypoints_stddev"] = (
                f"{stats['waypoints_stddev_center']:.4f} ± {stats['waypoints_stddev_hw']:.4f}"
            )

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows, index=splits)
        markdown_table = df.to_markdown()
        print("=" * 80)
        print(markdown_table)
        
        if results_dir:
            with open(os.path.join(results_dir, "stats.md"), "w", encoding="utf-8") as f:
                f.write(markdown_table)



def visualize_results(
    results: Dict[str, Any],
    dataset: TrajectoryDataset,
    use_extra_objects: bool,
    results_dir: str,
    generate_animation: bool = True,
    draw_indices: Optional[List[int]] = None,
    draw_spacing: int = 1,
    name_prefix: str = "task0",
):
    planner_visualizer = Visualizer(
        env=dataset.env, robot=dataset.robot, use_extra_objects=use_extra_objects
    )
    start_pos = results["sample"]["start_pos"]
    goal_pos = results["sample"]["goal_pos"]
    trajectories_iters = results["sample"]["trajectories_iters"]
    trajectories_final = results["sample"]["trajectories_final"]
    best_traj_idx = results["sample"]["best_traj_idx"]

    planner_visualizer.render_scene(
        trajectories=trajectories_final,
        best_traj_idx=best_traj_idx,
        start_pos=start_pos,
        goal_pos=goal_pos,
        draw_indices=draw_indices,
        draw_spacing=draw_spacing,
        save_path=os.path.join(results_dir, f"{name_prefix}-trajectories.png"),
    )

    if not generate_animation:
        return

    planner_visualizer.animate_robot_motion(
        trajectories=trajectories_final,
        best_traj_idx=best_traj_idx,
        start_pos=start_pos,
        goal_pos=goal_pos,
        draw_indices=draw_indices,
        draw_spacing=draw_spacing,
        save_path=os.path.join(results_dir, f"{name_prefix}-robot-motion.mp4"),
        n_frames=min(60, trajectories_final.shape[1]),
    )

    if trajectories_iters is not None and len(trajectories_iters) > 2:
        planner_visualizer.animate_optimization_iterations(
            trajectories=trajectories_iters,
            best_traj_idx=best_traj_idx,
            start_pos=start_pos,
            goal_pos=goal_pos,
            draw_indices=draw_indices,
            draw_spacing=draw_spacing,
            save_path=os.path.join(results_dir, f"{name_prefix}-opt-iters.mp4"),
            n_frames=min(60, len(trajectories_iters)),
        )


def create_test_subset(
    dataset: TrajectoryDataset,
    n_tasks: int,
    threshold_start_goal_pos: float,
    use_extra_objects: bool = False,
) -> Optional[Subset]:
    start_pos, goal_pos, success = dataset.robot.random_collision_free_start_goal(
        env=dataset.env,
        n_samples=n_tasks,
        threshold_start_goal_pos=threshold_start_goal_pos,
        use_extra_objects=use_extra_objects,
    )
    if not success:
        print(
            "Could not find sufficient collision-free start/goal pairs for test tasks, "
            "try reducing the threshold, robot margin or object density"
        )
        return None

    test_dataset = copy(dataset)
    test_dataset.n_trajs = n_tasks
    test_dataset.start_pos = start_pos
    test_dataset.goal_pos = goal_pos
    test_dataset.start_pos_normalized = test_dataset.normalizer.normalize(start_pos)
    test_dataset.goal_pos_normalized = test_dataset.normalizer.normalize(goal_pos)
    return Subset(test_dataset, list(range(n_tasks)))


def run_inference(
    model_config: ModelConfigBase,
    dataset: TrajectoryDataset,
    train_subset: Optional[Subset],
    val_subset: Optional[Subset],
    test_subset: Optional[Subset],
    results_dir: str,
    n_tasks: int,
    n_trajectories_per_task: int,
    draw_spacing: int,
    generate_animation: bool,
    debug: bool,
    tensor_args: Dict[str, Any],
) -> Dict[str, Any]:
    model_wrapper = model_config.prepare(
        dataset=dataset,
        tensor_args=tensor_args,
        n_trajectories_per_task=n_trajectories_per_task,
    )

    results = {}
    stats = {}

    print("=" * 80)
    print(f"Starting trajectory generation for {n_tasks} tasks per split")

    if train_subset is not None:
        print("=" * 80)
        print("Processing TRAIN split...")
        results["train"] = run_inference_on_dataset(
            subset=train_subset,
            n_tasks=n_tasks,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            debug=debug,
        )
        stats["train_stats"] = compute_stats(results["train"])
    if val_subset is not None:
        print("=" * 80)
        print("Processing VAL split...")
        results["val"] = run_inference_on_dataset(
            subset=val_subset,
            n_tasks=n_tasks,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            debug=debug,
        )
        stats["val_stats"] = compute_stats(results["val"])
    if test_subset is not None:
        print("=" * 80)
        print("Processing TEST split...")
        results["test"] = run_inference_on_dataset(
            subset=test_subset,
            n_tasks=n_tasks,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            debug=debug,
        )
        stats["test_stats"] = compute_stats(results["test"])

    print_stats(stats, n_tasks, n_trajectories_per_task, results_dir)

    print("Saving data...")
    with open(os.path.join(results_dir, "results.pickle"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    vis_results = None
    if "test" in results and results["test"].get("sample") is not None:
        vis_results = results["test"]
    elif "val" in results and results["val"].get("sample") is not None:
        vis_results = results["val"]
    elif "train" in results and results["train"].get("sample") is not None:
        vis_results = results["train"]

    if vis_results is not None:
        print("Saving visualization...")
        visualize_results(
            results=vis_results,
            dataset=dataset,
            use_extra_objects=model_config.use_extra_objects,
            results_dir=results_dir,
            draw_spacing=draw_spacing,
            generate_animation=generate_animation,
        )

    return results

import os
import traceback
from typing import Any, Dict, Set, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.autonotebook import tqdm

from mpd_s.dataset.data_transform import (
    NormalizerBase,
    TrivialNormalizer,
    get_data_transforms,
)
from mpd_s.dataset.filtering import get_filter_functions
from mpd_s.planning.costs import (
    CostJointAcceleration,
    CostJointJerk,
    CostJointVelocity,
    CostObstacles,
)
from mpd_s.planning.planners.gpmp2 import GPMP2
from mpd_s.planning.planners.gradient_optimization import GradientOptimization
from mpd_s.planning.planners.hybrid_planner import HybridPlanner
from mpd_s.planning.planners.rrt_connect import RRTConnect
from mpd_s.universe.environments import EnvBase, get_envs
from mpd_s.universe.robot import RobotBase, get_robots
from mpd_s.utils import fix_random_seed, save_config_to_yaml
from mpd_s.visualizer import Visualizer

NORMALIZERS = get_data_transforms()

ENVS = get_envs()
ROBOTS = get_robots()


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        env_name: str,
        robot_name: str,
        robot_margin: float,
        generating_robot_margin: float,
        n_support_points: int,
        duration: float,
        spline_degree: int,
        additional_robot_args: Dict[str, Any],
        tensor_args: Dict[str, Any],
    ):
        self.tensor_args = tensor_args
        self.env_name = env_name
        self.robot_name = robot_name
        self.n_support_points = n_support_points
        self.duration = duration
        self.robot_margin = robot_margin
        self.generating_robot_margin = generating_robot_margin
        self.spline_degree = spline_degree
        self.additional_robot_args = additional_robot_args
        self.env: EnvBase = ENVS[env_name](tensor_args=tensor_args)
        self.robot: RobotBase = ROBOTS[robot_name](
            margin=robot_margin,
            dt=duration / (n_support_points - 1),
            spline_degree=spline_degree,
            tensor_args=tensor_args,
            **additional_robot_args,
        )
        self.generating_robot: RobotBase = ROBOTS[robot_name](
            margin=generating_robot_margin,
            dt=duration / (n_support_points - 1),
            spline_degree=spline_degree,
            tensor_args=tensor_args,
            **additional_robot_args,
        )

        self.apply_augmentations: bool = None
        self.n_control_points: int = None
        self.trajectories: torch.Tensor = None
        self.control_points: torch.Tensor = None
        self.start_pos: torch.Tensor = None
        self.goal_pos: torch.Tensor = None
        self.normalizer: NormalizerBase = None
        self.trajectories_normalized: torch.Tensor = None
        self.control_points_normalized: torch.Tensor = None
        self.start_pos_normalized: torch.Tensor = None
        self.goal_pos_normalized: torch.Tensor = None
        self.n_trajectories_per_task: int = 0

    def filter_raw_data(
        self,
        task_start_idxs: torch.Tensor,
        filtering_config: Dict[str, Any],
    ) -> Set[int]:
        assert len(self.trajectories) > 0, (
            "Trajectories must be loaded before filtering"
        )
        filter_functions = get_filter_functions()
        indices_to_exclude = set()
        for filter_name, filter_params in filtering_config.items():
            if filter_params is None:
                continue
            filter_fn = filter_functions[filter_name]
            excluded = filter_fn(
                trajectories=self.trajectories,
                robot=self.robot,
                env=self.env,
                task_start_idxs=task_start_idxs,
                **filter_params,
            )
            indices_to_exclude.update(excluded)

        return indices_to_exclude

    def load_data(
        self,
        dataset_dir: str,
        apply_augmentations: bool,
        n_control_points: int = None,
        normalizer_name: str = "TrivialNormalizer",
        filtering_config: Dict[str, Any] = {},
        batch_size: int = 1,
        debug: bool = False,
    ) -> Tuple[Subset, DataLoader, Subset, DataLoader]:
        self.apply_augmentations = apply_augmentations
        self.n_control_points = n_control_points
        self.normalizer_name = normalizer_name
        self.filtering_config = filtering_config

        if debug:
            print("Loading trajectories from disk...")
        files = [
            f
            for f in os.listdir(dataset_dir)
            if f.startswith("trajectories_") and f.endswith(".pt")
        ]

        files_with_ids = []
        for f in files:
            try:
                id_ = int(f.split("_")[1].split(".")[0])
                files_with_ids.append((id_, f))
            except Exception:
                pass

        files_with_ids.sort(key=lambda x: x[0])

        trajectories_list = []
        for _, f in files_with_ids:
            t = torch.load(
                os.path.join(dataset_dir, f),
                map_location=self.tensor_args["device"],
            )
            trajectories_list.append(t)

        if not trajectories_list:
            print("No trajectories found!")
            return torch.empty(0)

        trajectories = torch.cat(trajectories_list, dim=0)

        if trajectories.numel() == 0:
            return

        self.n_trajectories, n_support_points, n_dim = trajectories.shape
        assert n_support_points == self.n_support_points and n_dim in (
            self.robot.n_dim,
            2 * self.robot.n_dim,
        )

        self.trajectories = trajectories[..., : self.robot.n_dim]
        self.start_pos = self.trajectories[..., 0, : self.robot.n_dim]
        self.goal_pos = self.trajectories[..., -1, : self.robot.n_dim]

        self.normalizer: NormalizerBase = NORMALIZERS[self.normalizer_name]()
        if self.n_control_points is not None:
            print("Fitting B-Splines to trajectories...")
            self.control_points = self.robot.fit_bsplines_to_trajectories(
                trajectories=self.trajectories,
                n_control_points=self.n_control_points,
            )
            self.normalizer.fit(self.control_points)
            self.control_points_normalized = self.normalizer.normalize(
                self.control_points
            )
        else:
            self.normalizer.fit(self.trajectories)
            self.trajectories_normalized = self.normalizer.normalize(self.trajectories)

        self.start_pos_normalized = self.normalizer.normalize(self.start_pos)
        self.goal_pos_normalized = self.normalizer.normalize(self.goal_pos)

        train_idx = torch.load(os.path.join(dataset_dir, "train_idx.pt"))
        val_idx = torch.load(os.path.join(dataset_dir, "val_idx.pt"))

        task_start_idxs = torch.load(os.path.join(dataset_dir, "task_start_idxs.pt"))

        indices_to_exclude = self.filter_raw_data(
            task_start_idxs=task_start_idxs, filtering_config=self.filtering_config
        )

        train_idx = torch.tensor(
            [idx for idx in train_idx.tolist() if idx not in indices_to_exclude][
                : len(train_idx) // batch_size * batch_size
            ]
        )
        val_idx = torch.tensor(
            [idx for idx in val_idx.tolist() if idx not in indices_to_exclude][
                : len(val_idx) // batch_size * batch_size
            ]
        )

        train_subset = Subset(self, train_idx)
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_subset = Subset(self, val_idx)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size)

        return train_subset, train_dataloader, val_subset, val_dataloader

    def __len__(self) -> int:
        return self.n_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_pos_normalized = self.start_pos_normalized[idx]
        goal_pos_normalized = self.goal_pos_normalized[idx]
        start_pos = self.start_pos[idx]
        goal_pos = self.goal_pos[idx]

        if self.n_control_points is not None:
            x = self.control_points_normalized[idx]
        else:
            x = self.trajectories_normalized[idx]

        if self.apply_augmentations and torch.rand(1).item() < 0.5:
            x = self.robot.invert_trajectories(x)
            start_pos_normalized, goal_pos_normalized = (
                goal_pos_normalized,
                start_pos_normalized,
            )
            start_pos, goal_pos = goal_pos, start_pos

        data = {
            "x": x,
            "start_pos_normalized": start_pos_normalized,
            "goal_pos_normalized": goal_pos_normalized,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
        }

        return data

    @staticmethod
    def _worker_process_task(args) -> Tuple[int, int, int, str]:
        (
            task_id,
            dataset_dir,
            env_name,
            robot_name,
            robot_margin,
            generating_robot_margin,
            n_support_points,
            duration,
            spline_degree,
            additional_robot_args,
            n_trajectories_per_task,
            threshold_start_goal_pos,
            n_sampling_steps,
            n_optimization_steps,
            smoothen,
            create_straight_line_trajectories,
            use_gpmp2,
            n_control_points,
            rrt_connect_max_radius,
            rrt_connect_n_points,
            rrt_connect_n_samples,
            gpmp2_n_interpolate,
            gpmp2_sigma_start,
            gpmp2_sigma_goal_prior,
            gpmp2_sigma_gp,
            gpmp2_sigma_collision,
            gpmp2_step_size,
            gpmp2_delta,
            gpmp2_method,
            grad_lambda_obstacles,
            grad_lambda_velocity,
            grad_lambda_acceleration,
            grad_lambda_jerk,
            grad_max_grad_norm,
            grad_n_interpolate,
            grid_map_sdf_fixed,
            grid_map_sdf_extra,
            tensor_args,
            seed,
            debug,
        ) = args

        try:
            env: EnvBase = ENVS[env_name](
                tensor_args=tensor_args,
                grid_map_sdf_fixed=grid_map_sdf_fixed,
                grid_map_sdf_extra=grid_map_sdf_extra,
            )
            robot: RobotBase = ROBOTS[robot_name](
                margin=robot_margin,
                dt=duration / (n_support_points - 1),
                spline_degree=spline_degree,
                tensor_args=tensor_args,
                **additional_robot_args,
            )
            generating_robot: RobotBase = ROBOTS[robot_name](
                margin=generating_robot_margin,
                dt=duration / (n_support_points - 1),
                spline_degree=spline_degree,
                tensor_args=tensor_args,
                **additional_robot_args,
            )
            normalizer = TrivialNormalizer()

            sampling_based_planner = RRTConnect(
                env=env,
                robot=generating_robot,
                n_trajectories=n_trajectories_per_task,
                max_radius=rrt_connect_max_radius,
                n_points=rrt_connect_n_points,
                n_samples=rrt_connect_n_samples,
                use_extra_objects=False,
                tensor_args=tensor_args,
            )
            if use_gpmp2:
                optimization_based_planner = GPMP2(
                    env=env,
                    robot=generating_robot,
                    n_dim=robot.n_dim,
                    n_support_points=n_support_points,
                    dt=generating_robot.dt,
                    n_interpolate=gpmp2_n_interpolate,
                    sigma_start=gpmp2_sigma_start,
                    sigma_gp=gpmp2_sigma_gp,
                    sigma_goal_prior=gpmp2_sigma_goal_prior,
                    sigma_collision=gpmp2_sigma_collision,
                    step_size=gpmp2_step_size,
                    delta=gpmp2_delta,
                    method=gpmp2_method,
                    use_extra_objects=False,
                    tensor_args=tensor_args,
                )
            else:
                collision_cost = None
                if grad_lambda_obstacles is not None:
                    collision_cost = CostObstacles(
                        robot=robot,
                        env=env,
                        n_support_points=n_support_points,
                        lambda_obstacles=grad_lambda_obstacles,
                        use_extra_objects=False,
                        tensor_args=tensor_args,
                    )

                velocity_cost = None
                if grad_lambda_velocity is not None:
                    velocity_cost = CostJointVelocity(
                        robot=robot,
                        n_support_points=n_support_points,
                        lambda_velocity=grad_lambda_velocity,
                        tensor_args=tensor_args,
                    )

                acceleration_cost = None
                if grad_lambda_acceleration is not None:
                    acceleration_cost = CostJointAcceleration(
                        robot=robot,
                        n_support_points=n_support_points,
                        lambda_acceleration=grad_lambda_acceleration,
                        tensor_args=tensor_args,
                    )

                jerk_cost = None
                if grad_lambda_jerk is not None:
                    jerk_cost = CostJointJerk(
                        robot=robot,
                        n_support_points=n_support_points,
                        lambda_jerk=grad_lambda_jerk,
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
                    env=env,
                    robot=generating_robot,
                    normalizer=normalizer,
                    n_support_points=n_support_points,
                    n_control_points=n_control_points,
                    costs=costs,
                    max_grad_norm=grad_max_grad_norm,
                    n_interpolate=grad_n_interpolate,
                    tensor_args=tensor_args,
                    use_extra_objects=False,
                )

            planner = HybridPlanner(
                sampling_based_planner=sampling_based_planner,
                optimization_based_planner=optimization_based_planner,
                smoothen=smoothen,
                create_straight_line_trajectories=create_straight_line_trajectories,
                n_trajectories=n_trajectories_per_task,
                n_support_points=n_support_points,
                n_control_points=n_control_points,
                tensor_args=tensor_args,
            )

            fix_random_seed(seed + task_id)

            start_pos, goal_pos, success = (
                generating_robot.random_collision_free_start_goal(
                    env=env,
                    n_samples=1,
                    threshold_start_goal_pos=threshold_start_goal_pos,
                )
            )

            if not success:
                return task_id, 0, 0, "failed_start_goal"

            start_pos = start_pos.squeeze(0)
            goal_pos = goal_pos.squeeze(0)

            trajectories = planner.optimize(
                start_pos=start_pos,
                goal_pos=goal_pos,
                n_sampling_steps=n_sampling_steps,
                n_optimization_steps=n_optimization_steps,
                debug=debug,
            )
            _, trajectories_free, _ = robot.get_trajectories_collision_and_free(
                env=env, trajectories=trajectories
            )
            n_free = len(trajectories_free)
            n_collision = len(trajectories) - n_free

            torch.save(
                trajectories.cpu(),
                os.path.join(dataset_dir, f"trajectories_{task_id}.pt"),
            )

            if debug:
                print(f"Task {task_id} - Free trajectories: {n_free}")
                planning_visualizer = Visualizer(
                    env=env, robot=robot, use_extra_objects=False
                )
                try:
                    print("Saving visualization...")
                    planning_visualizer.render_scene(
                        trajectories=trajectories,
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        save_path=os.path.join(
                            dataset_dir, f"trajectories_figure_{task_id}.png"
                        ),
                        draw_indices=[0],
                    )
                except Exception as e:
                    print(f"Visualization failed for task {task_id}: {e}")

            return task_id, n_collision, n_free, "success"

        except Exception as e:
            print(f"Task {task_id} failed with error: {e}")
            traceback.print_exc()
            return task_id, 0, 0, str(e)

    def generate_data(
        self,
        dataset_dir: str,
        n_tasks: int,
        n_trajectories_per_task: int,
        threshold_start_goal_pos: float,
        n_sampling_steps: int,
        n_optimization_steps: int,
        smoothen: bool,
        create_straight_line_trajectories: bool,
        use_gpmp2: bool,
        n_control_points: int,
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
        val_portion: float,
        n_processes: int,
        seed: int,
        debug: bool,
    ) -> None:
        os.makedirs(dataset_dir, exist_ok=True)

        init_config: dict = {
            "env_name": self.env_name,
            "robot_name": self.robot_name,
            "robot_margin": self.robot_margin,
            "generating_robot_margin": self.generating_robot_margin,
            "n_support_points": self.n_support_points,
            "duration": self.duration,
            "spline_degree": self.spline_degree,
            "additional_robot_args": self.additional_robot_args,
        }

        info_config = locals().copy()
        del info_config["self"]
        del info_config["dataset_dir"]

        save_config_to_yaml(init_config, os.path.join(dataset_dir, "init_config.yaml"))
        save_config_to_yaml(info_config, os.path.join(dataset_dir, "info_config.yaml"))

        print("Checking for existing data...")
        existing_tasks = {}
        if not os.path.exists(dataset_dir):
            return existing_tasks

        files = [
            f
            for f in os.listdir(dataset_dir)
            if f.startswith("trajectories_") and f.endswith(".pt")
        ]

        for f in files:
            try:
                task_id = int(f.split("_")[1].split(".")[0])
                trajectories = torch.load(
                    os.path.join(dataset_dir, f), map_location="cpu"
                )
                existing_tasks[task_id] = len(trajectories)
            except Exception:
                pass

        if existing_tasks:
            print(f"Found {len(existing_tasks)} existing tasks, will resume from there")

        task_start_idxs = []

        grid_map_sdf_fixed = self.env.grid_map_sdf_fixed
        grid_map_sdf_extra = self.env.grid_map_sdf_extra

        tasks_to_run = []
        n_skipped_tasks = 0
        for i in range(n_tasks):
            if i in existing_tasks:
                n_skipped_tasks += 1
                continue

            task_args = (
                i,
                dataset_dir,
                self.env_name,
                self.robot_name,
                self.robot_margin,
                self.generating_robot_margin,
                self.n_support_points,
                self.duration,
                self.spline_degree,
                self.additional_robot_args,
                n_trajectories_per_task,
                threshold_start_goal_pos,
                n_sampling_steps,
                n_optimization_steps,
                smoothen,
                create_straight_line_trajectories,
                use_gpmp2,
                n_control_points,
                rrt_connect_max_radius,
                rrt_connect_n_points,
                rrt_connect_n_samples,
                gpmp2_n_interpolate,
                gpmp2_sigma_start,
                gpmp2_sigma_goal_prior,
                gpmp2_sigma_gp,
                gpmp2_sigma_collision,
                gpmp2_step_size,
                gpmp2_delta,
                gpmp2_method,
                grad_lambda_obstacles,
                grad_lambda_velocity,
                grad_lambda_acceleration,
                grad_lambda_jerk,
                grad_max_grad_norm,
                grad_n_interpolate,
                grid_map_sdf_fixed,
                grid_map_sdf_extra,
                self.tensor_args,
                seed,
                debug,
            )
            tasks_to_run.append(task_args)

        print(f"{'=' * 80}")
        print(
            f"Starting trajectory generation for {n_tasks} tasks ({len(tasks_to_run)} new, {n_skipped_tasks} existing)"
        )
        print(f"Using {n_processes} processes")
        print(f"{'=' * 80}\n")

        ctx = mp.get_context("spawn")

        new_tasks = {}
        n_completed_tasks = 0
        n_failed_tasks = 0
        with tqdm(total=len(tasks_to_run), desc="Generating data") as pbar:
            pbar.update(n_skipped_tasks)

            if n_processes > 1:
                with ctx.Pool(processes=n_processes) as pool:
                    for res in pool.imap_unordered(
                        self._worker_process_task, tasks_to_run
                    ):
                        n_completed_tasks += 1
                        task_id, n_collision, n_free, status = res
                        new_tasks[task_id] = (n_collision + n_free, status)

                        if status != "success" or n_free == 0:
                            n_failed_tasks += 1

                        if (
                            n_failed_tasks > 10
                            and n_failed_tasks > n_completed_tasks * 0.1
                        ):
                            raise RuntimeError(
                                f"Too many tasks with 0 free trajectories ({n_failed_tasks}/{pbar.n})"
                            )

                        pbar.set_postfix(
                            {
                                "status": status,
                                "collision": n_collision,
                                "free": n_free,
                                "failed": n_failed_tasks,
                            }
                        )
                        pbar.update(1)

            else:
                for args in tasks_to_run:
                    res = self._worker_process_task(args)
                    n_completed_tasks += 1
                    task_id, n_collision, n_free, status = res
                    new_tasks[task_id] = (n_collision + n_free, status)

                    if status != "success" or n_free == 0:
                        n_failed_tasks += 1

                    if n_failed_tasks > 10 and n_failed_tasks > n_completed_tasks * 0.1:
                        raise RuntimeError(
                            f"Too many tasks with 0 free trajectories ({n_failed_tasks}/{pbar.n})"
                        )

                    pbar.set_postfix(
                        {
                            "status": status,
                            "collision": n_collision,
                            "free": n_free,
                            "failed": n_failed_tasks,
                        }
                    )
                    pbar.update(1)

        n_trajectories_total = 0
        for i in range(n_tasks):
            n_trajectories_i = 0
            if i in existing_tasks:
                n_trajectories_i = existing_tasks[i]
            elif i in new_tasks:
                n_trajectories_i, status = new_tasks[i]
                if status != "success":
                    n_trajectories_i = 0

            task_start_idxs.append(n_trajectories_total)
            n_trajectories_total += n_trajectories_i

        task_start_idxs.append(n_trajectories_total)

        task_start_idxs = torch.tensor(task_start_idxs, dtype=torch.long)
        train_tasks, val_tasks = random_split(
            range(n_tasks), [1 - val_portion, val_portion]
        )
        train_tasks_idxs = torch.tensor(train_tasks.indices)
        val_tasks_idxs = torch.tensor(val_tasks.indices)
        train_idxs = torch.tensor(
            [
                i
                for task_start_idx, task_end_idx in zip(
                    task_start_idxs[train_tasks_idxs],
                    task_start_idxs[train_tasks_idxs + 1],
                )
                for i in range(task_start_idx, task_end_idx)
            ]
        )
        val_idxs = torch.tensor(
            [
                i
                for task_start_idx, task_end_idx in zip(
                    task_start_idxs[val_tasks_idxs], task_start_idxs[val_tasks_idxs + 1]
                )
                for i in range(task_start_idx, task_end_idx)
            ]
        )

        torch.save(train_idxs, os.path.join(dataset_dir, "train_idx.pt"))
        torch.save(val_idxs, os.path.join(dataset_dir, "val_idx.pt"))
        torch.save(task_start_idxs, os.path.join(dataset_dir, "task_start_idxs.pt"))

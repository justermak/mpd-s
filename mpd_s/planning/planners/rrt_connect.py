from typing import Any, Dict, Optional

import torch

from mpd_s.planning.planners.classical_planner import ClassicalPlanner
from mpd_s.torch_timer import TimerCUDA
from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase


class RRTConnect(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
        n_trajectories: int,
        max_radius: float,
        n_points: float,
        n_samples: int,
        use_extra_objects: bool,
        tensor_args: Dict[str, Any],
        eps: float = 1e-6,
    ):
        super().__init__(
            name="RRTConnect",
            env=env,
            robot=robot,
            use_extra_objects=use_extra_objects,
            tensor_args=tensor_args,
        )
        self.n_points = n_points
        self.max_radius = max_radius
        self.n_samples = n_samples
        self.n_trajectories = n_trajectories
        self.n_trajectories_with_luft = int(1.1 * n_trajectories)
        self.use_extra_objects = use_extra_objects
        self.eps = eps

        self.samples: torch.Tensor = None
        ok = self._initialize_samples()
        if not ok:
            raise RuntimeError("Failed to initialize RRTConnect samples")

    def _initialize_samples(self) -> bool:
        samples, success = self.robot.random_collision_free_points(
            env=self.env,
            n_samples=self.n_samples,
            use_extra_objects=self.use_extra_objects,
        )
        if not success:
            print(
                "Could not find sufficient collision-free start/goal pairs for RRTConnect samples"
            )
        self.samples = samples
        return success

    def sample(self, n_samples) -> torch.Tensor:
        if self.samples_ptr + n_samples > len(self.samples):
            self._initialize_samples()
            self.samples_ptr = 0

        samples = self.samples[self.samples_ptr : self.samples_ptr + n_samples]
        self.samples_ptr += n_samples
        return samples

    def extend_and_cut(
        self,
        points: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        diff = targets - points
        dist = torch.linalg.norm(diff, dim=-1)
        direction = diff / (dist.unsqueeze(-1) + 1e-8)

        targets = points + direction * torch.minimum(
            dist, torch.tensor(self.max_radius, device=dist.device)
        ).unsqueeze(-1)

        targets = self.robot.enforce_rigid_constraints(targets)

        paths = self.robot.create_straight_line_trajectories(
            points, targets, n_support_points=self.n_points
        )

        in_collision = self.robot.get_collision_mask(
            env=self.env, points=paths, on_extra=self.use_extra_objects
        )

        has_collision = torch.any(in_collision, dim=-1)
        first_collision_idx = in_collision.float().argmax(dim=-1) - 1
        idx = torch.where(has_collision, first_collision_idx, paths.shape[1] - 1)

        res = paths[torch.arange(paths.shape[0]), idx]

        return res

    def optimize(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        n_sampling_steps: int,
        print_freq: int = 50,
        debug: bool = True,
    ) -> Optional[torch.Tensor]:
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.dim = start_pos.shape[-1]
        self.samples_ptr = 0
        self.n_sampling_steps = n_sampling_steps
        step = 0

        max_nodes = self.n_sampling_steps + 1

        tree_nodes = torch.zeros(
            (2, self.n_trajectories_with_luft, max_nodes, self.dim), **self.tensor_args
        )

        tree_nodes[0, :, 0, :] = start_pos
        tree_nodes[1, :, 0, :] = goal_pos

        self.tree_nodes = tree_nodes

        tree_parents = torch.zeros(
            (2, self.n_trajectories_with_luft, max_nodes),
            dtype=torch.long,
            device=start_pos.device,
        )

        tree_parents[0, :, 0] = -1
        tree_parents[1, :, 0] = -1

        self.tree_parents = tree_parents

        tree_counts = torch.ones(
            (2, self.n_trajectories_with_luft),
            dtype=torch.long,
            device=start_pos.device,
        )

        self.tree_counts = tree_counts

        active_mask = torch.ones(
            self.n_trajectories_with_luft, dtype=torch.bool, device=start_pos.device
        )

        cur = 1

        with TimerCUDA() as t:
            for step in range(1, n_sampling_steps + 1):
                cur = 1 - cur
                n_active = active_mask.sum().item()
                if self.n_trajectories_with_luft - n_active >= self.n_trajectories:
                    break

                if debug and step % print_freq == 0:
                    total_nodes = tree_counts.sum().item()
                    pad = len(str(self.n_sampling_steps))
                    print(
                        f"| Step: {step:>{pad}}/{self.n_sampling_steps:>{pad}} "
                        f"| Time: {t.elapsed:.3f}s "
                        f"| Nodes: {total_nodes} "
                        f"| Success: {self.n_trajectories_with_luft - n_active}/{self.n_trajectories}"
                    )

                max_len = tree_counts[cur].max().item()
                targets = self.sample(n_active)
                idx = torch.arange(max_len, device=targets.device).expand(n_active, -1)
                real_idx = torch.arange(
                    self.n_trajectories_with_luft, device=targets.device
                )[active_mask]

                cur_nodes = tree_nodes[cur, active_mask, :max_len, :]
                cur_counts = tree_counts[cur, active_mask]

                mask = idx < cur_counts.unsqueeze(-1)
                dists = torch.linalg.norm(cur_nodes - targets.unsqueeze(-2), dim=-1)
                dists_masked = torch.where(
                    mask, dists, torch.tensor(float("inf"), device=dists.device)
                )

                nearest_idxs = torch.min(dists_masked, dim=-1)[1]
                nearest_nodes = cur_nodes[torch.arange(n_active), nearest_idxs]
                new_nodes = self.extend_and_cut(
                    nearest_nodes,
                    targets,
                )

                upd_mask = (
                    torch.linalg.norm(new_nodes - nearest_nodes, dim=-1) > self.eps
                )

                row_idx = real_idx[upd_mask]
                col_idx = cur_counts[upd_mask]

                if len(row_idx) > 0:
                    tree_nodes[cur, row_idx, col_idx] = new_nodes[upd_mask]
                    tree_parents[cur, row_idx, col_idx] = nearest_idxs[upd_mask]
                    tree_counts[cur, row_idx] += 1

                max_len = tree_counts[1 - cur].max().item()
                targets2 = new_nodes[upd_mask]
                candidates_mask = active_mask.clone()
                candidates_mask[real_idx] = upd_mask
                n_candidates = candidates_mask.sum().item()

                if n_candidates == 0:
                    continue

                idx = torch.arange(max_len, device=targets.device).expand(
                    n_candidates, -1
                )
                real_idx = torch.arange(
                    self.n_trajectories_with_luft, device=targets.device
                )[candidates_mask]

                other_nodes = tree_nodes[1 - cur, candidates_mask, :max_len, :]
                other_counts = tree_counts[1 - cur, candidates_mask]

                mask = idx < other_counts.unsqueeze(-1)
                dists = torch.linalg.norm(other_nodes - targets2.unsqueeze(-2), dim=-1)
                dists_masked = torch.where(
                    mask, dists, torch.tensor(float("inf"), device=dists.device)
                )

                nearest_idxs = torch.min(dists_masked, dim=-1)[1]
                nearest_nodes = other_nodes[torch.arange(n_candidates), nearest_idxs]
                nodes_new_other = self.extend_and_cut(
                    nearest_nodes,
                    targets2,
                )

                upd_mask = (
                    torch.linalg.norm(nodes_new_other - nearest_nodes, dim=-1)
                    > self.eps
                )

                row_idx = real_idx[upd_mask]
                col_idx = other_counts[upd_mask]

                if len(row_idx) > 0:
                    tree_nodes[1 - cur, row_idx, col_idx] = nodes_new_other[upd_mask]
                    tree_parents[1 - cur, row_idx, col_idx] = nearest_idxs[upd_mask]
                    tree_counts[1 - cur, row_idx] += 1

                active_mask[real_idx] = (
                    torch.linalg.norm(nodes_new_other - targets2, dim=-1) > self.eps
                )

            trajectories = []

            tree_nodes = tree_nodes.cpu()
            tree_counts = tree_counts.cpu()
            tree_parents = tree_parents.cpu()
            success_mask = (~active_mask).cpu()

            cnt = 0
            for i in torch.where(success_mask)[0]:
                path1 = []
                cur = tree_counts[0, i].item() - 1
                while cur != -1:
                    path1.append(tree_nodes[0, i, cur])
                    cur = tree_parents[0, i, cur].item()
                path1 = path1[::-1]

                path2 = []
                cur = tree_counts[1, i].item() - 1
                while cur != -1:
                    path2.append(tree_nodes[1, i, cur])
                    cur = tree_parents[1, i, cur].item()
                if torch.allclose(path1[-1], path2[0], atol=self.eps):
                    path2 = path2[1:]
                trajectories.append(torch.stack(path1 + path2).to(**self.tensor_args))
                cnt += 1
                if cnt >= self.n_trajectories:
                    break

        return trajectories

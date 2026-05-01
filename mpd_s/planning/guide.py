from typing import List

import torch

from mpd_s.dataset.dataset import TrajectoryDataset
from mpd_s.planning.costs import Cost


class Guide:
    def __init__(
        self,
        dataset: TrajectoryDataset,
        costs: List[Cost],
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        self.dataset = dataset
        self.costs = costs
        self.n_interpolate = n_interpolate
        self.max_grad_norm = max_grad_norm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        trajectories_normalized = x.clone()

        with torch.enable_grad():
            trajectories_normalized.requires_grad_(True)

            trajectories = self.dataset.normalizer.unnormalize(trajectories_normalized)
            if self.dataset.n_control_points is not None:
                trajectories_pos = self.dataset.robot.get_trajectories_from_bsplines(
                    trajectories=trajectories,
                    n_support_points=self.dataset.n_support_points,
                )
            else:
                trajectories_pos = self.dataset.robot.get_position(
                    trajectories=trajectories,
                )
            cost = sum(
                cost(
                    trajectories=trajectories_pos,
                    n_interpolate=self.n_interpolate,
                ).sum()
                for cost in self.costs
            )

            grad = torch.autograd.grad(cost, trajectories_normalized)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

            n_fixed = 2 if self.dataset.n_control_points is not None else 1
            grad[..., :n_fixed, :] = 0.0
            grad[..., -n_fixed:, :] = 0.0

        return -grad

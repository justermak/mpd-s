from typing import Any, Dict

import torch
from torch.autograd.functional import jacobian

from mpd_s.universe.primitives import ObjectField


class GridMapSDF:
    def __init__(
        self,
        limits: torch.Tensor,
        cell_size: float,
        obj_field: ObjectField,
        tensor_args: Dict[str, Any],
    ) -> None:
        self.limits = limits
        self.tensor_args = tensor_args
        self.obj_field = obj_field

        self.workspace_size = torch.abs(limits[1] - limits[0])
        self.cell_size = cell_size
        self.grid_resolution = torch.ceil(self.workspace_size / cell_size).long()
        basis_ranges = [
            torch.linspace(
                self.limits[0][i],
                self.limits[1][i],
                self.grid_resolution[i],
                **self.tensor_args,
            )
            for i in range(limits.shape[1])
        ]

        points_for_sdf_meshgrid = torch.meshgrid(*basis_ranges, indexing="ij")
        self.points_for_sdf = torch.stack(points_for_sdf_meshgrid, dim=-1)

        def grad_fn(x):
            return self.obj_field.compute_signed_distance(x).sum(
                dim=tuple(range(x.dim() - 1))
            )

        sdf_batches = []
        grad_batches = []
        batch_size = 64

        for i in range(0, self.points_for_sdf.shape[0], batch_size):
            torch.cuda.empty_cache()
            points_batch = self.points_for_sdf[i : i + batch_size]
            sdf_batches.append(self.obj_field.compute_signed_distance(points_batch))
            grad_batches.append(jacobian(grad_fn, points_batch).permute(1, 2, 0, 3))

        torch.cuda.empty_cache()
        self.sdf_tensor = torch.cat(sdf_batches, dim=0)
        self.grad_sdf_tensor = torch.cat(grad_batches, dim=0)

    def compute_approx_signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        x_idx = (
            ((x - self.limits[0]) / self.workspace_size * self.grid_resolution)
            .round()
            .long()
        )

        max_idx = torch.tensor(self.points_for_sdf.shape[:-1], device=x.device) - 1
        x_idx = x_idx.clamp(torch.zeros_like(max_idx), max_idx)
        x_query = tuple(x_idx[..., i] for i in range(self.limits.shape[1]))

        sdf_vals = self.sdf_tensor[x_query]
        grad_sdf = self.grad_sdf_tensor[x_query]

        grid_point = self.points_for_sdf[x_query]
        sdf_vals = sdf_vals + ((x - grid_point).unsqueeze(-2) * grad_sdf).sum(-1)
        return sdf_vals

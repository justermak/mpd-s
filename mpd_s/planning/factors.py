from typing import Any, Dict, Tuple

import torch

from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase


class UnaryFactor:
    def __init__(
        self,
        dim: int,
        mean: torch.Tensor,
        tensor_args: Dict[str, Any] = None,
    ):
        self.dim = dim
        self.mean = mean
        self.tensor_args = tensor_args

    def get_error(
        self, x: torch.Tensor, calc_jacobian: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        error = self.mean - x

        if calc_jacobian:
            H = torch.eye(self.dim, **self.tensor_args).repeat(x.shape[0], 1, 1)
            return error.view(x.shape[0], self.dim, 1), H
        else:
            return error


class FieldFactor:
    def __init__(
        self,
        n_dim: int,
        sigma: float,
        use_extra_objects: bool = False,
    ) -> None:
        self.sigma = sigma
        self.n_dim = n_dim
        self.use_extra_objects = use_extra_objects

    def get_error(
        self,
        trajectories: torch.Tensor,
        env: EnvBase,
        robot: RobotBase,
        n_interpolate: int,
        calc_jacobian: bool = False,
        return_full_error: bool = True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        trajectories_interpolated = robot.linearly_interpolate_trajectories(
            trajectories, n_interpolate=n_interpolate
        )
        error_interpolated = robot.compute_cost(
            env=env,
            trajectories=trajectories_interpolated[:, 1:, :],
            on_extra=self.use_extra_objects,
        )
        error = (
            error_interpolated
            if return_full_error
            else robot.compute_cost(
                env=env,
                trajectories=trajectories[:, 1:, :],
                on_extra=self.use_extra_objects,
            )
        )

        if calc_jacobian:
            H = -torch.autograd.grad(
                error_interpolated.sum(), trajectories, retain_graph=True
            )[0][:, 1:, : self.n_dim]
            return error, H

        return error


class GPFactor:
    def __init__(
        self,
        dim: int,
        sigma: float,
        dt: float,
        num_factors: int,
        tensor_args: Dict[str, Any],
    ):
        self.dim = dim
        self.dt = dt
        self.tensor_args = tensor_args
        self.state_dim = self.dim * 2
        self.num_factors = num_factors
        self.idx1 = torch.arange(0, self.num_factors, device=tensor_args["device"])
        self.idx2 = torch.arange(1, self.num_factors + 1, device=tensor_args["device"])
        self.phi = self.calc_phi()
        Q_c_inv = torch.eye(dim, **tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(num_factors, dim, dim, **tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()  # shape: [num_factors, state_dim, state_dim]

        self.H1 = self.phi.repeat(self.num_factors, 1, 1)
        self.H2 = -1.0 * torch.eye(self.state_dim, **self.tensor_args).repeat(
            self.num_factors, 1, 1
        )

    def calc_phi(self):
        I = torch.eye(self.dim, **self.tensor_args)
        Z = torch.zeros(self.dim, self.dim, **self.tensor_args)
        phi_u = torch.cat((I, self.dt * I), dim=1)
        phi_l = torch.cat((Z, I), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi

    def calc_Q_inv(self):
        m1 = 12.0 * (self.dt**-3.0) * self.Q_c_inv
        m2 = -6.0 * (self.dt**-2.0) * self.Q_c_inv
        m3 = 4.0 * (self.dt**-1.0) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def get_error(self, x_traj, calc_jacobian=True):
        state_1 = torch.index_select(x_traj, 1, self.idx1).unsqueeze(-1)
        state_2 = torch.index_select(x_traj, 1, self.idx2).unsqueeze(-1)
        error = state_2 - self.phi @ state_1

        if calc_jacobian:
            H1 = self.H1
            H2 = self.H2
            return error, H1, H2
        else:
            return error

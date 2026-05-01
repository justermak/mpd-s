from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpd_s.dataset.dataset import TrajectoryDataset
from mpd_s.model.temporal_unet import TemporalUNet, TemporalUNetShortcut
from mpd_s.planning.planners.gradient_optimization import GradientOptimization


def get_models():
    return {
        "Diffusion": Diffusion,
        "DiffusionShortcut": DiffusionShortcut,
        "FlowMatchingShortcut": FlowMatchingShortcut,
        "Drift": Drift,
    }


def get_additional_init_args(model_name, args):
    additional_args = {}
    if model_name in ("Diffusion", "DiffusionShortcut"):
        additional_args["n_diffusion_steps"] = args["n_diffusion_steps"]
        additional_args["predict_noise"] = args["predict_noise"]
    if model_name in ("DiffusionShortcut", "FlowMatchingShortcut"):
        additional_args["n_diffusion_steps"] = args["n_diffusion_steps"]
        additional_args["bootstrap_fraction"] = args["bootstrap_fraction"]
        additional_args["dt_sampling_strategy"] = args["dt_sampling_strategy"]
    if model_name == "Drift":
        additional_args["temperature"] = args["temperature"]
    return additional_args


def get_additional_inference_args(model_name, args):
    additional_args = {}
    if model_name in ("Diffusion", "DiffusionShortcut", "FlowMatchingShortcut"):
        additional_args["n_inference_steps"] = args["n_inference_steps"]
    if model_name == "Diffusion":
        additional_args["eta"] = args["eta"]
    return additional_args


def cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999):
    trajectories = torch.linspace(0, n_diffusion_steps, n_diffusion_steps + 1)
    alphas_cumprod = (
        torch.cos(((trajectories / n_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5)
        ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = torch.clamp(betas, min=a_min, max=a_max)
    return betas_clipped


class GenerativeModel(nn.Module, ABC):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        cfg_fraction: float,
        cfg_scale: float,
    ):
        super().__init__()
        self.dataset = dataset
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.dim_mults = dim_mults if isinstance(dim_mults, tuple) else eval(dim_mults)
        self.kernel_size = kernel_size
        self.resnet_block_groups = resnet_block_groups
        self.positional_encoding = positional_encoding
        self.positional_encoding_dim = positional_encoding_dim
        self.attn_heads = attn_heads
        self.attn_head_dim = attn_head_dim
        self.context_dim = context_dim
        self.cfg_fraction = cfg_fraction
        self.cfg_scale = cfg_scale

        self.model = TemporalUNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )

    def build_context(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = torch.cat(
            [
                input_dict["start_pos_normalized"].view(-1, self.context_dim // 2),
                input_dict["goal_pos_normalized"].view(-1, self.context_dim // 2),
            ],
            dim=-1,
        )
        return context

    def get_model_prediction(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        use_cfg: bool = True,
        **conditioning,
    ) -> torch.Tensor:
        if use_cfg and self.cfg_scale != 1.0:
            model_output_cond = self.model(x_t, **conditioning, context=context)
            model_output_uncond = self.model(
                x_t, **conditioning, context=torch.zeros_like(context)
            )
            model_output = model_output_uncond + self.cfg_scale * (
                model_output_cond - model_output_uncond
            )
        else:
            model_output = self.model(x_t, **conditioning, context=context)
        return model_output

    @abstractmethod
    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        cfg_scale: float,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        pass


class Diffusion(GenerativeModel):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        cfg_fraction: float,
        cfg_scale: float,
        n_diffusion_steps: int,
        predict_noise: bool,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
            cfg_fraction=cfg_fraction,
            cfg_scale=cfg_scale,
        )
        self.n_diffusion_steps = n_diffusion_steps
        self.predict_noise = predict_noise

        betas = cosine_beta_schedule(n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def extract(
        self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(shape) - 1)))

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

        return sample

    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = input_dict["x"]
        device = x.device
        context = self.build_context(input_dict)
        batch_size, horizon, dim = x.shape

        if self.cfg_fraction is not None:
            mask = torch.rand(batch_size, device=device) < self.cfg_fraction
            context[mask] = 0

        t = torch.randint(
            0,
            self.n_diffusion_steps,
            (batch_size,),
            device=device,
        ).long()
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x_0=x, t=t, noise=noise)

        x_recon = self.get_model_prediction(x_noisy, context, t=t)

        if self.predict_noise:
            loss = F.mse_loss(x_recon, noise)
        else:
            loss = F.mse_loss(x_recon, x)

        loss_dict = {"loss": loss}

        return loss_dict

    @torch.no_grad()
    def ddpm_sample(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
    ) -> torch.Tensor:
        device = self.betas.device

        x_t = torch.randn((n_samples, self.horizon, self.state_dim), device=device)

        chain = [x_t]

        for time in reversed(range(self.n_diffusion_steps)):
            t = torch.full((n_samples,), time, device=device, dtype=torch.long)

            model_prediction = self.get_model_prediction(x_t, context, t=t)

            if self.predict_noise:
                x_recon = (
                    self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                    - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                    * model_prediction
                )
            else:
                x_recon = model_prediction

            x_recon = torch.clamp(x_recon, -4.0, 4.0)

            x_t = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_recon
                + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )

            log_variance = self.extract(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
            std = torch.exp(0.5 * log_variance)

            if (
                guide is not None
                and t_start_guide is not None
                and time <= t_start_guide
            ):
                x_t = guide.optimize(x_t, n_optimization_steps=n_guide_steps)

            noise = torch.randn_like(x_t)
            noise[t == 0] = 0

            x_t = x_t + std * noise

            chain.append(x_t)

        chain = torch.stack(chain, dim=0)
        return chain

    @torch.no_grad()
    def ddim_sample(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        eta: float,
        debug: bool = False,
    ) -> torch.Tensor:
        device = self.betas.device

        x_t = torch.randn((n_samples, self.horizon, self.state_dim), device=device)

        step_indices = (
            torch.linspace(self.n_diffusion_steps - 1, 0, n_inference_steps)
            .long()
            .flip(0)
            .to(device)
        )

        chain = [x_t]

        for i in reversed(range(len(step_indices))):
            t_idx = step_indices[i]
            prev_t_idx = step_indices[i - 1] if i > 0 else -1

            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

            model_prediction = self.get_model_prediction(x_t, context, t=t)

            alpha_bar = self.extract(self.alphas_cumprod, t, x_t.shape)

            if prev_t_idx >= 0:
                prev_t = torch.full(
                    (n_samples,), prev_t_idx, device=device, dtype=torch.long
                )
                alpha_bar_prev = self.extract(self.alphas_cumprod, prev_t, x_t.shape)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha_bar = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )

            if self.predict_noise:
                pred_x0 = (
                    x_t - sqrt_one_minus_alpha_bar * model_prediction
                ) / sqrt_alpha_bar
            else:
                pred_x0 = model_prediction

            pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)

            noise = (x_t - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha_bar

            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            )

            pred_dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * noise
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + pred_dir_xt

            if (
                guide is not None
                and t_start_guide is not None
                and prev_t_idx < t_start_guide
            ):
                x_prev = guide.optimize(x_prev, n_optimization_steps=n_guide_steps)

            if eta > 0:
                noise = torch.randn_like(x_prev)
                x_prev = x_prev + sigma_t * noise

            x_t = x_prev
            chain.append(x_t)

        chain = torch.stack(chain, dim=0)
        return chain

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        eta: float,
        debug: bool = False,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)

        if n_inference_steps is None:
            trajectories_chain_normalized = self.ddpm_sample(
                n_samples=n_samples,
                context=context,
                guide=guide,
                t_start_guide=t_start_guide,
                n_guide_steps=n_guide_steps,
                debug=debug,
            )
        else:
            trajectories_chain_normalized = self.ddim_sample(
                n_samples=n_samples,
                context=context,
                guide=guide,
                t_start_guide=t_start_guide,
                n_guide_steps=n_guide_steps,
                n_inference_steps=n_inference_steps,
                eta=eta,
                debug=debug,
            )

        return trajectories_chain_normalized


class DiffusionShortcut(Diffusion):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        cfg_fraction: float,
        cfg_scale: float,
        n_diffusion_steps: int,
        predict_noise: bool,
        bootstrap_fraction: float,
        dt_sampling_strategy: str,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
            n_diffusion_steps=n_diffusion_steps,
            predict_noise=predict_noise,
            cfg_fraction=cfg_fraction,
            cfg_scale=cfg_scale,
        )
        self.bootstrap_fraction = bootstrap_fraction
        self.dt_sampling_strategy = dt_sampling_strategy

        self.model = TemporalUNetShortcut(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )

    def get_alpha_bar(self, t):
        t_clamped = t.clamp(min=0)
        vals = self.alphas_cumprod.gather(0, t_clamped)
        vals = torch.where(t < 0, torch.ones_like(vals), vals)
        return vals

    def ddim_step(self, x, t, dt, model_prediction):
        alpha_bar_t = self.get_alpha_bar(t)
        t_prev = t - dt
        alpha_bar_prev = self.get_alpha_bar(t_prev)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1)

        if self.predict_noise:
            pred_x0 = (
                x - sqrt_one_minus_alpha_bar_t * model_prediction
            ) / sqrt_alpha_bar_t
        else:
            pred_x0 = model_prediction

        pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)

        noise = (x - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t

        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev).view(-1, 1, 1)

        x_prev = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * noise
        return x_prev

    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x_0 = input_dict["x"]
        device = x_0.device
        context = self.build_context(input_dict)
        batch_size, horizon, dim = x_0.shape

        n_bootstrap = int(batch_size * self.bootstrap_fraction)
        n_base = batch_size - n_bootstrap

        loss_dict = {}
        total_loss = 0.0

        if n_base > 0:
            x_0_base = x_0[n_bootstrap:]
            context_base = context[n_bootstrap:]

            t = torch.randint(
                0, self.n_diffusion_steps, (n_base,), device=device
            ).long()
            noise = torch.randn_like(x_0_base)
            x_t = self.q_sample(x_0=x_0_base, t=t, noise=noise)

            dt = torch.ones_like(t)
            model_output = self.get_model_prediction(x_t, context_base, t=t, dt=dt)

            if self.predict_noise:
                loss_base = F.mse_loss(model_output, noise)
            else:
                loss_base = F.mse_loss(model_output, x_0_base)

            loss_dict["loss_base"] = loss_base
            total_loss += loss_base * (n_base / batch_size)

        if n_bootstrap > 0:
            x_0_bootstrap = x_0[:n_bootstrap]
            context_bootstrap = context[:n_bootstrap]

            max_log2 = int(np.log2(self.n_diffusion_steps) + 1e-8)

            if self.dt_sampling_strategy == "uniform":
                k_exponents = torch.randint(
                    1, max_log2 + 1, (n_bootstrap,), device=device
                )
            elif self.dt_sampling_strategy == "weighted":
                possible_k = torch.arange(1, max_log2 + 1, device=device)
                possible_dt = 2**possible_k
                weights = 1.0 / possible_dt.float()
                k_indices = torch.multinomial(weights, n_bootstrap, replacement=True)
                k_exponents = possible_k[k_indices]
            else:
                raise ValueError(
                    f"Unknown dt_sampling_strategy: {self.dt_sampling_strategy}"
                )

            dt = 2**k_exponents

            max_compressed = self.n_diffusion_steps // dt
            compressed = (
                torch.rand((n_bootstrap,), device=device) * max_compressed
            ).long() + 1
            t = compressed * dt - 1

            noise = torch.randn_like(x_0_bootstrap)
            x_t = self.q_sample(x_0=x_0_bootstrap, t=t, noise=noise)
            dt_half = dt // 2

            with torch.no_grad():
                pred_mid = self.get_model_prediction(
                    x_t, context_bootstrap, t=t, dt=dt_half
                )
                x_mid = self.ddim_step(x_t, t, dt_half, pred_mid)
                x_mid = torch.clamp(x_mid, -4.0, 4.0)
                pred_target = self.get_model_prediction(
                    x_mid, context_bootstrap, t=t - dt_half, dt=dt_half
                )
                x_target = self.ddim_step(x_mid, t - dt_half, dt_half, pred_target)
                x_target = torch.clamp(x_target, -4.0, 4.0)

            pred = self.get_model_prediction(x_t, context_bootstrap, t=t, dt=dt)
            x_pred = self.ddim_step(x_t, t, dt, pred)

            loss_boot = F.mse_loss(x_pred, x_target)
            loss_dict["loss_bootstrap"] = loss_boot
            total_loss += loss_boot * (n_bootstrap / batch_size)

        loss_dict["loss"] = total_loss

        return loss_dict

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)
        device = self.betas.device

        trajectories = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )

        step_size = self.n_diffusion_steps // n_inference_steps
        if step_size < 1:
            step_size = 1

        current_t = self.n_diffusion_steps - 1

        while current_t >= 0:
            t_tensor = torch.full(
                (n_samples,), current_t, device=device, dtype=torch.long
            )
            dt_tensor = torch.full(
                (n_samples,), step_size, device=device, dtype=torch.long
            )
            model_out = self.get_model_prediction(
                trajectories, context=context, t=t_tensor, dt=dt_tensor
            )
            trajectories = self.ddim_step(trajectories, t_tensor, dt_tensor, model_out)
            current_t -= step_size
            if (
                guide is not None
                and t_start_guide is not None
                and current_t < t_start_guide
            ):
                trajectories = guide.optimize(
                    trajectories, n_optimization_steps=n_guide_steps
                )

        return trajectories.unsqueeze(0)


class FlowMatchingShortcut(GenerativeModel):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        cfg_fraction: float,
        cfg_scale: float,
        n_diffusion_steps: int,
        bootstrap_fraction: float,
        dt_sampling_strategy: str,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
            cfg_fraction=cfg_fraction,
            cfg_scale=cfg_scale,
        )

        self.model = TemporalUNetShortcut(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            dim_mults=self.dim_mults,
            kernel_size=self.kernel_size,
            resnet_block_groups=self.resnet_block_groups,
            positional_encoding=self.positional_encoding,
            positional_encoding_dim=self.positional_encoding_dim,
            attn_heads=self.attn_heads,
            attn_head_dim=self.attn_head_dim,
            context_dim=self.context_dim,
        )
        self.n_diffusion_steps = n_diffusion_steps
        self.bootstrap_fraction = bootstrap_fraction
        self.dt_sampling_strategy = dt_sampling_strategy
        self.min_dt = 1.0 / self.n_diffusion_steps
        self.eps = 1e-5

    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x_1 = input_dict["x"]
        device = x_1.device
        context = self.build_context(input_dict)
        batch_size, horizon, dim = x_1.shape
        x_0 = torch.randn_like(x_1)

        n_bootstrap = int(batch_size * self.bootstrap_fraction)
        n_base = batch_size - n_bootstrap

        loss_dict = {}

        total_loss = 0.0

        if n_base > 0:
            x_1_base = x_1[n_bootstrap:]
            x_0_base = x_0[n_bootstrap:]
            context_base = context[n_bootstrap:]

            t_base = torch.rand(n_base, device=device)
            t_base_exp = t_base.view(-1, 1, 1)

            x_t_base = (
                1 - (1 - self.eps) * t_base_exp
            ) * x_0_base + t_base_exp * x_1_base

            v_target_base = x_1_base - (1 - self.eps) * x_0_base

            dt_base = torch.full((n_base,), self.min_dt, device=device)

            v_pred_base = self.get_model_prediction(
                x_t_base, context_base, t=t_base, dt=dt_base
            )

            loss_base = F.mse_loss(v_pred_base, v_target_base)
            loss_dict["loss_base"] = loss_base
            total_loss += loss_base * (n_base / batch_size)

        if n_bootstrap > 0:
            x_0_bootstrap = x_0[:n_bootstrap]
            x_1_bootstrap = x_1[:n_bootstrap]
            context_bootstrap = context[:n_bootstrap]

            max_log2 = int(np.log2(self.n_diffusion_steps) + 1e-8)

            if self.dt_sampling_strategy == "uniform":
                k_exponents = torch.randint(
                    1, max_log2 + 1, (n_bootstrap,), device=device
                )
            elif self.dt_sampling_strategy == "weighted":
                possible_k = torch.arange(1, max_log2 + 1, device=device)
                possible_dt_int = 2**possible_k
                weights = 1.0 / possible_dt_int.float()
                k_indices = torch.multinomial(weights, n_bootstrap, replacement=True)
                k_exponents = possible_k[k_indices]
            else:
                raise ValueError(
                    f"Unknown dt_sampling_strategy: {self.dt_sampling_strategy}"
                )

            dt_int = 2**k_exponents
            dt = dt_int / self.n_diffusion_steps
            max_compressed = self.n_diffusion_steps // dt_int
            compressed = (
                torch.rand((n_bootstrap,), device=device) * max_compressed
            ).long() + 1
            t_int = compressed * dt_int - 1
            t = t_int / self.n_diffusion_steps
            t_exp = t.view(-1, 1, 1)

            x_t = (1 - (1 - self.eps) * t_exp) * x_0_bootstrap + t_exp * x_1_bootstrap
            dt_half = dt / 2.0

            with torch.no_grad():
                v_b1 = self.get_model_prediction(
                    x_t, context_bootstrap, t=t, dt=dt_half
                )
                x_mid = x_t + dt_half.view(-1, 1, 1) * v_b1
                x_mid = torch.clamp(x_mid, -4.0, 4.0)
                v_b2 = self.get_model_prediction(
                    x_mid, context_bootstrap, t=t + dt_half, dt=dt_half
                )
                v_target = (v_b1 + v_b2) / 2.0
                v_target = torch.clamp(v_target, -4.0, 4.0)

            v_pred = self.get_model_prediction(x_t, context_bootstrap, t=t, dt=dt)

            loss_bootstrap = F.mse_loss(v_pred, v_target)
            loss_dict["loss_bootstrap"] = loss_bootstrap

            total_loss += loss_bootstrap * (n_bootstrap / batch_size)

        loss_dict["loss"] = total_loss
        return loss_dict

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)
        device = context.device

        trajectories = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )

        dt_val = 1.0 / n_inference_steps
        dt = torch.full((n_samples,), dt_val, device=device)

        current_time = 0.0
        for _ in range(n_inference_steps):
            t = torch.full((n_samples,), current_time, device=device)
            v_pred = self.get_model_prediction(
                trajectories, context=context, t=t, dt=dt
            )
            trajectories = trajectories + dt_val * v_pred

            current_time += dt_val

            if (
                guide is not None
                and t_start_guide is not None
                and 1 - current_time < t_start_guide / self.n_diffusion_steps + 1e-8
            ):
                trajectories = guide.optimize(
                    trajectories, n_optimization_steps=n_guide_steps
                )

        trajectories_chain_normalized = trajectories.unsqueeze(0)

        return trajectories_chain_normalized


class Drift(GenerativeModel):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        cfg_fraction: float,
        cfg_scale: float,
        temperature: float,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
            cfg_fraction=cfg_fraction,
            cfg_scale=cfg_scale,
        )
        self.temperature = temperature

    def compute_drift(
        self, data_generated: torch.Tensor, data_positive: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute drift field V with attention-based kernel.

        Args:
            data_generated: Generated samples [batch_size, dim]
            data_positive: Data samples [x, dim]

        Returns:
            V: Drift vectors [batch_size, dim]
        """
        targets = torch.cat([data_generated, data_positive], dim=0)
        batch_size, dim = data_generated.shape

        dist = torch.cdist(data_generated, targets)
        dist[:, :batch_size].fill_diagonal_(1e8)
        kernel = (-dist / self.temperature).exp()

        normalizer = (
            kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
        )  # normalize along both dimensions, which we found to slightly improve performance
        normalizer = normalizer.clamp_min(1e-16).sqrt()
        normalized_kernel = kernel  # / normalizer

        pos_coeff = normalized_kernel[:, batch_size:] * normalized_kernel[
            :, :batch_size
        ].sum(dim=-1, keepdim=True)
        pos_V = pos_coeff @ targets[batch_size:]
        neg_coeff = normalized_kernel[:, :batch_size] * normalized_kernel[
            :, batch_size:
        ].sum(dim=-1, keepdim=True)
        neg_V = neg_coeff @ targets[:batch_size]

        return pos_V - neg_V

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        x_1 = input_dict["x"]
        device = x_1.device
        context = self.build_context(input_dict)
        batch_size, horizon, dim = x_1.shape
        x_0 = torch.randn_like(x_1)

        t = torch.zeros((batch_size,), device=device).long()

        x_pred = self.model(x_0, t, context)

        x_pred_flat = x_pred.reshape(batch_size, -1)
        x_data_flat = x_1.reshape(batch_size, -1)

        drift = self.compute_drift(x_pred_flat, x_data_flat)

        loss = F.mse_loss(x_pred_flat, (x_pred_flat + drift).detach())

        return {"loss": loss}

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: GradientOptimization,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)
        device = context.device

        noise = torch.randn((n_samples, self.horizon, self.state_dim), device=device)
        t = torch.zeros((n_samples,), device=device).long()

        trajectories = self.model(noise, t, context)

        if guide is not None and t_start_guide is not None and t_start_guide >= 0:
            trajectories = guide.optimize(
                trajectories, n_optimization_steps=n_guide_steps
            )

        trajectories_chain_normalized = trajectories.unsqueeze(0)

        return trajectories_chain_normalized

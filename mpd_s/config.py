import os

import torch

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

N_DIM = 2

DEFAULT_TENSOR_ARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

DEFAULT_TRAIN_ARGS = {
    # Dirs
    "checkpoints_dir": os.path.join(dir_path, "models"),
    "checkpoint_name": None,
    # Dataset
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvDense2D_2000_50",
    "normalizer_name": "TrivialNormalizer",
    "n_control_points": 24,  # set to None for non-spline models
    "spline_degree": 3,
    "apply_augmentations": True,
    "filter_collision": True,
    "filter_longest_portion": None,
    "filter_roughest_portion": None,
    # Generative model
    "model_name": "DiffusionShortcut",  # "Diffusion", "DiffusionShortcut", "FlowMatchingShortcut", "Drift"
    "cfg_fraction": None,
    "cfg_scale": 1.0,
    # Diffusion and/or FlowMatching
    "n_diffusion_steps": 32,
    "predict_noise": False,
    # Shortcut
    "bootstrap_fraction": 0.5,
    "dt_sampling_strategy": "weighted",  # "uniform", "weighted"
    # Drift
    "temperature": 1.0,
    # Inference
    "inference_args": {
        "n_inference_steps": 1,
        "eta": 0.0,
        # Guide
        "t_start_guide": 0,
        "n_guide_steps": 2,
        "lambda_obstacles": 5e-3,
        "lambda_velocity": 1e-6,
        "lambda_acceleration": 0,
        "lambda_jerk": 0,
        "max_grad_norm": 1.0,
        "n_interpolate": 10,
    },
    # Unet
    "state_dim": N_DIM,
    "horizon": 24,
    "hidden_dim": 64,
    "dim_mults": "(1, 2, 4)",
    "kernel_size": 5,
    "resnet_block_groups": 8,
    "positional_encoding": "sinusoidal",
    "positional_encoding_dim": 16,
    "attn_heads": 4,
    "attn_head_dim": 32,
    "context_dim": 2 * N_DIM,
    # Training
    "num_train_steps": 200000,
    "lr": 3e-5,
    "weight_decay": 0,
    "batch_size": 1024,
    "clip_grad_max_norm": 1.0,
    "use_amp": True,
    "use_ema": True,
    "ema_decay": 0.995,
    "ema_warmup": 40000,
    "ema_update_interval": 10,
    # Summary
    "log_interval": 2000,
    "checkpoint_interval": 20000,
    # Other
    "device": "cuda:6",
    "debug": False,
    "seed": 42,
}

DEFAULT_INFERENCE_ARGS = {
    # Dirs and general settings
    "generations_dir": os.path.join(dir_path, "runs"),
    "experiment_name": None,
    "n_tasks": 3,
    "n_trajectories_per_task": 10,
    "splits": '("test",)',  # '("train", "val", "test")',
    # Algorithm selection
    "algorithm": "rrt",  # Options: "generative-model", "mpd", "rrt", "rrt-smooth", "gpmp2", "grad", "grad-splines" "rrt-gpmp2", "rrt-grad", "rrt-grad-splines"
    # Dataset
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvDense2D_2000_50",
    "threshold_start_goal_pos": 2,
    "use_extra_objects": True,
    # Generative model
    "checkpoints_dir": os.path.join(dir_path, "models", "checkpoints"),
    "checkpoint_name": "",
    "checkpoint_iter": None,
    "n_inference_steps": 1,  # None for DDPM, otherwise DDIM or shortcut
    # DDIM
    "eta": 0.0,
    # Diffusion prior guide
    "t_start_guide": 0,
    "n_guide_steps": 2,
    "lambda_obstacles": 5e-3,
    "lambda_velocity": 5e-5,
    "lambda_acceleration": 0,
    "lambda_jerk": 0,
    "max_grad_norm": 1.0,
    "n_interpolate": 10,
    # MPD
    "mpd_checkpoints_dir": os.path.join(
        dir_path,
        "data_trained_models",
        "mpd",
        "EnvNarrowPassageDense2D-RobotPointMass",
        "checkpoints",
    ),
    "mpd_checkpoint_name": "ema_model_current.pth",
    "mpd_ddim": False,
    "mpd_start_guide_steps_fraction": 0.25,
    "mpd_n_guide_steps": 5,
    "mpd_sigma_collision": 1e1,
    "mpd_sigma_gp": 2e3,
    "mpd_max_grad_norm": 1.0,
    "mpd_n_interpolate": 5,
    # Classical algorithm
    "classical_n_dof": N_DIM,
    "classical_n_sampling_steps": 10000,
    "gpmp2_n_optimization_steps": 300,
    "grad_n_optimization_steps": 300,
    # RRT-Connect parameters
    "rrt_connect_max_radius": 0.3,
    "rrt_connect_n_points": 64,
    "rrt_connect_n_samples": 160000,
    # GPMP2 parameters
    "gpmp2_n_interpolate": 5,
    "gpmp2_sigma_start": 3e-2,
    "gpmp2_sigma_goal_prior": 3e-2,
    "gpmp2_sigma_gp": 1,
    "gpmp2_sigma_collision": 3e-3,
    "gpmp2_step_size": 1e-1,
    "gpmp2_delta": 1e-5,
    "gpmp2_method": "cholesky",
    # Gradient optimization parameters
    "grad_lambda_obstacles": 1e-2,
    "grad_lambda_velocity": 5e-5,
    "grad_lambda_acceleration": 1e-5,
    "grad_lambda_jerk": 5e-7,
    "grad_max_grad_norm": 0.01,
    "grad_n_interpolate": 10,
    # *-Grad-Splines parameters
    "grad_splines_n_control_points": 48,
    "grad_splines_spline_degree": 3,
    # Other
    "draw_spacing": 1,
    "generate_animation": False,
    "device": "cuda",
    "debug": False,
    "seed": 42,
}

DEFAULT_DATA_GENERATION_ARGS = {
    # Dataset initialization
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvSparse2D_20_50_L",
    "env_name": "EnvSparse2D",
    "robot_name": "L2D",
    "robot_margin": 0.05,
    "generating_robot_margin": 0.06,
    "n_support_points": 128,
    "duration": 5.0,
    "spline_degree": 3,
    "additional_robot_args": {"width": 0.3, "height": 0.4, "n_spheres": 15},
    # Task generation
    "n_tasks": 20,
    "n_trajectories_per_task": 50,
    "threshold_start_goal_pos": 1.5,
    # Planning parameters
    "n_sampling_steps": 10000,
    "n_optimization_steps": 300,
    "smoothen": True,
    "create_straight_line_trajectories": False,
    "use_gpmp2": False,
    "n_control_points": 24,
    # RRT-Connect parameters
    "rrt_connect_max_radius": 0.3,
    "rrt_connect_n_points": 64,
    "rrt_connect_n_samples": 160000,
    # GPMP2 parameters
    "gpmp2_n_interpolate": 5,
    "gpmp2_sigma_start": 3e-2,
    "gpmp2_sigma_goal_prior": 3e-2,
    "gpmp2_sigma_gp": 1,
    "gpmp2_sigma_collision": 3e-3,
    "gpmp2_step_size": 1e-1,
    "gpmp2_delta": 1e-5,
    "gpmp2_method": "cholesky",
    # Gradient optimization parameters
    "grad_lambda_obstacles": 1e-2,
    "grad_lambda_velocity": 5e-5,
    "grad_lambda_acceleration": 1e-5,
    "grad_lambda_jerk": 5e-7,
    "grad_max_grad_norm": 0.01,
    "grad_n_interpolate": 10,
    # Other
    "n_processes": 1,
    "val_portion": 0.05,
    "device": "cuda",
    "debug": False,
    "seed": 42,
}

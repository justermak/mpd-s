import os
from datetime import datetime

import configargparse
import torch

from mpd_s.config import DEFAULT_TRAIN_ARGS
from mpd_s.dataset.dataset import TrajectoryDataset
from mpd_s.model.generative_models import get_additional_init_args, get_models
from mpd_s.train import train
from mpd_s.utils import fix_random_seed, load_config_from_yaml, save_config_to_yaml

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print("-------- TRAINING STARTED --------")
    print(f"dataset: {args.dataset_name}")
    print(f"batch size: {args.batch_size}")
    print(f"apply augmentations: {args.apply_augmentations}")
    print(f"learning rate: {args.lr}")
    print(f"number of training steps: {args.num_train_steps}")

    if args.checkpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{args.model_name}__{args.dataset_name}__bs_{args.batch_size}__lr_{args.lr}__steps_{args.num_train_steps}__diffusion-steps_{args.n_diffusion_steps}__{timestamp}"
    else:
        checkpoint_name = args.checkpoint_name

    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_init_config_path = os.path.join(dataset_dir, "init_config.yaml")
    dataset_init_config = load_config_from_yaml(dataset_init_config_path)
    dataset = TrajectoryDataset(**dataset_init_config, tensor_args=tensor_args)

    dataset_usage_config = {
        "normalizer_name": args.normalizer_name,
        "apply_augmentations": args.apply_augmentations,
        "n_control_points": args.n_control_points,
        "filtering_config": {
            "filter_collision": {} if args.filter_collision else None,
            "filter_longest_trajectories": {"portion": args.filter_longest_portion}
            if args.filter_longest_portion is not None
            else None,
            "filter_roughest_trajectories": {"portion": args.filter_roughest_portion}
            if args.filter_roughest_portion is not None
            else None,
        },
    }

    print("\nDataset usage config:\n", dataset_usage_config)

    train_subset, train_dataloader, val_subset, val_dataloader = dataset.load_data(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        debug=args.debug,
        **dataset_usage_config,
    )

    assert args.state_dim == dataset.robot.n_dim
    assert args.horizon == (
        dataset.n_support_points
        if args.n_control_points is None
        else args.n_control_points
    )

    additional_args = get_additional_init_args(args.model_name, vars(args))
    model_config = {
        "state_dim": args.state_dim,
        "horizon": args.horizon,
        "hidden_dim": args.hidden_dim,
        "dim_mults": eval(args.dim_mults),
        "kernel_size": args.kernel_size,
        "resnet_block_groups": args.resnet_block_groups,
        "positional_encoding": args.positional_encoding,
        "positional_encoding_dim": args.positional_encoding_dim,
        "attn_heads": args.attn_heads,
        "attn_head_dim": args.attn_head_dim,
        "context_dim": args.context_dim,
        "cfg_fraction": args.cfg_fraction,
        "cfg_scale": args.cfg_scale,
        **additional_args,
    }
    model = MODELS[args.model_name](dataset=dataset, **model_config).to(device)

    # you can load a checkpoint here

    checkpoint_dir = os.path.join(args.checkpoints_dir, "checkpoints", checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset_usage_config_path = os.path.join(
        checkpoint_dir, "dataset_usage_config.yaml"
    )
    save_config_to_yaml(dataset_usage_config, dataset_usage_config_path)
    save_config_to_yaml(model_config, os.path.join(checkpoint_dir, "init_config.yaml"))
    save_config_to_yaml(vars(args), os.path.join(checkpoint_dir, "info_config.yaml"))

    train(
        checkpoint_name=checkpoint_name,
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        checkpoints_dir=args.checkpoints_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        num_train_steps=args.num_train_steps,
        clip_grad_max_norm=args.clip_grad_max_norm,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_warmup=args.ema_warmup,
        ema_update_interval=args.ema_update_interval,
        use_amp=args.use_amp,
        inference_args=args.inference_args,
        tensor_args=tensor_args,
        debug=args.debug,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {}

    for key, value in DEFAULT_TRAIN_ARGS.items():
        arg_name = f"--{key}"
        arg_type = type(value) if value is not None else str

        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no_{key}", dest=key, action="store_false")
        else:
            kwargs = {"type": arg_type, "default": value}
            if key in special_args:
                kwargs.update(special_args[key])
            parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()
    run(args)

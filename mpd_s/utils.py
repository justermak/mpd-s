import random

import numpy as np
import torch
import yaml


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_config_from_yaml(path: str):
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_config_to_yaml(params: dict, path: str):
    with open(path, "w") as outfile:
        yaml.dump(params, outfile, default_flow_style=False)

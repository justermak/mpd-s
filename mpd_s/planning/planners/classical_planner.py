from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

from mpd_s.universe.environments import EnvBase
from mpd_s.universe.robot import RobotBase


class ClassicalPlanner(ABC):
    def __init__(
        self,
        name: str,
        env: EnvBase,
        robot: RobotBase,
        use_extra_objects: bool,
        tensor_args: Dict[str, Any],
    ):
        self.name = name
        self.env = env
        self.robot = robot
        self.use_extra_objects = use_extra_objects
        self.tensor_args = tensor_args
        self.start_pos: torch.Tensor = None
        self.goal_pos: torch.Tensor = None

    @abstractmethod
    def optimize(self, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        pass

from abc import ABC
from typing import Any, Dict

import numpy as np
import torch

from mpd_s.torch_timer import TimerCUDA
from mpd_s.universe.grid_map_sdf import GridMapSDF
from mpd_s.universe.primitives import MultiBoxField, MultiSphereField, ObjectField


def create_workspace_boundary_boxes(limits: np.ndarray, r: float = 0.2) -> list:
    x_min, y_min = limits[0]
    x_max, y_max = limits[1]
    height = y_max - y_min
    width = x_max - x_min

    centers = np.array(
        [
            [x_min - r, (y_min + y_max) / 2],
            [x_max + r, (y_min + y_max) / 2],
            [(x_min + x_max) / 2, y_min - r],
            [(x_min + x_max) / 2, y_max + r],
        ]
    )

    half_sizes = np.array(
        [
            [r, height / 2 + r],
            [r, height / 2 + r],
            [width / 2 + r, r],
            [width / 2 + r, r],
        ]
    )

    return centers, half_sizes


class EnvBase(ABC):
    def __init__(
        self,
        limits: torch.Tensor,
        obj_field_fixed: ObjectField,
        obj_field_extra: ObjectField,
        sdf_cell_size: float,
        tensor_args: Dict[str, Any],
        grid_map_sdf_fixed: Any = None,
        grid_map_sdf_extra: Any = None,
    ):
        self.name = None
        self.tensor_args = tensor_args
        self.limits = limits
        self.limits_np = limits.cpu().numpy()

        self.obj_field_fixed = obj_field_fixed
        self.obj_field_extra = obj_field_extra

        if grid_map_sdf_fixed is not None and grid_map_sdf_extra is not None:
            self.grid_map_sdf_fixed = grid_map_sdf_fixed
            self.grid_map_sdf_extra = grid_map_sdf_extra
        else:
            with TimerCUDA() as t:
                self.grid_map_sdf_fixed = GridMapSDF(
                    self.limits,
                    sdf_cell_size,
                    self.obj_field_fixed,
                    tensor_args=self.tensor_args,
                )
                self.grid_map_sdf_extra = GridMapSDF(
                    self.limits,
                    sdf_cell_size,
                    self.obj_field_extra,
                    tensor_args=self.tensor_args,
                )

        self.distribution = torch.distributions.uniform.Uniform(
            self.limits[0], self.limits[1]
        )

    def random_points(self, shape) -> torch.Tensor:
        return self.distribution.sample(shape)


class EnvEmpty2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
        grid_map_sdf_fixed: Any = None,
        grid_map_sdf_extra: Any = None,
    ):
        self.name = "EnvEmpty2D"
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiBoxField(
                    boundary_centers,
                    boundary_half_sizes,
                    tensor_args=tensor_args,
                )
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array([[0.0, 0.0]]),
                    np.array([0.1]),
                    tensor_args=tensor_args,
                )
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            tensor_args=tensor_args,
            grid_map_sdf_fixed=grid_map_sdf_fixed,
            grid_map_sdf_extra=grid_map_sdf_extra,
        )


class EnvSparse2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
        grid_map_sdf_fixed: Any = None,
        grid_map_sdf_extra: Any = None,
    ):
        self.name = "EnvSparse2D"
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiSphereField(
                    centers=np.array(
                        [
                            [-0.6, 0.4],
                            [-0.5, -0.6],
                            [0.2, 0.1],
                            [0.6, -0.7],
                            [0.7, 0.7],
                        ]
                    ),
                    radii=np.array(
                        [
                            0.2,
                            0.2,
                            0.2,
                            0.2,
                            0.2,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    boundary_centers,
                    boundary_half_sizes,
                    tensor_args=tensor_args,
                ),
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [0.1, -0.5],
                            [0.0, 0.7],
                            [-0.5, -0.1],
                            [0.8, 0.0],
                        ]
                    ),
                    np.array(
                        [
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                        ]
                    ),
                    tensor_args=tensor_args,
                )
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            grid_map_sdf_fixed=grid_map_sdf_fixed,
            grid_map_sdf_extra=grid_map_sdf_extra,
            tensor_args=tensor_args,
        )


class EnvSimple2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
        grid_map_sdf_fixed: Any = None,
        grid_map_sdf_extra: Any = None,
    ):
        self.name = "EnvSimple2D"
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiSphereField(
                    centers=np.array(
                        [
                            [-0.43378472328186035, 0.3334643840789795],
                            [0.3313474655151367, 0.6288051009178162],
                            [-0.5656964778900146, -0.484994500875473],
                            [0.42124247550964355, -0.6656165719032288],
                            [0.05636655166745186, -0.5149664282798767],
                            [-0.36961784958839417, -0.12315540760755539],
                            [-0.8740217089653015, -0.4034936726093292],
                            [-0.6359214186668396, 0.6683124899864197],
                            [0.808782160282135, 0.5287870168685913],
                            [-0.023786112666130066, 0.4590069353580475],
                            [0.1455741971731186, 0.16420497000217438],
                            [0.628413736820221, -0.43461447954177856],
                            [0.17965620756149292, -0.8926276564598083],
                            [0.6775968670845032, 0.8817358016967773],
                            [-0.3608766794204712, 0.8313458561897278],
                        ]
                    ),
                    radii=np.array(
                        [
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    boundary_centers,
                    boundary_half_sizes,
                    tensor_args=tensor_args,
                ),
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.15, 0.15],
                            [-0.075, -0.85],
                            [-0.1, -0.1],
                            [0.45, -0.1],
                            [0.5, 0.35],
                            [-0.6, -0.85],
                            [0.05, 0.85],
                            [-0.8, 0.15],
                            [0.8, -0.8],
                        ]
                    ),
                    np.array(
                        [
                            0.05,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    np.array(
                        [
                            [0.45, -0.1],
                            [-0.25, -0.5],
                            [0.8, 0.1],
                        ]
                    ),
                    np.array(
                        [
                            [0.2, 0.2],
                            [0.15, 0.15],
                            [0.15, 0.15],
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            grid_map_sdf_fixed=grid_map_sdf_fixed,
            grid_map_sdf_extra=grid_map_sdf_extra,
            tensor_args=tensor_args,
        )


class EnvDense2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
        grid_map_sdf_fixed: Any = None,
        grid_map_sdf_extra: Any = None,
    ):
        self.name = "EnvDense2D"
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.43378472328186035, 0.3334643840789795],
                            [0.3313474655151367, 0.6288051009178162],
                            [-0.5656964778900146, -0.484994500875473],
                            [0.42124247550964355, -0.6656165719032288],
                            [0.05636655166745186, -0.5149664282798767],
                            [-0.36961784958839417, -0.12315540760755539],
                            [-0.8740217089653015, -0.4034936726093292],
                            [-0.6359214186668396, 0.6683124899864197],
                            [0.808782160282135, 0.5287870168685913],
                            [-0.023786112666130066, 0.4590069353580475],
                            [0.11544948071241379, -0.12676022946834564],
                            [0.1455741971731186, 0.16420497000217438],
                            [0.628413736820221, -0.43461447954177856],
                            [0.17965620756149292, -0.8926276564598083],
                            [0.6775968670845032, 0.8817358016967773],
                            [-0.3608766794204712, 0.8313458561897278],
                        ]
                    ),
                    np.array(
                        [
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    centers=np.array(
                        [
                            [0.607781708240509, 0.19512386620044708],
                            [0.5575312972068787, 0.5508843064308167],
                            [-0.3352295458316803, -0.6887519359588623],
                            [-0.6572632193565369, 0.31827881932258606],
                            [-0.664594292640686, -0.016457155346870422],
                            [0.8165988922119141, -0.19856023788452148],
                            [-0.8222246170043945, -0.6448580026626587],
                            [-0.2855989933013916, -0.36841487884521484],
                            [-0.8946458101272583, 0.8962447643280029],
                            [-0.23994405567646027, 0.6021060943603516],
                            [-0.006193588487803936, 0.8456171751022339],
                            [0.305103600025177, -0.3661990463733673],
                            [-0.10704007744789124, 0.1318950206041336],
                            [0.7156378626823425, -0.6923345923423767],
                            *boundary_centers,
                        ]
                    ),
                    half_sizes=np.array(
                        [
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            *boundary_half_sizes,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.4, 0.1],
                            [-0.075, -0.85],
                            [-0.1, -0.1],
                        ]
                    ),
                    np.array(
                        [
                            0.075,
                            0.1,
                            0.075,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    np.array(
                        [
                            [0.45, -0.1],
                            [0.35, 0.35],
                            [-0.6, -0.85],
                            [-0.65, -0.25],
                        ]
                    ),
                    np.array(
                        [
                            [0.1, 0.1],
                            [0.05, 0.075],
                            [0.05, 0.125],
                            [0.075, 0.05],
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            tensor_args=tensor_args,
            grid_map_sdf_fixed=grid_map_sdf_fixed,
            grid_map_sdf_extra=grid_map_sdf_extra,
        )


class EnvDenseNarrowPassage2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
        grid_map_sdf_fixed: Any = None,
        grid_map_sdf_extra: Any = None,
    ):
        self.name = "EnvDenseNarrowPassage2D"
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiSphereField(
                    centers=np.array(
                        [
                            [0.3313474655151367, 0.6288051009178162],
                            [-0.36961784958839417, -0.12315540760755539],
                            [-0.8740217089653015, -0.4034936726093292],
                            [0.808782160282135, 0.5287870168685913],
                            [0.6775968670845032, 0.8817358016967773],
                            [-0.3608766794204712, 0.8313458561897278],
                            [0.7156378626823425, -0.6923345923423767],
                            [0.35, 0],
                        ]
                    ),
                    radii=np.array(
                        [
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    centers=np.array(
                        [
                            [0.607781708240509, 0.19512386620044708],
                            [-0.3352295458316803, -0.6887519359588623],
                            [-0.6572632193565369, 0.41827881932258606],
                            [-0.664594292640686, 0.016457155346870422],
                            [0.8165988922119141, -0.19856023788452148],
                            [-0.8222246170043945, -0.6448580026626587],
                            [-0.8946458101272583, 0.8962447643280029],
                            [-0.23994405567646027, 0.6021060943603516],
                            [0.305103600025177, -0.3661990463733673],
                            [0.0, 0.5 + 0.05],
                            [0.0, -0.5 - 0.05],
                            *boundary_centers,
                        ]
                    ),
                    half_sizes=np.array(
                        [
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.5 - 0.05 / 4],
                            [0.1, 0.5 - 0.05 / 4],
                            *boundary_half_sizes,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.45, 0.2],
                            [-0.5, -0.4],
                            [0.6, -0.4],
                            [0.35, 0.35],
                            [0.4, -0.7],
                            [-0.65, 0.7],
                            [-0.225, 0.35],
                            [0.6, -0.1],
                        ]
                    ),
                    np.array(
                        [
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.075,
                            0.075,
                            0.05,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    np.array(
                        [
                            [0.2, -0.9],
                            [0.9, 0.1],
                            [0.35, 0.35],
                            [-0.6, -0.85],
                            [-0.70, -0.25],
                            [-0.9, 0.25],
                        ]
                    ),
                    np.array(
                        [
                            [0.125, 0.125],
                            [0.125, 0.125],
                            [0.1, 0.15],
                            [0.15, 0.15],
                            [0.1, 0.1],
                            [0.125, 0.125],
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            tensor_args=tensor_args,
            grid_map_sdf_fixed=grid_map_sdf_fixed,
            grid_map_sdf_extra=grid_map_sdf_extra,
        )


def get_envs():
    return {
        "EnvEmpty2D": EnvEmpty2D,
        "EnvSparse2D": EnvSparse2D,
        "EnvSimple2D": EnvSimple2D,
        "EnvDense2D": EnvDense2D,
        "EnvDenseNarrowPassage2D": EnvDenseNarrowPassage2D,
    }

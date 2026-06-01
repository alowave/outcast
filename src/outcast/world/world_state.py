"""World state configuration and management.

This module provides:
- WorldStateCfg: Configuration for generating world state
- WorldState: World state represented as dense NumPy arrays
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class WorldStateCfg:
    """Configuration for generating world state.

    This config defines:
    - Network sizes (number of UEs / UAVs / BSs)
    - Spatial boundary for sampling x-y coordinates (uniform distribution)
    - Fixed heights for UE/BS and min/max altitude range for UAVs

    Notes:
      - All distances/heights are in meters.
      - env_boundary is (width_m, height_m) and positions are sampled in:
        x in [0, width_m], y in [0, height_m]
    """

    n_ues: int = 200
    n_uavs: int = 20
    n_bss: int = 5
    env_boundary: tuple[float, float] = (1000.0, 1000.0)

    min_height_uav: float = 50.0
    max_height_uav: float = 100.0
    ue_height: float = 1.5
    bs_height: float = 25.0
    user_model: str | None = "obstacle_mobility"
    mobility: Any = None


@dataclass(slots=True)
class WorldState:
    """World state represented as dense NumPy arrays.

    Attributes:
        ue_pos: (N_UE, 3) array of UE positions [x, y, z] in meters.
        uav_pos: (N_UAV, 3) array of UAV positions [x, y, z] in meters.
        bs_pos: (N_BS, 3) array of BS positions [x, y, z] in meters.
        gn_load: Optional (N_UAV + N_BS,) array of generated loads.
        obstacles: Optional list of obstacles managed by the obstacle controller.
    """

    ue_pos: NDArray[np.float32]  # (N_UE, 3)
    uav_pos: NDArray[np.float32]  # (N_UAV, 3)
    bs_pos: NDArray[np.float32]  # (N_BS, 3)
    gn_load: NDArray[np.uint64] | None = None
    obstacles: list | None = None

    def randomize_ue_pos(self, cfg: WorldStateCfg, rng: np.random.Generator) -> None:
        """Sample UE positions uniformly within the environment boundaries.

        Writes results to self.ue_pos as [x, y, ue_height].
        """
        w, h = cfg.env_boundary
        ue_xy = rng.uniform([0.0, 0.0], [w, h], size=(cfg.n_ues, 2))
        ue_z = np.full((cfg.n_ues, 1), cfg.ue_height, dtype=np.float32)
        self.ue_pos = np.hstack([ue_xy, ue_z]).astype(np.float32)

    def randomize_uav_pos(self, cfg: WorldStateCfg, rng: np.random.Generator) -> None:
        """Sample UAV positions uniformly within the boundary and height limits."""
        w, h = cfg.env_boundary
        self.uav_pos = rng.uniform(
            [0.0, 0.0, cfg.min_height_uav],
            [w, h, cfg.max_height_uav],
            size=(cfg.n_uavs, 3),
        ).astype(np.float32)

    def randomize_bs_pos(self, cfg: WorldStateCfg, rng: np.random.Generator) -> None:
        """Sample Base Station positions uniformly within the environment x-y plane."""
        w, h = cfg.env_boundary
        bs_xy = rng.uniform([0.0, 0.0], [w, h], size=(cfg.n_bss, 2))
        bs_z = np.full((cfg.n_bss, 1), cfg.bs_height, dtype=np.float32)
        self.bs_pos = np.hstack([bs_xy, bs_z]).astype(np.float32)

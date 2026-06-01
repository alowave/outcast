"""Mock link layer implementation for UAV simulation.

This module provides a simplified mock implementation of the link layer that
computes distance matrices, in-range masks, and LOS (Line of Sight) information
for both access links (UE to serving nodes) and backhaul links (serving node
to serving node). It uses pre-allocated numpy arrays for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.outcast.geometry.los_cache import LosCacheCfg, LosSectorCache
from src.outcast.link_layer.link_data import AccessLinkData, BackhaulLinkData
from src.outcast.world.world_state import WorldState, WorldStateCfg

if TYPE_CHECKING:
    from src.outcast.geometry.obstacle import Obstacle


@dataclass(slots=True)
class LinkLayerCfg:
    """Link-layer configuration parameters.

    Attributes:
        access_range_m:
            Max UE-to-serving-node (UAV/BS) distance considered "in range".
        backhaul_range_m:
            Max serving-node-to-serving-node distance considered "in range"
            for the backhaul graph (e.g., UAV<->UAV, UAV<->BS, BS<->BS).
        rng_seed:
            Seed used for rng
    """

    access_range_m: float = 500.0
    backhaul_range_m: float = 1000.0
    rng_seed: int = 33


class MockLinkLayer:
    """Mock link-layer module.

    Responsibilities (in this simplified mock):
      - Compute and store distance matrices:
          * Access: UE -> (UAV + BS)
          * Backhaul: (UAV + BS) -> (UAV + BS)
      - Compute boolean in-range masks from configured ranges
      - Fill LOS matrices with random values (placeholder for a real LOS model)

    Notes:
      - This class assumes array sizes are fixed once initialized from WorldStateCfg.
      - Call initialize_data_arrays() once before calling update().
    """

    def __init__(self, cfg: LinkLayerCfg | None = None):
        self.cfg = cfg or LinkLayerCfg()

        self.fronthaul_data: AccessLinkData | None = None
        self.backhaul_data: BackhaulLinkData | None = None
        self._rng = np.random.default_rng(self.cfg.rng_seed)

        # Optional sector-based LoS cache.  Set via set_los_cache().
        self._los_cache: LosSectorCache | None = None
        self._obstacles: list[Obstacle] = []

    def set_los_cache(
        self,
        cfg: LosCacheCfg,
        obstacles: list[Obstacle],
    ) -> None:
        """
        Attach a sector-based LoS cache.

        Call this once after obstacles are loaded.  Subsequent calls to
        :meth:`update` will populate ``fronthaul_data.los`` and
        ``backhaul_data.los`` using the cache rather than random values.

        Args:
            cfg:       Cache configuration (sector size, env boundary).
            obstacles: Loaded obstacle list from ``ObstacleController``.
        """
        self._obstacles = obstacles
        self._los_cache = LosSectorCache(cfg)
        self._los_cache.reset()

    def update(self, ws: WorldState) -> None:
        """Update link layer data based on current world state.

        Computes distance matrices, in-range masks, height differences, and
        random LOS values for both access and backhaul links.

        Args:
            ws: Current world state containing UAV, BS, and UE positions.

        Raises:
            RuntimeError: If initialize_data_arrays() has not been called.
        """
        if self.fronthaul_data is None or self.backhaul_data is None:
            raise RuntimeError(
                "Call initialize_data_arrays(world_cfg) before update()."
            )

        # ------------------------------------------------------------------
        # IMPORTANT ORDER: S = [UAVs ; BSs]
        # Shapes:
        #   ws.uav_pos: (N_UAV, 3)
        #   ws.bs_pos : (N_BS, 3)
        #   s_pos     : (N_S, 3) where N_S = N_UAV + N_BS
        # ------------------------------------------------------------------
        s_pos = np.vstack([ws.uav_pos, ws.bs_pos])

        # -----------------------------
        # Access: UE -> S
        # -----------------------------
        ue_s_diff = ws.ue_pos[:, None, :] - s_pos[None, :, :]  # (N_UE, N_S, 3)

        # Fill squared distances directly into preallocated buffer, then sqrt in-place
        np.sum(
            ue_s_diff * ue_s_diff, axis=2, out=self.fronthaul_data.dist_m
        )  # (N_UE, N_S)
        np.sqrt(self.fronthaul_data.dist_m, out=self.fronthaul_data.dist_m)

        self.fronthaul_data.in_range[:] = (
            self.fronthaul_data.dist_m <= self.cfg.access_range_m
        )

        # Height difference: H_node - H_ue
        # ue_s_diff is (ue - node), so we take -ue_s_diff[:, :, 2]
        np.negative(ue_s_diff[:, :, 2], out=self.fronthaul_data.height_diff)
        np.copyto(self.fronthaul_data.ue_height, ws.ue_pos[:, 2])
        np.copyto(
            self.fronthaul_data.bs_height[: ws.uav_pos.shape[0]], ws.uav_pos[:, 2]
        )
        np.copyto(self.fronthaul_data.bs_height[ws.uav_pos.shape[0] :], ws.bs_pos[:, 2])

        # LoS mask — use the sector cache when available, else random (mock).
        if self.fronthaul_data.los is not None:
            if self._los_cache is not None:
                # positions_a = UE positions (N_UE, 3)
                # positions_b = serving-node positions S = [UAVs; BSs] (N_S, 3)
                self.fronthaul_data.los[:] = self._los_cache.compute_los_mask(
                    positions_a=ws.ue_pos,
                    positions_b=s_pos,
                    obstacles=self._obstacles,
                )
            else:
                self.fronthaul_data.los[:] = (
                    self._rng.random(self.fronthaul_data.los.shape) < 0.5
                )

        # -----------------------------
        # Backhaul: S -> S (symmetric)
        # -----------------------------
        s_s_diff = s_pos[:, None, :] - s_pos[None, :, :]  # (N_S, N_S, 3)

        np.sum(s_s_diff * s_s_diff, axis=2, out=self.backhaul_data.dist_m)  # (N_S, N_S)
        np.sqrt(self.backhaul_data.dist_m, out=self.backhaul_data.dist_m)

        self.backhaul_data.in_range[:] = (
            self.backhaul_data.dist_m <= self.cfg.backhaul_range_m
        )
        np.copyto(self.backhaul_data.bs_height[: ws.uav_pos.shape[0]], ws.uav_pos[:, 2])
        np.copyto(self.backhaul_data.bs_height[ws.uav_pos.shape[0] :], ws.bs_pos[:, 2])

        if self.backhaul_data.los is not None:
            if self._los_cache is not None:
                # Both axes of the backhaul matrix are serving-node positions.
                self.backhaul_data.los[:] = self._los_cache.compute_los_mask(
                    positions_a=s_pos,
                    positions_b=s_pos,
                    obstacles=self._obstacles,
                )
            else:
                self.backhaul_data.los[:] = (
                    self._rng.random(self.backhaul_data.los.shape) < 0.5
                )

    def initialize_data_arrays(self, world_cfg: WorldStateCfg) -> None:
        """Initialize pre-allocated data arrays based on world configuration.

        Creates numpy arrays for distances, in-range masks, LOS values, and
        other link metrics for both access (UE to serving nodes) and backhaul
        (serving node to serving node) links.

        Args:
            world_cfg: World configuration containing numbers of UEs, UAVs, and BSs.
        """
        n_ue = world_cfg.n_ues
        n_uav = world_cfg.n_uavs
        n_bs = world_cfg.n_bss

        n_s = n_uav + n_bs  # S = [UAVs ; BSs]

        # -----------------------------
        # Access (UE -> S)
        # -----------------------------
        dist_access = np.empty((n_ue, n_s), dtype=np.float32)
        in_range_access = np.empty((n_ue, n_s), dtype=np.bool_)
        los_access = np.empty((n_ue, n_s), dtype=np.bool_)

        # link_type matches S ordering:
        # 1 = UAV, 0 = BS  (change mapping if you prefer, but keep it consistent)
        link_type_access = np.empty((n_s,), dtype=np.uint8)
        link_type_access[:n_uav] = 0  # UAVs first
        link_type_access[n_uav:] = 1  # BSs after

        self.fronthaul_data = AccessLinkData(
            dist_m=dist_access,
            in_range=in_range_access,
            height_diff=np.empty((n_ue, n_s), dtype=np.float32),
            ue_height=np.empty(n_ue, dtype=np.float32),
            bs_height=np.empty(n_s, dtype=np.float32),
            los=los_access,
            link_type=link_type_access,
        )

        # -----------------------------
        # Backhaul (S -> S)
        # -----------------------------
        dist_bh = np.empty((n_s, n_s), dtype=np.float32)
        in_range_bh = np.empty((n_s, n_s), dtype=np.bool_)
        los_bh = np.empty((n_s, n_s), dtype=np.bool_)

        link_type_bh = np.empty((n_s,), dtype=np.uint8)
        link_type_bh[:n_uav] = 0
        link_type_bh[n_uav:] = 1

        self.backhaul_data = BackhaulLinkData(
            dist_m=dist_bh,
            in_range=in_range_bh,
            bs_height=np.empty(n_s, dtype=np.float32),
            los=los_bh,
            link_type=link_type_bh,
        )

"""
Backhaul Layer
--------------

This module implements the backhaul layer responsible for computing
channel characteristics between base stations.

It supports multiple backhaul channel models (FSO and mmWave) and
computes the path loss, received power, and achievable throughput
for each available backhaul link.

For a detailed mathematical description, see:
- docs/backhaul/channel-models.md
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from src.outcast.backhaul.bh_channel_data import BHChannelData
from src.outcast.backhaul.bh_config import BHLayerCfg
from src.outcast.backhaul.bh_fso_channel_model import StatisticalModel
from src.outcast.backhaul.bh_mmwave_channel_model import MmWaveModel
from src.outcast.geometry.coords import Coords3d
from src.outcast.link_layer.link_data import BackhaulLinkData
from src.outcast.utils.math_tools import lin2db
from src.outcast.world.world_state import WorldStateCfg


class BHLayer:
    def __init__(self, cfg: BHLayerCfg | None = None):
        self.cfg = BHLayerCfg() if cfg is None else cfg
        self.bh_channel_data: BHChannelData | None = None

        self._flow_residual: NDArray[np.float32] | None = None
        self._adjacency: NDArray[np.bool_] | None = None

        self.n_uavs: int = 0
        self.n_bss: int = 0

    def initialize_data_arrays(self, world_cfg: WorldStateCfg):
        """
        Preallocate arrays to optimize speed by avoiding allocation each timestep.
        """
        self.n_uavs = int(world_cfg.n_uavs)
        self.n_bss = int(world_cfg.n_bss)

        n = self.n_uavs + self.n_bss
        if n <= 0:
            raise ValueError("world_cfg.n_uavs + world_cfg.n_bss must be > 0")

        shape = (n, n)

        self.bh_channel_data = BHChannelData(
            path_loss_db=np.full(shape, np.inf, dtype=np.float32),
            received_power_dbm=np.full(shape, -np.inf, dtype=np.float32),
            throughput_bps=np.zeros(shape, dtype=np.uint64),
            flow_bps=np.zeros(shape, dtype=np.uint64),
            excess_bps=np.zeros(shape, dtype=np.uint64),
            missing_bps=np.zeros(shape, dtype=np.uint64),
        )

        self._flow_residual = np.zeros((n,), dtype=np.int64)
        self._adjacency = np.zeros(shape, dtype=np.bool_)

    def update_bh_channel_data(self, backhaul_link_data: BackhaulLinkData):
        """
        Using backhaul link data from the link layer, populate self.bh_channel_data in-place.

        Expected inputs:
        - in_range: (N, N)
        - dist_m: (N, N) full 3D distances
        - bs_height: (N,N) node heights
        - los: (N, N), optional
        """
        if self.bh_channel_data is None:
            raise RuntimeError(
                "Call initialize_data_arrays(world_cfg) before update_bh_channel_data().",
            )

        in_range = backhaul_link_data.in_range
        dist_m = backhaul_link_data.dist_m
        bs_height = backhaul_link_data.bs_height

        n = self.bh_channel_data.throughput_bps.shape[0]
        if (
            in_range.shape != (n, n)
            or dist_m.shape != (n, n)
            or bs_height.shape != (n,)
        ):
            raise ValueError(
                "BackhaulLinkData shapes do not match preallocated BHChannelData shape.",
            )

        # Clear outputs in-place
        self.bh_channel_data.path_loss_db[:, :].fill(np.inf)
        self.bh_channel_data.received_power_dbm[:, :].fill(-np.inf)
        self.bh_channel_data.throughput_bps[:, :].fill(0.0)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not bool(in_range[i, j]):
                    continue

                d_3d = float(dist_m[i, j])

                if self.cfg.channel_model == 0:
                    # Height fix for FSO:
                    # reconstruct horizontal distance from 3D distance and node heights
                    tx_h = float(bs_height[i])
                    rx_h = float(bs_height[j])
                    d_hor = float(np.sqrt(max(d_3d * d_3d - (tx_h - rx_h) ** 2, 0.0)))

                    tx = Coords3d(0.0, 0.0, tx_h)
                    rx = Coords3d(d_hor, 0.0, rx_h)

                    gain_db, rx_power_w, cap_bps = (
                        StatisticalModel.get_charge_power_and_capacity(
                            tx,
                            rx,
                            cfg=self.cfg.fso,
                        )
                    )

                    path_loss_db = -float(gain_db)
                    received_power_dbm = (
                        lin2db(float(rx_power_w) * 1e3) if rx_power_w > 0.0 else -np.inf
                    )
                    throughput = max(0.0, float(cap_bps))

                elif self.cfg.channel_model == 1:
                    # mmWave model
                    # If your mmWave model supports LOS/NLOS explicitly, adjust the call here.
                    path_loss_db, received_power_dbm, throughput = (
                        MmWaveModel.get_path_loss_power_and_capacity(
                            d_3d,
                            cfg=self.cfg.mmwave,
                        )
                    )

                else:
                    raise ValueError(
                        f"Unsupported backhaul channel model: {self.cfg.channel_model}",
                    )

                self.bh_channel_data.path_loss_db[i, j] = np.float32(path_loss_db)
                self.bh_channel_data.received_power_dbm[i, j] = np.float32(
                    received_power_dbm,
                )
                self.bh_channel_data.throughput_bps[i, j] = np.uint64(throughput)

    def set_flow(self, flow_bps: NDArray, adjacency: NDArray | None = None) -> None:
        """
        Store routing flow matrix in-place.

        Parameters
        ----------
        flow_bps : (N,N)
            Traffic flow matrix.
        adjacency : (N,N), optional
            Directed backhaul adjacency used by the controller/routing logic.
        """
        if self.bh_channel_data is None:
            raise RuntimeError("Call initialize_data_arrays() first.")

        if flow_bps.shape != self.bh_channel_data.flow_bps.shape:
            raise ValueError("flow_bps shape mismatch")

        np.copyto(self.bh_channel_data.flow_bps, flow_bps.astype(np.uint64, copy=False))

        if adjacency is not None:
            if self._adjacency is None:
                self._adjacency = np.zeros_like(adjacency, dtype=np.bool_)
            if adjacency.shape != self.bh_channel_data.flow_bps.shape:
                raise ValueError("adjacency shape mismatch")
            np.copyto(self._adjacency, adjacency.astype(np.bool_))

    def compute_excess_and_missing(self, in_range: NDArray | None = None) -> None:
        """
        Compute per-link excess/missing throughput in-place:

        Zero out values on non-used links if an adjacency matrix is available,
        otherwise optionally use in_range.
        """
        if self.bh_channel_data is None:
            raise RuntimeError("Call initialize_data_arrays() first.")

        cap = self.bh_channel_data.throughput_bps
        flow = self.bh_channel_data.flow_bps
        excess = self.bh_channel_data.excess_bps
        missing = self.bh_channel_data.missing_bps

        np.subtract(cap, flow, out=excess, where=cap >= flow)
        np.copyto(excess, 0, where=cap < flow)

        np.subtract(flow, cap, out=missing, where=flow >= cap)
        np.copyto(missing, 0, where=flow < cap)

        if self._adjacency is not None:
            mask = (~self._adjacency) | (flow == 0)
            excess[mask] = 0
            missing[mask] = 0
        elif in_range is not None:
            mask = (~in_range.astype(bool)) | (flow == 0)
            excess[mask] = 0
            missing[mask] = 0

    def check_flow_conservation(
        self,
        g_load: NDArray[np.uint64] | None = None,
        ignore_nodes: Iterable[int] | None = None,
        tol: float = 1e-3,
    ) -> tuple[bool, NDArray[np.float32], float]:
        """
        Check flow conservation:
            residual[k] = outflow(k) - inflow(k) - g_load(k)

        Here, g_load denotes the external injected traffic at node k.

        Returns:
            (is_conserved, residual_per_node, max_abs_residual)
        """
        if self.bh_channel_data is None or self._flow_residual is None:
            raise RuntimeError("Call initialize_data_arrays() first.")

        flow = self.bh_channel_data.flow_bps
        res = self._flow_residual
        n = res.shape[0]

        inflow = flow.sum(axis=0, dtype=np.int64)
        outflow = flow.sum(axis=1, dtype=np.int64)

        res[:] = outflow - inflow

        if g_load is not None:
            if g_load.shape[0] != n:
                raise ValueError("g_load shape mismatch")
            res[:] = res - g_load.astype(np.int64, copy=False)

        if ignore_nodes is not None:
            for k in ignore_nodes:
                if 0 <= k < n:
                    res[k] = 0.0

        max_abs = float(np.max(np.abs(res)))
        return (max_abs <= tol), res, max_abs

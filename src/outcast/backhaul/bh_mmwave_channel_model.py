"""
Backhaul mmWave Channel Model
-----------------------------

This module implements a simple mmWave backhaul channel model and computes:

- path loss [dB]
- received power [dBm]
- achievable throughput [bps]

The model is intended to be called by the backhaul layer, similarly to the
existing FSO channel model.
"""

from __future__ import annotations

import numpy as np

from src.uavnetsim.backhaul.bh_config import BHMmWaveChannelCfg


class MmWaveModel:
    """Simple LOS-only mmWave backhaul model."""

    @staticmethod
    def get_path_loss_power_and_capacity(
        distance_m: float,
        cfg: BHMmWaveChannelCfg | None = None,
    ) -> tuple[float, float, float]:
        """
        Compute mmWave path loss, received power, and achievable throughput.
        Implements the LOS-only mmWave model used in the simulator.
        Implements equation (17), (18) and (19) from [1].
        Args:
            distance_m: Link distance in meters
        Returns:
            tuple:
                path_loss_db (float)
                received_power_dbm (float)
                throughput_bps (float)
        """

        if distance_m <= 0.0:
            return np.inf, -np.inf, 0.0

        cfg = BHMmWaveChannelCfg() if cfg is None else cfg
        freq_ghz = cfg.frequency_hz / 1e9

        path_loss_db = (
            32.4
            + 20.0 * np.log10(freq_ghz)
            + 10.0 * cfg.path_loss_exponent * np.log10(distance_m)
        )

        path_loss_lin = 10.0 ** (path_loss_db / 10.0)

        rx_power_w = cfg.tx_power_w / path_loss_lin
        received_power_dbm = (
            10.0 * np.log10(rx_power_w * 1e3) if rx_power_w > 0.0 else -np.inf
        )

        throughput_bps = (
            cfg.efficiency
            * cfg.bandwidth_hz
            * np.log2(1.0 + cfg.tx_power_w / (path_loss_lin * cfg.noise_power_w))
        )

        if not np.isfinite(throughput_bps) or throughput_bps < 0.0:
            throughput_bps = 0.0

        return float(path_loss_db), float(received_power_dbm), float(throughput_bps)

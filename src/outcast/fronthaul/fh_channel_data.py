"""Fronthaul channel data containers for UAV simulation.

This module provides:
- FHChannelData: Dataclass for storing fronthaul channel metrics
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class FHChannelData:
    """Arrays describing UE -> UAV / BS links.

    Attributes:
        path_loss_db: Path loss in dB for each link (N_UE, N_S).
        total_received_power_dbm: Total received power in dBm (N_UE, N_S).
        ue_received_power_dbm: UE received power in dBm derived from PSD (N_UE, N_S).
        interference_dbm: Interference power in dBm (N_UE, N_S).
        snr_db: Signal-to-noise ratio in dB (N_UE, N_S).
        sinr_db: Signal-to-interference-plus-noise ratio in dB (N_UE, N_S).
        throughput_bps: Achievable throughput in bits per second (N_UE, N_S).
        assigned_bandwidth_hz: Assigned bandwidth per UE in Hz (N_UE,).
    """

    path_loss_db: NDArray[np.float32]  # (N_UE, N_S [UAV, BS])
    total_received_power_dbm: NDArray[np.float32]  # (N_UE, N_S [UAV, BS])
    ue_received_power_dbm: NDArray[
        np.float32
    ]  # (N_UE, N_S [UAV, BS]), derived from PSD
    interference_dbm: NDArray[np.float32]  # (N_UE, N_S [UAV, BS])
    snr_db: NDArray[np.float32]  # (N_UE, N_S [UAV, BS])
    sinr_db: NDArray[np.float32]  # (N_UE, N_S [UAV, BS])
    throughput_bps: NDArray[np.uint64]  # (N_UE, N_S [UAV, BS])
    assigned_bandwidth_hz: NDArray[np.uint32]  # (N_UE,)

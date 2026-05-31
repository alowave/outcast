"""
Arrays describing backhaul node <-> node links (N, N).
Convention:
- path_loss_db is a positive quantity (dB). We store it as: path_loss_db = -gain_db,
  where gain_db is the total channel gain returned by the FSO channel model.
- received_power_dbm is receiver electrical/optical power expressed in dBm.
- throughput_bps is the achievable link throughput/capacity in bps.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BHChannelData:
    path_loss_db: NDArray[np.float32]  # (N, N)
    received_power_dbm: NDArray[np.float32]  # (N, N)
    throughput_bps: NDArray[np.uint64]  # (N, N)

    flow_bps: NDArray[np.uint64]  # (N, N)
    excess_bps: NDArray[np.uint64]  # (N, N)
    missing_bps: NDArray[np.uint64]  # (N, N)

"""Link-layer data containers for access and backhaul link arrays."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class AccessLinkData:
    """Arrays describing UE -> UAV / BS links.

    Attributes:
        dist_m: 3D distance in meters (N_UE, N_S).
        in_range: Boolean mask for links within range (N_UE, N_S).
        height_diff: Height difference (Source_z - UE_z) in meters (N_UE, N_S).
        ue_height: UE heights in meters (N_UE,).
        bs_height: Source heights in meters (N_S,).
        los: Optional Line-of-Sight boolean mask (N_UE, N_S).
        link_type: Optional link type array (N_S,) where 0=UAV, 1=BS.
    """

    dist_m: NDArray[np.float32]  # (N_UE, N_S [UAV, BS])
    in_range: NDArray[np.bool_]  # (N_UE, N_S [UAV, BS]) bool
    height_diff: NDArray[np.float32]  # (N_UE, N_S [UAV, BS]) float32
    ue_height: NDArray[np.float32]  # (N_UE,) float32
    bs_height: NDArray[np.float32]  # (N_S [UAV, BS],) float32
    los: NDArray[np.bool_] | None = None  # (N_UE, N_S [UAV, BS]) bool (optional)
    link_type: NDArray[np.uint8] | None = (
        None  # (N_S [UAV, BS],) uint8 (0 - UAV, 1- BS, other..)
    )


@dataclass(slots=True)
class BackhaulLinkData:
    """Arrays describing UAV -> UAV (backhaul) links.

    Attributes:
        dist_m: 3D distance in meters (N_S, N_S).
        in_range: Boolean mask for links within range (N_S, N_S).
        bs_height: Source heights in meters (N_S,).
        los: Optional Line-of-Sight boolean mask (N_S, N_S).
        link_type: Optional link type array (N_S,) for backhaul links.
    """

    dist_m: NDArray[np.float32]  # (N_S [UAV, BS], N_S [UAV, BS]) float32
    in_range: NDArray[np.bool_]  # (N_S [UAV, BS], N_S [UAV, BS]) bool
    bs_height: NDArray[np.float32]  # (N_S [UAV, BS],) float32
    los: NDArray[np.bool_] | None = (
        None  # (N_S [UAV, BS], N_S [UAV, BS]) bool (optional)
    )
    link_type: NDArray[np.uint8] | None = (
        None  # (N_S [UAV, BS],) uint8 (0- UAV-UAV, 1- UAV-BS, other..)
    )

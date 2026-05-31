"""Fronthaul Channel Models Module.

This module implements the Probabilistic Line-of-Sight (PLOS) model for UAV-to-Ground
communication, and the Urban Macro (UMa) model for Ground-to-Ground communication, easing the calculation of path loss,
optimal deployment heights, and coverage radii.

For a detailed theoretical background, mathematical derivations, and references,
please refer to the documentation in:
/docs/fronthaul/channel-models.md
"""

import numpy as np
from numpy.typing import NDArray

from src.uavnetsim.fronthaul.fh_config import FHChannelCfg, FHLayerCfg
from src.uavnetsim.fronthaul.models.model_3gpp import UMaModel
from src.uavnetsim.fronthaul.models.model_simple_plos import PlosModel


class FHPathLossModelManager:
    """Manager class for Fronthaul Path Loss calculations.

    Delegates A2G links to the Probabilistic Line-of-Sight (PLOS) model
    and G2G links to the Urban Macro (UMa) model.
    """

    DEFAULT_CHANNEL_CFG = FHChannelCfg()
    DEFAULT_LAYER_CFG = FHLayerCfg()

    def __init__(self, channel_cfg: FHChannelCfg | None = None):
        """Initializes the Fronthaul Path Loss Model Manager.

        Args:
            channel_cfg: Configuration object for fronthaul channel parameters.
                If None, uses DEFAULT_CHANNEL_CFG.
        """
        self.channel_cfg = channel_cfg or self.DEFAULT_CHANNEL_CFG
        self.plos_model = PlosModel(self.channel_cfg)
        self.uma_model = UMaModel(self.channel_cfg)

    def get_vectorized_path_loss(
        self,
        dist_m: NDArray[np.float32],
        frequencies: NDArray[np.float32],
        height_diff: NDArray[np.float32] | None = None,
        is_los: NDArray[np.bool_] | None = None,
        link_type: NDArray[np.uint8] | None = None,
        ue_height: NDArray[np.float32] | None = None,
        bs_height: NDArray[np.float32] | None = None,
        uma_height_mode: str | None = None,
        enable_shadowing: bool | None = None,
        rng: np.random.Generator | None = None,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Calculates path loss using vectorized operations for A2G and G2G models.

        This method optimizes for time-sensitive simulations by using NumPy vectorized
        operations. It supports both the Probabilistic Line-of-Sight (PLOS) model for
        Air-to-Ground (A2G) links and the Urban Macro (UMa) model (TR 38.901) for
        Ground-to-Ground (G2G) links.

        Args:
            dist_m: 3D distance between UE and source (N_UE, N_S).
            frequencies: Carrier frequencies for each source in Hz (1, N_S).
            height_diff: Vertical height difference (Source_z - UE_z) in meters (N_UE, N_S).
                Required for A2G probabilistic calculation or G2G LOS probability.
            is_los: Optional boolean mask for Line-of-Sight conditions (N_UE, N_S).
                If provided, deterministic calculation is performed.
            link_type: Optional mask: 0 for UAV (A2G), 1 for BS (G2G). (1, N_S) or (N_S,).
                Defaults to A2G (0) for all links if None.
            ue_height: Optional per-user height array in meters with shape `(N_UE,)`.
                If omitted, a `(N_UE,)` array is created from `channel_cfg.default_ue_height_m`.
            bs_height: Optional per-source height array in meters with shape `(N_S,)`,
                ordered as `[UAVs, BSs]`. If provided, it can be used to derive
                `height_diff` when that matrix is not passed.
            uma_height_mode: Optional override for the UMa effective-environment-height
                behavior. Falls back to `channel_cfg.uma.effective_env_height_mode`.
            enable_shadowing: Optional override for UMa LOS/NLOS shadowing.
                Falls back to `channel_cfg.uma.enable_shadowing`.
            rng: Optional NumPy generator used when `enable_shadowing` is enabled.
            out: Optional output array to store results [N_UE, N_S].

        Returns:
            NDArray[np.float32]: Path loss in dB (N_UE, N_S).

        Raises:
            ValueError: If neither 'is_los' nor 'height_diff' is provided when needed.
        """
        n_ue = dist_m.shape[0]
        n_s = dist_m.shape[1]

        # Parse UE Heights
        if ue_height is None:
            ue_height = np.full(
                n_ue, self.channel_cfg.default_ue_height_m, dtype=np.float32
            )
        else:
            ue_height = np.asarray(ue_height, dtype=np.float32)
            if ue_height.ndim == 0:
                ue_height = np.full(n_ue, float(ue_height), dtype=np.float32)
            elif ue_height.shape != (n_ue,):
                raise ValueError(
                    f"ue_height must have shape ({n_ue},), got {ue_height.shape}."
                )

        # Parse Source Heights
        if bs_height is None:
            if link_type is None:
                bs_height = np.full(
                    n_s, self.channel_cfg.default_uav_height_m, dtype=np.float32
                )
            else:
                link_type_flat = np.atleast_1d(link_type).flatten()
                if link_type_flat.shape != (n_s,):
                    raise ValueError(
                        f"link_type must have shape ({n_s},), got {link_type_flat.shape}."
                    )
                bs_height = np.full(
                    n_s, self.channel_cfg.default_bs_height_m, dtype=np.float32
                )
                bs_height[link_type_flat == 0] = self.channel_cfg.default_uav_height_m
        else:
            bs_height = np.asarray(bs_height, dtype=np.float32)
            if bs_height.ndim == 0:
                bs_height = np.full(n_s, float(bs_height), dtype=np.float32)
            elif bs_height.shape != (n_s,):
                raise ValueError(
                    f"bs_height must have shape ({n_s},), got {bs_height.shape}."
                )

        if out is None:
            out = np.empty_like(dist_m)

        uma_cfg = self.channel_cfg.uma
        uma_height_mode = (
            uma_cfg.effective_env_height_mode
            if uma_height_mode is None
            else uma_height_mode
        )
        enable_shadowing = (
            uma_cfg.enable_shadowing if enable_shadowing is None else enable_shadowing
        )
        if uma_height_mode not in {"expected", "probabilistic"}:
            raise ValueError("uma_height_mode must be 'expected' or 'probabilistic'.")
        rng = rng or np.random.default_rng()

        # 1. Distinguish between A2G (UAV) and G2G (BS) links
        if link_type is None:
            mask_a2g = slice(None)
            any_a2g = True
            any_g2g = False
        else:
            link_type_flat = np.atleast_1d(link_type).flatten()
            mask_a2g = link_type_flat == 0
            mask_g2g = link_type_flat == 1
            any_a2g = np.any(mask_a2g)
            any_g2g = np.any(mask_g2g)

        # 2. Process A2G Links (PLOS Model)
        if any_a2g:
            self.plos_model.calculate_path_loss_a2g(
                dist_m=dist_m,
                frequencies=frequencies,
                height_diff=height_diff,
                is_los=is_los,
                ue_height=ue_height,
                bs_height=bs_height,
                mask_a2g=mask_a2g,
                out=out,
            )

        # 3. Process G2G Links (UMa Model TR 38.901)
        if any_g2g:
            self.uma_model.calculate_path_loss_g2g(
                dist_m=dist_m,
                frequencies=frequencies,
                height_diff=height_diff,
                is_los=is_los,
                ue_height=ue_height,
                bs_height=bs_height,
                mask_g2g=mask_g2g,
                uma_height_mode=uma_height_mode,
                enable_shadowing=enable_shadowing,
                rng=rng,
                out=out,
            )

        return out

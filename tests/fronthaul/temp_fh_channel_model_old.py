"""
Fronthaul Channel Models Module
===============================
This module implements the Probabilistic Line-of-Sight (PLOS) model for UAV-to-Ground
communication, easing the calculation of path loss, optimal deployment heights,
and coverage radiuses.

For a detailed theoretical background, mathematical derivations, and references,
please refer to the documentation in:
/docs/fronthaul/channel-models.md
"""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light
from scipy.optimize import brentq, fsolve

from src.outcast.fronthaul.fh_config import FHChannelCfg, FHLayerCfg
from src.outcast.geometry.coords import Coords3d
from src.outcast.utils.math_tools import db2lin, lin2db


class PlosModel:
    """
    Implementation of the Probabilistic Line-of-Sight (PLOS) model.

    This model characterizes the average path loss between a Low Altitude Platform (LAP)
    and a ground user by considering the weighted probability of LOS and NLOS conditions
    based on environmental geometry.

    References:
        - See /docs/fronthaul/channel-models.md for full details.
        - More: https://ieeexplore.ieee.org/document/6863654
    """

    DEFAULT_CHANNEL_CFG = FHChannelCfg()
    DEFAULT_LAYER_CFG = FHLayerCfg()

    def __init__(self, channel_cfg: FHChannelCfg | None = None):
        self.channel_cfg = channel_cfg or self.DEFAULT_CHANNEL_CFG
        self.env_type = self.channel_cfg.env_type
        self.env_a, self.env_b = self.get_a_b_params(channel_cfg=self.channel_cfg)

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
            out_a2g = out[:, mask_a2g]
            dist_a2g = dist_m[:, mask_a2g]
            freq_a2g = frequencies[:, mask_a2g]

            # Base FSPL: 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
            # np.maximum in case the distance is zero => log10(0) => error
            np.log10(np.maximum(dist_a2g, 1e-6), out=out_a2g)
            np.multiply(out_a2g, 20.0, out=out_a2g)

            const_part = 20 * np.log10((4 * np.pi * freq_a2g) / speed_of_light)
            np.add(out_a2g, const_part, out=out_a2g)

            # Handle unique frequencies for additional loss (eta_los, eta_nlos)
            unique_freqs = np.unique(freq_a2g)
            for f in unique_freqs:
                col_mask = freq_a2g[0] == f
                eta_los, eta_nlos = self.get_avg_loss(
                    env_type=self.env_type,
                    frequency=float(f),
                    channel_cfg=self.channel_cfg,
                )

                # Slices for this specific frequency band
                out_f = out_a2g[:, col_mask]

                if is_los is not None:
                    # Approach 1: Deterministic
                    additional_loss = np.where(
                        is_los[:, mask_a2g][:, col_mask], eta_los, eta_nlos
                    )
                elif height_diff is not None or bs_height is not None:
                    # Approach 2: Probabilistic
                    if height_diff is not None:
                        hd_f = height_diff[:, mask_a2g][:, col_mask]
                    else:
                        hd_f = (
                            bs_height[mask_a2g][np.newaxis, :][:, col_mask]
                            - ue_height[:, np.newaxis]
                        )
                    dist_2d = np.sqrt(
                        np.maximum(dist_a2g[:, col_mask] ** 2 - hd_f**2, 0)
                    )
                    theta_deg = np.degrees(np.arctan2(hd_f, dist_2d))
                    p_los = 1 / (
                        1 + self.env_a * np.exp(-self.env_b * (theta_deg - self.env_a))
                    )
                    additional_loss = (p_los * eta_los) + ((1 - p_los) * eta_nlos)
                else:
                    raise ValueError(
                        "For A2G, either 'is_los', 'height_diff', or 'bs_height' must be provided."
                    )

                out_f += additional_loss
                out_a2g[:, col_mask] = out_f  # __setitem__
            out[:, mask_a2g] = out_a2g  # __setitem__

        # 3. Process G2G Links (UMa Model TR 38.901)
        if any_g2g:
            dist_g2g = dist_m[:, mask_g2g]
            freq_g2g = frequencies[:, mask_g2g]  # Hz
            ue_height_col = ue_height[:, np.newaxis]
            bs_height_g2g = bs_height[mask_g2g][np.newaxis, :]
            g2g_result = np.empty_like(dist_g2g)

            # Conversion to GHz for UMa formulas (as per rf_g2g.py/TR 38.901)
            log_f = np.log10(freq_g2g / 1e9)

            if is_los is not None:
                if height_diff is not None:
                    hd_g2g = height_diff[:, mask_g2g]
                else:
                    hd_g2g = bs_height_g2g - ue_height_col
                distance_2d_g2g = np.sqrt(np.maximum(dist_g2g**2 - hd_g2g**2, 0))
                effective_env_height = self.get_uma_effective_environment_height(
                    distance_2d_g2g,
                    ue_height_col,
                    mode=uma_height_mode,
                    rng=rng,
                )
                effective_bs_height = bs_height_g2g - effective_env_height
                effective_ue_height = ue_height_col - effective_env_height
                breakpoint_distance = (
                    4.0
                    * effective_bs_height
                    * effective_ue_height
                    * freq_g2g
                    / speed_of_light
                )
                breakpoint_distance = np.maximum(breakpoint_distance, 1e-6)
                log_d_3d = np.log10(dist_g2g)
                pl1 = 28.0 + 22.0 * log_d_3d + 20.0 * log_f
                pl2 = (
                    28.0
                    + 40.0 * log_d_3d
                    + 20.0 * log_f
                    - 9.0 * np.log10(breakpoint_distance**2 + hd_g2g**2)
                )
                pl_los = np.where(distance_2d_g2g <= breakpoint_distance, pl1, pl2)
                pl_nlos = (
                    13.54 + 39.08 * log_d_3d + 20 * log_f - 0.6 * (ue_height_col - 1.5)
                )
                pl_nlos = np.maximum(pl_los, pl_nlos)
                if enable_shadowing:
                    pl_los = pl_los + rng.normal(
                        loc=0.0, scale=uma_cfg.shadowing_los_db, size=pl_los.shape
                    )
                    pl_nlos = pl_nlos + rng.normal(
                        loc=0.0, scale=uma_cfg.shadowing_nlos_db, size=pl_nlos.shape
                    )
                # Deterministic based on provided mask
                np.copyto(g2g_result, np.where(is_los[:, mask_g2g], pl_los, pl_nlos))
            elif height_diff is not None or bs_height is not None:
                # Probabilistic LOS for UMa
                if height_diff is not None:
                    hd_g2g = height_diff[:, mask_g2g]
                else:
                    hd_g2g = bs_height_g2g - ue_height_col
                dist_2d_g2g = np.sqrt(np.maximum(dist_g2g**2 - hd_g2g**2, 0))
                effective_env_height = self.get_uma_effective_environment_height(
                    dist_2d_g2g,
                    ue_height_col,
                    mode=uma_height_mode,
                    rng=rng,
                )
                effective_bs_height = bs_height_g2g - effective_env_height
                effective_ue_height = ue_height_col - effective_env_height
                breakpoint_distance = (
                    4.0
                    * effective_bs_height
                    * effective_ue_height
                    * freq_g2g
                    / speed_of_light
                )
                breakpoint_distance = np.maximum(breakpoint_distance, 1e-6)
                log_d_3d = np.log10(dist_g2g)
                pl1 = 28.0 + 22.0 * log_d_3d + 20.0 * log_f
                pl2 = (
                    28.0
                    + 40.0 * log_d_3d
                    + 20.0 * log_f
                    - 9.0 * np.log10(breakpoint_distance**2 + hd_g2g**2)
                )
                pl_los = np.where(dist_2d_g2g <= breakpoint_distance, pl1, pl2)
                pl_nlos = (
                    13.54
                    + 39.08 * log_d_3d
                    + 20.0 * log_f
                    - 0.6 * (ue_height_col - 1.5)
                )
                pl_nlos = np.maximum(pl_los, pl_nlos)
                if enable_shadowing:
                    pl_los = pl_los + rng.normal(
                        loc=0.0, scale=uma_cfg.shadowing_los_db, size=pl_los.shape
                    )
                    pl_nlos = pl_nlos + rng.normal(
                        loc=0.0, scale=uma_cfg.shadowing_nlos_db, size=pl_nlos.shape
                    )
                p_los_uma = self.get_uma_los_probability(dist_2d_g2g, ue_height_col)
                np.copyto(g2g_result, p_los_uma * pl_los + (1.0 - p_los_uma) * pl_nlos)
            else:
                raise ValueError(
                    "For G2G, either 'is_los', 'height_diff', or 'bs_height' must be provided."
                )

            out[:, mask_g2g] = g2g_result

        return out

    def get_path_loss(
        self,
        ue_coords: Coords3d,
        bs_coords: Coords3d = Coords3d(0, 0, 0),
        frequency: float = 2e9,
    ):
        """
        Calculates the expected path loss in linear scale.

        This relies on Equation (3) from the documentation, summing the
        Free Space Path Loss (FSPL) - Equation (2) - with the expected
        additional environmental losses based on the LOS probability.

        Args:
            ue_coords: Coordinates of the User Equipment (UE).
            bs_coords: Coordinates of the Base Station (BS). Defaults to (0, 0, 0).
            frequency: Carrier frequency in Hz. Defaults to DEFAULT_CARRIER_FREQ_MBS.

        Returns:
            The expected path loss as a linear value.
        """
        distance_2d = ue_coords.get_distance_to(bs_coords, flag_2d=True)
        distance_3d = np.sqrt(distance_2d**2 + (bs_coords.z - ue_coords.z) ** 2)
        los_probability = self.get_los_probability(
            abs(bs_coords.z - ue_coords.z),
            distance_2d,
        )
        avg_los_loss, avg_nlos_loss = self.channel_cfg.plos.resolve_avg_loss(
            self.env_type, frequency
        )
        path_loss = (
            20 * np.log10(4 * np.pi * frequency * distance_3d / speed_of_light)
            + los_probability * avg_los_loss
            + (1 - los_probability) * avg_nlos_loss
        )
        return db2lin(path_loss)

    def get_los_probability(
        self,
        height,
        distance_2d,
        a_param: float | None = None,
        b_param: float | None = None,
    ):
        """
        Computes the probability of having a clear Line-of-Sight (LOS) link.

        Implements Equation (1) from the documentation, which uses a modified
        sigmoid function dependent on the elevation angle and the environmental
        parameters alpha (a_param) and beta (b_param).

        Args:
            height: Vertical height difference between the UAV and ground user (m).
            distance_2d: Horizontal distance between the UAV and ground user (m).
            a_param: Environmental parameter alpha. Defaults to PLOS_A_PARAM.
            b_param: Environmental parameter beta. Defaults to PLOS_B_PARAM.

        Returns:
            The probability of a LOS link (0 to 1).
        """
        if a_param is None:
            a_param = self.env_a
        if b_param is None:
            b_param = self.env_b
        return 1 / (
            1
            + a_param
            * np.exp(
                -b_param * (180 / np.pi * np.arctan(height / distance_2d) - a_param)
            )
        )

    @staticmethod
    def get_uma_los_probability(
        distance_2d: NDArray[np.float32] | np.ndarray,
        ue_height: NDArray[np.float32] | np.ndarray,
    ) -> NDArray[np.float32]:
        p_los = np.ones_like(distance_2d, dtype=np.float32)
        mask_d_gt_18 = distance_2d > 18.0
        if np.any(mask_d_gt_18):
            d2d_m = distance_2d[mask_d_gt_18]
            ue_height_2d = np.broadcast_to(ue_height, distance_2d.shape)
            ue_h = ue_height_2d[mask_d_gt_18]
            c_prime = np.zeros_like(ue_h, dtype=np.float32)
            tall_mask = ue_h > 13.0
            c_prime[tall_mask] = ((ue_h[tall_mask] - 13.0) / 10.0) ** 1.5
            p_los[mask_d_gt_18] = (
                18.0 / d2d_m + np.exp(-d2d_m / 63.0) * (1.0 - 18.0 / d2d_m)
            ) * (1.0 + c_prime * 1.25 * (d2d_m / 100.0) ** 3 * np.exp(-d2d_m / 150.0))
        return np.clip(p_los, 0.0, 1.0)

    @staticmethod
    def get_uma_effective_environment_height(
        distance_2d: NDArray[np.float32] | np.ndarray,
        ue_height: NDArray[np.float32] | np.ndarray,
        mode: str = "expected",
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float32]:
        if mode == "expected":
            return PlosModel.get_uma_effective_environment_height_expected(
                distance_2d, ue_height
            )
        if mode == "probabilistic":
            return PlosModel.get_uma_effective_environment_height_probabilistic(
                distance_2d,
                ue_height,
                rng=rng,
            )
        raise ValueError("mode must be 'expected' or 'probabilistic'.")

    @staticmethod
    def get_uma_effective_environment_height_expected(
        distance_2d: NDArray[np.float32] | np.ndarray,
        ue_height: NDArray[np.float32] | np.ndarray,
    ) -> NDArray[np.float32]:
        effective_env_height = np.ones_like(distance_2d, dtype=np.float32)
        ue_height_2d = np.broadcast_to(ue_height, distance_2d.shape)
        eligible_mask = (ue_height >= 13.0) & (distance_2d > 18.0)
        if not np.any(eligible_mask):
            return effective_env_height

        eligible_dist = distance_2d[eligible_mask]
        eligible_height = ue_height_2d[eligible_mask]
        c_term = np.clip(
            1.25
            * (eligible_dist / 100.0) ** 3
            * np.exp(-eligible_dist / 150.0)
            * ((eligible_height - 13.0) / 10.0) ** 1.5,
            0.0,
            None,
        )
        probability_one_meter = 1.0 / (1.0 + c_term)
        max_env_height = np.maximum(eligible_height - 1.5, 12.0)
        mean_discrete_height = 0.5 * (12.0 + max_env_height)
        effective_env_height[eligible_mask] = (
            probability_one_meter * 1.0
            + (1.0 - probability_one_meter) * mean_discrete_height
        ).astype(np.float32)
        return effective_env_height

    @staticmethod
    def get_uma_effective_environment_height_probabilistic(
        distance_2d: NDArray[np.float32] | np.ndarray,
        ue_height: NDArray[np.float32] | np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float32]:
        rng = rng or np.random.default_rng()
        effective_env_height = np.ones_like(distance_2d, dtype=np.float32)
        ue_height_2d = np.broadcast_to(ue_height, distance_2d.shape)
        eligible_mask = (ue_height >= 13.0) & (distance_2d > 18.0)
        if not np.any(eligible_mask):
            return effective_env_height

        eligible_dist = distance_2d[eligible_mask]
        eligible_height = ue_height_2d[eligible_mask]
        c_term = np.clip(
            1.25
            * (eligible_dist / 100.0) ** 3
            * np.exp(-eligible_dist / 150.0)
            * ((eligible_height - 13.0) / 10.0) ** 1.5,
            0.0,
            None,
        )
        probability_one_meter = 1.0 / (1.0 + c_term)
        use_one_meter = (
            rng.random(size=probability_one_meter.shape) < probability_one_meter
        )
        sampled_height = np.empty_like(eligible_height, dtype=np.float32)
        sampled_height[use_one_meter] = 1.0

        non_one_mask = ~use_one_meter
        if np.any(non_one_mask):
            min_height = np.full(np.count_nonzero(non_one_mask), 12.0, dtype=np.float32)
            max_height = eligible_height[non_one_mask] - 1.5
            max_height = np.maximum(max_height, 12.0)
            random_steps = rng.integers(
                low=0,
                high=np.floor((max_height - min_height) / 3.0).astype(np.int32) + 1,
            )
            sampled_height[non_one_mask] = min_height + 3.0 * random_steps.astype(
                np.float32
            )

        effective_env_height[eligible_mask] = sampled_height
        return effective_env_height

    @staticmethod
    def get_a_b_params(
        env_type: str | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        channel_cfg: FHChannelCfg | None = None,
    ):
        """
        Retrieves or calculates the alpha and beta environmental parameters.

        If standard ITU parameters are not provided, it maps predefined urban
        environments to their respective alpha, beta, and gamma values. It then
        uses the surface polynomial fitting from Equation (12) and the coefficient
        Tables to compute the final S-curve parameters.

        Args:
            env_type: Type of environment ('Suburban', 'Urban', 'Dense Urban', or 'Highrise Urban'). Defaults to 'Urban'.
            alpha: Optional manually specified ITU alpha parameter.
            beta: Optional manually specified ITU beta parameter.
            gamma: Optional manually specified ITU gamma parameter.

        Returns:
            tuple (a, b) representing the fitted S-curve parameters.

        Raises:
            ValueError: If an undefined environment type is provided.

        More: https://ieeexplore.ieee.org/document/6863654
        """
        cfg = channel_cfg or PlosModel.DEFAULT_CHANNEL_CFG
        env_type = env_type or cfg.env_type
        if alpha is None and beta is None and gamma is None:
            if env_type == cfg.env_type:
                alpha, beta, gamma = cfg.plos.resolve_env_params(cfg.env_type)
            else:
                local_cfg = FHChannelCfg(
                    env_type=env_type,
                    default_ue_height_m=cfg.default_ue_height_m,
                    default_bs_height_m=cfg.default_bs_height_m,
                    default_uav_height_m=cfg.default_uav_height_m,
                    plos=cfg.plos,
                    uma=cfg.uma,
                )
                alpha, beta, gamma = local_cfg.plos.resolve_env_params(
                    local_cfg.env_type
                )
        elif alpha is None or beta is None or gamma is None:
            raise ValueError("alpha, beta, and gamma must all be provided together.")

        if env_type not in cfg.plos.env_profiles:
            raise ValueError(
                f"Undefined environment type: {env_type}. "
                "Choose one of: Suburban, Urban, Dense Urban, Highrise Urban."
            )

        cij_a = [
            [9.34e-1, 2.3e-1, -2.25e-3, 1.86e-5],
            [1.97e-2, 2.44e-3, 6.58e-6, 0],
            [-1.24e-4, -3.34e-6, 0, 0],
            [2.73e-7, 0, 0, 0],
        ]
        cij_b = [
            [1.17, -7.56e-2, 1.98e-3, -1.78e-5],
            [-5.79e-3, 1.81e-4, -1.65e-6, 0],
            [1.73e-5, -2.02e-7, 0, 0],
            [-2e-8, 0, 0, 0],
        ]

        def get_fitting_parameter(cij_matrix):
            sum_ = 0
            for i in range(4):
                for j in range(4):
                    sum_ += cij_matrix[i][j] * (alpha * beta) ** i * gamma**j
            return sum_

        a, b = get_fitting_parameter(cij_a), get_fitting_parameter(cij_b)
        return a, b

    @staticmethod
    def get_optimal_height_radius(
        min_snr: float | None = None,
        transmission_power: float | None = None,
        noise_power: float | None = None,
        carrier_freq: float | None = None,
        avg_loss_los: float | None = None,
        avg_loss_nlos: float | None = None,
        env_a: float | None = None,
        env_b: float | None = None,
        channel_cfg: FHChannelCfg | None = None,
        layer_cfg: FHLayerCfg | None = None,
    ):
        """
        Calculates the optimal UAV height and corresponding maximum coverage radius.

        By finding the optimal elevation angle, this function determines the
        geometric altitude that maximizes the footprint on the ground while
        satisfying the minimum SNR threshold requirement derived from Equations (5) and (6).

        Args:
            min_snr: Minimum SNR threshold requirement (linear). Defaults to DEFAULT_SNR_THRESHOLD.
            transmission_power: Drone transmission power (W). Defaults to DRONE_TX_POWER_RF.
            noise_power: Noise power (W). Defaults to NOISE_POWER_RF.
            carrier_freq: Carrier frequency for drone communications (Hz). Defaults to DEFAULT_CARRIER_FREQ_DRONE.
            avg_loss_los: Average additional path loss for LOS links (dB). Defaults to PLOS_AVG_LOS_LOSS.
            avg_loss_nlos: Average additional path loss for NLOS links (dB). Defaults to PLOS_AVG_NLOS_LOSS.
            env_a: Environmental parameter alpha. Defaults to PLOS_A_PARAM.
            env_b: Environmental parameter beta. Defaults to PLOS_B_PARAM.

        Returns:
            A tuple (height, radius) representing the optimal geometric deployment.
        """
        cfg = channel_cfg or PlosModel.DEFAULT_CHANNEL_CFG
        radio_cfg = layer_cfg or PlosModel.DEFAULT_LAYER_CFG
        min_snr = db2lin(radio_cfg.sinr_threshold_db) if min_snr is None else min_snr
        transmission_power = (
            db2lin(radio_cfg.tx_power_dbm_a2g - 30.0)
            if transmission_power is None
            else transmission_power
        )
        noise_power = radio_cfg.noise_power_rf_w if noise_power is None else noise_power
        carrier_freq = (
            radio_cfg.frequency_hz_a2g if carrier_freq is None else carrier_freq
        )
        avg_los_default, avg_nlos_default = cfg.plos.resolve_avg_loss(
            cfg.env_type, carrier_freq
        )
        env_a_default, env_b_default = PlosModel.get_a_b_params(channel_cfg=cfg)
        if avg_loss_los is None:
            avg_loss_los = avg_los_default
        if avg_loss_nlos is None:
            avg_loss_nlos = avg_nlos_default
        if env_a is None:
            env_a = env_a_default
        if env_b is None:
            env_b = env_b_default

        max_path_loss = lin2db(transmission_power / (min_snr * noise_power))
        A = avg_loss_los - avg_loss_nlos
        B = (
            20 * np.log10(carrier_freq)
            + 20 * np.log10(4 * np.pi / speed_of_light)
            + avg_loss_nlos
        )
        theta_opt = PlosModel.get_optimal_elevation_angle(A, env_a, env_b)

        R = (
            10
            ** (
                (
                    max_path_loss
                    - A
                    / (1 + env_a * np.exp(-env_b * (theta_opt * 180 / np.pi - env_a)))
                    - B
                )
                / 20
            )
        ) * np.cos(theta_opt)
        h = np.tan(theta_opt) * R
        return h, R

    @staticmethod
    def get_coverage_radius(
        uav_height: float | None = None,
        min_snr: float | None = None,
        transmission_power: float | None = None,
        noise_power: float | None = None,
        carrier_freq: float | None = None,
        avg_loss_los: float | None = None,
        avg_loss_nlos: float | None = None,
        env_a: float | None = None,
        env_b: float | None = None,
        ue_bandwidth: float | None = None,
        drone_bandwidth: float | None = None,
        channel_cfg: FHChannelCfg | None = None,
        layer_cfg: FHLayerCfg | None = None,
    ):
        """
        Finds the maximum ground coverage radius for a UAV at a fixed height.

        It numerically solves for the radius R where the path loss exactly equals
        the maximum allowable path loss dictated by the link budget. This utilizes
        the SNR limit from Equation (6), the received power from Equation (5),
        and the fundamental path loss model from Equation (3).

        Args:
            uav_height: Fixed deployment height of the UAV (m). Defaults to UAVS_HEIGHT.
            min_snr: Minimum SNR requirement (linear). Defaults to DEFAULT_SNR_THRESHOLD.
            transmission_power: Drone transmission power (W). Defaults to DRONE_TX_POWER_RF.
            noise_power: Noise power (W). Defaults to NOISE_POWER_RF.
            carrier_freq: Carrier frequency (Hz). Defaults to DEFAULT_CARRIER_FREQ_DRONE.
            avg_loss_los: Average additional path loss for LOS (dB). Defaults to PLOS_AVG_LOS_LOSS.
            avg_loss_nlos: Average additional path loss for NLOS (dB). Defaults to PLOS_AVG_NLOS_LOSS.
            env_a: Environmental parameter alpha. Defaults to PLOS_A_PARAM.
            env_b: Environmental parameter beta. Defaults to PLOS_B_PARAM.
            ue_bandwidth: Bandwidth allocated to a single user (Hz). Defaults to USER_BANDWIDTH.
            drone_bandwidth: Total drone backhaul bandwidth (Hz). Defaults to DRONE_BANDWIDTH.

        Returns:
            The maximum ground coverage radius (m).
        """
        cfg = channel_cfg or PlosModel.DEFAULT_CHANNEL_CFG
        radio_cfg = layer_cfg or PlosModel.DEFAULT_LAYER_CFG
        uav_height = cfg.default_uav_height_m if uav_height is None else uav_height
        min_snr = db2lin(radio_cfg.sinr_threshold_db) if min_snr is None else min_snr
        ue_bandwidth = (
            radio_cfg.user_bandwidth_hz if ue_bandwidth is None else ue_bandwidth
        )
        drone_bandwidth = (
            radio_cfg.drone_bandwidth_hz if drone_bandwidth is None else drone_bandwidth
        )
        transmission_power = (
            db2lin(radio_cfg.tx_power_dbm_a2g - 30.0)
            if transmission_power is None
            else transmission_power
        )
        noise_power = radio_cfg.noise_power_rf_w if noise_power is None else noise_power
        carrier_freq = (
            radio_cfg.frequency_hz_a2g if carrier_freq is None else carrier_freq
        )
        avg_los_default, avg_nlos_default = cfg.plos.resolve_avg_loss(
            cfg.env_type, carrier_freq
        )
        env_a_default, env_b_default = PlosModel.get_a_b_params(channel_cfg=cfg)
        if avg_loss_los is None:
            avg_loss_los = avg_los_default
        if avg_loss_nlos is None:
            avg_loss_nlos = avg_nlos_default
        if env_a is None:
            env_a = env_a_default
        if env_b is None:
            env_b = env_b_default

        tx_power = ue_bandwidth / drone_bandwidth * transmission_power
        max_path_loss = lin2db(tx_power / (min_snr * noise_power))
        A = avg_loss_los - avg_loss_nlos
        B = (
            20 * np.log10(carrier_freq)
            + 20 * np.log10(4 * np.pi / speed_of_light)
            + avg_loss_nlos
        )

        def func(R):
            return (
                max_path_loss
                - A
                / (
                    1
                    + env_a
                    * np.exp(-env_b * (np.arctan(uav_height / R) * 180 / np.pi - env_a))
                )
                - 10 * np.log10(uav_height**2 + R**2)
                - B
            )

        try:
            res = brentq(func, 1, 1e9)
        except ValueError:
            print("Infeasible solution for coverage radius!")
            return 0

        theta_2 = np.arctan(uav_height / res)
        max_pl = (
            A / (1 + env_a * np.exp(-env_b * (theta_2 * 180 / np.pi - env_a)))
            + 20 * np.log10(res / np.cos(theta_2))
            + B
        )
        assert res > 0 and abs(max_pl - max_path_loss) < 10
        return res

    @staticmethod
    def get_optimal_elevation_angle(
        A: float = -19.0, env_a: float = 9.61, env_b: float = 0.16
    ):
        """
        Calculates the optimal elevation angle for maximum coverage.

        Finds the root of the derivative of the coverage radius function with
        respect to the elevation angle. This analytically maximizes the physical
        footprint derived from the expected path loss calculation in Equation (3).

        Args:
            A: Average path loss difference between LOS and NLOS (dB).
            env_a: Environmental parameter alpha. Defaults to PLOS_A_PARAM.
            env_b: Environmental parameter beta. Defaults to PLOS_B_PARAM.

        Returns:
            The optimal elevation angle in radians.
        """

        def func(theta):
            return (np.pi / (9 * np.log(10))) * np.tan(
                theta
            ) + env_a * env_b * A * np.exp(-env_b * (theta * 180 / np.pi - env_a)) / (
                env_a * np.exp(-env_b * (theta * 180 / np.pi - env_a)) + 1
            ) ** 2

        return fsolve(func, np.pi / 4)[0]

    @staticmethod
    def get_avg_loss(
        env_type: str | None = None,
        frequency: float = 2e9,
        channel_cfg: FHChannelCfg | None = None,
    ):
        """Returns the standard average additional losses (eta_LOS, eta_NLOS).

        Provides tuple values for additional atmospheric and scattering losses
        beyond free space path loss. These represent the eta_LOS and eta_NLOS
        parameters directly utilized in Equation (3).

        Args:
            env_type: Type of environment ('Suburban', 'Urban', 'Dense Urban', or 'Highrise Urban'). Defaults to 'Urban'.
            frequency: Carrier frequency in Hz. Defaults to DEFAULT_CARRIER_FREQ_MBS.

        Returns:
            A tuple (eta_LOS, eta_NLOS) of standard average additional losses in dB.

        Raises:
            ValueError: If an undefined environment type is provided.
        """
        cfg = channel_cfg or PlosModel.DEFAULT_CHANNEL_CFG
        if env_type is not None and env_type != cfg.env_type:
            local_cfg = FHChannelCfg(
                env_type=env_type,
                default_ue_height_m=cfg.default_ue_height_m,
                default_bs_height_m=cfg.default_bs_height_m,
                default_uav_height_m=cfg.default_uav_height_m,
                plos=cfg.plos,
                uma=cfg.uma,
            )
            return local_cfg.plos.resolve_avg_loss(local_cfg.env_type, frequency)
        return cfg.plos.resolve_avg_loss(cfg.env_type, frequency)

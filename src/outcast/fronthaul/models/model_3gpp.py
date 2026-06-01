"""Urban Macro (UMa, TR 38.901) path loss model for G2G fronthaul links."""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light

from src.uavnetsim.fronthaul.fh_config import FHChannelCfg


class UMaModel:
    """Implementation of the Urban Macro (UMa) model (TR 38.901) for G2G Links."""

    def __init__(self, channel_cfg: FHChannelCfg):
        """Initializes the UMa model with channel configuration.

        Args:
            channel_cfg: Configuration object for fronthaul channel parameters.
        """
        self.channel_cfg = channel_cfg

    def calculate_path_loss_g2g(
        self,
        dist_m: NDArray[np.float32],
        frequencies: NDArray[np.float32],
        height_diff: NDArray[np.float32] | None,
        is_los: NDArray[np.bool_] | None,
        ue_height: NDArray[np.float32],
        bs_height: NDArray[np.float32],
        mask_g2g: NDArray[np.bool_] | slice,
        uma_height_mode: str,
        enable_shadowing: bool,
        rng: np.random.Generator,
        out: NDArray[np.float32],
    ) -> None:
        """Executes the exact immutable vectorization logic for G2G UMa model."""
        uma_cfg = self.channel_cfg.uma

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
                13.54 + 39.08 * log_d_3d + 20.0 * log_f - 0.6 * (ue_height_col - 1.5)
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

    @staticmethod
    def get_uma_los_probability(
        distance_2d: NDArray[np.float32] | np.ndarray,
        ue_height: NDArray[np.float32] | np.ndarray,
    ) -> NDArray[np.float32]:
        """Calculates the LOS probability for the UMa model (TR 38.901).

        Args:
            distance_2d: 2D horizontal distance between source and UE (m).
            ue_height: User Equipment height (m).

        Returns:
            The probability of LOS condition (0 to 1).
        """
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
        """Calculates the effective environment height (h_E) for UMa model.

        Args:
            distance_2d: 2D horizontal distance between source and UE (m).
            ue_height: User Equipment height (m).
            mode: Height calculation mode, either "expected" or "probabilistic".
            rng: Optional NumPy random generator for probabilistic sampling.

        Returns:
            The effective environment height (m).
        """
        if mode == "expected":
            return UMaModel.get_uma_effective_environment_height_expected(
                distance_2d, ue_height
            )
        if mode == "probabilistic":
            return UMaModel.get_uma_effective_environment_height_probabilistic(
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
        """Calculates the expected (mean) effective environment height for UMa model.

        Args:
            distance_2d: 2D horizontal distance between source and UE (m).
            ue_height: User Equipment height (m).

        Returns:
            The mean effective environment height (m).
        """
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
        """Probabilistically samples the effective environment height for UMa model.

        Args:
            distance_2d: 2D horizontal distance between source and UE (m).
            ue_height: User Equipment height (m).
            rng: Optional NumPy random generator for sampling.

        Returns:
            The sampled effective environment height (m).
        """
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

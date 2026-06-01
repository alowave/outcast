"""Probabilistic Line-of-Sight (PLOS) path loss model for A2G fronthaul links."""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light
from scipy.optimize import brentq, fsolve

from src.outcast.fronthaul.fh_config import FHChannelCfg, FHLayerCfg
from src.outcast.geometry.coords import Coords3d
from src.outcast.utils.math_tools import db2lin, lin2db


class PlosModel:
    """Implementation of the Probabilistic Line-of-Sight (PLOS) model for A2G Links.

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
        """Initializes the PLOS model with environmental parameters.

        Args:
            channel_cfg: Configuration object for fronthaul channel parameters.
                If None, uses DEFAULT_CHANNEL_CFG.
        """
        self.channel_cfg = channel_cfg or self.DEFAULT_CHANNEL_CFG
        self.env_type = self.channel_cfg.env_type
        self.env_a, self.env_b = self.get_a_b_params(channel_cfg=self.channel_cfg)

    def calculate_path_loss_a2g(
        self,
        dist_m: NDArray[np.float32],
        frequencies: NDArray[np.float32],
        height_diff: NDArray[np.float32] | None,
        is_los: NDArray[np.bool_] | None,
        ue_height: NDArray[np.float32],
        bs_height: NDArray[np.float32],
        mask_a2g: NDArray[np.bool_] | slice,
        out: NDArray[np.float32],
    ) -> None:
        """Executes the exact immutable vectorization logic for A2G PLOS model."""
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
                dist_2d = np.sqrt(np.maximum(dist_a2g[:, col_mask] ** 2 - hd_f**2, 0))
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

    def get_path_loss(
        self,
        ue_coords: Coords3d,
        bs_coords: Coords3d = Coords3d(0, 0, 0),
        frequency: float = 2e9,
    ):
        """Calculates the expected path loss in linear scale.

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
        """Computes the probability of having a clear Line-of-Sight (LOS) link.

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
    def get_a_b_params(
        env_type: str | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        channel_cfg: FHChannelCfg | None = None,
    ):
        """Retrieve or calculate the alpha and beta environmental parameters.

        If standard ITU parameters are not provided, maps predefined urban
        environments to their respective alpha, beta, and gamma values, then
        uses polynomial surface fitting to compute the final S-curve parameters.

        Args:
            env_type: Environment type ('Suburban', 'Urban', 'Dense Urban', or
                'Highrise Urban'). Defaults to the value in ``channel_cfg``.
            alpha: Optional manually specified ITU alpha parameter.
            beta: Optional manually specified ITU beta parameter.
            gamma: Optional manually specified ITU gamma parameter.
            channel_cfg: Optional channel configuration; falls back to
                ``DEFAULT_CHANNEL_CFG`` when ``None``.

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
            """Calculates a fitting parameter based on polynomial surface fitting coefficients."""
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
        """Calculate the optimal UAV height and corresponding maximum coverage radius.

        By finding the optimal elevation angle, this function determines the
        geometric altitude that maximises the footprint on the ground while
        satisfying the minimum SNR threshold requirement derived from Equations (5) and (6).

        Args:
            min_snr: Minimum SNR threshold requirement (linear). Defaults to
                the value in ``layer_cfg``.
            transmission_power: Drone transmission power (W). Defaults to the
                value in ``layer_cfg``.
            noise_power: Noise power (W). Defaults to the value in ``layer_cfg``.
            carrier_freq: Carrier frequency for drone communications (Hz).
                Defaults to the value in ``layer_cfg``.
            avg_loss_los: Average additional path loss for LOS links (dB).
                Resolved from ``channel_cfg`` when ``None``.
            avg_loss_nlos: Average additional path loss for NLOS links (dB).
                Resolved from ``channel_cfg`` when ``None``.
            env_a: Environmental parameter alpha. Resolved from ``channel_cfg``
                when ``None``.
            env_b: Environmental parameter beta. Resolved from ``channel_cfg``
                when ``None``.
            channel_cfg: Optional channel configuration; falls back to
                ``DEFAULT_CHANNEL_CFG`` when ``None``.
            layer_cfg: Optional layer configuration; falls back to
                ``DEFAULT_LAYER_CFG`` when ``None``.

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
        """Find the maximum ground coverage radius for a UAV at a fixed height.

        Numerically solves for the radius R where the path loss exactly equals
        the maximum allowable path loss dictated by the link budget, using
        the SNR limit from Equation (6) and the path loss model from Equation (3).

        Args:
            uav_height: Fixed deployment height of the UAV (m). Defaults to the
                value in ``channel_cfg``.
            min_snr: Minimum SNR requirement (linear). Defaults to the value in
                ``layer_cfg``.
            transmission_power: Drone transmission power (W). Defaults to the
                value in ``layer_cfg``.
            noise_power: Noise power (W). Defaults to the value in ``layer_cfg``.
            carrier_freq: Carrier frequency (Hz). Defaults to the value in
                ``layer_cfg``.
            avg_loss_los: Average additional path loss for LOS (dB). Resolved
                from ``channel_cfg`` when ``None``.
            avg_loss_nlos: Average additional path loss for NLOS (dB). Resolved
                from ``channel_cfg`` when ``None``.
            env_a: Environmental parameter alpha. Resolved from ``channel_cfg``
                when ``None``.
            env_b: Environmental parameter beta. Resolved from ``channel_cfg``
                when ``None``.
            ue_bandwidth: Bandwidth allocated to a single user (Hz). Defaults to
                the value in ``layer_cfg``.
            drone_bandwidth: Total drone backhaul bandwidth (Hz). Defaults to the
                value in ``layer_cfg``.
            channel_cfg: Optional channel configuration; falls back to
                ``DEFAULT_CHANNEL_CFG`` when ``None``.
            layer_cfg: Optional layer configuration; falls back to
                ``DEFAULT_LAYER_CFG`` when ``None``.

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

        # Radius equation to be solved from paper
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
        """Calculates the optimal elevation angle for maximum coverage.

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

        # Equation to find optimal elevation angle
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
        """Return the standard average additional losses (eta_LOS, eta_NLOS).

        Provides tuple values for additional atmospheric and scattering losses
        beyond free space path loss, corresponding to eta_LOS and eta_NLOS in
        Equation (3).

        Args:
            env_type: Environment type ('Suburban', 'Urban', 'Dense Urban', or
                'Highrise Urban'). Defaults to the value in ``channel_cfg``.
            frequency: Carrier frequency in Hz. Defaults to 2 GHz.
            channel_cfg: Optional channel configuration; falls back to
                ``DEFAULT_CHANNEL_CFG`` when ``None``.

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

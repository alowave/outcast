"""Fronthaul layer: vectorized channel, SINR, throughput, and fairness calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.outcast.fronthaul.fh_channel_data import FHChannelData
from src.outcast.fronthaul.fh_channel_model_manager import FHPathLossModelManager
from src.outcast.fronthaul.fh_config import FHLayerCfg
from src.outcast.link_layer.link_data import AccessLinkData
from src.outcast.utils.math_tools import db2lin
from src.outcast.world.world_state import WorldStateCfg


class FHLayer:
    """Fronthaul layer managing radio resource calculations for UAV/BS-to-UE links."""

    def __init__(self, cfg: FHLayerCfg):
        """Initializes the FHLayer.

        Args:
            cfg (FHLayerCfg): The configuration object for the fronthaul layer.
        """
        self.cfg = cfg
        self.path_loss_model = FHPathLossModelManager(cfg.channel)
        self.fh_channel_data: FHChannelData | None = None
        self.frequencies: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.tx_powers_dbm: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.freq_table: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.pwr_table: NDArray[np.float32] = np.array([], dtype=np.float32)

        # Pre-calculated noise PSD (linear mW/Hz)
        self.noise_psd_mw_per_hz: float = 0.0
        self.sinr_threshold_lin: float = 0.0

        # Arrays for fairness/coverage tracking
        self.coverage_score_history: NDArray[np.bool_] = np.array([], dtype=np.bool_)
        self.coverage_score_sum: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.coverage_score_idx: int = 0
        self.total_fairness: float = 0.0
        self.station_total_bandwidth_hz: NDArray[np.uint32] = np.array(
            [], dtype=np.uint32
        )

        # Pre-allocated scratch/buffer arrays for intermediate steps
        self.rx_power_lin: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.scratch_per_ue_lin: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.scratch_lin: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.snr_lin: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.snr_db: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.sinr_lin: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.serving_mask: NDArray[np.bool_] = np.array([], dtype=np.bool_)
        self.user_is_served: NDArray[np.bool_] = np.array([], dtype=np.bool_)
        self.scratch_user_mask: NDArray[np.bool_] = np.array([], dtype=np.bool_)
        self.scratch_link_mask: NDArray[np.bool_] = np.array([], dtype=np.bool_)
        self.row_idx: NDArray[np.intp] = np.array([], dtype=np.intp)
        self.best_idx: NDArray[np.intp] = np.array([], dtype=np.intp)
        self.n_associated_users: NDArray[np.int32] = np.array([], dtype=np.int32)
        self.n_unserved_users: NDArray[np.int32] = np.array([], dtype=np.int32)
        self.n_served_users_snr: NDArray[np.int32] = np.array([], dtype=np.int32)
        self.fairness_per_bs: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.scratch_snr_mask: NDArray[np.bool_] = np.array([], dtype=np.bool_)

    def update_fh_channel_data(self, fronthaul_link_data: AccessLinkData):
        """Updates the fronthaul channel data using the access link data.

        This method triggers a chain of vectorized calculations including path loss,
        received power, interference, SINR, association, throughput, and fairness.

        Args:
            fronthaul_link_data (AccessLinkData): The access link data from the link layer.

        Raises:
            RuntimeError: If initialize_data_arrays() has not been called before this method.
        """
        if self.fh_channel_data is None:
            raise RuntimeError(
                "Call initialize_data_arrays(world_cfg) before update_fh_channel_data()."
            )

        # Chain of vectorized calculations
        self._calculate_path_loss(fronthaul_link_data)
        self._calculate_received_power()
        self._calculate_interference(fronthaul_link_data.in_range)
        self._calculate_sinr()
        self._calculate_snr()
        self._associate_max_sinr(fronthaul_link_data.in_range)
        self._calculate_throughput()
        self._calculate_fairness()

    def _calculate_path_loss(self, link_data: AccessLinkData):
        """Calculates the path loss for all links.

        Uses the path loss model manager to perform a vectorized calculation of
        the path loss based on distance, frequencies, heights, and link type (A2G/G2G).

        Args:
            link_data (AccessLinkData): Information about the physical links including
                distance, locations, and line-of-sight status.
        """
        # Eq (2) & (3)
        self.fh_channel_data.path_loss_db = (
            self.path_loss_model.get_vectorized_path_loss(
                dist_m=link_data.dist_m,
                frequencies=self.frequencies,
                height_diff=link_data.height_diff,
                ue_height=link_data.ue_height,
                bs_height=link_data.bs_height,
                is_los=link_data.los,
                link_type=link_data.link_type,
            )
        )

    def _calculate_received_power(self):
        """Calculates the received power for all links.

        `total_received_power_dbm` stores the full-link received power after path loss.
        `ue_received_power_dbm` stores the received power captured in each UE's assigned
        bandwidth, derived via a PSD intermediate using the station total bandwidth.
        """
        # Eq (5) simplified form: P_R = P_T - L
        np.subtract(
            self.tx_powers_dbm,
            self.fh_channel_data.path_loss_db,
            out=self.fh_channel_data.total_received_power_dbm,
        )

        # Convert total received power from dBm to linear mW in the preallocated buffer.
        np.multiply(
            self.fh_channel_data.total_received_power_dbm, 0.1, out=self.rx_power_lin
        )
        np.power(10.0, self.rx_power_lin, out=self.rx_power_lin)

        # Convert full-link power to PSD by dividing by the station bandwidth.
        station_bw = self.station_total_bandwidth_hz[np.newaxis, :]
        np.divide(
            self.rx_power_lin, station_bw, out=self.rx_power_lin, where=station_bw > 0
        )
        np.copyto(self.rx_power_lin, 0.0, where=station_bw <= 0)

        # Convert PSD back to UE-band power by multiplying by the assigned UE bandwidth.
        np.multiply(
            self.rx_power_lin,
            self.fh_channel_data.assigned_bandwidth_hz[:, np.newaxis],
            out=self.rx_power_lin,
        )

        np.copyto(self.fh_channel_data.ue_received_power_dbm, self.rx_power_lin)
        np.log10(
            np.maximum(self.fh_channel_data.ue_received_power_dbm, 1e-15),
            out=self.fh_channel_data.ue_received_power_dbm,
        )
        np.multiply(
            self.fh_channel_data.ue_received_power_dbm,
            10.0,
            out=self.fh_channel_data.ue_received_power_dbm,
        )

    def _calculate_interference(self, in_range: NDArray[np.bool_]):
        """Calculates the interference for all users.

        Computes the total received power from all in-range base stations for each user
        and subtracts the received power of the direct link to find the interference.
        Also adds the noise power to the interference for calculating the SINR denominator.

        Args:
            in_range (NDArray[np.bool_]): Boolean array indicating which base stations
                are within range for each user.
        """
        # Sum of all power from BSs in range for each UE.
        np.sum(
            self.rx_power_lin,
            axis=1,
            keepdims=True,
            out=self.scratch_per_ue_lin,
            where=in_range,
        )

        # Interference I(j,n) = total_rx_pwr - P_R(j,n) [Linear mW].
        np.copyto(self.scratch_lin, self.scratch_per_ue_lin)
        np.subtract(
            self.scratch_lin, self.rx_power_lin, out=self.scratch_lin, where=in_range
        )

        # Persist interference in dBm for reporting/debugging while preserving the linear buffer
        np.copyto(self.fh_channel_data.interference_dbm, self.scratch_lin)
        np.log10(
            np.maximum(self.fh_channel_data.interference_dbm, 1e-15),
            out=self.fh_channel_data.interference_dbm,
        )  # TODO: check if clamping is needed
        np.multiply(
            self.fh_channel_data.interference_dbm,
            10.0,
            out=self.fh_channel_data.interference_dbm,
        )

        # Final SINR denominator buffer: I_lin + N_lin, where noise scales with the
        # UE assigned bandwidth rather than a fixed global user bandwidth.
        np.multiply(
            self.fh_channel_data.assigned_bandwidth_hz[:, np.newaxis],
            self.noise_psd_mw_per_hz,
            out=self.scratch_per_ue_lin,
        )
        np.add(self.scratch_lin, self.scratch_per_ue_lin, out=self.scratch_lin)

    def _calculate_sinr(self):
        """Calculates the Signal-to-Interference-plus-Noise Ratio (SINR).

        Computes the SINR by dividing the received signal power by the sum of
        interference and noise power (both in linear scale), and then converts it to dB.
        """
        # Eq (7)
        # Gamma_lin = P_R_lin / (I_lin + N_lin)
        np.divide(self.rx_power_lin, self.scratch_lin, out=self.sinr_lin)

        # Report/store SINR in dB
        np.log10(np.maximum(self.sinr_lin, 1e-15), out=self.fh_channel_data.sinr_db)
        np.multiply(
            self.fh_channel_data.sinr_db, 10.0, out=self.fh_channel_data.sinr_db
        )

    def _calculate_snr(self):
        """Calculates the Signal-to-Noise Ratio (SNR)."""
        np.multiply(
            self.fh_channel_data.assigned_bandwidth_hz[:, np.newaxis],
            self.noise_psd_mw_per_hz,
            out=self.scratch_per_ue_lin,
        )
        np.divide(self.rx_power_lin, self.scratch_per_ue_lin, out=self.snr_lin)

        np.log10(np.maximum(self.snr_lin, 1e-15), out=self.snr_db)
        np.multiply(self.snr_db, 10.0, out=self.snr_db)

    def _associate_max_sinr(self, in_range: NDArray[np.bool_]):
        """Associates each user with the base station providing the maximum SINR.

        Determines the best serving base station for each user based on the highest
        SINR among the in-range base stations. Updates the serving mask accordingly.

        Args:
            in_range (NDArray[np.bool_]): Boolean array indicating which base stations
                are within range for each user.
        """
        # Eq (9): Max-SINR Mask
        # We use scratch_lin to hold the masked SINR to find the best serving index.
        np.copyto(self.scratch_lin, self.sinr_lin)
        np.copyto(self.scratch_lin, -1.0, where=~in_range)

        # Associate each UE to exactly one source: the first max-SINR index.
        # If a UE has no in-range source, the selected entry stays False because
        # the assignment below copies the corresponding in_range value.
        self.best_idx[:] = np.argmax(self.scratch_lin, axis=1)
        self.serving_mask.fill(False)
        self.serving_mask[self.row_idx, self.best_idx] = in_range[
            self.row_idx, self.best_idx
        ]

    def _calculate_throughput(self):
        """Calculates the throughput for all served links.

        Computes the throughput using the Shannon-Hartley theorem based on the
        assigned per-user bandwidth and the linear SINR. Applies the serving mask
        so that only associated links receive throughput. The assigned bandwidth
        vector defaults to the layer-wide user bandwidth until a scheduler updates it.
        """
        # Eq (8)
        # R = B_n * log2(1 + Gamma_lin)
        np.copyto(self.scratch_lin, self.sinr_lin)
        np.add(self.scratch_lin, 1.0, out=self.scratch_lin)  # 1 + Gamma_lin
        np.log2(self.scratch_lin, out=self.scratch_lin)
        np.multiply(
            self.scratch_lin,
            self.fh_channel_data.assigned_bandwidth_hz[:, np.newaxis],
            out=self.scratch_lin,
        )

        # Final throughput is stored as uint64 in FHChannelData.
        # Final throughput reported only for serving links
        np.copyto(
            self.fh_channel_data.throughput_bps,
            self.scratch_lin.astype(np.uint64),
            where=self.serving_mask,
        )
        np.copyto(self.fh_channel_data.throughput_bps, 0, where=~self.serving_mask)

    def _calculate_fairness(self):
        """Calculates fairness metrics and updates coverage scores.

        Evaluates which links successfully serve users by checking against the SINR
        threshold. Updates the rolling coverage history and evaluates fairness metrics
        per base station as well as the overall system fairness using Jain's fairness index.
        """
        served_link_mask = self.scratch_link_mask
        user_served_mask = self.user_is_served

        # Each UE has at most one associated serving link. Extract that selected
        # SINR via the one-hot serving mask and compare only that metric against
        # the service threshold.
        np.multiply(self.sinr_lin, self.serving_mask, out=self.scratch_lin)
        np.sum(self.scratch_lin, axis=1, keepdims=True, out=self.scratch_per_ue_lin)
        np.greater_equal(
            self.scratch_per_ue_lin,
            self.sinr_threshold_lin,
            out=user_served_mask,
        )
        np.logical_and(self.serving_mask, user_served_mask, out=served_link_mask)

        # Update coverage over a fixed rolling window stored in the layer config.
        oldest_values = self.coverage_score_history[
            :, self.coverage_score_idx : self.coverage_score_idx + 1
        ]
        self.coverage_score_sum -= oldest_values
        self.coverage_score_sum += self.user_is_served
        self.coverage_score_history[
            :, self.coverage_score_idx : self.coverage_score_idx + 1
        ] = self.user_is_served
        self.coverage_score_idx = (
            self.coverage_score_idx + 1
        ) % self.cfg.coverage_history_window

        # Current coverage scores (C_i) for each user
        np.multiply(
            self.coverage_score_sum,
            1.0 / self.cfg.coverage_history_window,
            out=self.scratch_per_ue_lin,
        )

        # Calculate fairness for each source/BS slot.
        # scratch_lin = C_i * serving_mask
        np.multiply(self.scratch_per_ue_lin, self.serving_mask, out=self.scratch_lin)
        per_bs_sum = np.sum(self.scratch_lin, axis=0)

        # C_i^2
        np.square(self.scratch_per_ue_lin, out=self.scratch_per_ue_lin)
        np.multiply(self.scratch_per_ue_lin, self.serving_mask, out=self.scratch_lin)
        per_bs_squared_sum = np.sum(self.scratch_lin, axis=0)

        np.sum(self.serving_mask, axis=0, out=self.n_associated_users)

        # Count unserved users per source by broadcasting the per-user unserved flag back
        # over the serving association mask. Only the associated link contributes.
        np.logical_not(user_served_mask, out=self.scratch_user_mask)
        np.logical_and(
            self.scratch_user_mask,
            self.serving_mask,
            out=served_link_mask,
        )
        np.sum(served_link_mask, axis=0, out=self.n_unserved_users)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.fairness_per_bs[:] = (per_bs_sum**2) / (
                self.n_associated_users * per_bs_squared_sum
            )
            self.fairness_per_bs[:] = np.nan_to_num(self.fairness_per_bs, nan=0.0)

        # total fairness of the system (1 number)
        n_users_total = np.sum(self.n_associated_users)
        with np.errstate(divide="ignore", invalid="ignore"):
            total_fairness = (per_bs_sum.sum() ** 2) / (
                n_users_total * per_bs_squared_sum.sum()
            )
            self.total_fairness = float(np.nan_to_num(total_fairness, nan=0.0))

    def initialize_data_arrays(self, world_cfg: WorldStateCfg):
        """Initializes and pre-allocates data arrays for the simulation.

        Sets up the channel data structure and allocates buffer arrays used for
        vectorized calculations to avoid memory reallocation during the simulation loop.

        Args:
            world_cfg (WorldStateCfg): The configuration defining the number of UEs,
                UAVs, and ground base stations in the world.
        """
        n_ue = world_cfg.n_ues
        n_uavs = world_cfg.n_uavs
        n_bss = world_cfg.n_bss
        n_s = n_uavs + n_bss

        shape = (n_ue, n_s)
        self.fh_channel_data = FHChannelData(
            path_loss_db=np.zeros(shape, dtype=np.float32),
            total_received_power_dbm=np.zeros(shape, dtype=np.float32),
            ue_received_power_dbm=np.zeros(shape, dtype=np.float32),
            throughput_bps=np.zeros(shape, dtype=np.uint64),
            interference_dbm=np.zeros(shape, dtype=np.float32),
            snr_db=np.zeros(shape, dtype=np.float32),
            sinr_db=np.zeros(shape, dtype=np.float32),
            assigned_bandwidth_hz=np.full(
                n_ue, self.cfg.user_bandwidth_hz, dtype=np.uint32
            ),
        )

        # Pre-allocate buffer arrays to avoid relocation during simulation
        self.rx_power_lin = np.zeros(shape, dtype=np.float32)
        self.scratch_per_ue_lin = np.zeros((n_ue, 1), dtype=np.float32)
        self.scratch_lin = np.zeros(shape, dtype=np.float32)
        self.snr_lin = np.zeros(shape, dtype=np.float32)
        self.snr_db = self.fh_channel_data.snr_db
        self.sinr_lin = np.zeros(shape, dtype=np.float32)
        self.serving_mask = np.zeros(shape, dtype=np.bool_)
        self.user_is_served = np.zeros((n_ue, 1), dtype=np.bool_)
        self.scratch_user_mask = np.zeros((n_ue, 1), dtype=np.bool_)
        self.scratch_link_mask = np.zeros(shape, dtype=np.bool_)
        self.row_idx = np.arange(n_ue, dtype=np.intp)
        self.best_idx = np.zeros(n_ue, dtype=np.intp)
        self.n_associated_users = np.zeros(n_s, dtype=np.int32)
        self.n_unserved_users = np.zeros(n_s, dtype=np.int32)
        self.n_served_users_snr = np.zeros(n_s, dtype=np.int32)
        self.fairness_per_bs = np.zeros(n_s, dtype=np.float32)
        self.scratch_snr_mask = np.zeros(shape, dtype=np.bool_)
        # Total access bandwidth budget per serving node slot. A scheduler can
        # rewrite this array to reflect heterogeneous UAV/BS budgets.
        self.station_total_bandwidth_hz = np.full(
            n_s,
            self.cfg.drone_bandwidth_hz,
            dtype=np.uint32,
        )

        # Lookup tables for dynamic needs
        self.freq_table = np.array(
            [self.cfg.frequency_hz_a2g, self.cfg.frequency_hz_g2g], dtype=np.float32
        )
        self.pwr_table = np.array(
            [self.cfg.tx_power_dbm_a2g, self.cfg.tx_power_dbm_g2g], dtype=np.float32
        )

        # Lookup/Broadcastable arrays
        self.frequencies = np.empty((1, n_s), dtype=np.float32)
        self.tx_powers_dbm = np.empty((1, n_s), dtype=np.float32)

        self.frequencies[0, :n_uavs] = self.cfg.frequency_hz_a2g
        self.frequencies[0, n_uavs:] = self.cfg.frequency_hz_g2g

        self.tx_powers_dbm[0, :n_uavs] = self.cfg.tx_power_dbm_a2g
        self.tx_powers_dbm[0, n_uavs:] = self.cfg.tx_power_dbm_g2g

        # Pre-calculate noise PSD in linear mW/Hz.
        self.noise_psd_mw_per_hz = float(
            db2lin(self.cfg.noise_spectral_density_dbm_per_hz)
        )
        self.sinr_threshold_lin = float(db2lin(self.cfg.sinr_threshold_db))

        # Arrays for fairness/coverage tracking
        self.coverage_score_history = np.zeros(
            (n_ue, self.cfg.coverage_history_window),
            dtype=np.bool_,
        )
        self.coverage_score_sum = np.zeros((n_ue, 1), dtype=np.float32)
        self.coverage_score_idx = 0
        self.total_fairness = 0.0

    def get_n_served_users_sinr(self):
        """Return the number of served users per station based on the SINR threshold."""
        return self.n_associated_users - self.n_unserved_users

    def get_n_served_users_snr(self):
        """Return the number of served users per station based on the SNR threshold."""
        np.greater_equal(
            self.snr_db, self.cfg.snr_threshold_db, out=self.scratch_snr_mask
        )
        np.logical_and(
            self.scratch_snr_mask, self.serving_mask, out=self.scratch_snr_mask
        )
        np.sum(self.scratch_snr_mask, axis=0, out=self.n_served_users_snr)
        return self.n_served_users_snr

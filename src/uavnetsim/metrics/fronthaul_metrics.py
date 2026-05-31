"""Fronthaul metrics controller for UAV simulation.

This module implements a metrics controller for tracking fronthaul-related metrics
in the simulation, including SNR, SINR, throughput, fairness, and user association
statistics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray

from src.uavnetsim.fronthaul.fh_layer import FHLayer
from src.uavnetsim.metrics.base import BaseMetricsController, MetricValue


class FronthaulMetricsController(BaseMetricsController):
    """Collects, buffers, and saves Fronthaul metrics per time step.

    Inherits from BaseMetricsController to standardize batching,
    writing to disk, and general episode metric handling.
    """

    layer_name = "fh"

    def _allocate_batch_buffers(self) -> Dict[str, NDArray]:
        """Pre-allocates all necessary arrays for a single batch to maximize speed."""
        N = self.buf_size
        S = self.n_bs
        buffers = {}

        # The list of statistical attributes we want to save
        self.stats_to_track = [
            "min",
            "max",
            "mean",
            "median",
            "var",
            "std",
            "p5",
            "p10",
            "p90",
            "p95",
        ]

        metrics = ["snr", "sinr", "throughput"]

        for m in metrics:
            for s in self.stats_to_track:
                buffers[f"{m}_{s}"] = np.zeros(N, dtype=np.float32)

        # Fairness & Thresholds
        buffers["fairness_total"] = np.zeros(N, dtype=np.float32)
        buffers["fairness_per_bs"] = np.zeros((N, S), dtype=np.float32)

        buffers["sinr_thr_total"] = np.zeros(N, dtype=np.int32)
        buffers["sinr_thr_per_bs"] = np.zeros((N, S), dtype=np.int32)

        buffers["snr_thr_total"] = np.zeros(N, dtype=np.int32)
        buffers["snr_thr_per_bs"] = np.zeros((N, S), dtype=np.int32)

        buffers["assoc_ue_per_bs"] = np.zeros((N, S), dtype=np.int32)

        return buffers

    def _init_episode_trackers(self) -> None:
        """Initializes tracking arrays representing moving averages and min/max across the entire episode."""
        # Per User
        self.ep_tp_avg = np.zeros(self.n_ue, dtype=np.float64)
        self.ep_tp_min = np.full(self.n_ue, np.inf, dtype=np.float64)
        self.ep_tp_max = np.full(self.n_ue, -np.inf, dtype=np.float64)

        self.ep_snr_avg = np.zeros(self.n_ue, dtype=np.float64)
        self.ep_snr_min = np.full(self.n_ue, np.inf, dtype=np.float32)
        self.ep_snr_max = np.full(self.n_ue, -np.inf, dtype=np.float32)

        self.ep_sinr_avg = np.zeros(self.n_ue, dtype=np.float64)
        self.ep_sinr_min = np.full(self.n_ue, np.inf, dtype=np.float32)
        self.ep_sinr_max = np.full(self.n_ue, -np.inf, dtype=np.float32)

        # Per BS
        self.ep_assoc_avg = np.zeros(self.n_bs, dtype=np.float64)
        self.ep_assoc_min = np.full(self.n_bs, np.inf, dtype=np.int32)
        self.ep_assoc_max = np.full(self.n_bs, -np.inf, dtype=np.int32)

    def update_step(self, fh_layer: FHLayer) -> None:
        """Extracts metrics from the FHLayer and saves them into the pre-allocated buffer.

        Automatically saves the batch to disk if the buffer fills up.
        """
        idx = self.batch_step_idx
        bufs = self.batch_buffers

        # 1. Extract per-UE metrics based on the best connection or active service
        # Throughput: Only served links have >0 throughput. It is uint64.
        tp_ue = np.sum(fh_layer.fh_channel_data.throughput_bps, axis=1)

        # SNR/SINR: We take the value of the active served link. If unserved, it defaults to 0 via mask.
        snr_ue = np.sum(fh_layer.snr_db * fh_layer.serving_mask, axis=1)
        sinr_ue = np.sum(
            fh_layer.fh_channel_data.sinr_db * fh_layer.serving_mask, axis=1
        )

        # 2. Calculate and store Array Stats (SNR, SINR, Throughput)
        self._store_array_stats(bufs, "throughput", tp_ue, idx, self.stats_to_track)
        self._store_array_stats(bufs, "snr", snr_ue, idx, self.stats_to_track)
        self._store_array_stats(bufs, "sinr", sinr_ue, idx, self.stats_to_track)

        # 3. Store Fairness
        bufs["fairness_total"][idx] = fh_layer.total_fairness
        bufs["fairness_per_bs"][idx, :] = fh_layer.fairness_per_bs

        # 4. Store Threshold Achievements
        sinr_success_per_bs = fh_layer.get_n_served_users_sinr()
        bufs["sinr_thr_per_bs"][idx, :] = sinr_success_per_bs
        bufs["sinr_thr_total"][idx] = np.sum(sinr_success_per_bs)

        snr_success_per_bs = fh_layer.get_n_served_users_snr()
        bufs["snr_thr_per_bs"][idx, :] = snr_success_per_bs
        bufs["snr_thr_total"][idx] = np.sum(snr_success_per_bs)

        # 5. Store Association
        bufs["assoc_ue_per_bs"][idx, :] = fh_layer.n_associated_users

        # 6. Update Episode Running Trackers
        self._update_episode_trackers(
            tp_ue, snr_ue, sinr_ue, fh_layer.n_associated_users
        )

        # 7. Finalize Step
        self._finalize_step()

    def _update_episode_trackers(
        self, tp_ue: NDArray, snr_ue: NDArray, sinr_ue: NDArray, assoc_bs: NDArray
    ) -> None:
        """Updates the tracking variables representing the entire episode with moving average."""
        current_step = self.episode_step_idx + 1

        # Per User Tracking
        self.ep_tp_avg += (tp_ue - self.ep_tp_avg) / current_step
        np.minimum(self.ep_tp_min, tp_ue, out=self.ep_tp_min)
        np.maximum(self.ep_tp_max, tp_ue, out=self.ep_tp_max)

        self.ep_snr_avg += (snr_ue - self.ep_snr_avg) / current_step
        np.minimum(self.ep_snr_min, snr_ue, out=self.ep_snr_min)
        np.maximum(self.ep_snr_max, snr_ue, out=self.ep_snr_max)

        self.ep_sinr_avg += (sinr_ue - self.ep_sinr_avg) / current_step
        np.minimum(self.ep_sinr_min, sinr_ue, out=self.ep_sinr_min)
        np.maximum(self.ep_sinr_max, sinr_ue, out=self.ep_sinr_max)

        # Per BS Tracking
        self.ep_assoc_avg += (assoc_bs - self.ep_assoc_avg) / current_step
        np.minimum(self.ep_assoc_min, assoc_bs, out=self.ep_assoc_min)
        np.maximum(self.ep_assoc_max, assoc_bs, out=self.ep_assoc_max)

    def _build_episode_metrics(self) -> dict[str, MetricValue]:
        episode_metrics = {
            # Per User (Tracked across episode)
            "ep_tp_min_per_ue": self.ep_tp_min,
            "ep_tp_max_per_ue": self.ep_tp_max,
            "ep_tp_mean_per_ue": self.ep_tp_avg,
            "ep_snr_min_per_ue": self.ep_snr_min,
            "ep_snr_max_per_ue": self.ep_snr_max,
            "ep_snr_mean_per_ue": self.ep_snr_avg,
            "ep_sinr_min_per_ue": self.ep_sinr_min,
            "ep_sinr_max_per_ue": self.ep_sinr_max,
            "ep_sinr_mean_per_ue": self.ep_sinr_avg,
            # Per BS (Tracked across episode)
            "ep_assoc_min_per_bs": self.ep_assoc_min,
            "ep_assoc_max_per_bs": self.ep_assoc_max,
            "ep_assoc_mean_per_bs": self.ep_assoc_avg,
        }

        # Raise an error instead of silently replacing infinite values
        for key, val in episode_metrics.items():
            if np.any(np.isinf(val)):
                raise ValueError(
                    f"Found infinite values in metric '{key}', indicating it was never updated correctly."
                )

        return episode_metrics

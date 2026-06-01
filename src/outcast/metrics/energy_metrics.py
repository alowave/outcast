"""Energy metrics controller for UAV simulation.

This module implements a metrics controller for tracking energy-related metrics
for UAVs in the simulation, including energy levels, consumption, charging,
and critical states.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from src.outcast.metrics.base import BaseMetricsController, MetricValue
from src.outcast.world.energy_model.uav_battery import Battery


class EnergyMetricsController(BaseMetricsController):
    """Collects, buffers, and saves Energy metrics per time step.

    Tracks energy levels, consumption, charging, and critical states for UAVs.
    """

    layer_name = "energy"

    def _allocate_batch_buffers(self) -> Dict[str, NDArray]:
        """Pre-allocates all necessary arrays for a single batch."""
        N = self.buf_size
        n_uavs = self.n_uavs
        buffers = {}

        # Descriptive stats for energy levels across all UAVs
        self.stats_to_track = ["min", "max", "mean", "median", "std"]
        for s in self.stats_to_track:
            buffers[f"energy_level_{s}"] = np.zeros(N, dtype=np.float32)

        # Consumption and Charging Step Aggregates
        buffers["energy_consumed_total"] = np.zeros(N, dtype=np.float32)
        buffers["energy_consumed_mean"] = np.zeros(N, dtype=np.float32)
        buffers["energy_charged_total"] = np.zeros(N, dtype=np.float32)
        buffers["energy_charged_mean"] = np.zeros(N, dtype=np.float32)

        # Critical state tracking
        buffers["n_uavs_critical"] = np.zeros(N, dtype=np.int32)

        # We store the individual energy levels per UAV per step
        buffers["energy_levels_raw"] = np.zeros((N, n_uavs), dtype=np.float32)

        return buffers

    def _init_episode_trackers(self) -> None:
        """Initializes tracking arrays for per-UAV metrics across the episode."""
        n_uavs = self.world_cfg.n_uavs

        # Per UAV Episode Stats
        self.ep_energy_min = np.full(n_uavs, np.inf, dtype=np.float32)
        self.ep_energy_avg = np.zeros(n_uavs, dtype=np.float32)

        self.ep_total_consumed = np.zeros(n_uavs, dtype=np.float64)
        self.ep_total_charged = np.zeros(n_uavs, dtype=np.float64)

        # Last known energy levels to calculate deltas
        self._prev_energy_levels = np.zeros(n_uavs, dtype=np.float32)
        self._first_step = True

    def update_step(self, batteries: Sequence[Battery]) -> None:
        """Extracts energy metrics from a list of Battery objects.

        Args:
            batteries: List of Battery objects for the UAVs.
        """
        if not batteries or batteries[0].cfg.skip_energy_update:
            return

        idx = self.batch_step_idx
        bufs = self.batch_buffers

        # 1. Extract current levels
        energy_levels = np.array([b.energy_level for b in batteries], dtype=np.float32)

        # 2. Calculate Deltas (Consumed vs Charged)
        # Note: Because the Battery class can reset (recharge_count increment),
        # we calculate consumption and charging based on the internal logic deltas.
        if self._first_step:
            self._prev_energy_levels = np.array(
                [b.cfg.starting_energy_j for b in batteries]
            )
            self._first_step = False

        # In a real step, energy changes.
        # If level < prev: consumed. If level > prev: charged.
        # If level is reset (empty), we handle that via the battery's own recharge_count logic.
        deltas = energy_levels - self._prev_energy_levels

        charged_step = np.where(deltas > 0, deltas, 0.0)
        # For consumption, we look at negative deltas.
        # If a battery reset occurred, this delta logic is simplified by tracking
        # the battery's total consumption method or just positive/negative swings.
        consumed_step = np.where(deltas < 0, -deltas, 0.0)

        # 3. Store Per-Step Fleet Aggregates
        self._store_array_stats(
            bufs, "energy_level", energy_levels, idx, self.stats_to_track
        )

        bufs["energy_consumed_total"][idx] = np.sum(consumed_step)
        bufs["energy_consumed_mean"][idx] = np.mean(consumed_step)
        bufs["energy_charged_total"][idx] = np.sum(charged_step)
        bufs["energy_charged_mean"][idx] = np.mean(charged_step)

        # Critical State: Level below X% of max
        is_critical = energy_levels <= batteries[0].cfg.critical_energy_j
        bufs["n_uavs_critical"][idx] = np.sum(is_critical)
        bufs["energy_levels_raw"][idx, :] = energy_levels

        # 4. Update Episode Trackers
        recharge_counts = np.array(
            [b.recharge_count for b in batteries], dtype=np.int32
        )
        self._update_episode_trackers(
            energy_levels, consumed_step, charged_step, recharge_counts
        )

        # 5. Finalize
        self._prev_energy_levels = energy_levels.copy()
        self._finalize_step()

    def _update_episode_trackers(
        self,
        current_levels: NDArray,
        consumed: NDArray,
        charged: NDArray,
        recharge_counts: NDArray,
    ) -> None:
        """Updates the running episode-level totals and averages."""
        curr_step = self.episode_step_idx + 1

        # Energy Level trackers
        np.minimum(self.ep_energy_min, current_levels, out=self.ep_energy_min)
        self.ep_energy_avg += (current_levels - self.ep_energy_avg) / curr_step

        # Accumulators
        self.ep_total_consumed += consumed
        self.ep_total_charged += charged

        # We store the latest recharge counts (they are absolute counts from the Battery objects)
        self.ep_recharge_events = recharge_counts

    def _build_episode_metrics(self) -> dict[str, MetricValue]:
        """Builds final dictionary of metrics for the whole episode."""
        episode_metrics = {
            # Per UAV metrics
            "ep_energy_min_per_uav": self.ep_energy_min,
            "ep_energy_mean_per_uav": self.ep_energy_avg,
            "ep_energy_final_per_uav": self._prev_energy_levels,  # Last recorded step
            "ep_total_consumed_per_uav": self.ep_total_consumed,
            "ep_total_charged_per_uav": self.ep_total_charged,
            "ep_recharge_events_per_uav": self.ep_recharge_events,
        }

        # Validation
        for key, val in episode_metrics.items():
            if np.any(np.isinf(val)):
                raise ValueError(f"Infinite value in energy metric '{key}'.")

        return episode_metrics

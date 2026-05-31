"""Base metrics controller for UAV simulation.

This module provides an abstract base class for all metrics controllers in the UAV
simulation. It handles common functionality like batch buffering, saving to disk,
and episode-level metric tracking. Child classes must implement layer-specific
metric collection logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.uavnetsim.config.simulation_config import MetricsControllerCfg
from src.uavnetsim.world.world_state import WorldStateCfg

MetricValue = NDArray | np.generic | float | int | bool


class BaseMetricsController(ABC):
    """Abstract base class for all metrics controllers.

    Child classes must:
    - set `layer_name` (e.g. "fh", "bh", "energy")
    - allocate their own batch buffers
    - initialize their own episode trackers
    - implement one-step update logic
    - implement episode-tracker updates
    - implement final episode-metrics export
    """

    layer_name: str = ""

    def __init__(self, cfg: MetricsControllerCfg, world_cfg: WorldStateCfg):
        """Initialize the metrics controller.

        Args:
            cfg: Generic metrics configuration.
            world_cfg: World/simulation configuration object.
                Child classes can extract dimensions from it.
        """
        if not self.layer_name:
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty `layer_name`."
            )

        self.cfg = cfg
        self.world_cfg = world_cfg

        self.buf_size = self.cfg.buffer_size
        self.batch_step_idx = 0
        self.batch_count = 0
        self.episode_step_idx = 0

        self.n_ue = world_cfg.n_ues
        self.n_uavs = world_cfg.n_uavs
        self.n_bs = world_cfg.n_uavs + world_cfg.n_bss

        # Save path is separated by layer name:
        #   <save_dir>/<layer_name>/<layer_name>_metrics_batch_X.npz
        self.save_dir = Path(self.cfg.save_dir) / self.layer_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Per-timestep buffers
        self.batch_buffers: dict[str, NDArray] = self._allocate_batch_buffers()

        # Per-episode trackers
        self._init_episode_trackers()

    @classmethod
    def _store_array_stats(
        cls,
        bufs: dict[str, NDArray],
        prefix: str,
        data: NDArray,
        idx: int,
        stats: list[str],
    ) -> None:
        """Calculate and store descriptive statistics for a 1D/flat array."""
        data = np.asarray(data)

        if data.size == 0:
            for stat_name in stats:
                bufs[f"{prefix}_{stat_name}"][idx] = 0.0
            return

        pct_names = []
        pct_list = []
        for stat_name in stats:
            if stat_name == "min":
                bufs[f"{prefix}_min"][idx] = np.min(data)
            elif stat_name == "max":
                bufs[f"{prefix}_max"][idx] = np.max(data)
            elif stat_name == "mean":
                bufs[f"{prefix}_mean"][idx] = np.mean(data)
            elif stat_name == "median":
                bufs[f"{prefix}_median"][idx] = np.median(data)
            elif stat_name == "var":
                bufs[f"{prefix}_var"][idx] = np.var(data)
            elif stat_name == "std":
                bufs[f"{prefix}_std"][idx] = np.std(data)
            elif stat_name.startswith("p"):
                pct_names.append(stat_name)
                pct_list.append(float(stat_name[1:]))

        if pct_list:
            pcts = np.percentile(data, pct_list)
            for name, val in zip(pct_names, pcts):
                bufs[f"{prefix}_{name}"][idx] = val

    @abstractmethod
    def _allocate_batch_buffers(self) -> dict[str, NDArray]:
        """Pre-allocate all per-timestep batch buffers."""
        raise NotImplementedError

    @abstractmethod
    def _init_episode_trackers(self) -> None:
        """Initialize all per-episode running trackers."""
        raise NotImplementedError

    @abstractmethod
    def update_step(self, *args: Any, **kwargs: Any) -> None:
        """Collect metrics for a single simulation step.

        Child implementations should usually:
        1. read `idx = self.batch_step_idx`
        2. write into `self.batch_buffers`
        3. call `self._update_episode_trackers(...)`
        4. call `self._finalize_step()`
        """
        raise NotImplementedError

    @abstractmethod
    def _update_episode_trackers(self, *args: Any, **kwargs: Any) -> None:
        """Update the running episode-level trackers for the current step."""
        raise NotImplementedError

    @abstractmethod
    def _build_episode_metrics(self) -> dict[str, MetricValue]:
        """Build the final dictionary of episode metrics to save at sim end.

        Usually includes min/max/mean arrays or scalar summaries.
        """
        raise NotImplementedError

    def _finalize_step(self) -> None:
        """Shared end-of-step logic.

        Child classes should call this once at the end of `update_step()`.
        """
        self.batch_step_idx += 1
        self.episode_step_idx += 1

        if self.batch_step_idx >= self.buf_size:
            self.save_batch()

    def save_batch(self) -> None:
        """Save the populated subset of the current batch buffers to disk and reset write index."""
        if self.batch_step_idx == 0:
            return

        file_path = (
            self.save_dir / f"{self.layer_name}_metrics_batch_{self.batch_count}.npz"
        )

        # Save only the valid prefix [0:batch_step_idx]
        save_dict = {
            key: value[: self.batch_step_idx]
            for key, value in self.batch_buffers.items()
        }

        np.savez(file_path, **save_dict)

        self.batch_count += 1
        self.batch_step_idx = 0

    def episode_end(self) -> None:
        """Flush remaining batch data and save final episode-level metrics."""
        self.save_batch()

        if self.episode_step_idx == 0:
            return

        episode_metrics = self._build_episode_metrics()

        file_path = self.save_dir / f"{self.layer_name}_episode_metrics.npz"
        np.savez(file_path, **episode_metrics)

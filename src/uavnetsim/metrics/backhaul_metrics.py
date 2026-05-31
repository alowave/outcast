"""Backhaul Network Telemetry and Performance Metrics Controller.

Tracks, aggregates, and profiles step-level and episode-wide backhaul link
statistics, including capacity utilization, routing hop counts to the ground gateway,
and link payload deficit tracking.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from src.uavnetsim.backhaul.bh_layer import BHLayer
from src.uavnetsim.metrics.base import BaseMetricsController, MetricValue


class BackhaulMetricsController(BaseMetricsController):
    """
    Collects, buffers, and saves Backhaul metrics per time step.
    Handles link statistics, utilization, and hop counts from the Ground Node.
    """

    layer_name = "bh"

    def _allocate_batch_buffers(self) -> Dict[str, NDArray]:
        """Pre-allocates arrays for batch processing."""
        N = self.buf_size
        buffers = {}

        # Basic stats for Link Capacity, Excess, and Missing
        # We track: mean, min, max, median
        self.basic_stats = ["mean", "min", "max", "median"]
        metrics = ["capacity", "excess", "missing", "hops"]

        for m in metrics:
            for s in self.basic_stats:
                buffers[f"{m}_{s}"] = np.zeros(N, dtype=np.float32)

        # Utilization stats (includes variance)
        for s in self.basic_stats + ["var"]:
            buffers[f"utilization_{s}"] = np.zeros(N, dtype=np.float32)

        # Count of links with missing > 0
        buffers["n_links_missing_payload"] = np.zeros(N, dtype=np.int32)

        return buffers

    def _init_episode_trackers(self) -> None:
        """Initializes episode-wide tracking variables."""
        # Matrix to track duration (number of steps) where missing_bps > 0
        # shape: (n_nodes, n_nodes)
        self.ep_duration_missing = np.zeros((self.n_bs, self.n_bs), dtype=np.uint32)

        # Aggregators for episode-wide statistics
        # We store all values from all steps to compute final stats,
        # or we update running distributions. For simplicity and memory efficiency,
        # we will store running lists for the specific episode-wide distributions.
        self._ep_all_capacities = []
        self._ep_all_excess = []
        self._ep_all_missing = []
        self._ep_all_utilization = []
        self._ep_all_hops = []
        self._ep_all_missing_counts = []

    def update_step(self, bh_layer: BHLayer, gn_index: int | None = None) -> None:
        """
        Extracts metrics from BHLayer.

        Args:
            bh_layer: The backhaul layer instance.
            gn_index: Index of the Ground Node/Gateway to calculate hops to.
                When omitted, the first BS index is used.
        """
        if bh_layer.bh_channel_data is None:
            return

        idx = self.batch_step_idx
        bufs = self.batch_buffers
        data = bh_layer.bh_channel_data

        # 1. Prepare Masks
        # We only care about stats for links that actually exist/are in range
        # or have a defined capacity > 0.
        link_mask = data.throughput_bps > 0

        # 2. Extract Link Metrics (Flattened for stats)
        caps = data.throughput_bps[link_mask].astype(np.float32)
        excess = data.excess_bps[link_mask].astype(np.float32)
        missing = data.missing_bps[link_mask].astype(np.float32)

        # 3. Calculate Utilization (flow / capacity)
        # Avoid division by zero by using the link_mask
        flow = data.flow_bps[link_mask].astype(np.float32)
        utilization = np.divide(flow, caps, out=np.zeros_like(caps), where=caps > 0)

        # 4. Calculate Hops from Ground Node (GN)
        # Use the adjacency matrix provided by BHLayer
        hops_from_gn = self._calculate_hops(bh_layer, gn_index)
        # Filter hops to nodes that are actually reachable
        reachable_hops = hops_from_gn[np.isfinite(hops_from_gn) & (hops_from_gn > 0)]

        # 5. Store Per-Step Stats in Buffers
        self._store_array_stats(bufs, "capacity", caps, idx, self.basic_stats)
        self._store_array_stats(bufs, "excess", excess, idx, self.basic_stats)
        self._store_array_stats(bufs, "missing", missing, idx, self.basic_stats)
        self._store_array_stats(
            bufs, "utilization", utilization, idx, self.basic_stats + ["var"]
        )
        self._store_array_stats(bufs, "hops", reachable_hops, idx, self.basic_stats)

        # Count links with missing > 0
        n_missing = int(np.sum(data.missing_bps > 0))
        bufs["n_links_missing_payload"][idx] = n_missing

        # 6. Update Episode Trackers
        self._update_episode_trackers(
            caps,
            excess,
            missing,
            utilization,
            reachable_hops,
            n_missing,
            data.missing_bps,
        )

        self._finalize_step()

    def _calculate_hops(
        self,
        bh_layer: BHLayer,
        gn_index: int | None,
    ) -> NDArray:
        """
        Calculate hop counts from each node to the gateway over active links.

        The controller defines adjacency as ``A[i, j] = 1`` when node ``i``
        forwards traffic to node ``j``. To measure hops to the gateway, we run
        shortest-path on the transposed graph from the gateway node.
        """
        adj = bh_layer._adjacency
        if adj is None:
            return np.full(self.n_bs, np.inf)

        if gn_index is None:
            if bh_layer.n_bss <= 0:
                return np.full(self.n_bs, np.inf)
            gn_index = bh_layer.n_uavs

        if gn_index < 0 or gn_index >= adj.shape[0]:
            raise ValueError(
                f"gn_index {gn_index} is out of bounds for backhaul graph size {adj.shape[0]}."
            )

        reverse_graph = csr_matrix(adj.T)
        hop_counts = shortest_path(
            csgraph=reverse_graph,
            directed=True,
            unweighted=True,
            indices=gn_index,
        )
        return hop_counts

    def _update_episode_trackers(
        self,
        caps: NDArray,
        excess: NDArray,
        missing: NDArray,
        util: NDArray,
        hops: NDArray,
        n_missing: int,
        missing_matrix: NDArray,
    ) -> None:
        """Updates internal lists and matrices for episode-final calculations."""
        # Append data for episode-wide distribution analysis
        self._ep_all_capacities.extend(caps.tolist())
        self._ep_all_excess.extend(excess.tolist())
        self._ep_all_missing.extend(missing.tolist())
        self._ep_all_utilization.extend(util.tolist())
        self._ep_all_hops.extend(hops.tolist())
        self._ep_all_missing_counts.append(n_missing)

        # Update (N, N) duration matrix: increment if link has missing bps
        self.ep_duration_missing += (missing_matrix > 0).astype(np.uint32)

    def _build_episode_metrics(self) -> dict[str, MetricValue]:
        """Constructs the final episode summary metrics."""

        def get_dist_stats(data: list) -> dict:
            if not data:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
            arr = np.array(data)
            return {
                "mean": np.mean(arr),
                "min": np.min(arr),
                "max": np.max(arr),
                "median": np.median(arr),
            }

        # Calculate stats for all collected data points across the episode
        cap_stats = get_dist_stats(self._ep_all_capacities)
        exc_stats = get_dist_stats(self._ep_all_excess)
        miss_stats = get_dist_stats(self._ep_all_missing)
        util_stats = get_dist_stats(self._ep_all_utilization)
        hop_stats = get_dist_stats(self._ep_all_hops)

        episode_metrics = {
            # Capacity Episode Stats
            "ep_link_capacity_mean": cap_stats["mean"],
            "ep_link_capacity_min": cap_stats["min"],
            "ep_link_capacity_max": cap_stats["max"],
            "ep_link_capacity_median": cap_stats["median"],
            # Excess Episode Stats
            "ep_link_excess_mean": exc_stats["mean"],
            "ep_link_excess_min": exc_stats["min"],
            "ep_link_excess_max": exc_stats["max"],
            "ep_link_excess_median": exc_stats["median"],
            # Missing Episode Stats
            "ep_link_missing_mean": miss_stats["mean"],
            "ep_link_missing_min": miss_stats["min"],
            "ep_link_missing_max": miss_stats["max"],
            "ep_link_missing_median": miss_stats["median"],
            # Number of links with missing > 0
            "ep_total_links_with_missing_instances": np.sum(
                self._ep_all_missing_counts
            ),
            # Duration matrix (N, N)
            "ep_missing_duration_matrix": self.ep_duration_missing,
            # Utilization Episode Stats
            "ep_utilization_mean": util_stats["mean"],
            "ep_utilization_min": util_stats["min"],
            "ep_utilization_max": util_stats["max"],
            "ep_utilization_median": util_stats["median"],
            "ep_utilization_var": np.var(self._ep_all_utilization)
            if self._ep_all_utilization
            else 0.0,
            # Hops Episode Stats
            "ep_hops_mean": hop_stats["mean"],
            "ep_hops_min": hop_stats["min"],
            "ep_hops_max": hop_stats["max"],
            "ep_hops_median": hop_stats["median"],
        }

        return episode_metrics

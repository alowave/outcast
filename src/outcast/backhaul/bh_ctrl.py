from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from src.uavnetsim.backhaul.bh_layer import BHLayer
from src.uavnetsim.link_layer.link_data import BackhaulLinkData


class BHController:
    """
    Backhaul controller.

    """

    def __init__(
        self,
        bh_layer: BHLayer,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.bh_layer = bh_layer
        n = self.bh_layer.n_uavs + self.bh_layer.n_bss
        shape = (n, n)
        self.rng = np.random.default_rng() if rng is None else rng
        self.backhaul_outflow_matrix_bps = np.zeros(shape, dtype=np.uint64)
        self.backhaul_adjacency_matrix = np.zeros(shape, dtype=np.bool_)

    def randomize_connections(
        self,
        backhaul_link_data: BackhaulLinkData,
        k_nearest: int = 5,
        ensure_each_bs_used: bool = True,
    ) -> NDArray[np.bool_]:
        """
        Generate adjacency A (N,N), prioritizing short links.

        Convention:
          - Nodes: [UAV0..UAV(N_uav-1), BS0..BS(N_bs-1)]
          - A[i, j] = 1 means node i forwards traffic to node j (directed edge i -> j)

        Guarantees (for N_bs > 0):
          - Each UAV has exactly one outgoing edge.
          - Every UAV has a directed path ending in a BS (no cycles by construction).

        How reachability is guaranteed:
          - We start with all BS nodes considered "already connected".
          - Each UAV connects only to a node that is already connected.
          - Therefore every UAV eventually reaches a BS.
        Args:
            world_state: Provides positions (uav_pos, bs_pos).
            in_range: Optional (N,N) boolean mask; if provided, links i->j are only allowed
                      when in_range[i, j] is True (with simple fallbacks to keep reachability).
            k_nearest: Choose next hop uniformly among the k nearest feasible candidates.
                       (k_nearest=1 => always choose the nearest feasible next hop.)
            ensure_each_bs_used: If True, first try to connect at least one UAV into each BS
                                 (when enough UAVs exist).
            backhaul_link_data: Provides backhaul link data BHLinkData.

        Returns:
            A: (N,N) uint8 adjacency matrix.
            backhaul_link_data:
        """
        in_range = backhaul_link_data.in_range

        n_uav = self.bh_layer.n_uavs
        n_bs = self.bh_layer.n_bss
        n = n_uav + n_bs
        self.backhaul_adjacency_matrix.fill(False)
        self.backhaul_outflow_matrix_bps.fill(0)

        if n_uav == 0:
            return self.backhaul_adjacency_matrix

        if n_bs == 0:
            raise ValueError(
                "Cannot guarantee UAV-to-BS paths when there are zero BS nodes.",
            )

        if in_range is not None and in_range.shape != (n, n):
            raise ValueError("in_range must have shape (N,N) where N=n_uav+n_bs.")

        A = self.backhaul_adjacency_matrix

        uav_nodes = np.arange(0, n_uav, dtype=int)
        bs_nodes = np.arange(n_uav, n, dtype=int)

        # List of nodes that are guaranteed to reach a BS already.
        # Start with BS nodes as roots.
        connected: list[int] = list(bs_nodes)

        # We'll assign each UAV exactly one outgoing edge.
        unassigned_uavs = list(uav_nodes)
        self.rng.shuffle(unassigned_uavs)

        def choose_among_k_nearest(src: int, candidates: NDArray[np.int_]) -> int:
            """Pick one candidate, biased to short distance (uniform among k nearest)."""
            if candidates.size == 1:
                return int(candidates[0])

            d = backhaul_link_data.dist_m[candidates, src]
            k = int(min(max(k_nearest, 1), candidates.size))
            idx = np.argpartition(d, k - 1)[:k]
            return int(self.rng.choice(candidates[idx]))

        if ensure_each_bs_used:
            bs_order = bs_nodes.copy()
            self.rng.shuffle(bs_order)

            for bs in bs_order:
                if not unassigned_uavs:
                    break

                cand_uavs = np.array(unassigned_uavs, dtype=int)

                if in_range is not None:
                    mask = in_range[cand_uavs, bs]
                    cand_uavs_ok = cand_uavs[mask]
                else:
                    cand_uavs_ok = cand_uavs

                if cand_uavs_ok.size == 0:
                    continue

                uav = choose_among_k_nearest(src=bs, candidates=cand_uavs_ok)

                A[uav, bs] = 1
                connected.append(uav)
                unassigned_uavs.remove(uav)

        for uav in unassigned_uavs:
            cands = np.array(connected, dtype=int)

            uav_cands = cands[cands < n_uav]
            if uav_cands.size > 0:
                cands = uav_cands

            if in_range is not None:
                cands_ok = cands[in_range[uav, cands]]
            else:
                cands_ok = cands

            if cands_ok.size == 0:
                if in_range is not None:
                    bs_ok = bs_nodes[in_range[uav, bs_nodes]]
                    if bs_ok.size > 0:
                        cands_ok = bs_ok

            if cands_ok.size == 0:
                cands_ok = bs_nodes

            next_hop = choose_among_k_nearest(src=uav, candidates=cands_ok)

            A[uav, next_hop] = 1
            connected.append(uav)

        return A

    def distribute_random_load(self, g_load: np.ndarray) -> None:
        """
        A[i,j]=1 means directed edge i->j exists.
        g_load[i] is external injected load at node i.
        Requires A to be a DAG (no cycles).

        """
        self.backhaul_outflow_matrix_bps.fill(0)

        n_uav = self.bh_layer.n_uavs
        n_bs = self.bh_layer.n_bss

        N = n_uav + n_bs

        A = self.backhaul_adjacency_matrix

        inflow = np.zeros(N, dtype=np.uint64)

        indeg = A.sum(axis=0).astype(np.uint32)

        q = deque(np.flatnonzero(indeg == 0))
        processed = 0

        while q:
            i = q.popleft()
            processed += 1

            total = int(g_load[i]) + int(inflow[i])
            nbrs = np.flatnonzero(A[i])
            k = nbrs.size

            if k > 0 and total > 0:
                w = self.rng.random(k)
                w /= w.sum()
                flow_shares = np.floor(w * float(total)).astype(np.uint64)
                remainder = total - int(flow_shares.sum(dtype=np.uint64))
                flow_shares[-1] += np.uint64(remainder)

                for t, j in enumerate(nbrs):
                    fij = flow_shares[t]
                    self.backhaul_outflow_matrix_bps[i, j] = fij
                    inflow[j] += fij
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        q.append(j)
            else:
                for j in nbrs:
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        q.append(j)

        if processed != N:
            raise ValueError(
                "A has a directed cycle; DAG single-pass propagation is not valid.",
            )

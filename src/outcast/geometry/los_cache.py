"""
Sector-based Line-of-Sight cache.

The environment is divided into a uniform grid of square sectors, each with
side ``sector_size_m`` metres. For any pair of (2-D) points the cache stores
whether the straight path between the centres of their respective sectors is
clear.

Unlike the previous implementation, cache misses are not resolved by direct
segment-vs-edge intersection checks. Instead, the cache builds a visibility
graph whose endpoints are the active sector centres implied by the current
position arrays, then reads the clear-LOS pairs from that graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.uavnetsim.geometry.visibility_ctrl import VisibilityGraphCtrl
from src.uavnetsim.world.world_state import WorldStateCfg

if TYPE_CHECKING:
    from src.uavnetsim.geometry.obstacle import Obstacle
    from src.uavnetsim.geometry.visibility_graph import VisGraph


@dataclass(slots=True)
class LosCacheCfg:
    """
    Configuration for :class:`LosSectorCache`.

    Attributes:
        sector_size_m:
            Side length (metres) of each square sector. Smaller values give
            finer spatial resolution at the cost of a larger cache array.
    """

    sector_size_m: float = 100.0


class LosSectorCache:
    """
    Sector-granularity LoS cache backed by preallocated NumPy bool arrays.

    Cache misses are resolved in batches by building a visibility graph over
    the currently active sector centres and then populating the sector-pair
    cache from the graph's LOS edges.
    """

    def __init__(self, cfg: LosCacheCfg | None = None) -> None:
        self.cfg = cfg or LosCacheCfg()

        w, h = WorldStateCfg().env_boundary
        sz = self.cfg.sector_size_m

        self._nx: int = int(np.ceil(w / sz))
        self._ny: int = int(np.ceil(h / sz))

        shape = (self._nx, self._ny, self._nx, self._ny)
        self._cached: NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)
        self._los: NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)

        self._vis_ctrl: VisibilityGraphCtrl | None = None
        self._obstacles_ref: list[Obstacle] | None = None

    def reset(self) -> None:
        """Clear the entire cache."""
        self._cached.fill(False)
        self._los.fill(False)

    def compute_los_mask(
        self,
        positions_a: NDArray[np.float32],
        positions_b: NDArray[np.float32],
        obstacles: list[Obstacle],
    ) -> NDArray[np.bool_]:
        """
        Return a boolean LoS mask for every (A, B) pair.

        Results are evaluated at sector granularity: positions that fall into
        the same sector pair share the same LoS value computed from those
        sector centres.
        """
        sx_a, sy_a = self._positions_to_sectors(positions_a)
        sx_b, sy_b = self._positions_to_sectors(positions_b)

        unique_a = self._unique_sector_indices(sx_a, sy_a)
        unique_b = self._unique_sector_indices(sx_b, sy_b)

        self._populate_missing_pairs(unique_a, unique_b, obstacles)

        return self._materialize_mask(sx_a, sy_a, sx_b, sy_b)

    def _positions_to_sectors(
        self, positions: NDArray[np.float32]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        sz = self.cfg.sector_size_m
        sx = np.clip((positions[:, 0] / sz).astype(np.intp), 0, self._nx - 1)
        sy = np.clip((positions[:, 1] / sz).astype(np.intp), 0, self._ny - 1)
        return sx, sy

    def _sector_centre(self, sx: int, sy: int) -> tuple[float, float]:
        half = self.cfg.sector_size_m / 2.0
        return sx * self.cfg.sector_size_m + half, sy * self.cfg.sector_size_m + half

    def _unique_sector_indices(
        self,
        sx: NDArray[np.intp],
        sy: NDArray[np.intp],
    ) -> list[tuple[int, int]]:
        return list({(int(x), int(y)) for x, y in zip(sx, sy, strict=False)})

    def _populate_missing_pairs(
        self,
        sectors_a: list[tuple[int, int]],
        sectors_b: list[tuple[int, int]],
        obstacles: list[Obstacle],
    ) -> None:
        missing_pairs = [
            (sxa, sya, sxb, syb)
            for sxa, sya in sectors_a
            for sxb, syb in sectors_b
            if not self._cached[sxa, sya, sxb, syb]
        ]

        if not missing_pairs:
            return

        endpoint_sectors = {(sxa, sya) for sxa, sya, _, _ in missing_pairs} | {
            (sxb, syb) for _, _, sxb, syb in missing_pairs
        }

        self._ensure_visibility_ctrl(obstacles)
        if self._vis_ctrl is None:
            raise RuntimeError("Visibility controller was not initialized.")

        centre_by_sector = {
            (sx, sy): self._sector_centre(sx, sy) for sx, sy in endpoint_sectors
        }
        sector_by_centre = {
            centre: sector for sector, centre in centre_by_sector.items()
        }
        endpoints = list(centre_by_sector.values())
        self._vis_ctrl.update_endpoints(endpoints)
        vis_graph = self._vis_ctrl.build_graph()

        los_pairs = _collect_los_pairs(vis_graph, sector_by_centre)

        for sx1, sy1 in endpoint_sectors:
            for sx2, sy2 in endpoint_sectors:
                result = (sx1, sy1) == (sx2, sy2) or (
                    (sx1, sy1),
                    (sx2, sy2),
                ) in los_pairs
                self._store_pair(sx1, sy1, sx2, sy2, result)

    def _ensure_visibility_ctrl(self, obstacles: list[Obstacle]) -> None:
        if self._vis_ctrl is None or obstacles is not self._obstacles_ref:
            self._vis_ctrl = VisibilityGraphCtrl(obstacles=obstacles)
            self._obstacles_ref = obstacles

    def _store_pair(
        self,
        sxa: int,
        sya: int,
        sxb: int,
        syb: int,
        result: bool,
    ) -> None:
        self._cached[sxa, sya, sxb, syb] = True
        self._cached[sxb, syb, sxa, sya] = True
        self._los[sxa, sya, sxb, syb] = result
        self._los[sxb, syb, sxa, sya] = result

    def _materialize_mask(
        self,
        sx_a: NDArray[np.intp],
        sy_a: NDArray[np.intp],
        sx_b: NDArray[np.intp],
        sy_b: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        los_mask = np.empty((sx_a.shape[0], sx_b.shape[0]), dtype=np.bool_)

        for i, (sxa, sya) in enumerate(zip(sx_a, sy_a, strict=False)):
            for j, (sxb, syb) in enumerate(zip(sx_b, sy_b, strict=False)):
                los_mask[i, j] = self._los[sxa, sya, sxb, syb]

        return los_mask


def _collect_los_pairs(
    vis_graph: VisGraph,
    sector_by_centre: dict[tuple[float, float], tuple[int, int]],
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    los_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for edge in vis_graph.get_los_edges():
        a = sector_by_centre[(edge.p1.x, edge.p1.y)]
        b = sector_by_centre[(edge.p2.x, edge.p2.y)]
        los_pairs.add((a, b))
        los_pairs.add((b, a))

    return los_pairs

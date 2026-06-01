"""Static Thomas Cluster Process spatial Distribution Model.

Implements a stationary user equipment (UE) placement topology generated via a
Poisson Cluster Process (Thomas distribution). Disperses terminal node coordinates
as isotropic Gaussian child offsets surrounding uniformly distributed parent cluster centers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.outcast.geometry.coords import Coords3d
from src.outcast.world.user_model.base import UserModel

if TYPE_CHECKING:
    from src.outcast.world.world_state import WorldStateCfg


@dataclass(slots=True)
class ThomasClusterCfg:
    """Config specific to the Thomas Cluster Process distribution."""

    ue_cluster_density: float = 0.00001  # Clusters per m^2
    ue_sigma_cluster: float = 50.0  # Standard deviation of child points


class StaticThomasClusterModel(UserModel):
    """Static Poisson Cluster Process (Thomas distribution) user model."""

    def __init__(
        self,
        world_cfg: WorldStateCfg,
        cfg: ThomasClusterCfg | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.world_cfg = world_cfg
        self.cfg = cfg or ThomasClusterCfg()
        self.rng = rng or np.random.default_rng()

    def reset_users(
        self,
        locations: NDArray[np.float32] | list[Coords3d] | None = None,
    ) -> None:
        """Initialize user positions using TCP or from provided data."""
        if locations is not None:
            if isinstance(locations, list):
                self._locations = [loc.copy() for loc in locations]
            else:
                self._locations = [
                    Coords3d.from_array(row) for row in np.asarray(locations)
                ]
            return

        w, h = self.world_cfg.env_boundary
        n_ues = self.world_cfg.n_ues

        # Determine number of clusters
        n_clusters = max(1, self.rng.poisson(w * h * self.cfg.ue_cluster_density))
        ues_per_cluster = self.rng.multinomial(n_ues, [1 / n_clusters] * n_clusters)
        parents = self.rng.uniform([0.0, 0.0], [w, h], size=(n_clusters, 2))

        temp_locations: list[Coords3d] = []
        for i in range(n_clusters):
            if ues_per_cluster[i] > 0:
                offsets = self.rng.normal(
                    0, self.cfg.ue_sigma_cluster, size=(ues_per_cluster[i], 2)
                )
                points = parents[i] + offsets
                for p in points:
                    # Clipping ensures users stay within environment bounds
                    x = float(np.clip(p[0], 0.0, w))
                    y = float(np.clip(p[1], 0.0, h))
                    temp_locations.append(Coords3d(x, y, self.world_cfg.ue_height))

        self._locations = temp_locations

    def step(self, time_step: float) -> None:
        """Static model: positions remain unchanged."""
        pass

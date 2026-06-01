from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.outcast.world.world_state import WorldState


@dataclass(slots=True)
class LoadCfg:
    """
    Configuration for GN/cluster load generation.
    """

    min_random_gn_load: int = 10_000_000_000
    max_random_gn_load: int = 500_000_000_000


class LoadController:
    """
    Generates and applies random GN/cluster loads to a :class:`WorldState`.

    This controller owns how loads are sampled and writes the resulting vector
    into ``state.gn_load``.
    """

    def __init__(self, cfg: LoadCfg, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def apply_random_loads(
        self,
        state: WorldState,
        load_min: int | None = None,
        load_max: int | None = None,
    ) -> NDArray[np.uint64]:
        if load_min is None:
            load_min = self.cfg.min_random_gn_load
        if load_max is None:
            load_max = self.cfg.max_random_gn_load

        n = state.uav_pos.shape[0] + state.bs_pos.shape[0]
        loads = self.rng.integers(load_min, load_max + 1, size=n, dtype=np.uint64)
        state.gn_load = loads
        return loads

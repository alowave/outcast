"""Bounded Random Walk User Mobility Model.

Implements a stochastic user equipment (UE) movement pattern where terminal nodes
undergo uniform random spatial coordinate transitions during each simulation step,
constrained within the explicit configured Cartesian world environment boundary.
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
class RandomMovementCfg:
    step_range: tuple[float, float] = (-1.0, 1.0)


class RandomMovementUserModel(UserModel):
    """Move users by a bounded random x-y offset on each time step."""

    def __init__(
        self,
        world_cfg: WorldStateCfg,
        cfg: RandomMovementCfg | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.world_cfg = world_cfg
        self.cfg = cfg or RandomMovementCfg()
        self.rng = rng or np.random.default_rng()

        low, high = self.cfg.step_range
        if low > high:
            raise ValueError(
                f"random_movement.step_range must satisfy low <= high, got {self.cfg.step_range}."
            )

    def reset_users(
        self,
        locations: NDArray[np.float32] | list[Coords3d] | None = None,
    ) -> None:
        if locations is None:
            w, h = self.world_cfg.env_boundary
            xy = self.rng.uniform([0.0, 0.0], [w, h], size=(self.world_cfg.n_ues, 2))
            z = np.full(
                (self.world_cfg.n_ues, 1), self.world_cfg.ue_height, dtype=float
            )
            locations_array = np.hstack([xy, z])
            self._locations = [Coords3d.from_array(row) for row in locations_array]
            return

        if isinstance(locations, list):
            self._locations = [loc.copy() for loc in locations]
            return

        self._locations = [Coords3d.from_array(row) for row in np.asarray(locations)]

    def step(self, time_step: float) -> None:
        if time_step < 0:
            raise ValueError(f"time_step must be non-negative, got {time_step}.")

        if self._locations is None:
            raise ValueError("Locations not initialized.")

        if not self._locations:
            return

        low, high = self.cfg.step_range
        steps_xy = (
            self.rng.uniform(low, high, size=(len(self._locations), 2)) * time_step
        )
        max_x, max_y = self.world_cfg.env_boundary

        updated_locations: list[Coords3d] = []
        for location, delta_xy in zip(self._locations, steps_xy):
            next_x = float(np.clip(location.x + delta_xy[0], 0.0, max_x))
            next_y = float(np.clip(location.y + delta_xy[1], 0.0, max_y))
            updated_locations.append(Coords3d(next_x, next_y, location.z))

        self._locations = updated_locations

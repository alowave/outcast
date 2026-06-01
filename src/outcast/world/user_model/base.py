from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.outcast.geometry.coords import Coords3d


class UserModel(ABC):
    """Common base class for all user modelling implementations.

    Implementations may represent static or dynamic user models.
    They should all expose the same lifecycle:
    load -> reset -> step -> query state.
    """

    def __init__(self) -> None:
        self._locations: list[Coords3d] | None = None

    @property
    def locations(self) -> list[Coords3d]:
        """Return the current user locations."""
        if self._locations is None:
            raise ValueError("Locations not initialized.")
        return self._locations

    def get_locations_array(self) -> NDArray[np.float32]:
        """Return user locations as an array of shape (n_users, 3)."""
        locations_array = np.empty((len(self.locations), 3), dtype=np.float32)
        self.update_locations_array(locations_array)
        return locations_array

    def update_locations_array(self, locations_array: NDArray[np.float32]) -> None:
        """Write current user locations into an existing ``(n_users, 3)`` array."""
        expected_shape = (len(self.locations), 3)
        if locations_array.shape != expected_shape:
            raise ValueError(
                f"locations_array must have shape {expected_shape}, got {locations_array.shape}."
            )

        for idx, loc in enumerate(self.locations):
            locations_array[idx, 0] = loc.x
            locations_array[idx, 1] = loc.y
            locations_array[idx, 2] = loc.z

    def load_model(self, path: str | Path) -> None:
        """Load model parameters or trajectories from disk, if applicable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support load_model()."
        )

    @abstractmethod
    def reset_users(self, *args: Any, **kwargs: Any) -> None:
        """Reset user state for a new simulation episode or scenario.

        This may initialize user positions, trajectories, or any internal
        state needed by the model.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, time_step: float) -> None:
        """Advance the user model by one simulation time step.

        Static models may implement this as a no-op.

        Args:
            time_step: Duration of the simulation step in seconds.
        """
        raise NotImplementedError

"""3D Coordinate Mathematical Vector Utility.

Provides structural data containers and vector arithmetic operations for managing
3D positions, Euclidean distance steps, and trajectory updates in simulation space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Coords3d:
    """Represents a spatial 3D coordinate point with built-in vector arithmetic support."""

    __slots__ = ["x", "y", "z"]
    x: float
    y: float
    z: float

    def set(self, other_coords: Coords3d) -> None:
        self.x = other_coords.x
        self.y = other_coords.y
        self.z = other_coords.z

    def __str__(self) -> str:
        return f"{{x: {str(self.x)}, y:{str(self.y)}, z:{str(self.z)}}}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coords3d):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (int, float)):
            return Coords3d(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, np.ndarray):
            if len(other) == 2:
                return Coords3d(self.x + other[0], self.y + other[1], self.z)
            elif len(other) == 3:
                return Coords3d(self.x + other[0], self.y + other[1], self.z + other[2])
            raise ValueError(
                f"Array addition requires length 2 or 3, got length {len(other)}"
            )
        else:
            raise ValueError("Undefined operation for given operand!")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return Coords3d(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, np.ndarray):
            if len(other) == 2:
                return Coords3d(self.x - other[0], self.y - other[1], self.z)
            elif len(other) == 3:
                return Coords3d(self.x - other[0], self.y - other[1], self.z - other[2])
            raise ValueError(
                f"Array addition requires length 2 or 3, got length {len(other)}"
            )
        else:
            raise ValueError("Undefined operation for given operand!")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Coords3d(other - self.x, other - self.y, other - self.z)
        raise ValueError("Undefined operation for given operand!")

    def __truediv__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, (int, float)):
            return Coords3d(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __rtruediv__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(other.x / self.x, other.y / self.y, other.z / self.z)
        elif isinstance(other, (int, float)):
            return Coords3d(other / self.x, other / self.y, other / self.z)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __mul__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            return Coords3d(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __le__(self, other: Coords3d) -> bool:
        return self.x <= other.x and self.y <= other.y and self.z <= other.z

    def __rmul__(self, other):
        return self.__mul__(other)

    def get_distance_to(self, other_coords, flag_2d: bool = False) -> float:
        """Calculates the Euclidean absolute distance to an explicit target coordinate set."""
        if isinstance(other_coords, Coords3d):
            return float(
                np.sqrt(
                    (self.x - other_coords.x) ** 2
                    + (self.y - other_coords.y) ** 2
                    + ((self.z - other_coords.z) ** 2 if not flag_2d else 0)
                )
            )
        elif isinstance(other_coords, (tuple, list, np.ndarray)):
            squared_sum = (other_coords[0] - self.x) ** 2 + (
                other_coords[1] - self.y
            ) ** 2
            if len(other_coords) > 2 and not flag_2d:
                squared_sum += (other_coords[2] - self.z) ** 2
            return float(np.sqrt(squared_sum))
        else:
            raise ValueError("Unidentified input format!")

    def copy(self) -> Coords3d:
        return Coords3d(self.x, self.y, self.z)

    def np_array(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z))

    def as_2d_array(self) -> np.ndarray:
        return np.asarray((self.x, self.y))

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=dtype)

    def __len__(self) -> int:
        return 3

    def __getitem__(self, item: int) -> float:
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise ValueError("Out of bounds!")

    @staticmethod
    def from_array(array) -> Coords3d:
        return Coords3d(array[0], array[1], array[2])

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def norm(self) -> float:
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def update(self, destination: Coords3d, distance: float) -> tuple[bool, float]:
        """Steps coordinates toward a destination target vector by a given distance increment."""
        if self == destination:
            return True, 0.0

        direction = destination - self
        remaining_distance = direction.norm()
        direction = direction / remaining_distance

        if remaining_distance <= distance:
            self.set(destination)
            return True, distance - remaining_distance
        else:
            self.set(self + direction * distance)
            return False, 0.0

    def update_coords_from_array(self, np_array: np.ndarray) -> None:
        self.x = np_array[0]
        self.y = np_array[1]
        self.z = np_array[2]

    def in_boundary(self, min_coords: Coords3d, max_coords: Coords3d) -> bool:
        return (
            min_coords.x <= self.x <= max_coords.x
            and min_coords.y <= self.y <= max_coords.y
            and min_coords.z <= self.z <= max_coords.z
        )


def to_coords_3d(_array) -> Coords3d:
    return Coords3d(_array[0], _array[1], _array[2] if len(_array) > 2 else 0.0)

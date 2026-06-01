"""Geometric 3D Obstacle Representation.

Provides spatial data containers for physical obstacles (e.g., buildings)
and handles geometric collision detection using Shapely.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon

from src.uavnetsim.geometry.coords import Coords3d


class Obstacle:
    """
    Represents a single physical obstacle (e.g., building) in 3D space.

    This class handles the geometric math for a single building, including
    collision detection using Shapely.
    """

    def __init__(
        self,
        obstacle_id: int,
        height: float,
        vertices: list[tuple[float, float]],
        walls_normal: list[np.ndarray] | None = None,
    ) -> None:
        self.id = obstacle_id
        self.height = height
        self.vertices = vertices
        self.walls_normal = walls_normal if walls_normal is not None else []

        self._polygon = Polygon(self.vertices)

    def is_overlapping(self, coords: Coords3d) -> bool:
        """
        Check if a 3D point (Coords3d) is inside this obstacle.

        Args:
            coords: The 3D coordinates to check.

        Returns:
            True if the point is within the footprint and below the height.
        """
        if coords.z > self.height or coords.z < 0:
            return False

        point_2d = Point(coords.x, coords.y)
        return self._polygon.contains(point_2d)

    def __str__(self) -> str:
        return f"Obstacle(id={self.id}, height={self.height}, vertices={len(self.vertices)})"

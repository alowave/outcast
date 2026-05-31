"""Visibility graph controller for UAV simulation.

This module provides a lightweight controller for managing visibility graphs
in UAV simulation scenarios. It handles obstacle polygons and endpoints,
and builds visibility graphs for line-of-sight calculations.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.uavnetsim.geometry.obstacle import Obstacle
from src.uavnetsim.geometry.visibility_graph import Point, VisGraph

Coord = Sequence[float]


class VisibilityGraphCtrl:
    """Lightweight visibility-graph controller.

    This class only prepares obstacle polygons and endpoints,
    then explicitly builds a VisGraph.

    Args:
        obstacles: List of Obstacle objects. Each Obstacle is expected to have
            a Shapely Polygon stored in ``obstacle._polygon``.
        endpoints: Optional iterable of endpoint coordinates.
    """

    def __init__(
        self,
        obstacles: list[Obstacle] | None = None,
        endpoints: Iterable[Coord] | None = None,
    ):
        self.obstacles_list: list[Obstacle] = []
        self.obstacles_polys: list[list[Point]] = []
        self.endpoints: list[Point] = []
        self.vis_graph: VisGraph | None = None

        self.update_obstacles(obstacles or [])
        self.update_endpoints(endpoints or [])

    def update_obstacles(self, obstacles: list[Obstacle]) -> None:
        """Replace obstacle objects and rebuild their visibility polygons.

        build_graph() must be called explicitly after this.
        """
        self.obstacles_list = obstacles
        self.obstacles_polys = [
            self._obstacle_to_vis_polygon(obstacle) for obstacle in self.obstacles_list
        ]

        self.vis_graph = None

    def update_endpoints(self, endpoints: Iterable[Coord]) -> None:
        """Replace graph endpoints.

        build_graph() must be called explicitly after this.
        """
        self.endpoints = self._dedupe_points(
            self._to_point(endpoint) for endpoint in endpoints
        )

        self.vis_graph = None

    def build_graph(self) -> VisGraph:
        """Build visibility graph using obstacle polygons and endpoints.

        Only endpoints participate as visibility-graph nodes.
        Obstacle polygon vertices are used only as LOS blockers.
        """
        self.vis_graph = VisGraph()

        input_polys = self.obstacles_polys + [[endpoint] for endpoint in self.endpoints]

        self.vis_graph.build(input_polys)

        return self.vis_graph

    @property
    def all_points(self) -> list[Point]:
        """Return the deduplicated endpoint set used to build the visibility graph."""
        return list(self.endpoints)

    @property
    def n_points(self) -> int:
        """Return the total number of unique points in the visibility graph."""
        return len(self.all_points)

    @staticmethod
    def _obstacle_to_vis_polygon(obstacle: Obstacle) -> list[Point]:
        """Convert an Obstacle's Shapely polygon exterior into VisGraph Points.

        Shapely repeats the first coordinate at the end of exterior.coords;
        that duplicate closing coordinate is removed.
        """
        polygon = obstacle._polygon

        if polygon.is_empty:
            return []

        coords = list(polygon.exterior.coords)

        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        return [Point(x, y) for x, y in coords]

    @staticmethod
    def _to_point(coord: Coord | Point) -> Point:
        if isinstance(coord, Point):
            return coord

        if len(coord) != 2:
            raise ValueError(f"Expected a 2D coordinate, got: {coord}")

        return Point(coord[0], coord[1])

    @staticmethod
    def _dedupe_points(points: Iterable[Point]) -> list[Point]:
        seen: set[tuple[float, float]] = set()
        unique_points: list[Point] = []

        for point in points:
            key = (point.x, point.y)

            if key not in seen:
                seen.add(key)
                unique_points.append(point)

        return unique_points

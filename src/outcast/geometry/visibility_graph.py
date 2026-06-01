"""Visibility graph implementation for UAV line-of-sight calculations.

This module implements a visibility graph algorithm for computing line-of-sight
relationships between points in a 2D space with obstacles. It uses a rotational
sweep-line algorithm to efficiently determine visibility.

Classes:
    Point: 2D point representation with polygon ID tracking.
    Edge: Line segment between two points.
    Graph: Stores polygon boundary edges and visibility edges.
    VisGraph: Builds visibility graph and exposes clear-LOS edges.
    EdgeKey: Sort key for rotational sweep-line algorithm.

Functions:
    visible_vertices: Find all points visible from a given point.
    edge_in_polygon: Check if an edge lies inside a polygon.
    polygon_crossing: Point-in-polygon test using ray crossing.
    edge_distance: Euclidean distance between two points.
    intersect_point: Find intersection between line segment and edge.
    point_edge_distance: Distance from point to edge intersection.
    angle: Compute angle from center to point.
    angle2: Compute angle between three points.
    ccw: Determine orientation of three points (counter-clockwise/clockwise).
    on_segment: Check if point lies on line segment.
    edge_intersect: Check if two line segments intersect.
    insort: Insert value into sorted list maintaining order.
    bisect: Find insertion point in sorted list.
"""

from __future__ import annotations

from collections import defaultdict
from math import acos, atan, pi, sqrt
from multiprocessing import Pool
from typing import Iterable

INF = 1_000_000_000

# Floating-point tolerance used by orientation tests.
COLIN_TOLERANCE = 10
T = 10**COLIN_TOLERANCE
T2 = 10.0**COLIN_TOLERANCE


class Point:
    """2D point representation with polygon ID tracking.

    Attributes:
        x: X-coordinate.
        y: Y-coordinate.
        polygon_id: ID of the polygon this point belongs to (-1 if none).
    """

    """2D point representation with polygon ID tracking.

    Attributes:
        x: X-coordinate.
        y: Y-coordinate.
        polygon_id: ID of the polygon this point belongs to (-1 if none).
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other: object) -> bool:
        """Check equality based on x and y coordinates."""
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """Return hash based on x and y coordinates."""
        return hash(self.x) ^ hash(self.y)

    def __iter__(self):
        """Iterate over x and y coordinates."""
        yield self.x
        yield self.y

    def __str__(self) -> str:
        """Return string representation with 2 decimal precision."""
        return f"({self.x:.2f}, {self.y:.2f})"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"Point({self.x:.2f}, {self.y:.2f})"


class Edge:
    """Represents a line segment between two points.

    Attributes:
        p1: First point of the edge.
        p2: Second point of the edge.
    """

    __slots__ = ("p1", "p2")

    def __init__(self, point1: Point, point2: Point):
        self.p1 = point1
        self.p2 = point2

    def get_adjacent(self, point: Point) -> Point:
        """Return the other point of the edge.

        Args:
            point: One of the edge's endpoints.

        Returns:
            The other endpoint of the edge.
        """
        if point == self.p1:
            return self.p2
        return self.p1

    def as_tuple(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return edge as tuple of coordinate pairs.

        Returns:
            Tuple containing ((x1, y1), (x2, y2)).
        """
        return (self.p1.x, self.p1.y), (self.p2.x, self.p2.y)

    def __contains__(self, point: Point) -> bool:
        """Check if point is one of the edge's endpoints."""
        return self.p1 == point or self.p2 == point

    def __eq__(self, other: object) -> bool:
        """Check equality based on endpoints (order-independent)."""
        if not isinstance(other, Edge):
            return False

        return (self.p1 == other.p1 and self.p2 == other.p2) or (
            self.p1 == other.p2 and self.p2 == other.p1
        )

    def __hash__(self) -> int:
        """Return hash based on endpoints."""
        return hash(self.p1) ^ hash(self.p2)

    def __str__(self) -> str:
        """Return string representation of edge."""
        return f"({self.p1}, {self.p2})"

    def __repr__(self) -> str:
        """Return detailed string representation of edge."""
        return f"Edge({self.p1!r}, {self.p2!r})"


class Graph:
    """Stores polygon boundary edges and visibility edges.

    Input format:
        [
            [Point(...), Point(...), Point(...)],  # obstacle polygon
            [Point(...)],                          # endpoint
        ]

    Polygons with 3 or more points are treated as obstacles.
    Single-point polygons are treated as standalone endpoints.
    """

    def __init__(self, polygons: Iterable[Iterable[Point]]):
        self.graph: dict[Point, set[Edge]] = defaultdict(set)
        self.edges: set[Edge] = set()
        self.polygons: dict[int, set[Edge]] = defaultdict(set)
        self.point_polygons: dict[Point, set[int]] = defaultdict(set)
        self.nodes: set[Point] = set()

        polygon_id = 0

        for polygon in polygons:
            points = list(polygon)

            if not points:
                continue

            if len(points) > 1 and points[0] == points[-1]:
                points = points[:-1]

            if len(points) == 1:
                self.add_point(points[0], is_node=True)
                continue

            is_obstacle_polygon = len(points) > 2

            for i, point in enumerate(points):
                sibling_point = points[(i + 1) % len(points)]
                edge = Edge(point, sibling_point)

                if is_obstacle_polygon:
                    self.point_polygons[point].add(polygon_id)
                    self.point_polygons[sibling_point].add(polygon_id)
                    self.polygons[polygon_id].add(edge)

                self.add_edge(edge, register_nodes=not is_obstacle_polygon)

            if is_obstacle_polygon:
                polygon_id += 1

    def add_point(self, point: Point, *, is_node: bool = False) -> None:
        """Add a standalone point to the graph.

        Args:
            point: Point to add to the graph.
        """
        self.graph[point]
        if is_node:
            self.nodes.add(point)

    def add_edge(self, edge: Edge, *, register_nodes: bool = True) -> None:
        """Add an edge to the graph.

        Args:
            edge: Edge to add to the graph.
        """
        self.graph[edge.p1].add(edge)
        self.graph[edge.p2].add(edge)
        self.edges.add(edge)

        if register_nodes:
            self.nodes.add(edge.p1)
            self.nodes.add(edge.p2)

    def get_adjacent_points(self, point: Point) -> list[Point]:
        """Return all points connected to the given point.

        Args:
            point: Point to get adjacent points for.

        Returns:
            List of adjacent points.
        """
        return [edge.get_adjacent(point) for edge in self[point]]

    def get_points(self) -> list[Point]:
        """Return all points in the graph.

        Returns:
            List of all points.
        """
        return list(self.graph)

    def get_nodes(self) -> list[Point]:
        return list(self.nodes)

    def get_edges(self) -> set[Edge]:
        """Return all edges in the graph.

        Returns:
            Set of all edges.
        """
        return self.edges

    def __contains__(self, item: object) -> bool:
        """Check if point or edge is in the graph."""
        if isinstance(item, Point):
            return item in self.graph

        if isinstance(item, Edge):
            return item in self.edges

        return False

    def __getitem__(self, point: Point) -> set[Edge]:
        """Return edges connected to the given point.

        Args:
            point: Point to get edges for.

        Returns:
            Set of edges connected to the point.
        """
        return self.graph.get(point, set())

    def __str__(self) -> str:
        """Return string representation of graph adjacency."""
        result = ""

        for point in self.graph:
            result += f"\n{point}: "
            for edge in self.graph[point]:
                result += str(edge)

        return result

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return self.__str__()


class VisGraph:
    """Builds a visibility graph and exposes clear-LOS edges.

    `self.graph` stores obstacle boundary geometry.
    `self.visgraph` stores visibility edges.
    """

    def __init__(self):
        self.graph: Graph | None = None
        self.visgraph: Graph | None = None

    def build(
        self,
        polygons: list[list[Point]],
        *,
        workers: int = 1,
        batch_size: int = 10,
    ) -> Graph:
        """Build the visibility graph.

        Parameters
        ----------
        polygons:
            List of obstacle polygons and endpoint singletons.

        Example:
                [
                    [Point(0, 0), Point(10, 0), Point(10, 10)],
                    [Point(20, 20)],
                ]

        workers:
            Number of multiprocessing workers. Use 1 for sequential execution.

        batch_size:
            Number of points processed per visibility batch.
        """
        self.graph = Graph(polygons)
        self.visgraph = Graph([])

        points = self.graph.get_nodes()
        batches = [
            points[i : i + batch_size] for i in range(0, len(points), batch_size)
        ]

        if workers == 1:
            for batch in batches:
                for edge in _vis_graph(self.graph, batch):
                    self.visgraph.add_edge(edge)
        else:
            with Pool(workers) as pool:
                args = [(self.graph, batch) for batch in batches]

                for visible_edges in pool.imap(_vis_graph_wrapper, args):
                    for edge in visible_edges:
                        self.visgraph.add_edge(edge)

        return self.visgraph

    def find_visible(self, point: Point) -> list[Point]:
        """Return graph points that have clear LOS from `point`."""
        self._ensure_built()
        return visible_vertices(point, self.graph)

    def get_los_edges(self) -> list[Edge]:
        """Return all visibility-graph edges.

        These are the edges with clear LOS according to the visibility graph.
        """
        self._ensure_built()
        return list(self.visgraph.get_edges())

    def get_los_edge_coordinates(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return all clear-LOS edges as coordinate pairs."""
        return [edge.as_tuple() for edge in self.get_los_edges()]

    def _ensure_built(self) -> None:
        if self.graph is None or self.visgraph is None:
            raise RuntimeError(
                "Visibility graph has not been built. Call build() first."
            )


def _vis_graph_wrapper(args: tuple[Graph, list[Point]]) -> list[Edge]:
    return _vis_graph(*args)


def _vis_graph(graph: Graph, points: list[Point]) -> list[Edge]:
    visible_edges = []

    for p1 in points:
        for p2 in visible_vertices(p1, graph, candidate_points=points, scan="full"):
            visible_edges.append(Edge(p1, p2))

    return visible_edges


def visible_vertices(
    point: Point,
    graph: Graph,
    candidate_points: Iterable[Point] | None = None,
    scan: str = "full",
) -> list[Point]:
    """Return graph points visible from `point`.

    scan="full":
        Checks all angular directions.

    scan="half":
        Checks only half the sweep. Useful when constructing symmetric graphs,
        but `full` is clearer and safer for the current use case.
    """
    if scan not in {"full", "half"}:
        raise ValueError("scan must be either 'full' or 'half'.")

    candidate_list = (
        list(candidate_points) if candidate_points is not None else graph.get_nodes()
    )
    candidate_set = set(candidate_list)
    sweep_points = list(set(candidate_list + graph.get_points()))
    sweep_points.sort(key=lambda p: (angle(point, p), edge_distance(point, p)))

    open_edges: list[EdgeKey] = []
    point_inf = Point(INF, point.y)

    for edge in graph.get_edges():
        if point in edge:
            continue

        if edge_intersect(point, point_inf, edge):
            if on_segment(point, edge.p1, point_inf):
                continue

            if on_segment(point, edge.p2, point_inf):
                continue

            insort(open_edges, EdgeKey(point, point_inf, edge))

    visible = []
    previous = None
    previous_visible = None

    for candidate in sweep_points:
        if candidate == point:
            continue

        if scan == "half" and angle(point, candidate) > pi:
            break

        if open_edges:
            for edge in graph[candidate]:
                if ccw(point, candidate, edge.get_adjacent(candidate)) == -1:
                    key = EdgeKey(point, candidate, edge)
                    index = bisect(open_edges, key) - 1

                    if index >= 0 and open_edges[index] == key:
                        del open_edges[index]

        is_visible = False

        if (
            previous is None
            or ccw(point, previous, candidate) != 0
            or not on_segment(point, previous, candidate)
        ):
            if len(open_edges) == 0:
                is_visible = True
            elif not edge_intersect(point, candidate, open_edges[0].edge):
                is_visible = True

        elif not previous_visible:
            is_visible = False

        else:
            is_visible = True

            for edge_key in open_edges:
                if previous not in edge_key.edge and edge_intersect(
                    previous,
                    candidate,
                    edge_key.edge,
                ):
                    is_visible = False
                    break

            if is_visible and edge_in_polygon(previous, candidate, graph):
                is_visible = False

        if is_visible and candidate not in graph.get_adjacent_points(point):
            is_visible = not edge_in_polygon(point, candidate, graph)

        if is_visible and candidate in candidate_set:
            visible.append(candidate)

        for edge in graph[candidate]:
            if (
                point not in edge
                and ccw(point, candidate, edge.get_adjacent(candidate)) == 1
            ):
                insort(open_edges, EdgeKey(point, candidate, edge))

        previous = candidate
        previous_visible = is_visible

    return visible


def edge_in_polygon(p1: Point, p2: Point, graph: Graph) -> bool:
    """Check if an edge lies inside a polygon.

    Args:
        p1: First point of the edge.
        p2: Second point of the edge.
        graph: Graph containing polygon edge information.

    Returns:
        True if the edge midpoint lies inside the polygon, False otherwise.
    """
    """Check if an edge lies inside a polygon.

    Args:
        p1: First point of the edge.
        p2: Second point of the edge.
        graph: Graph containing polygon edge information.

    Returns:
        True if the edge midpoint lies inside the polygon, False otherwise.
    """
    shared_polygons = graph.point_polygons[p1] & graph.point_polygons[p2]

    if not shared_polygons:
        return False

    midpoint = Point(
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2,
    )

    for poly_id in shared_polygons:
        if polygon_crossing(midpoint, graph.polygons[poly_id]):
            return True

    return False


def polygon_crossing(point: Point, polygon_edges: Iterable[Edge]) -> bool:
    """Return True if `point` is inside the polygon represented by `polygon_edges`."""
    point_inf = Point(INF, point.y)

    intersect_count = 0
    collinear_flag = False
    collinear_direction = 0

    for edge in polygon_edges:
        if point.y < edge.p1.y and point.y < edge.p2.y:
            continue

        if point.y > edge.p1.y and point.y > edge.p2.y:
            continue

        collinear_p1 = ccw(point, edge.p1, point_inf) == 0 and edge.p1.x > point.x
        collinear_p2 = ccw(point, edge.p2, point_inf) == 0 and edge.p2.x > point.x

        if collinear_p1 and collinear_p2:
            continue

        collinear_point = edge.p1 if collinear_p1 else edge.p2

        if collinear_p1 or collinear_p2:
            if edge.get_adjacent(collinear_point).y > point.y:
                collinear_direction += 1
            else:
                collinear_direction -= 1

            if collinear_flag:
                if collinear_direction == 0:
                    intersect_count += 1

                collinear_flag = False
                collinear_direction = 0
            else:
                collinear_flag = True

        elif edge_intersect(point, point_inf, edge):
            intersect_count += 1

    return intersect_count % 2 == 1


def edge_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Euclidean distance between p1 and p2.
    """
    return sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def intersect_point(p1: Point, p2: Point, edge: Edge) -> Point | None:
    """Return the intersection point between segment p1-p2 and `edge`."""
    if p1 in edge:
        return p1

    if p2 in edge:
        return p2

    if edge.p1.x == edge.p2.x:
        if p1.x == p2.x:
            return None

        p_slope = (p1.y - p2.y) / (p1.x - p2.x)
        intersect_x = edge.p1.x
        intersect_y = p_slope * (intersect_x - p1.x) + p1.y

        return Point(intersect_x, intersect_y)

    if p1.x == p2.x:
        edge_slope = (edge.p1.y - edge.p2.y) / (edge.p1.x - edge.p2.x)
        intersect_x = p1.x
        intersect_y = edge_slope * (intersect_x - edge.p1.x) + edge.p1.y

        return Point(intersect_x, intersect_y)

    p_slope = (p1.y - p2.y) / (p1.x - p2.x)
    edge_slope = (edge.p1.y - edge.p2.y) / (edge.p1.x - edge.p2.x)

    if edge_slope == p_slope:
        return None

    intersect_x = (edge_slope * edge.p1.x - p_slope * p1.x + p1.y - edge.p1.y) / (
        edge_slope - p_slope
    )

    intersect_y = edge_slope * (intersect_x - edge.p1.x) + edge.p1.y

    return Point(intersect_x, intersect_y)


def point_edge_distance(p1: Point, p2: Point, edge: Edge) -> float:
    """Return distance from p1 to the intersection point with `edge`."""
    intersection = intersect_point(p1, p2, edge)

    if intersection is not None:
        return edge_distance(p1, intersection)

    return 0


def angle(center: Point, point: Point) -> float:
    """Compute angle from center to point in radians.

    Args:
        center: Center point.
        point: Target point.

    Returns:
        Angle in radians [0, 2π).
    """
    dx = point.x - center.x
    dy = point.y - center.y

    if dx == 0:
        if dy < 0:
            return pi * 3 / 2

        return pi / 2

    if dy == 0:
        if dx < 0:
            return pi

        return 0

    if dx < 0:
        return pi + atan(dy / dx)

    if dy < 0:
        return 2 * pi + atan(dy / dx)

    return atan(dy / dx)


def angle2(point_a: Point, point_b: Point, point_c: Point) -> float:
    """Compute angle at point_b formed by point_a and point_c.

    Args:
        point_a: First point.
        point_b: Vertex point.
        point_c: Third point.

    Returns:
        Angle in radians.
    """
    a = (point_c.x - point_b.x) ** 2 + (point_c.y - point_b.y) ** 2
    b = (point_c.x - point_a.x) ** 2 + (point_c.y - point_a.y) ** 2
    c = (point_b.x - point_a.x) ** 2 + (point_b.y - point_a.y) ** 2

    cos_value = (a + c - b) / (2 * sqrt(a) * sqrt(c))
    cos_value = max(-1.0, min(1.0, cos_value))

    return acos(int(cos_value * T) / T2)


def ccw(point_a: Point, point_b: Point, point_c: Point) -> int:
    """Determine orientation of three points.

    Returns:
        1 if counter-clockwise.
        -1 if clockwise.
        0 if collinear.
    """
    area = (
        int(
            (
                (point_b.x - point_a.x) * (point_c.y - point_a.y)
                - (point_b.y - point_a.y) * (point_c.x - point_a.x)
            )
            * T
        )
        / T2
    )

    if area > 0:
        return 1

    if area < 0:
        return -1

    return 0


def on_segment(point_a: Point, point_b: Point, point_c: Point) -> bool:
    """Given collinear points A, B, C, return True if B lies on segment AC."""
    return min(point_a.x, point_c.x) <= point_b.x <= max(point_a.x, point_c.x) and min(
        point_a.y, point_c.y
    ) <= point_b.y <= max(point_a.y, point_c.y)


def edge_intersect(p1: Point, q1: Point, edge: Edge) -> bool:
    """Check if line segment p1-q1 intersects with edge.

    Args:
        p1: First point of first segment.
        q1: Second point of first segment.
        edge: Edge to check intersection against.

    Returns:
        True if segments intersect, False otherwise.
    """
    p2 = edge.p1
    q2 = edge.p2

    o1 = ccw(p1, q1, p2)
    o2 = ccw(p1, q1, q2)
    o3 = ccw(p2, q2, p1)
    o4 = ccw(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def insort(items: list, value) -> None:
    """Insert value into sorted list while maintaining sort order.

    Args:
        items: Sorted list to insert into.
        value: Value to insert.
    """
    lo = 0
    hi = len(items)

    while lo < hi:
        mid = (lo + hi) // 2

        if value < items[mid]:
            hi = mid
        else:
            lo = mid + 1

    items.insert(lo, value)


def bisect(items: list, value) -> int:
    """Find insertion point for value in sorted list.

    Args:
        items: Sorted list to search.
        value: Value to find insertion point for.

    Returns:
        Index where value should be inserted to maintain sort order.
    """
    lo = 0
    hi = len(items)

    while lo < hi:
        mid = (lo + hi) // 2

        if value < items[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo


class EdgeKey:
    """Sort key for the rotational sweep-line visibility algorithm."""

    def __init__(self, p1: Point, p2: Point, edge: Edge):
        self.p1 = p1
        self.p2 = p2
        self.edge = edge

    def __eq__(self, other: object) -> bool:
        """Check equality based on edge."""
        return isinstance(other, EdgeKey) and self.edge == other.edge

    def __lt__(self, other: "EdgeKey") -> bool:
        """Compare edge keys for sorting in sweep-line algorithm."""
        if self.edge == other.edge:
            return False

        if not edge_intersect(self.p1, self.p2, other.edge):
            return True

        self_distance = point_edge_distance(self.p1, self.p2, self.edge)
        other_distance = point_edge_distance(self.p1, self.p2, other.edge)

        if self_distance > other_distance:
            return False

        if self_distance < other_distance:
            return True

        same_point = None

        if self.edge.p1 in other.edge:
            same_point = self.edge.p1
        elif self.edge.p2 in other.edge:
            same_point = self.edge.p2

        if same_point is None:
            return False

        self_angle = angle2(
            self.p1,
            self.p2,
            self.edge.get_adjacent(same_point),
        )

        other_angle = angle2(
            self.p1,
            self.p2,
            other.edge.get_adjacent(same_point),
        )

        return self_angle < other_angle

    def __repr__(self) -> str:
        """Return string representation of EdgeKey."""
        return f"EdgeKey(edge={self.edge!r}, p1={self.p1!r}, p2={self.p2!r})"

"""Voronoi Roadmap Generation and Dijkstra Routing Graph.

Computes open-space navigation topologies from physical obstacle vertex hulls
using Scipy 2D Voronoi tessellations. Provides graph structures and Dijkstra
shortest-path algorithms to calculate collision-free paths.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import Voronoi

from src.outcast.geometry.coords import Coords3d


class Graph:
    def __init__(self):
        self.vertices: dict[int, Coords3d] = {}
        self.adj: dict[int, dict[int, float]] = {}

    def add_edge(self, u_id: int, u_coords: Coords3d, v_id: int, v_coords: Coords3d):
        if u_id not in self.vertices:
            self.vertices[u_id] = u_coords
            self.adj[u_id] = {}
        if v_id not in self.vertices:
            self.vertices[v_id] = v_coords
            self.adj[v_id] = {}

        # Distance remains 2D-based for the mobility graph
        dist = u_coords.get_distance_to(v_coords, flag_2d=True)
        self.adj[u_id][v_id] = dist
        self.adj[v_id][u_id] = dist

    def get_shortest_path(self, start_id: int, end_id: int) -> list[Coords3d]:
        """Runs Dijkstra to find the shortest path between graph vertices."""
        distances = {v: float("inf") for v in self.vertices}
        previous = {v: None for v in self.vertices}
        distances[start_id] = 0
        pq = [(0, start_id)]

        while pq:
            curr_dist, u = heapq.heappop(pq)
            if u == end_id:
                break
            if curr_dist > distances[u]:
                continue

            for v, weight in self.adj[u].items():
                new_dist = curr_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    previous[v] = u
                    heapq.heappush(pq, (new_dist, v))

        path = []
        curr = end_id
        while curr is not None:
            path.append(self.vertices[curr].copy())
            curr = previous[curr]
        return path[::-1]


def build_voronoi_graph(obstacles_list, boundary) -> Graph:
    """
    Generates the roadmap from obstacles using a 2D Voronoi diagram.
    The boundary is typically passed from world_cfg.env_boundary.
    """
    graph = Graph()
    all_pts = []

    # Extract 2D vertices from obstacles
    for obs in obstacles_list:
        all_pts.extend([v[:2] for v in obs.vertices])

    pts = np.unique(np.array(all_pts), axis=0)
    if pts.shape[0] < 3:
        return graph

    vor = Voronoi(pts)
    w, h = boundary

    for simplex in vor.ridge_vertices:
        # Only consider ridges with finite vertices (non-infinite)
        if np.all(np.array(simplex) >= 0):
            v1_raw, v2_raw = vor.vertices[simplex[0]], vor.vertices[simplex[1]]

            # Check if both Voronoi vertices are within the map boundary
            if (
                0 <= v1_raw[0] <= w
                and 0 <= v1_raw[1] <= h
                and 0 <= v2_raw[0] <= w
                and 0 <= v2_raw[1] <= h
            ):
                # Height is set to 0.0; specific UE height is handled by the mobility model or simulation
                graph.add_edge(
                    simplex[0],
                    Coords3d(v1_raw[0], v1_raw[1], 0.0),
                    simplex[1],
                    Coords3d(v2_raw[0], v2_raw[1], 0.0),
                )
    return graph

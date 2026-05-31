"""Spatial Obstacle Lifecycle and Environment Bounds Controller.

Manages the deserialization, instantiation, and geometric querying of physical
map structures (e.g., NumPy boundary matrices) for collision avoidance pipelines
and world simulation boundary scaling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from src.uavnetsim.geometry.coords import Coords3d
from src.uavnetsim.geometry.obstacle import Obstacle


@dataclass(slots=True)
class ObstacleCfg:
    """Configuration for obstacle loading."""

    enabled: bool = True
    map_title: str = "Poznan"
    boundary_margin: int = 20
    default_height: float = 50.0


class ObstacleController:
    """
    Controller responsible for loading and managing the collection of obstacles.
    """

    def __init__(self, cfg: ObstacleCfg) -> None:
        self.cfg = cfg or ObstacleCfg()
        self.obstacles_list: list[Obstacle] = []

    def load_obstacles(self, map_title: str | None = None) -> None:
        """
        Populates self.obstacles_list based on the map title.
        """
        if not self.cfg.enabled:
            return

        map_title = map_title or self.cfg.map_title

        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_x = os.path.join(base_dir, "saved_maps", map_title, "xs.npy")
        path_y = os.path.join(base_dir, "saved_maps", map_title, "ys.npy")
        path_z = os.path.join(base_dir, "saved_maps", map_title, "zs.npy")

        try:
            _obs_x = np.load(path_x, allow_pickle=True)
            _obs_y = np.load(path_y, allow_pickle=True)
            if os.path.exists(path_z):
                _obs_z = np.load(path_z, allow_pickle=True)
            else:
                _obs_z = np.full(len(_obs_x), self.cfg.default_height, dtype=float)

            if len(_obs_x) != len(_obs_y) or len(_obs_x) != len(_obs_z):
                raise ValueError(
                    f"Mismatched obstacle data lengths for map '{map_title}': "
                    f"xs={len(_obs_x)}, ys={len(_obs_y)}, zs={len(_obs_z)}"
                )

            self._setup_obstacles(zip(_obs_x, _obs_y, _obs_z))

            print(
                f"ObstacleController: Loaded {len(self.obstacles_list)} obstacles from {map_title}"
            )

        except FileNotFoundError:
            print(f"Warning: Obstacle data not found at {path_x}")

    def _setup_obstacles(self, obstacles_data_list) -> None:
        """Internal method to create Obstacle objects."""
        for obstacle_id, obstacle_data in enumerate(obstacles_data_list):
            vertices = []
            obstacle_xs, obstacle_ys, height = obstacle_data

            for idx in range(len(obstacle_xs)):
                vertices.append((obstacle_xs[idx], obstacle_ys[idx]))

            new_obstacle = Obstacle(
                obstacle_id=obstacle_id, height=float(height), vertices=vertices
            )
            self.obstacles_list.append(new_obstacle)

    def check_overlap(self, coords: Coords3d) -> bool:
        """Checks if a point overlaps with any managed obstacle."""
        for obs in self.obstacles_list:
            if obs.is_overlapping(coords):
                return True
        return False

    def get_boundaries(self) -> list[list[float]]:
        if not self.obstacles_list:
            raise ValueError("No obstacles loaded")
        all_v = [v for obs in self.obstacles_list for v in obs.vertices]
        xs, ys = zip(*all_v)
        return [
            [min(xs) - self.cfg.boundary_margin, max(xs) + self.cfg.boundary_margin],
            [min(ys) - self.cfg.boundary_margin, max(ys) + self.cfg.boundary_margin],
        ]

    def crop(self, xs: list[float], ys: list[float]) -> None:
        self.obstacles_list = [
            obs
            for obs in self.obstacles_list
            if all(
                xs[0] <= v[0] <= xs[1] and ys[0] <= v[1] <= ys[1] for v in obs.vertices
            )
        ]

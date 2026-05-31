"""Golden Petal UAV Simulation Scenario.

This module defines a UAV network simulation scenario where drone base stations
are deployed in a Vogel Spiral pattern and fly along mathematical Rose Curve paths.
Features a central cross obstacle configuration scaled using the Golden Ratio,
and concentric rings of ground users rotating in alternating directions.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from src.uavnetsim.geometry.coords import Coords3d
from src.uavnetsim.world.environment_geometry.obstacles import ObstacleCfg
from src.uavnetsim.world.plotting.plot_ctrl import PlotCfg
from src.uavnetsim.world.uav_ctrl import DroneStation, UavGraphController
from src.uavnetsim.world.user_model.base import UserModel
from src.uavnetsim.world.world_ctrl import WorldController
from src.uavnetsim.world.world_state import WorldState, WorldStateCfg

GOLDEN_ANGLE_DEG = 137.508
GOLDEN_PETAL_TRACE_COLORS = [
    (255, 160, 0),
    (255, 50, 0),
    (243, 203, 33),
    (0, 247, 255),
]
GOLDEN_BUILDING_COLOR = (224, 56, 10)
GOLDEN_PETAL_PATH_RESOLUTION = 96
GOLDEN_PETAL_TRACE_LENGTH = 400


def build_vogel_spiral_centers(
    num_uavs: int = 30,
    spacing_scale: float = 35.0,
    height: float = 50.0,
    origin_x: float = 350.0,
    origin_y: float = 350.0,
) -> list[Coords3d]:
    golden_angle_rad = np.deg2rad(GOLDEN_ANGLE_DEG)
    centers = []
    for idx in range(num_uavs):
        radius = spacing_scale * np.sqrt(idx)
        theta = idx * golden_angle_rad
        centers.append(
            Coords3d(
                origin_x + radius * np.cos(theta),
                origin_y + radius * np.sin(theta),
                height,
            )
        )
    return centers


def build_rose_curve_path(
    center: Coords3d,
    petal_length: float = 15.0,
    petal_k: int = 2,
    resolution: int = GOLDEN_PETAL_PATH_RESOLUTION,
) -> list[Coords3d]:
    if resolution < 4:
        raise ValueError(f"resolution must be at least 4, got {resolution}.")
    if petal_length <= 0:
        raise ValueError(f"petal_length must be positive, got {petal_length}.")
    if petal_k <= 0:
        raise ValueError(f"petal_k must be positive, got {petal_k}.")

    thetas = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    radius = petal_length * np.cos(petal_k * thetas)
    xs = center.x + radius * np.cos(thetas)
    ys = center.y + radius * np.sin(thetas)
    return [Coords3d(float(x), float(y), center.z) for x, y in zip(xs, ys)]


def build_golden_petal_paths(
    centers: list[Coords3d],
    petal_length: float = 15.0,
    petal_ks: tuple[int, ...] = (2, 3, 4, 5),
    resolution: int = GOLDEN_PETAL_PATH_RESOLUTION,
) -> list[list[Coords3d]]:
    return [
        build_rose_curve_path(
            center=center,
            petal_length=petal_length,
            petal_k=petal_ks[idx % len(petal_ks)],
            resolution=resolution,
        )
        for idx, center in enumerate(centers)
    ]


def build_rotated_square(
    center_x: float,
    center_y: float,
    side_length: float,
    rotation_rad: float,
) -> tuple[list[float], list[float]]:
    half_side = side_length / 2.0
    local_vertices = np.asarray(
        [
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ],
        dtype=float,
    )
    rotation = np.asarray(
        [
            [np.cos(rotation_rad), -np.sin(rotation_rad)],
            [np.sin(rotation_rad), np.cos(rotation_rad)],
        ],
        dtype=float,
    )
    vertices = local_vertices @ rotation.T
    vertices += np.asarray([center_x, center_y], dtype=float)
    return vertices[:, 0].tolist(), vertices[:, 1].tolist()


def _triangle_area(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    side_ab = point_b - point_a
    side_ac = point_c - point_a
    return float(abs(side_ab[0] * side_ac[1] - side_ab[1] * side_ac[0]) / 2.0)


def build_rotated_rectangle(
    center_x: float,
    center_y: float,
    length_x: float,
    length_y: float,
    rotation_rad: float = 0.0,
) -> tuple[list[float], list[float]]:
    half_x = length_x / 2.0
    half_y = length_y / 2.0
    local_vertices = np.asarray(
        [
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y],
        ],
        dtype=float,
    )
    rotation = np.asarray(
        [
            [np.cos(rotation_rad), -np.sin(rotation_rad)],
            [np.sin(rotation_rad), np.cos(rotation_rad)],
        ],
        dtype=float,
    )
    vertices = local_vertices @ rotation.T
    vertices += np.asarray([center_x, center_y], dtype=float)
    return vertices[:, 0].tolist(), vertices[:, 1].tolist()


def build_central_cross_obstacles(
    center: Coords3d,
    max_radius: float,
    base_length: float = 30.0,
    min_length: float = 10.0,
    rect_width: float = 8.0,
    obstacle_height: float = 35.0,
    gap: float = 6.0,
    center_square_side: float = 12.0,
) -> list[tuple[list[float], list[float], float]]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    obstacles_data_list: list[tuple[list[float], list[float], float]] = []

    # Central obstacle
    xs, ys = build_rotated_rectangle(
        center_x=center.x,
        center_y=center.y,
        length_x=center_square_side,
        length_y=center_square_side,
        rotation_rad=0.0,
    )
    obstacles_data_list.append((xs, ys, obstacle_height))

    # Build one symmetric golden-ratio sequence of lengths
    lengths: list[float] = []
    current_length = base_length
    accumulated_distance = center_square_side / 2.0 + gap

    while accumulated_distance + current_length / 2.0 <= max_radius:
        length = max(min_length, current_length)
        lengths.append(length)
        accumulated_distance += length + gap
        current_length /= phi

        if length <= min_length and accumulated_distance > max_radius:
            break

    # Place rectangles on each of the four arms
    running_offset = center_square_side / 2.0 + gap
    for length in lengths:
        center_offset = running_offset + length / 2.0

        # Left arm: horizontal rectangle
        xs, ys = build_rotated_rectangle(
            center_x=center.x - center_offset,
            center_y=center.y,
            length_x=length,
            length_y=rect_width,
            rotation_rad=0.0,
        )
        obstacles_data_list.append((xs, ys, obstacle_height))

        # Right arm: horizontal rectangle
        xs, ys = build_rotated_rectangle(
            center_x=center.x + center_offset,
            center_y=center.y,
            length_x=length,
            length_y=rect_width,
            rotation_rad=0.0,
        )
        obstacles_data_list.append((xs, ys, obstacle_height))

        # Bottom arm: vertical rectangle
        xs, ys = build_rotated_rectangle(
            center_x=center.x,
            center_y=center.y - center_offset,
            length_x=rect_width,
            length_y=length,
            rotation_rad=0.0,
        )
        obstacles_data_list.append((xs, ys, obstacle_height))

        # Top arm: vertical rectangle
        xs, ys = build_rotated_rectangle(
            center_x=center.x,
            center_y=center.y + center_offset,
            length_x=rect_width,
            length_y=length,
            rotation_rad=0.0,
        )
        obstacles_data_list.append((xs, ys, obstacle_height))

        running_offset += length + gap

    return obstacles_data_list


class RotatingUserModel(UserModel):
    def __init__(
        self,
        ue_positions: NDArray[np.float32],
        origin: tuple[float, float],
        base_angular_velocity: float = 0.1,
    ):
        super().__init__()
        self.origin = np.array(origin, dtype=np.float32)
        self.base_angular_velocity = base_angular_velocity

        diff = ue_positions[:, :2] - self.origin
        self.radii = np.linalg.norm(diff, axis=1)
        self.angles = np.arctan2(diff[:, 1], diff[:, 0])
        self.heights = ue_positions[:, 2]

        self._setup_velocities()
        self._update_coords()

    def _setup_velocities(self) -> None:
        # Use rounding to handle potential floating point precision issues with radii
        rounded_radii = np.round(self.radii, decimals=3)
        unique_radii = np.unique(rounded_radii)
        self.angular_velocities = np.zeros_like(self.radii, dtype=np.float32)
        for i, r in enumerate(unique_radii):
            # Alternate directions across successive disks (circles)
            # This ensures symmetry and alternating rotation as requested
            direction = 1 if i % 2 == 0 else -1
            self.angular_velocities[rounded_radii == r] = (
                direction * self.base_angular_velocity
            )

    def reset_users(self, ue_pos: NDArray[np.float32], *args, **kwargs) -> None:
        diff = ue_pos[:, :2] - self.origin
        self.radii = np.linalg.norm(diff, axis=1)
        self.angles = np.arctan2(diff[:, 1], diff[:, 0])
        self.heights = ue_pos[:, 2]
        self._setup_velocities()
        self._update_coords()

    def _update_coords(self):
        xs = self.origin[0] + self.radii * np.cos(self.angles)
        ys = self.origin[1] + self.radii * np.sin(self.angles)
        self._locations = [
            Coords3d(float(x), float(y), float(z))
            for x, y, z in zip(xs, ys, self.heights)
        ]

    def step(self, time_step: float) -> None:
        self.angles += self.angular_velocities * time_step
        self._update_coords()


def build_golden_petal_uav_world(plot_enabled: bool = True) -> WorldController:
    centers = build_vogel_spiral_centers()
    paths = build_golden_petal_paths(centers)
    stations = [
        DroneStation(drone_id=station_idx, coords=path[0])
        for station_idx, path in enumerate(paths)
    ]
    uav_ctrl = UavGraphController(
        stations=stations,
        paths=paths,
        speeds=12.0,
        cyclic=True,
    )

    # Calculate max radius to surround all UAV paths
    max_radius = 0.0
    origin_x, origin_y = 350.0, 350.0
    for path in paths:
        for p in path:
            dist = np.sqrt((p.x - origin_x) ** 2 + (p.y - origin_y) ** 2)
            if dist > max_radius:
                max_radius = dist

    # Create 5 circles of users
    ue_positions = []
    # Using 5 circles, from small to max_radius
    radii = np.linspace(max_radius / 10.0, max_radius, 5)
    for r in radii:
        # Roughly 1.5m between users for a dense look
        n_users_circle = max(8, int(2 * np.pi * r / 12))
        thetas = np.linspace(0, 2 * np.pi, n_users_circle, endpoint=False)
        for t in thetas:
            ue_positions.append(
                [origin_x + r * np.cos(t), origin_y + r * np.sin(t), 0.0]
            )

    ue_pos_array = np.array(ue_positions, dtype=np.float32)

    user_model = RotatingUserModel(ue_pos_array, origin=(origin_x, origin_y))

    state = WorldState(
        ue_pos=ue_pos_array,
        uav_pos=uav_ctrl.get_locations_array(),
        bs_pos=np.empty((0, 3), dtype=np.float32),
        obstacles=[],
    )
    world_cfg = WorldStateCfg(
        n_ues=len(ue_positions),
        n_uavs=len(centers),
        n_bss=0,
        env_boundary=(700.0, 700.0),
        user_model="rotating",  # Custom name
    )

    world_controller = WorldController(
        world_cfg=world_cfg,
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(
            enabled=plot_enabled,
            uav_trace_enabled=True,
            uav_trace_max_length=GOLDEN_PETAL_TRACE_LENGTH,
        ),
        uav_ctrl=uav_ctrl,
        user_model=user_model,
        state=state,
    )
    obstacles_data_list = build_central_cross_obstacles(
        center=Coords3d(350.0, 350.0, 50.0),
        max_radius=max_radius,
        base_length=55.0,
        min_length=11.0,
        rect_width=11.0,
        obstacle_height=35.0,
        gap=11,
        center_square_side=12.0,
    )

    world_controller.obstacle_controller._setup_obstacles(obstacles_data_list)
    world_controller.state.obstacles = (
        world_controller.obstacle_controller.obstacles_list
    )
    if world_controller.plt_controller is not None:
        world_controller.plt_controller.set_uav_trace_colors(GOLDEN_PETAL_TRACE_COLORS)
        world_controller.plt_controller.set_obstacle_color(GOLDEN_BUILDING_COLOR)
        world_controller.plt_controller.update_obstacles(
            world_controller.state.obstacles
        )
    return world_controller


def main() -> None:
    controller = build_golden_petal_uav_world()
    time_step = 0.05
    sleep_time = 0.02

    try:
        while True:
            controller.simulate_time_step(time_step=time_step)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

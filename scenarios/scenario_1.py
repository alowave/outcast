"""Manhattan Grid Street and Square Flight UAV Simulation.

This module defines a structured urban network simulation scenario featuring
a grid-based environment layout. UAV base stations are arranged in a regular
rectangular matrix and execute synchronized square patrol trajectories.

Ground users are constrained to a Manhattan-style grid network mimicking city
streets (horizontal and vertical thoroughfares), navigating around rectangular
grid-cell block obstacles placed symmetrically between the traffic lanes.
"""

from __future__ import annotations

import time

import numpy as np

from src.outcast.geometry.coords import Coords3d
from src.outcast.world.environment_geometry.obstacles import ObstacleCfg
from src.outcast.world.plotting.plot_ctrl import PlotCfg
from src.outcast.world.uav_ctrl import SquareGraphUavController
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldState, WorldStateCfg

FIERY_UAV_TRACE_COLORS = [
    (255, 160, 0),
    (255, 50, 0),
    (243, 203, 33),
    (0, 247, 255),
]
BUILDING_COLOR = (224, 56, 10)


def build_square_grid_centers(
    rows: int = 3,
    cols: int = 6,
    spacing: float = 100.0,
    height: float = 50.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[Coords3d]:
    return [
        Coords3d(origin_x + x_idx * spacing, origin_y + y_idx * spacing, height)
        for x_idx in range(1, cols + 1)
        for y_idx in range(1, rows + 1)
    ]


def build_grid_cell_obstacles() -> list[tuple[list[float], list[float], float]]:
    obstacles_data_list = []
    for obstacle_x in [150.0, 250.0, 350.0, 450.0, 550.0]:
        for obstacle_y in [150.0, 250.0]:
            xs = [
                obstacle_x - 10.0,
                obstacle_x + 10.0,
                obstacle_x + 10.0,
                obstacle_x - 10.0,
            ]
            ys = [
                obstacle_y - 10.0,
                obstacle_y - 10.0,
                obstacle_y + 10.0,
                obstacle_y + 10.0,
            ]
            obstacles_data_list.append((xs, ys, 40.0))
    return obstacles_data_list


def build_street_user_positions(
    n_users: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if n_users < 0:
        raise ValueError(f"n_users must be non-negative, got {n_users}.")

    rng = rng or np.random.default_rng(33)
    horizontal_count = n_users // 2
    vertical_count = n_users - horizontal_count
    ue_positions = []

    for _ in range(horizontal_count):
        y = rng.choice([100.0, 200.0, 300.0])
        x = rng.uniform(100.0, 600.0)
        ue_positions.append([x, y, 1.5])

    for _ in range(vertical_count):
        x = rng.choice([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        y = rng.uniform(100.0, 300.0)
        ue_positions.append([x, y, 1.5])

    return np.asarray(ue_positions, dtype=np.float32)


def build_square_uav_world(
    plot_enabled: bool = True, seed: int = 33
) -> WorldController:
    centers = build_square_grid_centers()
    uav_ctrl = SquareGraphUavController(
        centers=centers,
        side_length=20.0,
        speeds=10.0,
        cyclic=True,
    )

    state = WorldState(
        ue_pos=build_street_user_positions(rng=np.random.default_rng(seed)),
        uav_pos=uav_ctrl.get_locations_array(),
        bs_pos=np.empty((0, 3), dtype=np.float32),
        obstacles=[],
    )
    world_cfg = WorldStateCfg(
        n_ues=200,
        n_uavs=len(centers),
        n_bss=0,
        env_boundary=(700.0, 400.0),
        user_model="random_movement",
    )

    world_controller = WorldController(
        seed=seed,
        world_cfg=world_cfg,
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(
            enabled=plot_enabled,
            uav_trace_enabled=True,
            uav_trace_max_length=40,
        ),
        uav_ctrl=uav_ctrl,
        state=state,
    )
    world_controller.obstacle_controller._setup_obstacles(build_grid_cell_obstacles())
    world_controller.state.obstacles = (
        world_controller.obstacle_controller.obstacles_list
    )
    if world_controller.plt_controller is not None:
        world_controller.plt_controller.set_uav_trace_colors(FIERY_UAV_TRACE_COLORS)
        world_controller.plt_controller.set_obstacle_color(BUILDING_COLOR)
        world_controller.plt_controller.update_obstacles(
            world_controller.state.obstacles
        )
    return world_controller


def main() -> None:
    controller = build_square_uav_world()
    time_step = 0.1
    sleep_time = 0.03

    try:
        while True:
            controller.simulate_time_step(time_step=time_step)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

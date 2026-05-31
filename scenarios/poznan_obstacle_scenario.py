"""Poznan City Map UAV Patrol Simulation.

This module sets up a multi-UAV network simulation integrated with real-world
city layouts (Poznan). Drones fly a coordinated circular patrol orbit at
a fixed altitude around the city center, while ground users move dynamically
using an obstacle-aware mobility model that respects building boundaries.
"""

import time

import numpy as np

from src.uavnetsim.geometry.coords import Coords3d
from src.uavnetsim.world.environment_geometry.obstacles import (
    ObstacleCfg,
)
from src.uavnetsim.world.plotting.plot_ctrl import PlotCfg
from src.uavnetsim.world.uav_ctrl import DroneStation, UavGraphController
from src.uavnetsim.world.user_model.obstacle_mobility_model import (
    ObstacleMobilityCfg,
    ObstacleMobilityUserModel,
)
from src.uavnetsim.world.world_ctrl import WorldController
from src.uavnetsim.world.world_state import WorldStateCfg


def build_poznan_scenario(plot_enabled: bool = True) -> WorldController:
    n_uavs = 5
    n_ues = 100
    uav_height = 100.0

    rng = np.random.default_rng(seed=42)
    m_cfg = ObstacleMobilityCfg()

    obstacle_cfg = ObstacleCfg(enabled=True, map_title="Poznan")

    world_cfg = WorldStateCfg(
        n_ues=n_ues,
        n_uavs=n_uavs,
        n_bss=0,
        env_boundary=(1000, 1000),
        user_model="obstacle_mobility",
    )

    world_controller = WorldController(
        world_cfg=world_cfg,
        obstacle_cfg=obstacle_cfg,
        plot_cfg=PlotCfg(
            enabled=plot_enabled, uav_trace_enabled=True, uav_trace_max_length=100
        ),
    )

    obstacles = world_controller.state.obstacles
    real_bounds = world_controller.obstacle_controller.get_boundaries()

    min_x, max_x = real_bounds[0]
    min_y, max_y = real_bounds[1]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    world_cfg.env_boundary = (max_x + 100, max_y + 100)

    uav_paths = []
    t_vals = np.linspace(0, 2 * np.pi, 120)
    patrol_radius = 700.0

    for i in range(n_uavs):
        phase_shift = (2 * np.pi * i) / n_uavs
        path = [
            Coords3d(
                center_x + patrol_radius * np.cos(t + phase_shift),
                center_y + patrol_radius * np.sin(t + phase_shift),
                uav_height,
            )
            for t in t_vals
        ]
        uav_paths.append(path)

    stations = [DroneStation(drone_id=i, coords=uav_paths[i][0]) for i in range(n_uavs)]
    uav_ctrl = UavGraphController(
        stations=stations, paths=uav_paths, speeds=20.0, cyclic=True
    )

    user_model = ObstacleMobilityUserModel(
        world_cfg=world_cfg,
        obstacles=obstacles,
        obstacle_cfg=obstacle_cfg,
        mobility_cfg=m_cfg,
        rng=rng,  # Now resolved
    )
    user_model.duration = 10000
    user_model.speed_range = (15.0, 25.0)
    user_model.pause_range = (0.0, 0.5)

    initial_ue_pos = np.zeros((n_ues, 3), dtype=np.float32)
    user_model.reset_users(initial_ue_pos)

    world_controller.uav_ctrl = uav_ctrl
    world_controller.user_model = user_model

    world_controller.state.uav_pos = uav_ctrl.get_locations_array()
    world_controller.state.ue_pos = user_model.get_locations_array()
    world_controller.state.bs_pos = np.empty((0, 3), dtype=np.float32)

    if world_controller.plt_controller:
        world_controller.plt_controller.set_obstacle_color((224, 56, 10))
        world_controller.plt_controller.update_obstacles(obstacles)
        world_controller.plt_controller.ue_items.setSymbolSize(7)

    return world_controller


def main() -> None:
    controller = build_poznan_scenario(plot_enabled=True)
    time_step = 0.1
    sleep_time = 0.02

    try:
        while True:
            controller.simulate_time_step(time_step=time_step)

            if controller.plt_controller:
                controller.plt_controller.process_events()

            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")


if __name__ == "__main__":
    main()

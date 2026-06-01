"""Synchronized Golden Petal Drone Cluster Simulation.

This module defines a specialized UAV network scenario focused on coordinated,
localized cluster flight. A fleet of autonomous drones operates around a single
central anchor point, tracking intricate, interlocking "golden petal" geometry
paths at a fixed altitude.

Unlike macro urban scenarios, this configuration utilizes optimized golden
start-offsets to maintain precise separation, running in an obstacle-free
environment with zero active ground user nodes to evaluate pure flight
controller tracking performance.
"""

from __future__ import annotations

import time

import numpy as np

from src.outcast.geometry.coords import Coords3d
from src.outcast.world.environment_geometry.obstacles import ObstacleCfg
from src.outcast.world.plotting.plot_ctrl import PlotCfg
from src.outcast.world.uav_ctrl import GoldenPetalGraphUavController
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldState, WorldStateCfg

GOLDEN_PETAL_TRACE_COLORS = [
    (255, 160, 0),
    (255, 50, 0),
    (243, 203, 33),
    (0, 247, 255),
]
GOLDEN_PETAL_CENTER = Coords3d(300.0, 300.0, 50.0)
GOLDEN_PETAL_NUM_UAVS = 10
GOLDEN_PETAL_OUTER_RADIUS = 120.0
GOLDEN_PETAL_HALF_ANGLE_DEG = 18.0
GOLDEN_PETAL_SAMPLES_PER_SIDE = 48
GOLDEN_PETAL_TRACE_LENGTH = 700


def build_golden_petal_uav_world(plot_enabled: bool = True) -> WorldController:
    uav_ctrl = GoldenPetalGraphUavController(
        center=GOLDEN_PETAL_CENTER,
        n_uavs=GOLDEN_PETAL_NUM_UAVS,
        outer_radius=GOLDEN_PETAL_OUTER_RADIUS,
        petal_half_angle_deg=GOLDEN_PETAL_HALF_ANGLE_DEG,
        speed=14.0,
        samples_per_side=GOLDEN_PETAL_SAMPLES_PER_SIDE,
        cyclic=True,
        use_golden_start_offsets=True,
    )

    state = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=uav_ctrl.get_locations_array(),
        bs_pos=np.empty((0, 3), dtype=np.float32),
        obstacles=[],
    )
    world_cfg = WorldStateCfg(
        n_ues=0,
        n_uavs=GOLDEN_PETAL_NUM_UAVS,
        n_bss=0,
        env_boundary=(600.0, 600.0),
        user_model=None,
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
        state=state,
    )
    if world_controller.plt_controller is not None:
        world_controller.plt_controller.set_uav_trace_colors(GOLDEN_PETAL_TRACE_COLORS)
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

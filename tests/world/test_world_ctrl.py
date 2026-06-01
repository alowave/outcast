import os
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.outcast.geometry.coords import Coords3d
from src.outcast.world.environment_geometry.obstacles import ObstacleCfg
from src.outcast.world.plotting.plot_ctrl import PlotCfg
from src.outcast.world.uav_ctrl import (
    DroneStation,
    StaticUavController,
    UavGraphController,
)
from src.outcast.world.user_model.random_movement import (
    RandomMovementCfg,
    RandomMovementUserModel,
)
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldState, WorldStateCfg


def test_world_controller_uses_obstacle_boundaries_for_world_env_boundary():
    world_cfg = WorldStateCfg(env_boundary=(100.0, 200.0))

    with (
        patch(
            "src.outcast.world.world_ctrl.ObstacleController.load_obstacles"
        ) as load_obstacles,
        patch(
            "src.outcast.world.world_ctrl.ObstacleController.get_boundaries",
            return_value=[[5.0, 400.0], [7.0, 900.0]],
        ),
    ):
        controller = WorldController(
            world_cfg=world_cfg,
            obstacle_cfg=ObstacleCfg(enabled=True),
            plot_cfg=PlotCfg(enabled=False),
        )

    load_obstacles.assert_called_once()
    assert controller.world_cfg.env_boundary == (400.0, 900.0)


def test_world_controller_initializes_plot_controller_when_enabled():
    state = WorldState(
        ue_pos=np.array([[1.0, 2.0, 1.5]], dtype=np.float32),
        uav_pos=np.array([[10.0, 20.0, 80.0]], dtype=np.float32),
        bs_pos=np.array([[100.0, 200.0, 25.0]], dtype=np.float32),
        obstacles=[],
    )

    controller = WorldController(
        world_cfg=WorldStateCfg(),
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(enabled=True),
        state=state,
    )

    assert controller.plt_controller is not None

    ue_x, ue_y = controller.plt_controller.ue_items.getData()
    assert list(ue_x) == [1.0]
    assert list(ue_y) == [2.0]


def test_world_controller_simulate_time_step_refreshes_plot_positions():
    controller = WorldController(
        world_cfg=WorldStateCfg(n_ues=0, n_uavs=0, n_bss=0, user_model=None),
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(enabled=True),
        state=WorldState(
            ue_pos=np.empty((0, 3), dtype=np.float32),
            uav_pos=np.empty((0, 3), dtype=np.float32),
            bs_pos=np.empty((0, 3), dtype=np.float32),
            obstacles=[],
        ),
    )

    controller.state.ue_pos = np.array([[3.0, 4.0, 1.5]], dtype=np.float32)
    controller.state.uav_pos = np.array([[30.0, 40.0, 70.0]], dtype=np.float32)
    controller.state.bs_pos = np.array([[300.0, 400.0, 25.0]], dtype=np.float32)

    controller.simulate_time_step()

    ue_x, ue_y = controller.plt_controller.ue_items.getData()
    uav_x, uav_y = controller.plt_controller.uav_items.getData()
    bs_x, bs_y = controller.plt_controller.bs_items.getData()

    assert list(ue_x) == [3.0]
    assert list(ue_y) == [4.0]
    assert list(uav_x) == [30.0]
    assert list(uav_y) == [40.0]
    assert list(bs_x) == [300.0]
    assert list(bs_y) == [400.0]


def test_world_controller_loads_random_movement_user_model_and_updates_ues():
    world_cfg = WorldStateCfg(
        n_ues=1,
        n_uavs=0,
        n_bss=0,
        env_boundary=(10.0, 10.0),
        user_model="random_movement",
    )

    controller = WorldController(
        world_cfg=world_cfg,
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(enabled=False),
        user_model=RandomMovementUserModel(
            world_cfg,
            RandomMovementCfg(step_range=(0.25, 0.25)),
            np.random.default_rng(33),
        ),
        state=WorldState(
            ue_pos=np.array([[1.0, 2.0, 1.5]], dtype=np.float32),
            uav_pos=np.empty((0, 3), dtype=np.float32),
            bs_pos=np.empty((0, 3), dtype=np.float32),
            obstacles=[],
        ),
    )
    ue_pos_before = controller.state.ue_pos

    controller.simulate_time_step()

    assert controller.state.ue_pos is ue_pos_before
    np.testing.assert_allclose(
        controller.state.ue_pos,
        np.array([[1.25, 2.25, 1.5]], dtype=np.float32),
    )


def test_world_controller_default_uav_ctrl_does_not_move_uavs():
    uav_pos = np.array([[10.0, 20.0, 80.0]], dtype=np.float32)
    controller = WorldController(
        world_cfg=WorldStateCfg(n_ues=0, n_uavs=1, n_bss=0, user_model=None),
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(enabled=False),
        state=WorldState(
            ue_pos=np.empty((0, 3), dtype=np.float32),
            uav_pos=uav_pos.copy(),
            bs_pos=np.empty((0, 3), dtype=np.float32),
            obstacles=[],
        ),
    )

    assert isinstance(controller.uav_ctrl, StaticUavController)
    controller.simulate_time_step(time_step=5.0)

    np.testing.assert_allclose(controller.state.uav_pos, uav_pos)


def test_world_controller_uses_custom_uav_ctrl_to_update_uavs():
    uav_pos = np.array([[0.0, 0.0, 80.0]], dtype=np.float32)
    start = Coords3d(0.0, 0.0, 80.0)
    end = Coords3d(10.0, 0.0, 80.0)
    station = DroneStation(coords=start)
    uav_ctrl = UavGraphController(
        stations=[station],
        paths=[[start, end]],
        speeds=5.0,
    )
    controller = WorldController(
        world_cfg=WorldStateCfg(n_ues=0, n_uavs=1, n_bss=0, user_model=None),
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(enabled=False),
        uav_ctrl=uav_ctrl,
        state=WorldState(
            ue_pos=np.empty((0, 3), dtype=np.float32),
            uav_pos=uav_pos.copy(),
            bs_pos=np.empty((0, 3), dtype=np.float32),
            obstacles=[],
        ),
    )

    controller.simulate_time_step(time_step=1.0)

    np.testing.assert_allclose(
        controller.state.uav_pos,
        np.array([[5.0, 0.0, 80.0]], dtype=np.float32),
    )

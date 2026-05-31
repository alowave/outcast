import os

import numpy as np
from pyqtgraph.Qt import QtGui

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.uavnetsim.geometry.obstacle import Obstacle
from src.uavnetsim.world.plotting.plot_ctrl import (
    PlotCfg,
    PlotController,
    make_drone_symbol,
)
from src.uavnetsim.world.world_state import WorldState


def test_make_drone_symbol_builds_qpainter_path():
    symbol = make_drone_symbol()

    assert isinstance(symbol, QtGui.QPainterPath)
    assert not symbol.isEmpty()


def test_plot_ctrl_builds_entity_items_and_obstacles_from_world_state():
    state = WorldState(
        ue_pos=np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 1.5]], dtype=np.float32),
        uav_pos=np.array([[10.0, 20.0, 80.0]], dtype=np.float32),
        bs_pos=np.array([[100.0, 200.0, 25.0]], dtype=np.float32),
        obstacles=[
            Obstacle(
                obstacle_id=7,
                height=42.0,
                vertices=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
            )
        ],
    )

    plot_ctrl = PlotController(state, show=False)

    ue_x, ue_y = plot_ctrl.ue_items.getData()
    uav_x, uav_y = plot_ctrl.uav_items.getData()
    bs_x, bs_y = plot_ctrl.bs_items.getData()

    assert isinstance(plot_ctrl.uav_items.opts["symbol"], QtGui.QPainterPath)
    assert plot_ctrl.uav_items.opts["symbolSize"] == 12
    assert plot_ctrl.uav_items.opts["symbolPen"].color().getRgb() == (0, 0, 0, 255)
    assert list(ue_x) == [1.0, 3.0]
    assert list(ue_y) == [2.0, 4.0]
    assert list(uav_x) == [10.0]
    assert list(uav_y) == [20.0]
    assert list(bs_x) == [100.0]
    assert list(bs_y) == [200.0]

    assert len(plot_ctrl.obstacles) == 1

    obstacle_x, obstacle_y = plot_ctrl.obstacles[0].getData()
    assert list(obstacle_x) == [0.0, 5.0, 5.0, 0.0, 0.0]
    assert list(obstacle_y) == [0.0, 0.0, 5.0, 5.0, 0.0]


def test_plot_ctrl_clear_removes_all_cached_items():
    state = WorldState(
        ue_pos=np.array([[1.0, 2.0, 1.5]], dtype=np.float32),
        uav_pos=np.array([[10.0, 20.0, 80.0]], dtype=np.float32),
        bs_pos=np.array([[100.0, 200.0, 25.0]], dtype=np.float32),
        obstacles=[],
    )

    plot_ctrl = PlotController(state, show=False)
    plot_ctrl.update_ues(np.empty((0, 3), dtype=np.float32))
    plot_ctrl.update_uavs(np.empty((0, 3), dtype=np.float32))
    plot_ctrl.update_bs(np.empty((0, 3), dtype=np.float32))
    plot_ctrl.update_obstacles([])

    ue_x, ue_y = plot_ctrl.ue_items.getData()
    uav_x, uav_y = plot_ctrl.uav_items.getData()
    bs_x, bs_y = plot_ctrl.bs_items.getData()

    assert ue_x is None
    assert ue_y is None
    assert uav_x is None
    assert uav_y is None
    assert bs_x is None
    assert bs_y is None
    assert plot_ctrl.obstacles == []


def test_plot_ctrl_uav_trace_keeps_bounded_history():
    state = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=np.array([[0.0, 0.0, 80.0]], dtype=np.float32),
        bs_pos=np.empty((0, 3), dtype=np.float32),
        obstacles=[],
    )
    plot_ctrl = PlotController(
        state,
        cfg=PlotCfg(uav_trace_enabled=True, uav_trace_max_length=2),
        show=False,
    )

    plot_ctrl.update_uavs(np.array([[1.0, 1.0, 80.0]], dtype=np.float32))
    plot_ctrl.update_uavs(np.array([[2.0, 2.0, 80.0]], dtype=np.float32))

    assert plot_ctrl.uav_history is not None
    assert plot_ctrl.uav_history.shape == (2, 1, 2)
    assert plot_ctrl.uav_history_size == 2
    assert len(plot_ctrl.uav_trace_items) == 1

    trace_x, trace_y = plot_ctrl.uav_trace_items[0].getData()
    assert list(trace_x) == [1.0, 2.0]
    assert list(trace_y) == [1.0, 2.0]


def test_plot_ctrl_uav_trace_rebuilds_when_uav_count_changes():
    plot_ctrl = PlotController(
        cfg=PlotCfg(uav_trace_enabled=True),
        show=False,
    )

    plot_ctrl.update_uavs(
        np.array(
            [
                [0.0, 0.0, 80.0],
                [10.0, 10.0, 80.0],
            ],
            dtype=np.float32,
        )
    )
    assert len(plot_ctrl.uav_trace_items) == 2

    plot_ctrl.update_uavs(np.array([[20.0, 20.0, 80.0]], dtype=np.float32))
    assert plot_ctrl.uav_history is not None
    assert plot_ctrl.uav_history.shape == (50, 1, 2)
    assert plot_ctrl.uav_history_size == 1
    assert len(plot_ctrl.uav_trace_items) == 1

    plot_ctrl.update_uavs(np.array([[21.0, 21.0, 80.0]], dtype=np.float32))

    trace_x, trace_y = plot_ctrl.uav_trace_items[0].getData()
    assert list(trace_x) == [20.0, 21.0]
    assert list(trace_y) == [20.0, 21.0]


def test_plot_ctrl_uav_trace_uses_runtime_per_uav_colors():
    plot_ctrl = PlotController(
        cfg=PlotCfg(
            uav_trace_enabled=True,
            uav_trace_max_length=3,
        ),
        show=False,
    )
    plot_ctrl.set_uav_trace_colors([[255, 98, 0], [30, 144, 255, 160]])

    plot_ctrl.update_uavs(
        np.array(
            [
                [0.0, 0.0, 80.0],
                [10.0, 10.0, 80.0],
            ],
            dtype=np.float32,
        )
    )
    plot_ctrl.update_uavs(
        np.array(
            [
                [1.0, 0.0, 80.0],
                [11.0, 10.0, 80.0],
            ],
            dtype=np.float32,
        )
    )

    first_pen = plot_ctrl.uav_trace_items[0].opts["pen"].color().getRgb()
    second_pen = plot_ctrl.uav_trace_items[1].opts["pen"].color().getRgb()

    assert first_pen == (255, 98, 0, 180)
    assert second_pen == (30, 144, 255, 160)


def test_plot_ctrl_uses_runtime_obstacle_color():
    obstacle = Obstacle(
        obstacle_id=7,
        height=42.0,
        vertices=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
    )
    plot_ctrl = PlotController(show=False)
    plot_ctrl.set_obstacle_color((224, 56, 10))

    plot_ctrl.update_obstacles([obstacle])

    obstacle_pen = plot_ctrl.obstacles[0].opts["pen"].color().getRgb()
    assert obstacle_pen == (224, 56, 10, 255)

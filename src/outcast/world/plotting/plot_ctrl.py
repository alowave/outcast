"""Real-Time 2D PyQtGraph Network Visualization Engine.

Manages rendering elements for simulation assets, including ground user equipment
(UEs), base stations (BSs), custom quadcopter symbols with active tail trace histories,
and closed polygon obstacle footprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets

from src.uavnetsim.geometry.obstacle import Obstacle
from src.uavnetsim.world.world_state import WorldState


def make_drone_symbol() -> QtGui.QPainterPath:
    """Generates a custom multi-rotor vector shape for PyQtGraph scatter symbols."""
    path = QtGui.QPainterPath()
    petal_radius = 0.33
    petal_offset = 0.33

    path.addEllipse(
        -petal_radius,
        -petal_offset - petal_radius,
        2 * petal_radius,
        2 * petal_radius,
    )
    path.addEllipse(
        -petal_radius,
        petal_offset - petal_radius,
        2 * petal_radius,
        2 * petal_radius,
    )
    path.addEllipse(
        -petal_offset - petal_radius,
        -petal_radius,
        2 * petal_radius,
        2 * petal_radius,
    )
    path.addEllipse(
        petal_offset - petal_radius,
        -petal_radius,
        2 * petal_radius,
        2 * petal_radius,
    )
    path.addEllipse(-0.10, -0.10, 0.20, 0.20)
    return path


@dataclass(slots=True)
class PlotCfg:
    enabled: bool = True
    uav_trace_enabled: bool = False
    uav_trace_max_length: int = 50


class PlotController:
    """
    Thin pyqtgraph controller for world-entity visualization.

    Attributes are exposed explicitly so callers can access the individual plot
    items for UEs, BSs, UAVs, and obstacle outlines.
    """

    def __init__(
        self,
        state: WorldState | None = None,
        cfg: PlotCfg | None = None,
        show: bool = False,
    ) -> None:
        self.cfg = cfg or PlotCfg()
        self.uav_trace_colors: list[tuple[int, int, int, int]] = [(60, 200, 120, 80)]
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=show)
        self.plot = self.win.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True)

        self.ue_items = self.plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=4,
            symbolBrush=(60, 140, 255),
        )
        self.bs_items = self.plot.plot(
            [],
            [],
            pen=None,
            symbol="x",
            symbolSize=10,
            symbolPen=pg.mkPen(color=(240, 90, 40), width=2),
        )
        drone_symbol = make_drone_symbol()
        self.uav_items = self.plot.plot(
            [],
            [],
            pen=None,
            symbol=drone_symbol,
            symbolSize=12,
            symbolBrush=(60, 200, 120),
            symbolPen=pg.mkPen("k", width=1),
        )
        self.obstacles: list[Any] = []
        self.obstacle_color: tuple[int, int, int] = (110, 110, 110)
        self.uav_history: np.ndarray | None = None
        self.uav_history_size = 0
        self.uav_trace_items: list[Any] = []

        if state is not None:
            self.refresh(state)

    def refresh(self, state: WorldState) -> None:
        """Update every plot layer from a world state snapshot."""
        self.update_ues(state.ue_pos)
        self.update_bs(state.bs_pos)
        self.update_uavs(state.uav_pos)
        self.update_obstacles(state.obstacles)

    def update_ues(self, coords: np.ndarray) -> None:
        self._set_scatter_data(self.ue_items, coords)

    def update_bs(self, coords: np.ndarray) -> None:
        self._set_scatter_data(self.bs_items, coords)

    def update_uavs(self, coords: np.ndarray) -> None:
        """Updates drone locations and pushes historical positions to trajectory trace paths."""
        self._set_scatter_data(self.uav_items, coords)
        if not self.cfg.uav_trace_enabled:
            return

        if coords.size == 0:
            self._clear_uav_traces()
            return

        coords_xy = np.asarray(coords[:, :2], dtype=np.float32)
        self._append_uav_history(coords_xy)
        self._ensure_uav_trace_items(coords_xy.shape[0])
        self._update_uav_traces()

    def update_obstacles(self, obstacles: list[Obstacle] | None) -> None:
        """Reconstructs closed geometric border lines for physical map obstacles."""
        for item in self.obstacles:
            self.plot.removeItem(item)

        self.obstacles = []
        if not obstacles:
            return

        for obstacle in obstacles:
            vertices = np.asarray(obstacle.vertices, dtype=float)
            if vertices.size == 0:
                continue

            closed_vertices = np.vstack([vertices, vertices[0]])
            item = self.plot.plot(
                closed_vertices[:, 0],
                closed_vertices[:, 1],
                pen=pg.mkPen(color=self.obstacle_color, width=2),
            )
            self.obstacles.append(item)

    def process_events(self) -> None:
        self.app.processEvents()

    def set_uav_trace_colors(
        self, colors: list[list[int]] | list[tuple[int, ...]]
    ) -> None:
        if not colors:
            raise ValueError("uav trace colors must not be empty.")

        trace_colors = []
        for color in colors:
            if len(color) not in {3, 4}:
                raise ValueError(
                    f"uav trace colors must be RGB or RGBA values, got {color}."
                )
            alpha = int(color[3]) if len(color) == 4 else 180
            trace_colors.append((int(color[0]), int(color[1]), int(color[2]), alpha))

        self.uav_trace_colors = trace_colors
        for uav_idx, item in enumerate(self.uav_trace_items):
            item.setPen(pg.mkPen(color=self._uav_trace_color(uav_idx), width=2))

    def set_obstacle_color(self, color: tuple[int, int, int] | list[int]) -> None:
        if len(color) != 3:
            raise ValueError(f"obstacle color must be an RGB value, got {color}.")
        self.obstacle_color = (int(color[0]), int(color[1]), int(color[2]))

    def _ensure_uav_trace_items(self, num_uavs: int) -> None:
        if len(self.uav_trace_items) == num_uavs:
            return

        for item in self.uav_trace_items:
            self.plot.removeItem(item)

        self.uav_trace_items = [
            self.plot.plot(
                [],
                [],
                pen=pg.mkPen(color=self._uav_trace_color(uav_idx), width=2),
            )
            for uav_idx in range(num_uavs)
        ]

    def _append_uav_history(self, coords_xy: np.ndarray) -> None:
        history_shape = (
            self.cfg.uav_trace_max_length,
            coords_xy.shape[0],
            coords_xy.shape[1],
        )
        if self.uav_history is None or self.uav_history.shape != history_shape:
            self.uav_history = np.empty(history_shape, dtype=np.float32)
            self.uav_history_size = 0

        if self.uav_history_size < self.cfg.uav_trace_max_length:
            self.uav_history[self.uav_history_size] = coords_xy
            self.uav_history_size += 1
            return

        self.uav_history[:-1] = self.uav_history[1:]
        self.uav_history[-1] = coords_xy

    def _update_uav_traces(self) -> None:
        if self.uav_history is None:
            return

        history = self.uav_history[: self.uav_history_size]
        for uav_idx, item in enumerate(self.uav_trace_items):
            path = history[:, uav_idx, :]
            item.setData(x=path[:, 0], y=path[:, 1])

    def _clear_uav_traces(self) -> None:
        self.uav_history = None
        self.uav_history_size = 0
        for item in self.uav_trace_items:
            item.setData(x=np.empty(0, dtype=float), y=np.empty(0, dtype=float))

    def _uav_trace_color(self, uav_idx: int) -> tuple[int, int, int, int]:
        return self.uav_trace_colors[uav_idx % len(self.uav_trace_colors)]

    @staticmethod
    def _set_scatter_data(item: Any, coords: np.ndarray) -> None:
        if coords.size == 0:
            item.setData(x=np.empty(0, dtype=float), y=np.empty(0, dtype=float))
            return

        item.setData(x=coords[:, 0], y=coords[:, 1])

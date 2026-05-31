import numpy as np
from shapely.geometry import Point as ShapelyPoint

from src.uavnetsim.geometry.visibility_ctrl import VisibilityGraphCtrl
from src.uavnetsim.world.environment_geometry.obstacles import (
    ObstacleCfg,
    ObstacleController,
)


def _sample_positions_outside_obstacles(
    count: int,
    z_range: tuple[float, float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    obstacles,
    rng: np.random.Generator,
) -> np.ndarray:
    positions: list[list[float]] = []
    attempts = 0
    max_attempts = 10_000

    while len(positions) < count:
        if attempts >= max_attempts:
            raise RuntimeError(
                "Failed to sample positions outside obstacle footprints."
            )

        attempts += 1
        x = float(rng.uniform(*x_bounds))
        y = float(rng.uniform(*y_bounds))
        footprint_point = ShapelyPoint(x, y)

        if any(obstacle._polygon.covers(footprint_point) for obstacle in obstacles):
            continue

        z = float(rng.uniform(*z_range))
        positions.append([x, y, z])

    return np.asarray(positions, dtype=np.float32)


def test_visibility_graph_ctrl_builds_graph_for_random_poznan_uavs_and_ues():
    obstacle_controller = ObstacleController(ObstacleCfg(map_title="Madrid_square"))
    obstacle_controller.load_obstacles()

    assert obstacle_controller.obstacles_list

    x_limits, y_limits = obstacle_controller.get_boundaries()
    rng = np.random.default_rng(33)

    uav_pos = _sample_positions_outside_obstacles(
        count=4,
        z_range=(50.0, 100.0),
        x_bounds=(x_limits[0], x_limits[1]),
        y_bounds=(y_limits[0], y_limits[1]),
        obstacles=obstacle_controller.obstacles_list,
        rng=rng,
    )
    ue_pos = _sample_positions_outside_obstacles(
        count=4,
        z_range=(1.5, 1.5),
        x_bounds=(x_limits[0], x_limits[1]),
        y_bounds=(y_limits[0], y_limits[1]),
        obstacles=obstacle_controller.obstacles_list,
        rng=rng,
    )

    endpoints = [
        *uav_pos[:, :2].tolist(),
        *ue_pos[:, :2].tolist(),
    ]
    controller = VisibilityGraphCtrl(
        obstacles=obstacle_controller.obstacles_list,
        endpoints=endpoints,
    )

    vis_graph = controller.build_graph()
    edges = vis_graph.get_los_edge_coordinates()
    endpoint_keys = {tuple(endpoint) for endpoint in endpoints}

    assert controller.endpoints
    assert len(controller.endpoints) == 8
    assert vis_graph is controller.vis_graph
    assert edges
    assert all(len(edge) == 2 for edge in edges)
    assert any(
        edge_start in endpoint_keys or edge_end in endpoint_keys
        for edge_start, edge_end in edges
    )

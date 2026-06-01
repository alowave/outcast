from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.outcast.backhaul.bh_config import BHLayerCfg
from src.outcast.config.simulation_config import (
    MetricsControllerCfg,
    OrchestratorCfg,
    SimulationCfg,
)
from src.outcast.link_layer.mock_link_layer import LinkLayerCfg
from src.outcast.orchestrator import SCENARIO_REGISTRY, Orchestrator
from src.outcast.world.environment_geometry.obstacles import ObstacleCfg
from src.outcast.world.plotting.plot_ctrl import PlotCfg
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldStateCfg


def _build_tiny_world(plot_enabled: bool, seed: int) -> WorldController:
    return WorldController(
        seed=seed,
        world_cfg=WorldStateCfg(
            n_ues=4,
            n_uavs=2,
            n_bss=1,
            env_boundary=(100.0, 100.0),
            user_model=None,
        ),
        obstacle_cfg=ObstacleCfg(enabled=False),
        plot_cfg=PlotCfg(enabled=plot_enabled),
    )


def test_orchestrator_runs_full_layer_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SCENARIO_REGISTRY, "tiny", _build_tiny_world)
    sim_cfg = SimulationCfg(
        link_layer=LinkLayerCfg(access_range_m=5000.0, backhaul_range_m=5000.0),
        backhaul=BHLayerCfg(channel_model=1),
    )
    cfg = OrchestratorCfg(
        scenario_name="tiny",
        time_step=0.1,
        total_steps=2,
        simulation=sim_cfg,
        metrics=MetricsControllerCfg(
            enabled=True,
            buffer_size=1,
            save_dir=str(tmp_path / "metrics"),
        ),
    )
    cfg.simulation.plot.enabled = False

    orchestrator = Orchestrator(cfg)
    orchestrator.run()

    assert orchestrator.world_ctrl is not None
    assert orchestrator.link_layer is not None
    assert orchestrator.fh_layer is not None
    assert orchestrator.bh_layer is not None
    assert orchestrator.bh_ctrl is not None

    world_cfg = orchestrator.world_ctrl.world_cfg
    n_stations = world_cfg.n_uavs + world_cfg.n_bss

    assert orchestrator.link_layer.fronthaul_data is not None
    assert orchestrator.link_layer.backhaul_data is not None
    assert orchestrator.fh_layer.fh_channel_data is not None
    assert orchestrator.bh_layer.bh_channel_data is not None

    assert orchestrator.link_layer.fronthaul_data.dist_m.shape == (
        world_cfg.n_ues,
        n_stations,
    )
    assert orchestrator.link_layer.backhaul_data.dist_m.shape == (
        n_stations,
        n_stations,
    )
    assert orchestrator.fh_layer.fh_channel_data.throughput_bps.shape == (
        world_cfg.n_ues,
        n_stations,
    )
    assert orchestrator.bh_layer.bh_channel_data.throughput_bps.shape == (
        n_stations,
        n_stations,
    )

    uav_routes = orchestrator.bh_ctrl.backhaul_adjacency_matrix[: world_cfg.n_uavs]
    assert np.all(uav_routes.sum(axis=1) == 1)
    assert np.any(orchestrator.bh_layer.bh_channel_data.flow_bps > 0)

    metrics_dir = tmp_path / "metrics" / "fh"
    assert (metrics_dir / "fh_metrics_batch_0.npz").exists()
    assert (metrics_dir / "fh_metrics_batch_1.npz").exists()
    assert (metrics_dir / "fh_episode_metrics.npz").exists()


def test_orchestrator_switches_registered_scenarios() -> None:
    scenario_1_cfg = OrchestratorCfg(
        scenario_name="scenario_1",
        metrics=MetricsControllerCfg(enabled=False),
    )
    scenario_1_cfg.simulation.plot.enabled = False
    scenario_1 = Orchestrator(scenario_1_cfg)

    scenario_2_cfg = OrchestratorCfg(
        scenario_name="scenario_2",
        metrics=MetricsControllerCfg(enabled=False),
    )
    scenario_2_cfg.simulation.plot.enabled = False
    scenario_2 = Orchestrator(scenario_2_cfg)

    assert scenario_1.world_ctrl is not None
    assert scenario_2.world_ctrl is not None
    assert scenario_1.world_ctrl.world_cfg.n_ues == 200
    assert scenario_1.world_ctrl.world_cfg.n_uavs == 18
    assert scenario_2.world_ctrl.world_cfg.n_ues == 0
    assert scenario_2.world_ctrl.world_cfg.n_uavs == 10

    scenario_1.close()
    scenario_2.close()

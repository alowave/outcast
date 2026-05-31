from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.uavnetsim.backhaul.bh_layer import BHLayer
from src.uavnetsim.fronthaul.fh_layer import FHLayer, FHLayerCfg
from src.uavnetsim.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
from src.uavnetsim.metrics.backhaul_metrics import BackhaulMetricsController
from src.uavnetsim.metrics.base import MetricsControllerCfg
from src.uavnetsim.metrics.energy_metrics import EnergyMetricsController
from src.uavnetsim.metrics.fronthaul_metrics import FronthaulMetricsController
from src.uavnetsim.world.environment_geometry.obstacles import ObstacleCfg
from src.uavnetsim.world.plotting.plot_ctrl import PlotCfg
from src.uavnetsim.world.uav_ctrl import DroneStationController
from src.uavnetsim.world.world_ctrl import WorldController
from src.uavnetsim.world.world_state import WorldStateCfg


@pytest.mark.parametrize("time_steps", [5])
def test_integrated_simulation_updates_metrics(time_steps: int, tmp_path: Path) -> None:

    world_cfg = WorldStateCfg(n_ues=8, n_uavs=2, n_bss=1)
    obstacle_cfg = ObstacleCfg(enabled=False)
    plot_cfg = PlotCfg(enabled=False)

    controller = WorldController(
        world_cfg=world_cfg,
        obstacle_cfg=obstacle_cfg,
        plot_cfg=plot_cfg,
    )
    controller.init_random()

    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)

    fh_layer = FHLayer(FHLayerCfg())
    fh_layer.initialize_data_arrays(world_cfg)

    bh_layer = BHLayer()
    bh_layer.initialize_data_arrays(world_cfg)

    n_stations = world_cfg.n_uavs + world_cfg.n_bss

    metrics_cfg = MetricsControllerCfg(
        save_dir=str(tmp_path / "metrics"), buffer_size=3
    )
    metrics_controller = FronthaulMetricsController(metrics_cfg, world_cfg)
    bh_metrics_controller = BackhaulMetricsController(metrics_cfg, world_cfg)

    uav_ctrl = DroneStationController(locations=controller.state.uav_pos)
    controller.uav_ctrl = uav_ctrl

    energy_metrics_controller = EnergyMetricsController(metrics_cfg, world_cfg)

    for _ in range(time_steps):
        controller.simulate_time_step()
        ll.update(controller.state)
        fh_layer.update_fh_channel_data(ll.fronthaul_data)

        # Update Backhaul Channel Data and Dependencies
        bh_layer.update_bh_channel_data(ll.backhaul_data)
        # Generate flows (mocking empty flow matrix matching dimensions)
        mock_flow = np.zeros_like(bh_layer.bh_channel_data.throughput_bps)
        bh_layer.set_flow(mock_flow)
        bh_layer.compute_excess_and_missing()

        # Capture the current state of batteries from the UAV controller
        batteries = [station.battery for station in uav_ctrl.stations]

        metrics_controller.update_step(fh_layer)
        bh_metrics_controller.update_step(bh_layer)
        energy_metrics_controller.update_step(batteries)

    metrics_controller.episode_end()
    bh_metrics_controller.episode_end()
    energy_metrics_controller.episode_end()

    # Original World Validations
    assert controller.state.ue_pos.shape == (world_cfg.n_ues, 3)
    assert controller.state.uav_pos.shape == (world_cfg.n_uavs, 3)
    assert controller.state.bs_pos.shape == (world_cfg.n_bss, 3)
    assert controller.state.ue_pos.size > 0
    assert controller.state.uav_pos.size > 0
    assert controller.state.bs_pos.size > 0

    assert ll.fronthaul_data is not None
    assert ll.backhaul_data is not None

    assert ll.fronthaul_data.dist_m.shape == (world_cfg.n_ues, n_stations)
    assert ll.fronthaul_data.in_range.shape == (world_cfg.n_ues, n_stations)
    assert ll.backhaul_data.dist_m.shape == (n_stations, n_stations)
    assert ll.backhaul_data.in_range.shape == (n_stations, n_stations)
    assert ll.fronthaul_data.dist_m.size > 0
    assert ll.backhaul_data.dist_m.size > 0

    # New Metrics Verification
    metrics_dir = tmp_path / "metrics" / "fh"
    assert metrics_dir.exists(), "Metrics directory should be created."

    batch_0 = metrics_dir / "fh_metrics_batch_0.npz"
    batch_1 = metrics_dir / "fh_metrics_batch_1.npz"
    ep_metrics = metrics_dir / "fh_episode_metrics.npz"

    assert batch_0.exists(), "First metric batch file was not produced."
    assert batch_1.exists(), "Second metric batch file was not produced."
    assert ep_metrics.exists(), "Episode metrics were not finalized."

    d0 = np.load(batch_0)
    assert d0["snr_mean"].shape == (
        3,
    )  # Expecting length corresponding to buffer flush

    d1 = np.load(batch_1)
    assert d1["snr_mean"].shape == (time_steps - 3,)  # Size of remaining elements

    ep = np.load(ep_metrics)
    assert ep["ep_tp_mean_per_ue"].shape == (world_cfg.n_ues,)
    assert ep["ep_assoc_mean_per_bs"].shape == (n_stations,)

    ### === Backhaul ===
    bh_dir = tmp_path / "metrics" / "bh"
    assert bh_dir.exists(), "Backhaul metrics directory should be created."

    assert (bh_dir / "bh_metrics_batch_0.npz").exists()
    assert (bh_dir / "bh_episode_metrics.npz").exists()

    # Step-batch checks
    d0_bh = np.load(bh_dir / "bh_metrics_batch_0.npz")
    assert "capacity_mean" in d0_bh
    assert "utilization_mean" in d0_bh
    assert "n_links_missing_payload" in d0_bh  # Updated key name

    # Episode-wide summary checks
    ep_bh = np.load(bh_dir / "bh_episode_metrics.npz")
    assert ep_bh["ep_link_capacity_mean"] >= 0  # Updated key name
    assert ep_bh["ep_utilization_mean"] >= 0
    assert ep_bh["ep_total_links_with_missing_instances"] >= 0  # Updated key name
    assert ep_bh["ep_missing_duration_matrix"].shape == (n_stations, n_stations)

    ### === Energy ===
    # Energy Metrics Verification
    energy_dir = tmp_path / "metrics" / "energy"
    assert energy_dir.exists(), "Energy metrics directory should be created."

    # Check batch and episode files
    assert (energy_dir / "energy_metrics_batch_0.npz").exists()
    assert (energy_dir / "energy_episode_metrics.npz").exists()

    ep_e = np.load(energy_dir / "energy_episode_metrics.npz")

    # Verify Per-UAV episode metrics
    assert ep_e["ep_energy_mean_per_uav"].shape == (world_cfg.n_uavs,)
    assert ep_e["ep_total_consumed_per_uav"].shape == (world_cfg.n_uavs,)
    assert ep_e["ep_recharge_events_per_uav"].shape == (world_cfg.n_uavs,)

    # Verify per-timestep batch metrics
    d0_e = np.load(energy_dir / "energy_metrics_batch_0.npz")
    assert "energy_level_mean" in d0_e
    assert "energy_consumed_total" in d0_e
    assert "n_uavs_critical" in d0_e

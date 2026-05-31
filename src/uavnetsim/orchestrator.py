"""UAV simulation orchestrator for coordinating world, link, fronthaul, and backhaul layers.

This module provides:
- Orchestrator: Main coordinator for simulation components
- Scenario registry and builders for different simulation scenarios
- Configuration management and metrics collection
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from scenarios.city_of_the_sun import (
    build_golden_petal_uav_world as build_city_of_the_sun_world,
)
from scenarios.poznan_obstacle_scenario import (
    build_poznan_scenario as build_poznan_world,
)
from scenarios.scenario_1 import build_square_uav_world
from scenarios.scenario_2 import (
    build_golden_petal_uav_world as build_scenario_2_world,
)
from src.uavnetsim.backhaul.bh_ctrl import BHController
from src.uavnetsim.backhaul.bh_layer import BHLayer
from src.uavnetsim.config.hydra_registry import register_configs
from src.uavnetsim.config.simulation_config import (
    MetricsControllerCfg,
    OrchestratorCfg,
    build_orchestrator_cfg,
)
from src.uavnetsim.fronthaul.fh_layer import FHLayer
from src.uavnetsim.link_layer.mock_link_layer import MockLinkLayer
from src.uavnetsim.metrics.backhaul_metrics import BackhaulMetricsController
from src.uavnetsim.metrics.energy_metrics import EnergyMetricsController
from src.uavnetsim.metrics.fronthaul_metrics import FronthaulMetricsController
from src.uavnetsim.world.world_ctrl import WorldController

ScenarioBuilder = Callable[[bool, int], WorldController]


def _build_city_of_the_sun(plot_enabled: bool, seed: int) -> WorldController:
    del seed
    return build_city_of_the_sun_world(plot_enabled=plot_enabled)


def _build_scenario_1(plot_enabled: bool, seed: int) -> WorldController:
    return build_square_uav_world(plot_enabled=plot_enabled, seed=seed)


def _build_scenario_2(plot_enabled: bool, seed: int) -> WorldController:
    del seed
    return build_scenario_2_world(plot_enabled=plot_enabled)


def _build_poznan_scenario(plot_enabled: bool, seed: int) -> WorldController:
    del seed
    return build_poznan_world(plot_enabled=plot_enabled)


SCENARIO_REGISTRY: dict[str, ScenarioBuilder] = {
    "city_of_the_sun": _build_city_of_the_sun,
    "scenario_1": _build_scenario_1,
    "scenario_2": _build_scenario_2,
    "poznan_obstacle": _build_poznan_scenario,  # New entry
}

USE_LOCAL_OVERRIDES = True


class Orchestrator:
    """Coordinates world, link, fronthaul, backhaul, and metrics updates."""

    def __init__(
        self, cfg: OrchestratorCfg | DictConfig | dict[str, Any] | None = None
    ) -> None:
        self.cfg = self._coerce_cfg(cfg)
        self.sim_cfg = self.cfg.simulation

        self.world_ctrl: WorldController | None = None
        self.link_layer: MockLinkLayer | None = None
        self.fh_layer: FHLayer | None = None
        self.bh_layer: BHLayer | None = None
        self.bh_ctrl: BHController | None = None
        self.fh_metrics_ctrl: FronthaulMetricsController | None = None
        self.bh_metrics_ctrl: BackhaulMetricsController | None = None
        self.energy_metrics_ctrl: EnergyMetricsController | None = None

        self.batteries = None

        self.current_step = 0
        self._closed = False

        self.setup()

    @classmethod
    def available_scenarios(cls) -> tuple[str, ...]:
        """Return a tuple of available scenario names.

        Returns:
            Sorted tuple of scenario names registered in SCENARIO_REGISTRY.
        """
        return tuple(sorted(SCENARIO_REGISTRY))

    def setup(self) -> None:
        """Initialize all simulation components based on configuration."""
        if self.cfg.scenario_name != "tiny":
            self.sim_cfg.world.user_model = getattr(
                self.sim_cfg.world, "user_model", "obstacle_mobility"
            )
            self.sim_cfg.obstacle.map_title = getattr(
                self.sim_cfg.obstacle, "map_title", "poznan_obstacle"
            )

        builder = self._get_scenario_builder(self.cfg.scenario_name)
        self.world_ctrl = builder(self.sim_cfg.plot.enabled, self.sim_cfg.seed)
        self._ensure_gn_load()

        world_cfg = self.world_ctrl.world_cfg

        self.link_layer = MockLinkLayer(self.sim_cfg.link_layer)
        self.link_layer.initialize_data_arrays(world_cfg)

        # Activate real sector-based LoS computation when obstacles are loaded.
        obstacles = getattr(self.world_ctrl.state, "obstacles", None)  # = world_cfg_obs
        if obstacles is None and hasattr(self.world_ctrl, "obstacle_ctrl"):
            obstacles = self.world_ctrl.obstacle_ctrl.obstacles_list
        if obstacles:
            from src.uavnetsim.geometry.los_cache import LosCacheCfg

            los_cfg = LosCacheCfg(sector_size_m=self.sim_cfg.los_cache.sector_size_m)
            self.link_layer.set_los_cache(los_cfg, obstacles)

        self.fh_layer = FHLayer(self.sim_cfg.fronthaul)
        self.fh_layer.initialize_data_arrays(world_cfg)

        self.bh_layer = BHLayer(self.sim_cfg.backhaul)
        self.bh_layer.initialize_data_arrays(world_cfg)

        self.bh_ctrl = BHController(
            bh_layer=self.bh_layer,
            rng=np.random.default_rng(self.sim_cfg.seed),
        )

        self._update_backhaul_routing(self.bh_ctrl, self.link_layer.backhaul_data)

        if self.cfg.metrics.enabled:
            self.fh_metrics_ctrl = FronthaulMetricsController(
                self.cfg.metrics, world_cfg
            )
            self.bh_metrics_ctrl = BackhaulMetricsController(
                self.cfg.metrics, world_cfg
            )
            self.energy_metrics_ctrl = EnergyMetricsController(
                self.cfg.metrics, world_cfg
            )
        else:
            self.fh_metrics_ctrl = None
            self.bh_metrics_ctrl = None
            self.energy_metrics_ctrl = None

        self.batteries = [_uav.battery for _uav in self.world_ctrl.uav_ctrl.stations]
        self.current_step = 0
        self._closed = False

    def step(self) -> None:
        """Advance the simulation by one time step."""
        world_ctrl = self._require(self.world_ctrl, "world_ctrl")
        link_layer = self._require(self.link_layer, "link_layer")
        fh_layer = self._require(self.fh_layer, "fh_layer")
        bh_layer = self._require(self.bh_layer, "bh_layer")
        bh_ctrl = self._require(self.bh_ctrl, "bh_ctrl")

        world_ctrl.simulate_time_step(time_step=self.cfg.time_step)
        self._ensure_gn_load()

        link_layer.update(world_ctrl.state)
        if link_layer.fronthaul_data is None or link_layer.backhaul_data is None:
            raise RuntimeError("Link-layer data arrays were not initialized.")

        fh_layer.update_fh_channel_data(link_layer.fronthaul_data)

        # self._update_backhaul_routing(bh_ctrl, link_layer.backhaul_data)
        bh_layer.update_bh_channel_data(link_layer.backhaul_data)
        bh_layer.set_flow(
            bh_ctrl.backhaul_outflow_matrix_bps,
            bh_ctrl.backhaul_adjacency_matrix,
        )
        bh_layer.compute_excess_and_missing()

        if self.fh_metrics_ctrl is not None:
            self.fh_metrics_ctrl.update_step(fh_layer)
        if self.bh_metrics_ctrl is not None:
            self.bh_metrics_ctrl.update_step(bh_layer)
        if self.energy_metrics_ctrl is not None:
            self.energy_metrics_ctrl.update_step(self.batteries)

        self.current_step += 1

    def run(self, total_steps: int | None = None) -> None:
        """Run the simulation for the specified number of steps.

        Args:
            total_steps: Number of steps to run. If None, uses cfg.total_steps.
        """
        steps = self.cfg.total_steps if total_steps is None else total_steps
        try:
            for _ in range(steps):
                self.step()
        finally:
            self.close()

    def close(self) -> None:
        """Clean up resources and finalize metrics collection."""
        if self._closed:
            return

        if self.fh_metrics_ctrl is not None:
            self.fh_metrics_ctrl.episode_end()
        if self.bh_metrics_ctrl is not None:
            self.bh_metrics_ctrl.episode_end()

        self._closed = True

    def _get_scenario_builder(self, scenario_name: str) -> ScenarioBuilder:
        try:
            return SCENARIO_REGISTRY[scenario_name]
        except KeyError as exc:
            available = ", ".join(self.available_scenarios())
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Available scenarios: {available}."
            ) from exc

    def _ensure_gn_load(self) -> None:
        world_ctrl = self._require(self.world_ctrl, "world_ctrl")
        n_nodes = world_ctrl.state.uav_pos.shape[0] + world_ctrl.state.bs_pos.shape[0]
        if world_ctrl.state.gn_load is None or world_ctrl.state.gn_load.shape != (
            n_nodes,
        ):
            world_ctrl.randomize_loads()

    def _update_backhaul_routing(
        self,
        bh_ctrl: BHController,
        backhaul_data,
    ) -> None:
        bh_ctrl.backhaul_adjacency_matrix.fill(False)
        bh_ctrl.backhaul_outflow_matrix_bps.fill(0)

        world_ctrl = self._require(self.world_ctrl, "world_ctrl")
        if bh_ctrl.bh_layer.n_bss == 0:
            return

        if world_ctrl.state.gn_load is None:
            raise RuntimeError("World generated-node load is not initialized.")

        bh_ctrl.randomize_connections(backhaul_data)
        bh_ctrl.distribute_random_load(world_ctrl.state.gn_load)

    @staticmethod
    def _require(value, name: str):
        if value is None:
            raise RuntimeError(f"Orchestrator {name} is not initialized.")
        return value

    @staticmethod
    def _coerce_cfg(
        cfg: OrchestratorCfg | DictConfig | dict[str, Any] | None,
    ) -> OrchestratorCfg:
        if cfg is None:
            return OrchestratorCfg()
        if isinstance(cfg, DictConfig):
            return build_orchestrator_cfg(
                OmegaConf.to_container(cfg, resolve=True),
            )
        return build_orchestrator_cfg(cfg)


def _apply_default_metrics_dir(cfg: OrchestratorCfg) -> None:
    default_metrics_cfg = MetricsControllerCfg()
    if cfg.metrics.save_dir != default_metrics_cfg.save_dir:
        return

    save_dir = Path(HydraConfig.get().runtime.output_dir) / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics.save_dir = str(save_dir)


def _apply_local_overrides(cfg: OrchestratorCfg) -> None:
    """Optional in-code overrides for local development.

    Keep this disabled by default and change values here only when you want
    code-level defaults to win over Hydra config and CLI overrides.
    """
    cfg.scenario_name = "poznan_obstacle"

    cfg.simulation.world.user_model = "obstacle_mobility"
    cfg.simulation.obstacle.enabled = True
    cfg.simulation.obstacle.map_title = "poznan_center"

    cfg.total_steps = 10000
    cfg.time_step = 0.1
    cfg.simulation.plot.enabled = True

    cfg.simulation.link_layer.access_range_m = 500.0
    cfg.simulation.link_layer.backhaul_range_m = 1000.0

    cfg.metrics.enabled = True
    cfg.metrics.buffer_size = 1000
    cfg.simulation.backhaul.channel_model = 1


register_configs()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the UAV simulation orchestrator.

    Args:
        cfg: Hydra configuration object.
    """
    orchestrator_cfg = build_orchestrator_cfg(OmegaConf.to_container(cfg, resolve=True))
    _apply_default_metrics_dir(orchestrator_cfg)
    if USE_LOCAL_OVERRIDES:
        _apply_local_overrides(orchestrator_cfg)
    orchestrator = Orchestrator(orchestrator_cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()

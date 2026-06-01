"""Simulation configuration classes for UAV simulation.

This module defines the main configuration dataclasses used throughout the UAV
simulation, including simulation parameters, metrics controller settings, and
orchestrator configuration. It also provides builder functions for merging
configuration with OmegaConf.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf

from src.outcast.backhaul.bh_config import BHLayerCfg
from src.outcast.fronthaul.fh_config import FHLayerCfg
from src.outcast.geometry.los_cache import LosCacheCfg
from src.outcast.link_layer.mock_link_layer import LinkLayerCfg
from src.outcast.world.energy_model.uav_battery import BatteryCfg
from src.outcast.world.environment_geometry.obstacles import ObstacleCfg
from src.outcast.world.plotting.plot_ctrl import PlotCfg
from src.outcast.world.user_model.load_ctrl import LoadCfg
from src.outcast.world.user_model.random_movement import RandomMovementCfg
from src.outcast.world.world_state import WorldStateCfg


@dataclass(slots=True)
class SimulationCfg:
    """Main simulation configuration.

    Attributes:
        seed: Random seed for reproducibility.
        world: World state configuration.
        load: User load configuration.
        obstacle: Obstacle configuration.
        plot: Plotting configuration.
        random_movement: Random movement configuration.
        link_layer: Link layer configuration.
        fronthaul: Fronthaul layer configuration.
        backhaul: Backhaul layer configuration.
        uav_battery: UAV battery configuration.
    """

    seed: int = 33
    world: WorldStateCfg = field(default_factory=WorldStateCfg)
    load: LoadCfg = field(default_factory=LoadCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)
    plot: PlotCfg = field(default_factory=PlotCfg)
    random_movement: RandomMovementCfg = field(default_factory=RandomMovementCfg)
    link_layer: LinkLayerCfg = field(default_factory=LinkLayerCfg)
    fronthaul: FHLayerCfg = field(default_factory=FHLayerCfg)
    backhaul: BHLayerCfg = field(default_factory=BHLayerCfg)
    uav_battery: BatteryCfg = field(default_factory=BatteryCfg)
    los_cache: LosCacheCfg = field(default_factory=LosCacheCfg)


@dataclass(slots=True)
class MetricsControllerCfg:
    """Metrics controller configuration.

    Attributes:
        enabled: Whether metrics collection is enabled.
        buffer_size: Number of steps to buffer before saving to disk.
        save_dir: Directory path for saving metrics files.
    """

    enabled: bool = True
    buffer_size: int = 1000
    save_dir: str = "./outputs/metrics"


@dataclass(slots=True)
class OrchestratorCfg:
    """Orchestrator configuration for running simulations.

    Attributes:
        scenario_name: Name of the scenario to run.
        time_step: Time step duration in seconds.
        total_steps: Total number of simulation steps.
        simulation: Simulation configuration.
        metrics: Metrics controller configuration.
    """

    scenario_name: str = "city_of_the_sun"
    time_step: float = 1.0
    total_steps: int = 1000
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    metrics: MetricsControllerCfg = field(default_factory=MetricsControllerCfg)


def _mapping_or_empty(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return {} if data is None else data


def build_metrics_cfg(
    data: MetricsControllerCfg | Mapping[str, Any] | None = None,
) -> MetricsControllerCfg:
    """Build metrics controller configuration from defaults and optional data.

    Args:
        data: Optional configuration data to merge with defaults.

    Returns:
        Merged MetricsControllerCfg instance.
    """
    defaults = MetricsControllerCfg()
    if data is None:
        return defaults
    if isinstance(data, MetricsControllerCfg):
        return data

    merged = OmegaConf.merge(
        OmegaConf.structured(defaults), dict(_mapping_or_empty(data))
    )
    return OmegaConf.to_object(merged)


def build_simulation_cfg(
    data: SimulationCfg | Mapping[str, Any] | None = None,
) -> SimulationCfg:
    """Build simulation configuration from defaults and optional data.

    Args:
        data: Optional configuration data to merge with defaults.

    Returns:
        Merged SimulationCfg instance.
    """
    defaults = SimulationCfg()
    if data is None:
        return defaults
    if isinstance(data, SimulationCfg):
        return data

    merged = OmegaConf.merge(
        OmegaConf.structured(defaults), dict(_mapping_or_empty(data))
    )
    return OmegaConf.to_object(merged)


def build_orchestrator_cfg(
    data: OrchestratorCfg | Mapping[str, Any] | None = None,
) -> OrchestratorCfg:
    """Build orchestrator configuration from defaults and optional data.

    Args:
        data: Optional configuration data to merge with defaults.

    Returns:
        Merged OrchestratorCfg instance.
    """
    defaults = OrchestratorCfg()
    if data is None:
        return defaults
    if isinstance(data, OrchestratorCfg):
        return data

    merged = OmegaConf.merge(
        OmegaConf.structured(defaults), dict(_mapping_or_empty(data))
    )
    return OmegaConf.to_object(merged)

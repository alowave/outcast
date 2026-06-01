"""Hydra configuration registry for UAV simulation.

This module registers structured configuration classes with Hydra's ConfigStore,
allowing them to be used as optional structured-config aliases without colliding
with file-based configs.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from src.outcast.backhaul.bh_config import BHLayerCfg
from src.outcast.config.simulation_config import (
    MetricsControllerCfg,
    OrchestratorCfg,
    SimulationCfg,
)
from src.outcast.fronthaul.fh_config import FHLayerCfg
from src.outcast.link_layer.mock_link_layer import LinkLayerCfg
from src.outcast.world.energy_model.uav_battery import BatteryCfg
from src.outcast.world.environment_geometry.obstacles import ObstacleCfg
from src.outcast.world.plotting.plot_ctrl import PlotCfg
from src.outcast.world.user_model.load_ctrl import LoadCfg
from src.outcast.world.user_model.random_movement import RandomMovementCfg
from src.outcast.world.world_state import WorldStateCfg


def register_configs() -> None:
    """Register optional structured-config aliases without colliding with file configs."""
    cs = ConfigStore.instance()
    cs.store(group="schema", name="orchestrator", node=OrchestratorCfg)
    cs.store(group="schema", name="simulation", node=SimulationCfg)
    cs.store(group="schema/metrics", name="structured", node=MetricsControllerCfg)
    cs.store(group="schema/world", name="structured", node=WorldStateCfg)
    cs.store(group="schema/load", name="structured", node=LoadCfg)
    cs.store(group="schema/obstacle", name="structured", node=ObstacleCfg)
    cs.store(group="schema/plot", name="structured", node=PlotCfg)
    cs.store(
        group="schema/random_movement",
        name="structured",
        node=RandomMovementCfg,
    )
    cs.store(group="schema/link_layer", name="structured", node=LinkLayerCfg)
    cs.store(group="schema/fronthaul", name="structured", node=FHLayerCfg)
    cs.store(group="schema/backhaul", name="structured", node=BHLayerCfg)
    cs.store(group="schema/uav_battery", name="structured", node=BatteryCfg)

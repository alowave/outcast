"""Simulation Orchestration Engine and Core World Controller.

Acts as the primary orchestrator for the simulation execution lifecycle. Coordinates
spatial entity state modifications, physical obstacle constraints, terminal-user mobility
factories, traffic load distribution, and live GUI plot refreshing.
"""

from __future__ import annotations

import numpy as np

from src.outcast.world.environment_geometry.obstacles import (
    ObstacleCfg,
    ObstacleController,
)
from src.outcast.world.plotting.plot_ctrl import PlotCfg, PlotController
from src.outcast.world.uav_ctrl import BaseUavController, StaticUavController
from src.outcast.world.user_model.base import UserModel
from src.outcast.world.user_model.load_ctrl import LoadCfg, LoadController
from src.outcast.world.user_model.obstacle_mobility_model import (
    ObstacleMobilityUserModel,
)
from src.outcast.world.user_model.random_movement import RandomMovementUserModel
from src.outcast.world.world_state import WorldState, WorldStateCfg


class WorldController:
    """
    Orchestrator for world state generation and simulation updates.

    Coordinates between spatial models, load controllers, and visualization.
    """

    def __init__(
        self,
        seed: int = 33,
        world_cfg: WorldStateCfg | None = None,
        load_cfg: LoadCfg | None = None,
        obstacle_cfg: ObstacleCfg | None = None,
        plot_cfg: PlotCfg | None = None,
        uav_ctrl: BaseUavController | None = None,
        user_model: UserModel | None = None,
        state: WorldState | None = None,
    ) -> None:
        self.world_cfg = world_cfg or WorldStateCfg()
        self.load_cfg = load_cfg or LoadCfg()
        self.obstacle_cfg = obstacle_cfg or ObstacleCfg()
        self.plot_cfg = plot_cfg or PlotCfg()

        self.rng = np.random.default_rng(seed)
        self.load_controller = LoadController(self.load_cfg, self.rng)
        self.obstacle_controller = ObstacleController(self.obstacle_cfg)
        self.plt_controller: PlotController | None = None
        self.uav_ctrl = uav_ctrl

        if self.obstacle_cfg.enabled:
            self.obstacle_controller.load_obstacles()
            self._sync_env_boundary_from_obstacles()

        if state is not None:
            self.state = state
        else:
            self.state = WorldState(
                ue_pos=np.empty((0, 3), dtype=np.float32),
                uav_pos=np.empty((0, 3), dtype=np.float32),
                bs_pos=np.empty((0, 3), dtype=np.float32),
                obstacles=self.obstacle_controller.obstacles_list,
            )

        self.user_model = user_model or self._build_user_model()

        if state is None:
            self.randomize_positions()
            self.load_controller.apply_random_loads(self.state)

        if self.uav_ctrl is None:
            self.uav_ctrl = StaticUavController(self.state.uav_pos)

        if self.user_model is not None:
            self.user_model.reset_users(self.state.ue_pos)

        if self.plot_cfg.enabled:
            self.plt_controller = PlotController(
                self.state, cfg=self.plot_cfg, show=True
            )

    def init_random(self) -> None:
        """
        Initialize all entity positions and loads with random values.
        """
        self.randomize_positions()
        self.randomize_loads()

    def randomize_positions(self) -> None:
        """
        Refresh positions for all world entities and sync to the user model.
        """
        self.state.randomize_ue_pos(self.world_cfg, self.rng)
        self.state.randomize_uav_pos(self.world_cfg, self.rng)
        self.state.randomize_bs_pos(self.world_cfg, self.rng)
        if self.user_model is not None:
            self.user_model.reset_users(self.state.ue_pos)
        if self.uav_ctrl is not None:
            self.uav_ctrl.set_locations(self.state.uav_pos)

    def randomize_loads(
        self,
        min_load: int | None = None,
        max_load: int | None = None,
    ) -> None:
        """
        Apply random traffic loads to UAVs and Base Stations.
        """
        self.load_controller.apply_random_loads(self.state, min_load, max_load)

    def _sync_env_boundary_from_obstacles(self) -> None:
        """
        Update world config boundary based on the loaded obstacle dataset.
        """
        boundaries = self.obstacle_controller.get_boundaries()
        self.world_cfg.env_boundary = (
            float(boundaries[0][1]),
            float(boundaries[1][1]),
        )

    def _build_user_model(self) -> UserModel | None:
        """
        Factory method to select the appropriate UserModel implementation.
        """
        if self.world_cfg.user_model == "obstacle_mobility":
            m_cfg = self.world_cfg.mobility
            if m_cfg is None:
                from src.outcast.world.user_model.obstacle_mobility_model import (
                    ObstacleMobilityCfg,
                )

                m_cfg = ObstacleMobilityCfg()
            return ObstacleMobilityUserModel(
                world_cfg=self.world_cfg,
                obstacles=self.state.obstacles,
                obstacle_cfg=self.obstacle_controller.cfg,
                mobility_cfg=m_cfg,
                rng=self.rng,
            )

        if self.world_cfg.user_model is None:
            return None

        if self.world_cfg.user_model == "random_movement":
            return RandomMovementUserModel(self.world_cfg, rng=self.rng)

        if self.world_cfg.user_model == "static_thomas_cluster":
            from src.outcast.world.user_model.user_static_model import (
                StaticThomasClusterModel,
            )

            return StaticThomasClusterModel(self.world_cfg, rng=self.rng)

        raise ValueError(f"Unsupported user model: {self.world_cfg.user_model}")

    def simulate_time_step(self, time_step: float = 1.0) -> None:
        """
        Progress simulation time, updating user movement and UI visualizations.
        """
        if self.user_model is not None:
            self.user_model.step(time_step)
            self.user_model.update_locations_array(self.state.ue_pos)

        if self.uav_ctrl is not None:
            self.uav_ctrl.step(time_step)
            if len(self.uav_ctrl.locations) != self.state.uav_pos.shape[0]:
                if self.uav_ctrl.stations:
                    raise ValueError(
                        "uav_ctrl station count does not match state.uav_pos shape."
                    )
                self.uav_ctrl.set_locations(self.state.uav_pos)
            self.uav_ctrl.update_locations_array(self.state.uav_pos)

        if not self.plot_cfg.enabled or self.plt_controller is None:
            return

        self.plt_controller.update_ues(self.state.ue_pos)
        self.plt_controller.update_bs(self.state.bs_pos)
        self.plt_controller.update_uavs(self.state.uav_pos)
        self.plt_controller.process_events()

import numpy as np

from src.uavnetsim.world.world_ctrl import WorldController
from src.uavnetsim.world.world_state import WorldState, WorldStateCfg


def test_world_controller_randomize_positions_matches_split_randomizers():
    cfg = WorldStateCfg(n_ues=4, n_uavs=3, n_bss=2, env_boundary=(100.0, 200.0))
    controller = WorldController(world_cfg=cfg)
    split_state = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=np.empty((0, 3), dtype=np.float32),
        bs_pos=np.empty((0, 3), dtype=np.float32),
    )
    controller.state = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=np.empty((0, 3), dtype=np.float32),
        bs_pos=np.empty((0, 3), dtype=np.float32),
    )
    controller.rng = np.random.default_rng(33)
    controller.randomize_positions()

    rng_split = np.random.default_rng(33)
    split_state.randomize_ue_pos(cfg, rng_split)
    split_state.randomize_uav_pos(cfg, rng_split)
    split_state.randomize_bs_pos(cfg, rng_split)

    np.testing.assert_allclose(controller.state.ue_pos, split_state.ue_pos)
    np.testing.assert_allclose(controller.state.uav_pos, split_state.uav_pos)
    np.testing.assert_allclose(controller.state.bs_pos, split_state.bs_pos)


def test_world_state_split_randomizers_preserve_expected_shapes_and_heights():
    cfg = WorldStateCfg(n_ues=5, n_uavs=6, n_bss=3)
    rng = np.random.default_rng(7)
    state = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=np.empty((0, 3), dtype=np.float32),
        bs_pos=np.empty((0, 3), dtype=np.float32),
    )

    state.randomize_ue_pos(cfg, rng)
    state.randomize_uav_pos(cfg, rng)
    state.randomize_bs_pos(cfg, rng)

    assert state.ue_pos.shape == (cfg.n_ues, 3)
    assert state.uav_pos.shape == (cfg.n_uavs, 3)
    assert state.bs_pos.shape == (cfg.n_bss, 3)

    assert state.ue_pos.dtype == np.float32
    assert state.uav_pos.dtype == np.float32
    assert state.bs_pos.dtype == np.float32

    assert np.all(state.ue_pos[:, 2] == cfg.ue_height)
    assert np.all(state.bs_pos[:, 2] == cfg.bs_height)
    assert np.all(state.uav_pos[:, 2] >= cfg.min_height_uav)
    assert np.all(state.uav_pos[:, 2] <= cfg.max_height_uav)

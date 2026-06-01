import numpy as np

from src.outcast.world.user_model.random_movement import (
    RandomMovementCfg,
    RandomMovementUserModel,
)
from src.outcast.world.world_state import WorldStateCfg


def test_random_movement_step_applies_bounded_offset_and_clips_to_boundary():
    cfg = WorldStateCfg(
        n_ues=2,
        env_boundary=(5.0, 6.0),
        ue_height=1.5,
        user_model="random_movement",
    )
    model = RandomMovementUserModel(
        cfg,
        RandomMovementCfg(step_range=(0.5, 0.5)),
        np.random.default_rng(1),
    )

    model.reset_users(
        np.array(
            [
                [1.0, 2.0, 1.5],
                [4.8, 5.9, 1.5],
            ],
            dtype=np.float32,
        )
    )

    model.step(1.0)
    positions = model.get_locations_array()

    np.testing.assert_allclose(
        positions,
        np.array(
            [
                [1.5, 2.5, 1.5],
                [5.0, 6.0, 1.5],
            ],
            dtype=np.float32,
        ),
    )


def test_random_movement_updates_existing_array_in_place():
    cfg = WorldStateCfg(
        n_ues=1,
        env_boundary=(5.0, 6.0),
        ue_height=1.5,
        user_model="random_movement",
    )
    model = RandomMovementUserModel(
        cfg,
        RandomMovementCfg(step_range=(0.5, 0.5)),
        np.random.default_rng(1),
    )
    target = np.array([[1.0, 2.0, 1.5]], dtype=np.float32)

    model.reset_users(target)
    model.step(1.0)
    model.update_locations_array(target)

    np.testing.assert_allclose(target, np.array([[1.5, 2.5, 1.5]], dtype=np.float32))

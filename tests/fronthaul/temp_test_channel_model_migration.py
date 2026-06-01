import numpy as np
from numpy.testing import assert_allclose

# Import the original monolithic model
from temp_fh_channel_model_old import PlosModel as OldModel

# Import the new refactored manager
from src.outcast.fronthaul.fh_channel_model_manager import (
    FHPathLossModelManager as NewModelManager,
)


def test_vectorized_path_loss_refactoring_equivalence():
    """
    Tests that the refactored FronthaulPathLossModel produces the exact same
    mathematical output as the original monolithic PlosModel for a mixed
    A2G (UAV) and G2G (Base Station) deployment.
    """

    # 1. Setup Testcase Data
    n_ue = 10  # 10 Ground Users
    n_s = 4  # 4 Sources (2 UAVs, 2 Base Stations)

    # 3D Distances (meters) - Random realistic distances between 50m and 1000m
    np.random.seed(42)
    dist_m = np.random.uniform(50.0, 1000.0, size=(n_ue, n_s)).astype(np.float32)

    # Frequencies (Hz) - 2 GHz for UAVs, 3.5 GHz for Base Stations
    frequencies = np.array([[2e9, 2e9, 3.5e9, 3.5e9]], dtype=np.float32)

    # Link Types: 0 for A2G (UAV), 1 for G2G (BS)
    link_type = np.array([0, 0, 1, 1], dtype=np.uint8)

    # Heights (meters)
    ue_height = np.random.uniform(1.5, 20.0, size=n_ue).astype(np.float32)
    bs_height = np.array(
        [100.0, 100.0, 30.0, 30.0], dtype=np.float32
    )  # UAVs @ 100m, BSs @ 30m

    # 2. Instantiate Models
    old_model = OldModel()
    new_model = NewModelManager()

    # 3. Execution - Old Implementation
    # We pass a seeded RNG to ensure shadow fading and UMa probabilities are deterministic
    rng_old = np.random.default_rng(seed=999)
    out_old = old_model.get_vectorized_path_loss(
        dist_m=dist_m.copy(),
        frequencies=frequencies.copy(),
        link_type=link_type,
        ue_height=ue_height,
        bs_height=bs_height,
        enable_shadowing=True,
        uma_height_mode="probabilistic",
        rng=rng_old,
    )

    # 4. Execution - New Refactored Implementation
    # We must re-instantiate the RNG with the EXACT same seed so it generates the
    # identical random sequence for the new model's internal calls.
    rng_new = np.random.default_rng(seed=999)
    out_new = new_model.get_vectorized_path_loss(
        dist_m=dist_m.copy(),
        frequencies=frequencies.copy(),
        link_type=link_type,
        ue_height=ue_height,
        bs_height=bs_height,
        enable_shadowing=True,
        uma_height_mode="probabilistic",
        rng=rng_new,
    )

    # 5. Assertion
    # assert_allclose checks that arrays are equal up to a very tight floating-point tolerance
    assert_allclose(
        out_old,
        out_new,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Refactored FronthaulPathLossModel output does not match the original PlosModel.",
    )

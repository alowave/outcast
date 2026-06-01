import numpy as np
import pytest

from src.outcast.fronthaul.fh_channel_model_manager import FHPathLossModelManager
from src.outcast.fronthaul.fh_config import FHChannelCfg


def test_manual_validation_path_loss():
    # 1. Setup Configuration
    channel_cfg = FHChannelCfg(
        env_type="Urban",
        default_ue_height_m=1.5,
        default_bs_height_m=25.0,
        default_uav_height_m=50.0,
    )
    manager = FHPathLossModelManager(channel_cfg)

    # 2. Define Scenarios
    # Column 0: A2G (UAV), Column 1: G2G (BS)
    dist_m = np.array([[111.144, 500.551]], dtype=np.float32)
    frequencies = np.array([[2.0e9, 2.0e9]], dtype=np.float32)
    height_diff = np.array([[48.5, 23.5]], dtype=np.float32)
    link_type = np.array([0, 1], dtype=np.uint8)  # 0=UAV, 1=BS

    # CASE A: A2G Probabilistic
    # We pass is_los=None to trigger probabilistic calculation
    out = manager.get_vectorized_path_loss(
        dist_m=dist_m,
        frequencies=frequencies,
        height_diff=height_diff,
        link_type=link_type,
        is_los=None,
    )

    expected_a2g_prob = 88.49  # From manual calculation
    assert out[0, 0] == pytest.approx(expected_a2g_prob, abs=0.1)

    # CASE B: G2G Deterministic LOS
    is_los_mask = np.array([[True, True]], dtype=np.bool_)
    out_los = manager.get_vectorized_path_loss(
        dist_m=dist_m,
        frequencies=frequencies,
        height_diff=height_diff,
        link_type=link_type,
        is_los=is_los_mask,
        enable_shadowing=False,  # Disable for deterministic comparison
    )

    expected_g2g_los = 96.89  # From manual calculation (PL2 formula)
    assert out_los[0, 1] == pytest.approx(expected_g2g_los, abs=0.1)


def test_plos_optimal_params():
    """Validates the geometric optimization logic."""
    manager = FHPathLossModelManager()
    h, r = manager.plos_model.get_optimal_height_radius()

    # Basic sanity checks for Urban environment
    assert h > 0
    assert r > 0
    assert h < r  # Typically optimal altitude is lower than radius for Urban

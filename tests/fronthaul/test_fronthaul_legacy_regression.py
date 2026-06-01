from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from src.outcast.fronthaul.fh_layer import FHLayer, FHLayerCfg
from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldStateCfg

DATA_DIR = Path(__file__).parent / "legacy_reference"


def test_fronthaul_legacy_regression_2():
    """
    Regression test comparing current results (dBm/dB) against legacy results
    converted to log scale.
    """
    data_init = np.load(DATA_DIR / "iteration_0.npz")
    world_cfg = WorldStateCfg(
        n_ues=len(data_init["ue_coords"]), n_uavs=len(data_init["uav_coords"]), n_bss=0
    )

    ws = WorldController(world_cfg=world_cfg).state
    ll = MockLinkLayer(LinkLayerCfg(access_range_m=500000))
    ll.initialize_data_arrays(world_cfg)
    fh_layer = FHLayer(FHLayerCfg())
    fh_layer.initialize_data_arrays(world_cfg)

    for i in range(5):
        data = np.load(DATA_DIR / f"iteration_{i}.npz")
        ws.ue_pos[:] = data["ue_coords"]
        ws.uav_pos[:] = data["uav_coords"]

        ll.update(ws)
        ll.fronthaul_data.los = None  # Force probabilistic mode
        fh_layer.update_fh_channel_data(ll.fronthaul_data)

        # --- Extraction of Current Results (Already in dBm/dB) ---
        rows = fh_layer.row_idx
        cols = fh_layer.best_idx
        actual_dbm = fh_layer.fh_channel_data.ue_received_power_dbm[rows, cols]
        actual_sinr_db = fh_layer.fh_channel_data.sinr_db[rows, cols]

        # --- Conversion of Legacy Data to Log Scale ---
        # 1. Power: Watts -> dBm
        legacy_dbm = 10.0 * np.log10(np.maximum(data["ue_rx_power"], 1e-15)) + 30.0

        # 2. SINR: Ratio -> dB
        legacy_sinr_db = 10.0 * np.log10(np.maximum(data["ue_sinr"], 1e-15))

        # --- Verification (Assertions) ---

        # A. Association
        assert_allclose(
            fh_layer.best_idx,
            data["bs_id"],
            err_msg=f"Iteration {i}: UE Association (bs_id) mismatch",
        )

        # B. Link Quality
        # atol=3.0 means we allow the power to be off by up to 3 dB
        assert_allclose(
            actual_dbm,
            legacy_dbm,
            atol=3.0,
            err_msg=f"Iteration {i}: UE Rx Power mismatch (>3dB)",
        )

        assert_allclose(
            actual_sinr_db,
            legacy_sinr_db,
            atol=3.0,
            err_msg=f"Iteration {i}: UE SINR mismatch (>3dB)",
        )

        # C. Global Metrics (Fairness and Reliability)
        actual_total_reliability = fh_layer.user_is_served.sum() / ws.ue_pos.shape[0]
        assert_allclose(
            actual_total_reliability,
            data["total_reliability"].item(),
            rtol=0.2,
            err_msg=f"Iteration {i}: Total Reliability mismatch",
        )

        assert_allclose(
            fh_layer.total_fairness,
            data["total_fairness"].item(),
            rtol=0.2,
            err_msg=f"Iteration {i}: Total Fairness mismatch",
        )

        assert_allclose(
            fh_layer.fairness_per_bs,
            data["per_uav_fairness"],
            rtol=0.2,
            err_msg=f"Iteration {i}: Per-UAV Fairness mismatch",
        )

import numpy as np
import pytest

from src.outcast.fronthaul.fh_layer import FHLayer, FHLayerCfg
from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
from src.outcast.utils.math_tools import db2lin
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldStateCfg


def test_fh_layer_1():
    world_cfg = WorldStateCfg()
    ws = WorldController(world_cfg=world_cfg).state
    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.update(ws)

    fh_layer = FHLayer(FHLayerCfg())
    fh_layer.initialize_data_arrays(world_cfg)
    fh_layer.update_fh_channel_data(ll.fronthaul_data)

    link_data = ll.fronthaul_data
    channel_data = fh_layer.fh_channel_data

    # 1. Verify Path Loss Integration
    # Check that path loss is calculated and physically plausible (> 0 and not NaN)
    assert not np.any(np.isnan(channel_data.path_loss_db))
    assert np.all(channel_data.path_loss_db > 0)

    # 2. Verify Data Consistency across types
    # Ensure Link Layer and FH Layer are looking at the same dimensions
    n_ue, n_s = link_data.dist_m.shape
    assert channel_data.path_loss_db.shape == (n_ue, n_s)

    # 3. Verify Association Logic vs SINR
    # Association must happen with the source providing the highest SINR among those in range
    for ue_idx in range(n_ue):
        ue_in_range = link_data.in_range[ue_idx]
        if np.any(ue_in_range):
            # Find which source index was actually chosen
            chosen_source = np.where(fh_layer.serving_mask[ue_idx])[0]

            # If the user is served, it must be exactly one source
            if len(chosen_source) > 0:
                s_idx = chosen_source[0]
                # The chosen source MUST be in range
                assert ue_in_range[s_idx]
                # The chosen source MUST have the max SINR among in-range sources
                current_sinr = channel_data.sinr_db[ue_idx, s_idx]
                max_possible_sinr = np.max(channel_data.sinr_db[ue_idx, ue_in_range])
                assert current_sinr == pytest.approx(max_possible_sinr)

    # 4. Verify Throughput Enforcement
    # Throughput should be 0 for any link that is NOT the serving link
    non_serving_mask = ~fh_layer.serving_mask
    assert np.all(channel_data.throughput_bps[non_serving_mask] == 0)

    # 5. Verify Fairness Bounds
    # Jain's fairness index must always be between 0 and 1.0
    assert 0.0 <= fh_layer.total_fairness <= 1.0

    # 6. Verify SINR Calculation (Signal - (Interference + Noise))
    # We'll check one random UE's serving link
    ue_idx = 0
    if np.any(fh_layer.serving_mask[ue_idx]):
        s_idx = np.where(fh_layer.serving_mask[ue_idx])[0][0]
        p_rx_dbm = channel_data.ue_received_power_dbm[ue_idx, s_idx]

        # Calculate expected noise + interference in linear
        total_pwr_lin = np.sum(
            db2lin(
                channel_data.ue_received_power_dbm[ue_idx, link_data.in_range[ue_idx]]
                - 30
            )
            * 1000
        )
        p_rx_lin = db2lin(p_rx_dbm - 30) * 1000
        interf_lin = total_pwr_lin - p_rx_lin

        expected_noise_lin = (
            fh_layer.fh_channel_data.assigned_bandwidth_hz[ue_idx]
            * fh_layer.noise_psd_mw_per_hz
        )
        expected_sinr_lin = p_rx_lin / (interf_lin + expected_noise_lin)
        expected_sinr_db = 10 * np.log10(expected_sinr_lin)
        expected_snr_lin = p_rx_lin / expected_noise_lin
        expected_snr_db = 10 * np.log10(expected_snr_lin)

        assert channel_data.sinr_db[ue_idx, s_idx] == pytest.approx(
            expected_sinr_db, abs=0.1
        )
        assert channel_data.sinr_db[ue_idx, s_idx] == pytest.approx(
            expected_sinr_db, abs=0.1
        )
        assert channel_data.snr_db[ue_idx, s_idx] == pytest.approx(
            expected_snr_db, abs=0.1
        )
        assert fh_layer.snr_lin[ue_idx, s_idx] >= fh_layer.sinr_lin[ue_idx, s_idx]


def test_fh_layer_logic_verification():
    """
    Validates the mathematical integration:
    P_R -> Interference -> SINR -> Association -> Throughput -> Fairness.
    """
    # 1. Setup (2 UEs, 1 UAV, 1 BS)
    world_cfg = WorldStateCfg(n_ues=2, n_uavs=1, n_bss=1)
    layer_cfg = FHLayerCfg(
        tx_power_dbm_a2g=23.0,
        tx_power_dbm_g2g=46.0,
        user_bandwidth_hz=500e3,
        sinr_threshold_db=10.0,
    )
    fh_layer = FHLayer(layer_cfg)
    fh_layer.initialize_data_arrays(world_cfg)

    in_range = np.array([[True, True], [True, True]])
    # link_type = np.array([0, 1], dtype=np.uint8)

    # 2. Inject Controlled Path Loss
    # UE0: UAV PL=70, BS PL=120 (UAV wins, BS interference is low)
    # UE1: UAV PL=110, BS PL=70 (BS wins, UAV interference is low)
    path_loss_input = np.array([[70.0, 120.0], [110.0, 70.0]], dtype=np.float32)
    fh_layer.fh_channel_data.path_loss_db = path_loss_input

    # 3. Execute Pipeline
    fh_layer._calculate_received_power()
    fh_layer._calculate_interference(in_range)
    fh_layer._calculate_snr()
    fh_layer._calculate_sinr()
    fh_layer._associate_max_sinr(in_range)
    fh_layer._calculate_throughput()
    fh_layer._calculate_fairness()

    # --- ASSERTIONS ---

    # A. Check Association
    # UE0 -> UAV (Index 0), UE1 -> BS (Index 1)
    assert fh_layer.serving_mask[0, 0]
    assert fh_layer.serving_mask[1, 1]

    # B. Check SINR vs Threshold (Internal Logic)
    # UE0 SINR is ~27dB, UE1 SINR is ~93dB. Both > 10dB threshold.
    # Therefore, both must be served.
    expected_noise_lin = (
        fh_layer.fh_channel_data.assigned_bandwidth_hz[:, np.newaxis]
        * fh_layer.noise_psd_mw_per_hz
    )
    expected_snr_lin = fh_layer.rx_power_lin / expected_noise_lin
    assert np.allclose(fh_layer.snr_lin, expected_snr_lin)
    assert np.all(fh_layer.snr_lin >= fh_layer.sinr_lin)

    # C. Test Throughput Synchronization
    actual_sinr_ue1 = fh_layer.sinr_lin[1, 1]
    expected_thr_ue1 = np.uint32(500e3 * np.log2(1.0 + actual_sinr_ue1))
    obtained_thr_ue1 = fh_layer.fh_channel_data.throughput_bps[1, 1]
    assert float(obtained_thr_ue1) == pytest.approx(float(expected_thr_ue1), abs=2)

    # D. Test Fairness
    # Since both users are served in this timestep, coverage scores are equal.
    # Jain's Index for equal scores must be 1.0.
    assert fh_layer.total_fairness == pytest.approx(1.0)


def test_fh_layer_unserved_outage():
    """
    Validates behavior when SINR is below the threshold.
    """
    world_cfg = WorldStateCfg(n_ues=1, n_uavs=1, n_bss=0)
    # Set threshold very high (50dB) so user is in outage
    layer_cfg = FHLayerCfg(sinr_threshold_db=50.0)
    fh_layer = FHLayer(layer_cfg)
    fh_layer.initialize_data_arrays(world_cfg)

    # Force SINR to ~0.4 dB (1.1 linear)
    fh_layer.sinr_lin[0, 0] = 1.1
    fh_layer.serving_mask[0, 0] = True

    fh_layer._calculate_throughput()
    fh_layer._calculate_fairness()

    # 1. Fairness should be 0 because the user did not clear the 50dB threshold
    assert fh_layer.total_fairness == 0.0

    # 2. user_is_served should preserve its meaning after fairness evaluation.
    assert not fh_layer.user_is_served[0, 0]


def test_fh_layer_serving_link_only_controls_service_state():
    """Validates that only the associated link can mark a user as served."""
    world_cfg = WorldStateCfg(n_ues=1, n_uavs=2, n_bss=0)
    fh_layer = FHLayer(FHLayerCfg(sinr_threshold_db=10.0))
    fh_layer.initialize_data_arrays(world_cfg)

    fh_layer.sinr_lin[:] = np.array([[1.0, 1000.0]], dtype=np.float32)
    fh_layer.serving_mask[:] = np.array([[True, False]], dtype=np.bool_)

    fh_layer._calculate_fairness()

    assert fh_layer.total_fairness == 0.0
    assert not fh_layer.user_is_served[0, 0]
    assert np.all(fh_layer.n_unserved_users == np.array([1, 0], dtype=np.int32))
    assert not fh_layer.user_is_served[0, 0]


def test_fh_layer_interference_summation():
    """Validates that interference is the sum of other in-range sources."""
    world_cfg = WorldStateCfg(n_ues=1, n_uavs=2, n_bss=0)
    fh_layer = FHLayer(FHLayerCfg())
    fh_layer.initialize_data_arrays(world_cfg)

    # UE0 receives -50dBm from UAV0 and -50dBm from UAV1
    fh_layer.fh_channel_data.ue_received_power_dbm = np.array(
        [[-50.0, -50.0]], dtype=np.float32
    )
    np.multiply(
        fh_layer.fh_channel_data.ue_received_power_dbm, 0.1, out=fh_layer.rx_power_lin
    )
    np.power(10.0, fh_layer.rx_power_lin, out=fh_layer.rx_power_lin)
    in_range = np.array([[True, True]])

    fh_layer._calculate_interference(in_range)

    # For UAV0, interference should be exactly the power of UAV1
    # P_R = -50dBm -> 1e-5 mW.
    # Interference_dbm should be -50.0
    assert fh_layer.fh_channel_data.interference_dbm[0, 0] == pytest.approx(
        -50.0, abs=0.1
    )


def test_fh_layer_association_clamping():
    """Tests that users out of range are not associated even with high SINR."""
    world_cfg = WorldStateCfg(n_ues=1, n_uavs=1, n_bss=0)
    layer_cfg = FHLayerCfg()
    fh_layer = FHLayer(layer_cfg)
    fh_layer.initialize_data_arrays(world_cfg)

    # Force high SINR but set in_range to False
    fh_layer.sinr_lin = np.array([[1000.0]], dtype=np.float32)
    in_range = np.array([[False]])

    fh_layer._associate_max_sinr(in_range)
    assert not fh_layer.serving_mask[0, 0]

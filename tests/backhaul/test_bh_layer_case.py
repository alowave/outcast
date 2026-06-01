import numpy as np

from src.outcast.backhaul.bh_config import BHLayerCfg
from src.outcast.backhaul.bh_layer import BHLayer
from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
from src.outcast.world.world_state import WorldState, WorldStateCfg


def test_bh_precalculated_case():
    # Small deterministic network:
    # UAV0 -> UAV1 -> BS0

    world_cfg = WorldStateCfg(n_ues=0, n_uavs=2, n_bss=1)

    ws = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=np.array(
            [
                [0.0, 0.0, 80.0],  # UAV0
                [100.0, 0.0, 80.0],  # UAV1
            ],
            dtype=np.float32,
        ),
        bs_pos=np.array(
            [
                [200.0, 0.0, 25.0],  # BS0
            ],
            dtype=np.float32,
        ),
    )

    ll = MockLinkLayer(LinkLayerCfg(backhaul_range_m=300.0))
    ll.initialize_data_arrays(world_cfg)
    ll.update(ws)

    bh_layer = BHLayer(BHLayerCfg())
    bh_layer.initialize_data_arrays(world_cfg)
    bh_layer.update_bh_channel_data(ll.backhaul_data)

    bh_data = bh_layer.bh_channel_data
    assert bh_data is not None

    assert np.isclose(ll.backhaul_data.dist_m[0, 1], 100.0, atol=1e-3)
    assert np.isclose(
        ll.backhaul_data.dist_m[1, 2], np.sqrt(100.0**2 + 55.0**2), atol=1e-3
    )
    assert np.isclose(
        ll.backhaul_data.dist_m[0, 2], np.sqrt(200.0**2 + 55.0**2), atol=1e-3
    )

    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 2] = True

    # Manual flow from hand calculation
    flow = np.zeros((3, 3), dtype=np.uint64)
    flow[0, 1] = 10
    flow[1, 2] = 30

    bh_layer.set_flow(flow, adjacency)

    # Overwrite capacities with deterministic values for exact checking
    bh_data.throughput_bps.fill(0)
    bh_data.throughput_bps[0, 1] = 15
    bh_data.throughput_bps[1, 2] = 25

    bh_layer.compute_excess_and_missing()

    # Manual expected results
    # link 0->1: capacity 15, flow 10 => excess 5, missing 0
    # link 1->2: capacity 25, flow 30 => excess 0, missing 5
    assert bh_data.excess_bps[0, 1] == 5
    assert bh_data.missing_bps[0, 1] == 0

    assert bh_data.excess_bps[1, 2] == 0
    assert bh_data.missing_bps[1, 2] == 5

    unused_links = ~adjacency
    assert np.all(bh_data.excess_bps[unused_links] == 0)
    assert np.all(bh_data.missing_bps[unused_links] == 0)

    # Conservation check:
    g_load = np.array([10, 20, 0], dtype=np.uint64)

    # BS0 is a sink, so ignore it in the conservation check
    is_conserved, leak, absolute_leak = bh_layer.check_flow_conservation(
        g_load,
        ignore_nodes=[2],
    )

    assert is_conserved
    assert absolute_leak == 0
    assert np.allclose(leak, 0)


def test_bh_precalculated_case_branching():
    # Topology:
    # UAV0 -> UAV2
    # UAV1 -> UAV2
    # UAV2 -> BS0
    #
    # Loads:
    # UAV0 = 10
    # UAV1 = 20
    # UAV2 = 5
    # BS0  = 0
    # BS1  = 0
    #
    # Manual flows:
    # f(0,2) = 10
    # f(1,2) = 20
    # f(2,3) = 10 + 20 + 5 = 35

    world_cfg = WorldStateCfg(n_ues=0, n_uavs=3, n_bss=2)

    ws = WorldState(
        ue_pos=np.empty((0, 3), dtype=np.float32),
        uav_pos=np.array(
            [
                [0.0, 0.0, 80.0],  # UAV0
                [0.0, 100.0, 80.0],  # UAV1
                [100.0, 50.0, 80.0],  # UAV2
            ],
            dtype=np.float32,
        ),
        bs_pos=np.array(
            [
                [200.0, 50.0, 25.0],  # BS0
                [300.0, 50.0, 25.0],  # BS1
            ],
            dtype=np.float32,
        ),
    )

    ll = MockLinkLayer(LinkLayerCfg(backhaul_range_m=400.0))
    ll.initialize_data_arrays(world_cfg)
    ll.update(ws)

    bh_layer = BHLayer(BHLayerCfg())
    bh_layer.initialize_data_arrays(world_cfg)
    bh_layer.update_bh_channel_data(ll.backhaul_data)

    bh_data = bh_layer.bh_channel_data
    assert bh_data is not None

    n = world_cfg.n_uavs + world_cfg.n_bss
    assert bh_data.path_loss_db.shape == (n, n)
    assert bh_data.received_power_dbm.shape == (n, n)
    assert bh_data.throughput_bps.shape == (n, n)

    # Manual adjacency:
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    adjacency = np.zeros((5, 5), dtype=bool)
    adjacency[0, 2] = True
    adjacency[1, 2] = True
    adjacency[2, 3] = True

    flow = np.zeros((5, 5), dtype=np.uint64)
    flow[0, 2] = 10
    flow[1, 2] = 20
    flow[2, 3] = 35

    bh_layer.set_flow(flow, adjacency)

    # Deterministic capacities for exact checking
    bh_data.throughput_bps.fill(0)
    bh_data.throughput_bps[0, 2] = 15  # excess = 5
    bh_data.throughput_bps[1, 2] = 10  # missing = 10
    bh_data.throughput_bps[2, 3] = 40  # excess = 5

    bh_layer.compute_excess_and_missing()

    # Expected excess / missing
    assert bh_data.excess_bps[0, 2] == 5
    assert bh_data.missing_bps[0, 2] == 0

    assert bh_data.excess_bps[1, 2] == 0
    assert bh_data.missing_bps[1, 2] == 10

    assert bh_data.excess_bps[2, 3] == 5
    assert bh_data.missing_bps[2, 3] == 0

    # Unused links
    unused_links = ~adjacency
    assert np.all(bh_data.excess_bps[unused_links] == 0)
    assert np.all(bh_data.missing_bps[unused_links] == 0)

    # Conservation:
    # g_load = traffic generated at nodes
    g_load = np.array([10, 20, 5, 0, 0], dtype=np.uint64)

    # BS0 and BS1 are sinks / gateways
    is_conserved, leak, absolute_leak = bh_layer.check_flow_conservation(
        g_load,
        ignore_nodes=[3, 4],
    )

    assert is_conserved
    assert absolute_leak == 0
    assert np.allclose(leak, 0)


def test_bh_precalculated_large_chain_case():
    # Larger deterministic testcase based on a manually calculated path.
    #
    # Path:
    # n0 -> n1 -> n2 -> n3 -> n4 -> n5 -> n6 -> n7
    #
    # Manual accumulated loads on links:
    #  520, 1080, 1920, 2760, 3560, 4440, 5760
    #
    # Manual link capacities:
    #  5624, 5509, 5120, 5718, 5474, 4599, 6594
    #
    # Manual node loads are increments:
    # [520, 560, 840, 840, 800, 880, 1320, 0]

    world_cfg = WorldStateCfg(n_ues=0, n_uavs=7, n_bss=1)

    bh_layer = BHLayer(BHLayerCfg())
    bh_layer.initialize_data_arrays(world_cfg)

    bh_data = bh_layer.bh_channel_data
    assert bh_data is not None

    n = world_cfg.n_uavs + world_cfg.n_bss
    assert n == 8

    adjacency = np.zeros((n, n), dtype=bool)
    flow = np.zeros((n, n), dtype=np.uint64)

    # Chain adjacency
    adjacency[0, 1] = True
    adjacency[1, 2] = True
    adjacency[2, 3] = True
    adjacency[3, 4] = True
    adjacency[4, 5] = True
    adjacency[5, 6] = True
    adjacency[6, 7] = True

    # Manual accumulated flows
    flow[0, 1] = 520
    flow[1, 2] = 1080
    flow[2, 3] = 1920
    flow[3, 4] = 2760
    flow[4, 5] = 3560
    flow[5, 6] = 4440
    flow[6, 7] = 5760

    bh_layer.set_flow(flow, adjacency)

    # Manual capacities from the figure
    bh_data.throughput_bps.fill(0)
    bh_data.throughput_bps[0, 1] = 5624
    bh_data.throughput_bps[1, 2] = 5509
    bh_data.throughput_bps[2, 3] = 5120
    bh_data.throughput_bps[3, 4] = 5718
    bh_data.throughput_bps[4, 5] = 5474
    bh_data.throughput_bps[5, 6] = 4599
    bh_data.throughput_bps[6, 7] = 6594

    bh_layer.compute_excess_and_missing()

    # Expected excess / missing
    assert bh_data.excess_bps[0, 1] == 5624 - 520
    assert bh_data.missing_bps[0, 1] == 0

    assert bh_data.excess_bps[1, 2] == 5509 - 1080
    assert bh_data.missing_bps[1, 2] == 0

    assert bh_data.excess_bps[2, 3] == 5120 - 1920
    assert bh_data.missing_bps[2, 3] == 0

    assert bh_data.excess_bps[3, 4] == 5718 - 2760
    assert bh_data.missing_bps[3, 4] == 0

    assert bh_data.excess_bps[4, 5] == 5474 - 3560
    assert bh_data.missing_bps[4, 5] == 0

    assert bh_data.excess_bps[5, 6] == 4599 - 4440
    assert bh_data.missing_bps[5, 6] == 0

    assert bh_data.excess_bps[6, 7] == 6594 - 5760
    assert bh_data.missing_bps[6, 7] == 0

    # Unused links must remain zero
    unused_links = ~adjacency
    assert np.all(bh_data.excess_bps[unused_links] == 0)
    assert np.all(bh_data.missing_bps[unused_links] == 0)

    # Manual node loads
    g_load = np.array([520, 560, 840, 840, 800, 880, 1320, 0], dtype=np.uint64)

    # Last node is the BS sink
    is_conserved, leak, absolute_leak = bh_layer.check_flow_conservation(
        g_load,
        ignore_nodes=[7],
    )

    assert is_conserved
    assert absolute_leak == 0
    assert np.allclose(leak, 0)

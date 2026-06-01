import numpy as np

from src.outcast.backhaul.bh_ctrl import BHController
from src.outcast.backhaul.bh_layer import BHLayer
from src.outcast.link_layer.mock_link_layer import MockLinkLayer
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldStateCfg


def test_bh_layer_1():
    world_cfg = WorldStateCfg()
    ws = WorldController(world_cfg=world_cfg).state
    ll = MockLinkLayer()
    ll.initialize_data_arrays(world_cfg)
    ll.update(ws)

    bh_layer = BHLayer()
    bh_layer.initialize_data_arrays(world_cfg)
    bh_layer.update_bh_channel_data(ll.backhaul_data)

    bh_data = bh_layer.bh_channel_data
    assert bh_data is not None

    n = world_cfg.n_uavs + world_cfg.n_bss

    assert bh_data.path_loss_db.shape == (n, n)
    assert bh_data.received_power_dbm.shape == (n, n)
    assert bh_data.throughput_bps.shape == (n, n)

    in_range = ll.backhaul_data.in_range

    mask = in_range.copy()
    np.fill_diagonal(mask, False)

    assert np.all(bh_data.throughput_bps[~mask] == 0)
    assert np.all(np.isfinite(bh_data.throughput_bps[mask]))
    assert np.all(bh_data.throughput_bps[mask] >= 0)


def test_bh_layer_flow_conservation():
    world_cfg = WorldStateCfg()
    world_ctrl = WorldController(world_cfg=world_cfg)
    world_ctrl.init_random()

    ll = MockLinkLayer()
    ll.initialize_data_arrays(world_cfg)
    ll.update(world_ctrl.state)

    bh_layer = BHLayer()
    bh_layer.initialize_data_arrays(world_cfg)
    bh_layer.update_bh_channel_data(ll.backhaul_data)

    bh_ctrl = BHController(bh_layer=bh_layer)

    world_ctrl.randomize_loads(100e9, 500e9)
    bh_ctrl.randomize_connections(backhaul_link_data=ll.backhaul_data)
    bh_ctrl.distribute_random_load(g_load=world_ctrl.state.gn_load)
    bh_layer.set_flow(
        bh_ctrl.backhaul_outflow_matrix_bps,
        bh_ctrl.backhaul_adjacency_matrix,
    )
    bh_layer.compute_excess_and_missing()
    is_conserved, leak, absolute_leak = bh_layer.check_flow_conservation(
        world_ctrl.state.gn_load
    )

    unused_links = ~bh_ctrl.backhaul_adjacency_matrix
    assert np.all(bh_layer.bh_channel_data.excess_bps[unused_links] == 0)
    print("is_conserved:", is_conserved)
    print("absolute_leak:", absolute_leak)
    print("leak:", leak)

    assert leak.shape[0] == world_cfg.n_uavs + world_cfg.n_bss
    print("gn_load:", world_ctrl.state.gn_load)
    print("outflow sum per node:", bh_layer.bh_channel_data.flow_bps.sum(axis=1))
    print("inflow sum per node:", bh_layer.bh_channel_data.flow_bps.sum(axis=0))

    external_injection = world_ctrl.state.gn_load.copy()
    external_injection[world_cfg.n_uavs :] = 0
    is_conserved, leak, absolute_leak = bh_layer.check_flow_conservation(
        external_injection
    )

    print("is_conserved:", is_conserved)
    print("absolute_leak:", absolute_leak)
    print("leak:", leak)
    assert leak.shape[0] == world_cfg.n_uavs + world_cfg.n_bss
    print("gn_load:", world_ctrl.state.gn_load)

    # for min_load, max_load in zip(np.arange(1e9, 500e9, 1e6),
    #                               np.arange(500e9, 2000e9, 1e6)):
    #     world_ctrl.randomize_loads(min_load, max_load)
    #     bh_ctrl.randomize_connections(backhaul_link_data=ll.backhaul_data)
    #     bh_ctrl.distribute_random_load(g_load=world_ctrl.state.gn_load)
    #     bh_layer.set_flow(bh_ctrl.backhaul_outflow_matrix_bps)
    #     bh_layer.compute_excess_and_missing()
    #     is_conserved, leak, absolute_leak = bh_layer.check_flow_conservation(world_ctrl.state.gn_load)

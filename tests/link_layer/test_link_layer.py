from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
from src.outcast.world.world_ctrl import WorldController
from src.outcast.world.world_state import WorldStateCfg


def test_link_layer_1():
    world_cfg = WorldStateCfg()
    ws = WorldController(world_cfg=world_cfg).state
    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.update(ws)
    assert 1
    # TODO: continue

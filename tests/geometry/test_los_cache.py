"""
Tests for src.outcast.geometry.los_cache
==========================================

Coverage
--------
- LosCacheCfg defaults and custom values
- LosSectorCache initialisation (sector counts, preallocated array shapes)
- _positions_to_sectors: correct mapping, boundary clamping
- _sector_centre: correct world-space centre coordinates
- Cache hit / miss logic and symmetric write-back
- compute_los_mask: no-obstacle field, single blocker, partial blocking, output shape
- reset() clears both arrays
- MockLinkLayer.set_los_cache integration: real LoS replaces random mock
"""

from __future__ import annotations

import math

import numpy as np

from src.outcast.geometry.los_cache import (
    LosCacheCfg,
    LosSectorCache,
)
from src.outcast.geometry.obstacle import Obstacle

# ---------------------------------------------------------------------------
# Shared test geometry
# ---------------------------------------------------------------------------

# A 10 m × 10 m square obstacle centred near (50, 50) in world space.
# Vertices: (45,45), (55,45), (55,55), (45,55)
_BLOCK_VERTICES = [(45.0, 45.0), (55.0, 45.0), (55.0, 55.0), (45.0, 55.0)]
_BLOCK_HEIGHT = 20.0


def _make_obstacle(vertices=_BLOCK_VERTICES, height=_BLOCK_HEIGHT) -> Obstacle:
    return Obstacle(obstacle_id=0, height=height, vertices=vertices)


def _make_cache(
    sector_size: float = 100.0,
) -> LosSectorCache:
    return LosSectorCache(LosCacheCfg(sector_size_m=sector_size))


def _pos(x: float, y: float, z: float = 0.0) -> np.ndarray:
    """Return a (1, 3) float32 position array."""
    return np.array([[x, y, z]], dtype=np.float32)


# ---------------------------------------------------------------------------
# LosCacheCfg
# ---------------------------------------------------------------------------


def test_los_cache_cfg_defaults():
    cfg = LosCacheCfg()

    assert cfg.sector_size_m == 100.0


def test_los_cache_cfg_custom():
    cfg = LosCacheCfg(sector_size_m=50.0)

    assert cfg.sector_size_m == 50.0


# ---------------------------------------------------------------------------
# LosSectorCache – initialisation
# ---------------------------------------------------------------------------


def test_cache_sector_counts_exact_division():
    # 1000 / 100 = 10 sectors each axis
    cache = _make_cache(sector_size=100.0)

    assert cache._nx == 10
    assert cache._ny == 10


def test_cache_sector_counts_non_exact_division():
    # 1000 / 300 = 3.33 → ceil → 4 sectors
    cache = _make_cache(sector_size=300.0)

    assert cache._nx == 4
    assert cache._ny == 4


def test_cache_preallocated_array_shapes():
    cache = _make_cache(sector_size=100.0)
    expected = (10, 10, 10, 10)

    assert cache._cached.shape == expected
    assert cache._los.shape == expected
    assert cache._cached.dtype == np.bool_
    assert cache._los.dtype == np.bool_


def test_cache_initialised_all_false():
    cache = _make_cache()

    assert not cache._cached.any()
    assert not cache._los.any()


# ---------------------------------------------------------------------------
# _positions_to_sectors
# ---------------------------------------------------------------------------


def test_positions_to_sectors_basic():
    # sector_size=100: x=150 → sector 1, y=250 → sector 2
    cache = _make_cache(sector_size=100.0)
    pos = np.array([[150.0, 250.0, 0.0]], dtype=np.float32)
    sx, sy = cache._positions_to_sectors(pos)

    assert sx[0] == 1
    assert sy[0] == 2


def test_positions_to_sectors_origin_maps_to_zero():
    cache = _make_cache(sector_size=100.0)
    pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    sx, sy = cache._positions_to_sectors(pos)

    assert sx[0] == 0
    assert sy[0] == 0


def test_positions_to_sectors_clamped_at_upper_boundary():
    # x = 1000 → would be sector 10, but max index is 9 → clamp to 9
    cache = _make_cache(sector_size=100.0)
    pos = np.array([[1000.0, 1000.0, 0.0]], dtype=np.float32)
    sx, sy = cache._positions_to_sectors(pos)

    assert sx[0] == 9
    assert sy[0] == 9


def test_positions_to_sectors_clamped_below_zero():
    # Negative coordinates (outside boundary) should clamp to 0
    cache = _make_cache(sector_size=100.0)
    pos = np.array([[-50.0, -1.0, 0.0]], dtype=np.float32)
    sx, sy = cache._positions_to_sectors(pos)

    assert sx[0] == 0
    assert sy[0] == 0


# ---------------------------------------------------------------------------
# _sector_centre
# ---------------------------------------------------------------------------


def test_sector_centre_first_sector():
    cache = _make_cache(sector_size=100.0)
    cx, cy = cache._sector_centre(0, 0)

    assert math.isclose(cx, 50.0, abs_tol=1e-9)
    assert math.isclose(cy, 50.0, abs_tol=1e-9)


def test_sector_centre_arbitrary_sector():
    # sector (3, 5) with size 100 → centre at (350, 550)
    cache = _make_cache(sector_size=100.0)
    cx, cy = cache._sector_centre(3, 5)

    assert math.isclose(cx, 350.0, abs_tol=1e-9)
    assert math.isclose(cy, 550.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Cache hit / miss and symmetric write-back
# ---------------------------------------------------------------------------


def test_cache_miss_then_hit_open_field():
    cache = _make_cache(sector_size=100.0)

    # First call: cache miss for sector (0,0) → (1,0)
    a = _pos(50.0, 50.0)  # sector (0, 0)
    b = _pos(150.0, 50.0)  # sector (1, 0)
    result1 = cache.compute_los_mask(a, b, obstacles=[])

    assert result1[0, 0] is np.bool_(True)
    # Cache should now be populated
    assert cache._cached[0, 0, 1, 0]

    # Second call must return same result (cache hit)
    result2 = cache.compute_los_mask(a, b, obstacles=[])
    assert result2[0, 0] == result1[0, 0]


def test_cache_symmetric_write():
    cache = _make_cache(sector_size=100.0)
    a = _pos(50.0, 50.0)  # sector (0, 0)
    b = _pos(150.0, 50.0)  # sector (1, 0)

    cache.compute_los_mask(a, b, obstacles=[])

    # Both (0,0,1,0) and (1,0,0,0) should be cached
    assert cache._cached[0, 0, 1, 0]
    assert cache._cached[1, 0, 0, 0]
    assert cache._los[0, 0, 1, 0] == cache._los[1, 0, 0, 0]


def test_cache_hit_count_same_sector_pair():
    """Points in the same sector pair should always hit after first call."""
    cache = _make_cache(sector_size=200.0)  # large sectors → fewer unique pairs
    obstacles = []

    # First call populates sector (0,0) → (1,0)
    cache.compute_los_mask(_pos(10.0, 10.0), _pos(210.0, 10.0), obstacles)

    # Second call with different points in the same sectors is a hit
    cached_before = cache._cached.sum()
    cache.compute_los_mask(_pos(50.0, 50.0), _pos(250.0, 50.0), obstacles)
    cached_after = cache._cached.sum()

    # No new entries should have been added
    assert cached_after == cached_before


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_cache():
    cache = _make_cache()
    cache.compute_los_mask(_pos(50.0, 50.0), _pos(150.0, 50.0), obstacles=[])

    assert cache._cached.any()

    cache.reset()

    assert not cache._cached.any()
    assert not cache._los.any()


# ---------------------------------------------------------------------------
# compute_los_mask – output shape and values
# ---------------------------------------------------------------------------


def test_compute_los_mask_shape():
    cache = _make_cache()
    n_a, n_b = 5, 3
    pos_a = np.zeros((n_a, 3), dtype=np.float32)
    pos_b = np.zeros((n_b, 3), dtype=np.float32)
    mask = cache.compute_los_mask(pos_a, pos_b, obstacles=[])

    assert mask.shape == (n_a, n_b)
    assert mask.dtype == np.bool_


def test_compute_los_mask_open_field_all_true():
    cache = _make_cache()
    pos_a = np.array([[50.0, 50.0, 0.0], [150.0, 50.0, 0.0]], dtype=np.float32)
    pos_b = np.array([[550.0, 50.0, 0.0], [750.0, 800.0, 0.0]], dtype=np.float32)
    mask = cache.compute_los_mask(pos_a, pos_b, obstacles=[])

    assert mask.all(), "Open field with no obstacles should be all-LoS"


def test_compute_los_mask_blocking_obstacle():
    # Obstacle at (45,45)-(55,55).  Sector size 100 m.
    # Sector (0,0) centre = (50,50) — inside the obstacle footprint; the
    # sector-centre ray from (50,50) to itself trivially has LoS.
    # Use a large obstacle that fully blocks sector-centre to sector-centre.
    big_obs = Obstacle(
        obstacle_id=0,
        height=50.0,
        vertices=[(80.0, 0.0), (120.0, 0.0), (120.0, 1000.0), (80.0, 1000.0)],
    )
    # Sector (0,0) centre = (50,50),  sector (1,0) centre = (150,50).
    # The obstacle wall at x=80–120 crosses the ray from 50→150 at y=50.
    cache = _make_cache(sector_size=100.0)
    a = _pos(50.0, 50.0)  # sector (0,0)
    b = _pos(150.0, 50.0)  # sector (1,0)
    mask = cache.compute_los_mask(a, b, obstacles=[big_obs])

    assert not mask[0, 0], "Ray should be blocked by wide vertical obstacle"


def test_compute_los_mask_same_sector_both_points():
    # Two points in the exact same sector → sector-centre to itself → always LoS
    cache = _make_cache(sector_size=100.0)
    a = _pos(10.0, 10.0)  # sector (0,0)
    b = _pos(90.0, 90.0)  # sector (0,0)
    mask = cache.compute_los_mask(a, b, obstacles=[_make_obstacle()])

    assert mask[0, 0], "Same-sector pair maps to same sector centre → LoS with itself"


def test_compute_los_mask_multiple_sources_and_targets():
    cache = _make_cache(sector_size=100.0)

    # 3 sources in sector (0,0); 2 targets: one clear (sector (9,9)), one may vary
    pos_a = np.array(
        [[10.0, 10.0, 0.0], [20.0, 20.0, 0.0], [30.0, 30.0, 0.0]], dtype=np.float32
    )
    pos_b = np.array([[950.0, 950.0, 0.0], [50.0, 50.0, 0.0]], dtype=np.float32)
    mask = cache.compute_los_mask(pos_a, pos_b, obstacles=[])

    assert mask.shape == (3, 2)
    # All sources are in the same sector (0,0); all results for a given target column
    # must be identical (same cached sector pair)
    assert mask[0, 0] == mask[1, 0] == mask[2, 0]
    assert mask[0, 1] == mask[1, 1] == mask[2, 1]


# ---------------------------------------------------------------------------
# MockLinkLayer integration
# ---------------------------------------------------------------------------


def test_mock_link_layer_without_cache_uses_random_los():
    """Without set_los_cache(), the LOS arrays must be filled (not all one value)
    on average across many seeds — but the key property is they are bool arrays."""
    from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
    from src.outcast.world.world_ctrl import WorldController
    from src.outcast.world.world_state import WorldStateCfg

    world_cfg = WorldStateCfg(n_ues=10, n_uavs=3, n_bss=2)
    ws = WorldController(world_cfg=world_cfg).state
    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.update(ws)

    assert ll.fronthaul_data.los is not None
    assert ll.fronthaul_data.los.shape == (10, 5)
    assert ll.fronthaul_data.los.dtype == np.bool_

    assert ll.backhaul_data.los is not None
    assert ll.backhaul_data.los.shape == (5, 5)


def test_mock_link_layer_with_cache_produces_deterministic_los():
    """set_los_cache() makes LOS deterministic: two update() calls with the
    same world state must yield identical LOS arrays."""
    from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
    from src.outcast.world.world_ctrl import WorldController
    from src.outcast.world.world_state import WorldStateCfg

    world_cfg = WorldStateCfg(n_ues=8, n_uavs=2, n_bss=1)
    ws = WorldController(world_cfg=world_cfg).state

    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.set_los_cache(
        LosCacheCfg(sector_size_m=100.0),
        obstacles=[],
    )

    ll.update(ws)
    fh_los_1 = ll.fronthaul_data.los.copy()
    bh_los_1 = ll.backhaul_data.los.copy()

    ll.update(ws)
    fh_los_2 = ll.fronthaul_data.los.copy()
    bh_los_2 = ll.backhaul_data.los.copy()

    np.testing.assert_array_equal(fh_los_1, fh_los_2)
    np.testing.assert_array_equal(bh_los_1, bh_los_2)


def test_mock_link_layer_with_cache_shapes():
    from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
    from src.outcast.world.world_ctrl import WorldController
    from src.outcast.world.world_state import WorldStateCfg

    world_cfg = WorldStateCfg(n_ues=6, n_uavs=3, n_bss=2)
    ws = WorldController(world_cfg=world_cfg).state

    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.set_los_cache(
        LosCacheCfg(sector_size_m=100.0),
        obstacles=[_make_obstacle()],
    )
    ll.update(ws)

    n_s = world_cfg.n_uavs + world_cfg.n_bss  # 5
    assert ll.fronthaul_data.los.shape == (world_cfg.n_ues, n_s)
    assert ll.backhaul_data.los.shape == (n_s, n_s)


def test_mock_link_layer_cache_open_field_all_los():
    """With no obstacles all LoS entries must be True."""
    from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
    from src.outcast.world.world_ctrl import WorldController
    from src.outcast.world.world_state import WorldStateCfg

    world_cfg = WorldStateCfg(n_ues=5, n_uavs=2, n_bss=1)
    ws = WorldController(world_cfg=world_cfg).state

    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.set_los_cache(
        LosCacheCfg(sector_size_m=100.0),
        obstacles=[],
    )
    ll.update(ws)

    assert ll.fronthaul_data.los.all(), "No obstacles → all fronthaul LoS True"
    assert ll.backhaul_data.los.all(), "No obstacles → all backhaul LoS True"


def test_mock_link_layer_set_cache_then_reset_then_update():
    """Calling reset() between updates clears the cache but compute still works."""
    from src.outcast.link_layer.mock_link_layer import LinkLayerCfg, MockLinkLayer
    from src.outcast.world.world_ctrl import WorldController
    from src.outcast.world.world_state import WorldStateCfg

    world_cfg = WorldStateCfg(n_ues=4, n_uavs=2, n_bss=1)
    ws = WorldController(world_cfg=world_cfg).state

    ll = MockLinkLayer(LinkLayerCfg())
    ll.initialize_data_arrays(world_cfg)
    ll.set_los_cache(
        LosCacheCfg(sector_size_m=100.0),
        obstacles=[],
    )

    ll.update(ws)
    ll._los_cache.reset()
    # Second update after reset must still produce valid bool arrays
    ll.update(ws)

    assert ll.fronthaul_data.los.dtype == np.bool_
    assert ll.fronthaul_data.los.all()

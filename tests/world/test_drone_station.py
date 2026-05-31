import numpy as np
import pytest

from src.uavnetsim.geometry.coords import Coords3d
from src.uavnetsim.utils.math_tools import db2lin
from src.uavnetsim.world import (
    DroneStation,
    DroneStationController,
    GoldenPetalGraphUavController,
    SquareGraphUavController,
    UavGraphController,
)
from src.uavnetsim.world.energy_model.uav_battery import BatteryCfg


def test_controller_sets_station_tx_power_from_dbm():
    controller = DroneStationController(
        stations=[
            DroneStation(coords=Coords3d(0.0, 0.0, 10.0)),
            DroneStation(coords=Coords3d(1.0, 0.0, 10.0)),
        ]
    )

    tx_powers_dbm = np.array([30.0, 20.0, 10.0], dtype=np.float32)
    controller.set_tx_powers_dbm(tx_powers_dbm)

    assert controller.stations[0].tx_power == pytest.approx(db2lin(0.0))
    assert controller.stations[1].tx_power == pytest.approx(db2lin(-10.0))
    assert np.array_equal(controller.tx_powers_dbm, tx_powers_dbm[:2])


def test_drone_station_uses_received_power_for_energy_harvesting():
    station = DroneStation(
        battery_cfg=BatteryCfg(skip_energy_charge=False),
        received_power=3.0,
    )
    station.battery.energy_level = 100.0
    station.battery.skip_movement_energy = True

    assert station.update_energy(2.0) is True
    assert station.battery.energy_level == pytest.approx(106.0)
    assert station.battery.received_power == pytest.approx(3.0)


def test_drone_station_controller_steps_movement_and_energy():
    station = DroneStation(
        coords=Coords3d(0.0, 0.0, 10.0),
        battery_cfg=BatteryCfg(skip_energy_charge=True),
    )
    station.set_waypoint(Coords3d(10.0, 0.0, 10.0), travel_duration=2.0)
    energy_before = station.battery.energy_level
    controller = DroneStationController(stations=[station])

    assert controller.step(1.0) == [False]
    assert station.coords == Coords3d(5.0, 0.0, 10.0)
    assert station.battery.energy_level < energy_before

    assert controller.step(1.0) == [True]
    assert station.coords == Coords3d(10.0, 0.0, 10.0)


def test_uav_graph_controller_moves_station_along_path():
    station = DroneStation(
        coords=Coords3d(0.0, 0.0, 10.0),
        battery_cfg=BatteryCfg(skip_energy_charge=True),
    )
    controller = UavGraphController(
        stations=[station],
        paths=[[Coords3d(0.0, 0.0, 10.0), Coords3d(10.0, 0.0, 10.0)]],
        speeds=5.0,
    )

    assert station.next_waypoint == Coords3d(10.0, 0.0, 10.0)
    assert controller.current_waypoint_indices == [2]

    assert controller.step(1.0) == [False]
    assert station.coords == Coords3d(5.0, 0.0, 10.0)

    assert controller.step(1.0) == [True]
    assert station.coords == Coords3d(10.0, 0.0, 10.0)
    assert station.next_waypoint is None


def test_uav_graph_controller_cycles_path():
    station = DroneStation(
        coords=Coords3d(0.0, 0.0, 10.0),
        battery_cfg=BatteryCfg(skip_energy_charge=True),
    )
    controller = UavGraphController(
        stations=[station],
        paths=[[Coords3d(0.0, 0.0, 10.0), Coords3d(10.0, 0.0, 10.0)]],
        speeds=[10.0],
        cyclic=True,
    )

    assert controller.step(1.0) == [True]
    assert station.coords == Coords3d(10.0, 0.0, 10.0)
    assert station.next_waypoint == Coords3d(0.0, 0.0, 10.0)

    assert controller.step(1.0) == [True]
    assert station.coords == Coords3d(0.0, 0.0, 10.0)
    assert station.next_waypoint == Coords3d(10.0, 0.0, 10.0)


def test_uav_graph_controller_rejects_inconsistent_inputs():
    station = DroneStation(coords=Coords3d(0.0, 0.0, 10.0))

    with pytest.raises(ValueError, match="one path per station"):
        UavGraphController(stations=[station], paths=[], speeds=1.0)

    with pytest.raises(ValueError, match="must be positive"):
        UavGraphController(stations=[station], paths=[[station.coords]], speeds=0.0)


def test_square_graph_controller_builds_square_paths_and_moves():
    controller = SquareGraphUavController(
        centers=[Coords3d(100.0, 100.0, 50.0)],
        side_length=20.0,
        speeds=10.0,
    )
    station = controller.stations[0]

    assert controller.paths[0] == [
        Coords3d(90.0, 110.0, 50.0),
        Coords3d(110.0, 110.0, 50.0),
        Coords3d(110.0, 90.0, 50.0),
        Coords3d(90.0, 90.0, 50.0),
    ]
    assert station.coords == Coords3d(90.0, 110.0, 50.0)
    assert station.next_waypoint == Coords3d(110.0, 110.0, 50.0)

    assert controller.step(1.0) == [False]
    assert station.coords == Coords3d(100.0, 110.0, 50.0)

    assert controller.step(1.0) == [True]
    assert station.coords == Coords3d(110.0, 110.0, 50.0)
    assert station.next_waypoint == Coords3d(110.0, 90.0, 50.0)


def test_square_graph_controller_rejects_invalid_side_length():
    with pytest.raises(ValueError, match="side_length must be positive"):
        SquareGraphUavController(
            centers=[Coords3d(100.0, 100.0, 50.0)],
            side_length=0.0,
            speeds=10.0,
        )


def test_golden_petal_graph_controller_builds_symmetric_petal_paths():
    center = Coords3d(300.0, 300.0, 50.0)
    controller = GoldenPetalGraphUavController(
        center=center,
        n_uavs=10,
        outer_radius=120.0,
        petal_half_angle_deg=18.0,
        speed=14.0,
        samples_per_side=8,
        use_golden_start_offsets=False,
    )

    phi = GoldenPetalGraphUavController.PHI
    expected_base = Coords3d(300.0 + 120.0 / phi**2, 300.0, 50.0)
    expected_tip = Coords3d(420.0, 300.0, 50.0)

    assert len(controller.stations) == 10
    assert len(controller.paths) == 10
    assert all(len(path) == 16 for path in controller.paths)
    assert controller.paths[0][0] == expected_base
    assert controller.paths[0][8] == expected_tip
    assert controller.stations[0].coords == controller.paths[0][0]
    assert controller.stations[0].next_waypoint == controller.paths[0][1]


def test_golden_petal_graph_controller_can_stagger_start_phase():
    center = Coords3d(300.0, 300.0, 50.0)
    unstaggered_path = GoldenPetalGraphUavController.build_petal_path(
        center=center,
        petal_idx=1,
        n_petals=10,
        outer_radius=120.0,
        petal_half_angle_deg=18.0,
        samples_per_side=8,
    )
    controller = GoldenPetalGraphUavController(
        center=center,
        n_uavs=10,
        outer_radius=120.0,
        petal_half_angle_deg=18.0,
        speed=14.0,
        samples_per_side=8,
        use_golden_start_offsets=True,
    )

    assert controller.paths[1][0] != unstaggered_path[0]
    assert sorted(controller.paths[1], key=hash) == sorted(unstaggered_path, key=hash)


def test_golden_petal_graph_controller_rejects_invalid_inputs():
    center = Coords3d(300.0, 300.0, 50.0)

    with pytest.raises(ValueError, match="n_uavs"):
        GoldenPetalGraphUavController(center, 0, 120.0, 18.0, 14.0, 8)
    with pytest.raises(ValueError, match="outer_radius"):
        GoldenPetalGraphUavController(center, 10, 0.0, 18.0, 14.0, 8)
    with pytest.raises(ValueError, match="petal_half_angle_deg"):
        GoldenPetalGraphUavController(center, 10, 120.0, 0.0, 14.0, 8)
    with pytest.raises(ValueError, match="samples_per_side"):
        GoldenPetalGraphUavController(center, 10, 120.0, 18.0, 14.0, 1)

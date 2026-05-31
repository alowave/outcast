import pytest

from src.uavnetsim.world.energy_model.uav_battery import Battery, BatteryCfg


def test_battery_implements_battery_api():
    battery = Battery(cfg=BatteryCfg())

    assert battery.energy_level == pytest.approx(799200.0)


def test_battery_step_delegates_energy_update():
    battery = Battery(cfg=BatteryCfg())
    energy_before = battery.energy_level

    assert battery.step(1.0, speed_x=1.0) is True
    assert battery.energy_level < energy_before


def test_battery_uses_generic_received_power_for_charging():
    cfg = BatteryCfg(skip_energy_charge=False)
    battery = Battery(cfg=cfg, starting_energy=100.0)
    battery.skip_movement_energy = True
    energy_before = battery.energy_level

    assert battery.step(2.0, received_power=3.0) is True
    assert battery.energy_level == pytest.approx(energy_before + 6.0)


def test_battery_rejects_negative_step():
    battery = Battery(cfg=BatteryCfg())

    with pytest.raises(ValueError, match="time_step must be non-negative"):
        battery.step(-1.0)

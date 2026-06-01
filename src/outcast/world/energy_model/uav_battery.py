"""UAV Aerodynamic Energy Consumption and Battery Model.

Implements a quadcopter physical energy consumption model based on the framework
published in https://ieeexplore.ieee.org/document/8648498. Models dynamic level
flight power, blade profile drag, vertical ascent/descent power, and communication
circuit overhead under steady velocity states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.outcast.utils.math_tools import wh_to_joules


@dataclass(slots=True)
class BatteryCfg:
    """Default physical specifications and thresholds for the drone's power source."""

    mass_kg: float = 4.0
    propeller_radius_m: float = 0.25
    number_of_propellers: int = 4
    air_density_kg_m3: float = 1.225
    gravitation_acceleration_m_s2: float = 9.80665
    profile_drag_coefficient: float = 0.08
    starting_energy_j: float = field(default_factory=lambda: wh_to_joules(222.0))
    max_energy_j: float = field(default_factory=lambda: wh_to_joules(222.0))
    min_energy_j: float = 0.0
    critical_energy_j: float = field(default_factory=lambda: wh_to_joules(11.0))
    travel_speed_m_s: float = 13.0
    skip_energy_update: bool = False
    skip_energy_charge: bool = True


class Battery:
    rotor_disks_area = None

    def __init__(
        self,
        cfg: BatteryCfg | None = None,
        starting_energy: float | None = None,
    ):
        self.cfg = cfg or BatteryCfg()
        self._energy_level = (
            self.cfg.starting_energy_j if starting_energy is None else starting_energy
        )
        self.uav_mass = self.cfg.mass_kg
        self.set_rotor_disks_area()
        self.recharge_count = 0
        self.received_power = 0.0
        self.skip_movement_energy = False

    @property
    def energy_level(self) -> float:
        return self._energy_level

    @energy_level.setter
    def energy_level(self, value: float) -> None:
        self._energy_level = value

    def step(
        self,
        time_step: float,
        speed_x: float = 0.0,
        speed_y: float = 0.0,
        speed_z: float = 0.0,
        tx_power: float = 0.0,
        received_power: float | None = None,
    ) -> bool:
        """Executes a single simulation tick evaluation, updating the dynamic battery capacity state."""
        if time_step < 0:
            raise ValueError(f"time_step must be non-negative, got {time_step}.")
        return self.update_energy(
            time_step,
            speed_x=speed_x,
            speed_y=speed_y,
            speed_z=speed_z,
            tx_power=tx_power,
            received_power=received_power,
        )

    def update_energy(
        self,
        time_step: float,
        speed_x: float = 0.0,
        speed_y: float = 0.0,
        speed_z: float = 0.0,
        tx_power: float = 0.0,
        received_power: float | None = None,
    ) -> bool:
        """Applies internal power depletion models and updates the remaining joule pools."""
        if self.cfg.skip_energy_update:
            return True
        if received_power is None:
            received_power = self.received_power
        else:
            self.received_power = received_power

        self.discharge_energy(time_step, speed_x, speed_y, speed_z, tx_power)
        self.recharge_energy(time_step, received_power)
        self.energy_level = min(self.energy_level, self.cfg.max_energy_j)

        if self.energy_level <= self.cfg.min_energy_j:
            self.energy_empty()
            return False
        return True

    def energy_empty(self) -> None:
        self.energy_level = self.cfg.starting_energy_j + self.energy_level
        self.recharge_count += 1

    def get_total_energy_consumption(self) -> float:
        return self.recharge_count * self.cfg.starting_energy_j + (
            self.cfg.starting_energy_j - self.energy_level
        )

    def recharge_energy(self, time_step: float, received_power: float) -> None:
        if self.cfg.skip_energy_charge:
            return
        self.energy_level += received_power * time_step

    def discharge_energy(
        self,
        time_step: float,
        speed_x: float = 0.0,
        speed_y: float = 0.0,
        speed_z: float = 0.0,
        tx_power: float = 0.0,
    ) -> None:
        self.energy_level -= (
            self.get_dynamic_consumed_energy(time_step, speed_x, speed_y, speed_z)
            + tx_power * time_step
        )

    def get_level_flight_power(
        self, speed_x: float = 0.0, speed_y: float = 0.0
    ) -> float:
        """Calculates dynamic level-flight aerodynamic power consumption using blade induced thrust profiles."""
        uav_weight = self.uav_mass * self.cfg.gravitation_acceleration_m_s2

        # Hover state execution block
        if abs(speed_x) + abs(speed_y) == 0.0:
            return uav_weight ** (3 / 2) / np.sqrt(
                2 * self.cfg.air_density_kg_m3 * self.rotor_disks_area
            )

        # Active forward velocity execution block
        horizontal_speed = np.sqrt(speed_x**2 + speed_y**2)
        _power = (
            uav_weight**2
            / (np.sqrt(2) * self.cfg.air_density_kg_m3 * self.rotor_disks_area)
            / np.sqrt(
                horizontal_speed**2
                + np.sqrt(
                    horizontal_speed**4
                    + 4
                    * (
                        np.sqrt(
                            uav_weight
                            / (2 * self.cfg.air_density_kg_m3 * self.rotor_disks_area)
                        )
                    )
                    ** 4
                )
            )
        )
        return _power

    def get_vertical_flight_power(self, speed_z: float) -> float:
        return speed_z * self.uav_mass * self.cfg.gravitation_acceleration_m_s2

    def get_blade_drag_power(self, speed_x: float = 0.0, speed_y: float = 0.0) -> float:
        return (
            1
            / 8
            * self.cfg.profile_drag_coefficient
            * self.cfg.air_density_kg_m3
            * self.rotor_disks_area
            * np.sqrt(speed_x**2 + speed_y**2) ** 3
        )

    def set_rotor_disks_area(
        self,
        area: float | None = None,
        propellers_radius: float | None = None,
        number_of_uav_propellers: int | None = None,
    ) -> None:
        if propellers_radius is None:
            propellers_radius = self.cfg.propeller_radius_m
        if number_of_uav_propellers is None:
            number_of_uav_propellers = self.cfg.number_of_propellers
        if area is None:
            self.rotor_disks_area = (
                propellers_radius**2 * np.pi * number_of_uav_propellers
            )
        else:
            self.rotor_disks_area = area

    def get_consumption_power(
        self, speed_x: float = 0.0, speed_y: float = 0.0, speed_z: float = 0.0
    ) -> float:
        _power = self.get_level_flight_power(speed_x, speed_y)
        if speed_x + speed_y != 0.0:
            _power += self.get_blade_drag_power(speed_x, speed_y)
        if speed_z == 0.0:
            return _power
        return _power + self.get_vertical_flight_power(speed_z)

    def get_dynamic_consumed_energy(
        self,
        time_step: float,
        speed_x: float = 0.0,
        speed_y: float = 0.0,
        speed_z: float = 0.0,
    ) -> float:
        if self.skip_movement_energy:
            self.skip_movement_energy = False
            return 0.0
        dynamic_power = self.get_consumption_power(speed_x, speed_y, speed_z)
        return time_step * dynamic_power

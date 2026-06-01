"""UAV movement controllers and drone station management.

This module provides:
- BaseUavController: Abstract base for UAV movement control
- StaticUavController: Controller that keeps UAVs stationary
- DroneStation: Individual drone with battery and movement capabilities
- DroneStationController: Default controller using DroneStation objects
- UavGraphController: Move UAVs along predefined waypoint paths
- SquareGraphUavController: Move UAVs around square paths
- GoldenPetalGraphUavController: Move UAVs in golden-ratio rosette patterns
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from src.outcast.geometry.coords import Coords3d
from src.outcast.utils.math_tools import db2lin
from src.outcast.world.energy_model.uav_battery import Battery, BatteryCfg


class BaseUavController(ABC):
    """Common base class for UAV movement controllers."""

    def __init__(
        self,
        locations: Iterable[Coords3d] | NDArray[np.float32] | Coords3d | None = None,
        stations: list | None = None,
    ) -> None:
        self.stations = stations or []
        self.tx_powers_dbm: NDArray[np.float32] | None = None
        self._locations: list[Coords3d] = []
        if locations is not None:
            self.set_locations(locations)

    @property
    def locations(self) -> list[Coords3d]:
        """Return the current UAV locations."""
        if self.stations:
            return [station.coords for station in self.stations]
        return self._locations

    def get_locations_array(self) -> NDArray[np.float32]:
        """Return UAV locations as an array of shape (n_uavs, 3)."""
        locations_array = np.empty((len(self.locations), 3), dtype=np.float32)
        self.update_locations_array(locations_array)
        return locations_array

    def update_locations_array(self, locations_array: NDArray[np.float32]) -> None:
        """Write current UAV locations into an existing ``(n_uavs, 3)`` array."""
        expected_shape = (len(self.locations), 3)
        if locations_array.shape != expected_shape:
            raise ValueError(
                f"locations_array must have shape {expected_shape}, got {locations_array.shape}."
            )

        for idx, loc in enumerate(self.locations):
            locations_array[idx, 0] = loc.x
            locations_array[idx, 1] = loc.y
            locations_array[idx, 2] = loc.z

    def set_locations(
        self,
        locations: Iterable[Coords3d] | NDArray[np.float32] | Coords3d,
    ) -> None:
        """Set UAV locations from Coords3d objects or arrays."""
        if isinstance(locations, Coords3d):
            self._locations = [locations.copy()]
            return

        if isinstance(locations, np.ndarray):
            locations_array = np.asarray(locations, dtype=np.float32)
            if locations_array.ndim == 1:
                if locations_array.shape != (3,):
                    raise ValueError(
                        f"locations must have shape (3,) or (n, 3), got {locations_array.shape}."
                    )
                self._locations = [Coords3d.from_array(locations_array)]
                return

            if locations_array.ndim == 2 and locations_array.shape[1] == 3:
                self._locations = [Coords3d.from_array(row) for row in locations_array]
                return

            raise ValueError(
                f"locations must have shape (3,) or (n, 3), got {locations_array.shape}."
            )

        self._locations = [loc.copy() for loc in locations]

    def set_tx_powers_dbm(self, tx_powers_dbm: NDArray[np.float32]) -> None:
        """Set per-UAV station transmit powers from FHLayer-style dBm values."""
        tx_powers = np.asarray(tx_powers_dbm, dtype=np.float32).reshape(-1)
        n_stations = len(self.stations)
        if tx_powers.shape[0] < n_stations:
            raise ValueError(
                f"tx_powers_dbm must contain at least {n_stations} values, got {tx_powers.shape[0]}."
            )

        self.tx_powers_dbm = tx_powers[:n_stations].copy()
        for station, tx_power_dbm in zip(self.stations, self.tx_powers_dbm):
            station.tx_power = float(db2lin(float(tx_power_dbm) - 30.0))

    @abstractmethod
    def step(self, time_step: float, *args, **kwargs):
        """Advance UAV movement by one simulation time step."""
        raise NotImplementedError


class StaticUavController(BaseUavController):
    """Default controller that leaves UAV positions unchanged."""

    def step(self, time_step: float) -> None:
        """Advance simulation by one time step without moving UAVs.

        Args:
            time_step: Time elapsed in seconds.

        Raises:
            ValueError: If time_step is negative.
        """
        if time_step < 0:
            raise ValueError(f"time_step must be non-negative, got {time_step}.")
        return None


class DroneStation:
    """Represents a single UAV drone station with battery and movement capabilities.

    Attributes:
        id: Unique identifier for the drone.
        coords: Current 3D position of the drone.
        battery: Battery model for energy tracking.
        tx_power: Transmit power in watts.
        received_power: Power received from other nodes.
        next_waypoint: Target destination for movement.
        speeds: 3D velocity vector.
        speed: Scalar speed magnitude.
        moving_flag: Whether the drone is currently moving.
    """

    def __init__(
        self,
        drone_id: int = 0,
        coords: Coords3d | None = None,
        battery: Battery | None = None,
        battery_cfg: BatteryCfg | None = None,
        tx_power: float = 0.0,
        received_power: float = 0.0,
    ) -> None:
        self.id = drone_id
        self.coords = (coords or Coords3d(0.0, 0.0, 0.0)).copy()
        self.battery = battery or Battery(cfg=battery_cfg)
        self.tx_power = tx_power
        self.received_power = received_power
        self.next_waypoint: Coords3d | None = None
        self.speeds = Coords3d(0.0, 0.0, 0.0)
        self.speed = 0.0
        self.moving_flag = False

    def update_energy(self, time_step: float) -> bool | None:
        """Update battery energy based on current power consumption and movement.

        Args:
            time_step: Time elapsed in seconds.

        Returns:
            Battery status from update_energy call.
        """
        self.battery.received_power = self.received_power
        return self.battery.update_energy(
            time_step,
            speed_x=self.speeds.x,
            speed_y=self.speeds.y,
            speed_z=self.speeds.z,
            tx_power=self.tx_power,
            received_power=self.received_power,
        )

    def move(self, time_step: float) -> bool:
        """Move toward the current waypoint and update energy for this time step."""
        if time_step < 0:
            raise ValueError(f"time_step must be non-negative, got {time_step}.")

        if self.next_waypoint is None or self.next_waypoint == self.coords:
            self.next_waypoint = None
            self.moving_flag = False
            self.speeds = Coords3d(0.0, 0.0, 0.0)
            self.speed = 0.0
            self.update_energy(time_step)
            return True

        if self.speed <= 0:
            raise ValueError("DroneStation speed must be positive while moving.")

        distance = time_step * self.speed
        required_distance = self.coords.get_distance_to(self.next_waypoint)
        arrived, _ = self.coords.update(self.next_waypoint, distance)

        if arrived:
            travel_time = min(required_distance / self.speed, time_step)
            self.update_energy(travel_time)
            self.speed = 0.0
            self.speeds = Coords3d(0.0, 0.0, 0.0)
            self.next_waypoint = None
            self.moving_flag = False

            remaining_time = time_step - travel_time
            if remaining_time > 0:
                self.update_energy(remaining_time)
            return True

        self.moving_flag = True
        self.update_energy(time_step)
        return False

    def set_waypoint(self, waypoint: Coords3d, travel_duration: float) -> None:
        """Set the next waypoint and calculate required speed.

        Args:
            waypoint: Target destination coordinates.
            travel_duration: Time in seconds to reach the waypoint.

        Raises:
            ValueError: If travel_duration is not positive.
        """
        if travel_duration <= 0:
            raise ValueError(
                f"travel_duration must be positive, got {travel_duration}."
            )
        self.next_waypoint = waypoint.copy()
        self.speeds = (self.next_waypoint - self.coords) / travel_duration
        self.speed = float(np.linalg.norm(self.speeds))
        self.moving_flag = True


class DroneStationController(BaseUavController):
    """Default UAV controller backed by DroneStation objects."""

    def __init__(
        self,
        stations: list[DroneStation] | None = None,
        locations=None,
        battery_cfg: BatteryCfg | None = None,
    ) -> None:
        if stations is None:
            if locations is None:
                stations = []
            else:
                locations_array = np.asarray(locations, dtype=np.float32)
                if locations_array.ndim == 1:
                    locations_array = locations_array.reshape(1, 3)
                stations = [
                    DroneStation(
                        drone_id=station_idx,
                        coords=Coords3d.from_array(location),
                        battery_cfg=battery_cfg,
                    )
                    for station_idx, location in enumerate(locations_array)
                ]
        super().__init__(stations=stations)

    def set_received_powers(self, received_powers) -> None:
        """Set received power for each drone station.

        Args:
            received_powers: Array of received power values.

        Raises:
            ValueError: If number of powers doesn't match number of stations.
        """
        powers = np.asarray(received_powers, dtype=np.float32).reshape(-1)
        if powers.shape[0] != len(self.stations):
            raise ValueError(
                f"received_powers must contain {len(self.stations)} values, got {powers.shape[0]}."
            )
        for station, received_power in zip(self.stations, powers):
            station.received_power = float(received_power)

    def step(self, time_step: float) -> list[bool]:
        """Advance all drone stations by one time step.

        Args:
            time_step: Time elapsed in seconds.

        Returns:
            List of arrival status for each station (True if arrived at waypoint).
        """
        return [station.move(time_step) for station in self.stations]


class UavGraphController(DroneStationController):
    """Move UAV stations along predefined waypoint paths."""

    def __init__(
        self,
        stations: list[DroneStation],
        paths: list[list[Coords3d]],
        speeds: list[float] | float,
        cyclic: bool = False,
    ) -> None:
        if len(paths) != len(stations):
            raise ValueError(
                f"paths must contain one path per station, got {len(paths)} paths "
                f"for {len(stations)} stations."
            )

        super().__init__(stations=stations)
        self.paths = [[waypoint.copy() for waypoint in path] for path in paths]
        self.cyclic = cyclic
        self.current_waypoint_indices = [0] * len(stations)

        if isinstance(speeds, (int, float)):
            self.speeds = [float(speeds)] * len(stations)
        else:
            if len(speeds) != len(stations):
                raise ValueError(
                    f"speeds must contain one speed per station, got {len(speeds)} "
                    f"speeds for {len(stations)} stations."
                )
            self.speeds = [float(speed) for speed in speeds]

        for speed in self.speeds:
            if speed <= 0:
                raise ValueError(f"UAV graph speeds must be positive, got {speed}.")

        for station_idx in range(len(self.stations)):
            self._set_next_waypoint(station_idx)

    def _set_next_waypoint(self, station_idx: int) -> None:
        station = self.stations[station_idx]
        path = self.paths[station_idx]
        if not path:
            return

        while True:
            waypoint_idx = self.current_waypoint_indices[station_idx]
            if waypoint_idx >= len(path):
                if not self.cyclic:
                    return
                waypoint_idx = 0
                self.current_waypoint_indices[station_idx] = 0

            next_waypoint = path[waypoint_idx]
            self.current_waypoint_indices[station_idx] += 1

            distance = float(station.coords.get_distance_to(next_waypoint))
            if distance <= 1e-6:
                if not self.cyclic and self.current_waypoint_indices[
                    station_idx
                ] >= len(path):
                    return
                continue

            travel_duration = distance / self.speeds[station_idx]
            station.set_waypoint(next_waypoint, travel_duration)
            return

    def step(self, time_step: float) -> list[bool]:
        """Advance all drone stations and set next waypoints upon arrival.

        Args:
            time_step: Time elapsed in seconds.

        Returns:
            List of arrival status for each station (True if arrived at waypoint).
        """
        arrived_list = super().step(time_step)
        for station_idx, arrived in enumerate(arrived_list):
            if arrived and not self.stations[station_idx].moving_flag:
                self._set_next_waypoint(station_idx)
        return arrived_list


class SquareGraphUavController(UavGraphController):
    """Move each UAV around a square path centered on a configured point."""

    def __init__(
        self,
        centers: Iterable[Coords3d],
        side_length: float,
        speeds: list[float] | float,
        cyclic: bool = True,
        battery_cfg: BatteryCfg | None = None,
    ) -> None:
        center_list = [center.copy() for center in centers]
        paths = [self.build_square_path(center, side_length) for center in center_list]
        stations = [
            DroneStation(
                drone_id=station_idx,
                coords=path[0],
                battery_cfg=battery_cfg,
            )
            for station_idx, path in enumerate(paths)
        ]
        super().__init__(stations=stations, paths=paths, speeds=speeds, cyclic=cyclic)

    @staticmethod
    def build_square_path(center: Coords3d, side_length: float) -> list[Coords3d]:
        """Build a square path centered at the given coordinates.

        Args:
            center: Center point of the square.
            side_length: Length of each side of the square.

        Returns:
            List of four corner coordinates forming the square.

        Raises:
            ValueError: If side_length is not positive.
        """
        if side_length <= 0:
            raise ValueError(f"side_length must be positive, got {side_length}.")

        half_side = side_length / 2.0
        return [
            Coords3d(center.x - half_side, center.y + half_side, center.z),
            Coords3d(center.x + half_side, center.y + half_side, center.z),
            Coords3d(center.x + half_side, center.y - half_side, center.z),
            Coords3d(center.x - half_side, center.y - half_side, center.z),
        ]


class GoldenPetalGraphUavController(UavGraphController):
    """Move UAVs around one shared golden-ratio rosette."""

    PHI = (1.0 + np.sqrt(5.0)) / 2.0

    def __init__(
        self,
        center: Coords3d,
        n_uavs: int,
        outer_radius: float,
        petal_half_angle_deg: float,
        speed: list[float] | float,
        samples_per_side: int,
        cyclic: bool = True,
        use_golden_start_offsets: bool = True,
        battery_cfg: BatteryCfg | None = None,
    ) -> None:
        if n_uavs <= 0:
            raise ValueError(f"n_uavs must be positive, got {n_uavs}.")
        if outer_radius <= 0:
            raise ValueError(f"outer_radius must be positive, got {outer_radius}.")
        if samples_per_side < 2:
            raise ValueError(
                f"samples_per_side must be at least 2, got {samples_per_side}."
            )
        if petal_half_angle_deg <= 0:
            raise ValueError(
                f"petal_half_angle_deg must be positive, got {petal_half_angle_deg}."
            )

        paths = [
            self.build_petal_path(
                center=center,
                petal_idx=petal_idx,
                n_petals=n_uavs,
                outer_radius=outer_radius,
                petal_half_angle_deg=petal_half_angle_deg,
                samples_per_side=samples_per_side,
            )
            for petal_idx in range(n_uavs)
        ]
        if use_golden_start_offsets:
            paths = [
                self.rotate_path_by_golden_offset(path, petal_idx)
                for petal_idx, path in enumerate(paths)
            ]

        stations = [
            DroneStation(
                drone_id=station_idx,
                coords=path[0],
                battery_cfg=battery_cfg,
            )
            for station_idx, path in enumerate(paths)
        ]
        super().__init__(stations=stations, paths=paths, speeds=speed, cyclic=cyclic)

    @classmethod
    def build_petal_path(
        cls,
        center: Coords3d,
        petal_idx: int,
        n_petals: int,
        outer_radius: float,
        petal_half_angle_deg: float,
        samples_per_side: int,
    ) -> list[Coords3d]:
        """Build a petal-shaped path for golden-ratio rosette patterns.

        Args:
            center: Center point of the rosette.
            petal_idx: Index of this petal in the rosette.
            n_petals: Total number of petals in the rosette.
            outer_radius: Maximum radius of the petal.
            petal_half_angle_deg: Half-angle of the petal in degrees.
            samples_per_side: Number of sample points per petal side.

        Returns:
            List of coordinates forming the petal path.
        """
        alpha = 2.0 * np.pi * petal_idx / n_petals
        beta = np.deg2rad(petal_half_angle_deg)
        base = cls.polar(center, outer_radius / cls.PHI**2, alpha)
        left_shoulder = cls.polar(center, outer_radius / cls.PHI, alpha - beta)
        tip = cls.polar(center, outer_radius, alpha)
        right_shoulder = cls.polar(center, outer_radius / cls.PHI, alpha + beta)

        left_side = cls.quadratic_bezier(
            base,
            left_shoulder,
            tip,
            samples_per_side,
        )
        right_side = cls.quadratic_bezier(
            tip,
            right_shoulder,
            base,
            samples_per_side,
        )
        return left_side + right_side

    @staticmethod
    def polar(center: Coords3d, radius: float, angle_rad: float) -> Coords3d:
        """Convert polar coordinates to Cartesian coordinates.

        Args:
            center: Origin point for the polar coordinate system.
            radius: Distance from the center.
            angle_rad: Angle in radians from the positive x-axis.

        Returns:
            Cartesian coordinates at the specified polar position.
        """
        return Coords3d(
            center.x + radius * np.cos(angle_rad),
            center.y + radius * np.sin(angle_rad),
            center.z,
        )

    @staticmethod
    def quadratic_bezier(
        start: Coords3d,
        control: Coords3d,
        end: Coords3d,
        samples: int,
    ) -> list[Coords3d]:
        """Generate points along a quadratic Bezier curve.

        Args:
            start: Starting point of the curve.
            control: Control point determining curve shape.
            end: Ending point of the curve.
            samples: Number of sample points to generate.

        Returns:
            List of coordinates along the Bezier curve.
        """
        points = []
        for t in np.linspace(0.0, 1.0, samples, endpoint=False):
            one_minus_t = 1.0 - t
            points.append(
                Coords3d(
                    one_minus_t**2 * start.x
                    + 2.0 * one_minus_t * t * control.x
                    + t**2 * end.x,
                    one_minus_t**2 * start.y
                    + 2.0 * one_minus_t * t * control.y
                    + t**2 * end.y,
                    start.z,
                )
            )
        return points

    @classmethod
    def rotate_path_by_golden_offset(
        cls,
        path: list[Coords3d],
        petal_idx: int,
    ) -> list[Coords3d]:
        """Rotate a path by a golden-ratio based offset.

        Args:
            path: List of coordinates to rotate.
            petal_idx: Index used to calculate the offset.

        Returns:
            Rotated list of coordinates.
        """
        if not path:
            return path
        golden_angle = 2.0 * np.pi / cls.PHI**2
        offset = int((petal_idx * golden_angle / (2.0 * np.pi)) * len(path))
        offset %= len(path)
        return path[offset:] + path[:offset]


DefaultDroneStationController = DroneStationController
BaseUAVController = BaseUavController

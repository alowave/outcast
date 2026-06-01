"""
Backhaul Module
---------------

This module implements the backhaul layer functionality, including the
FSO-based channel model and computation of achievable link throughput
between base stations under gain and distance restrictions.

For a detailed mathematical description, see:
- docs/backhaul/channel-models.md
- docs/backhaul/overview.md
"""

from __future__ import annotations

from math import erf

import numpy as np

from src.outcast.backhaul.bh_config import BHFsoChannelCfg
from src.outcast.geometry.coords import Coords3d
from src.outcast.utils.math_tools import lin2db


class StatisticalModel:
    """As in https://ieeexplore.ieee.org/document/9040587"""

    """Power and capacity."""

    @staticmethod
    def get_charge_power_and_capacity(
        tx_coords: Coords3d,
        rx_coords: Coords3d,
        cfg: BHFsoChannelCfg | None = None,
        get_gain: bool = False,
        fixed_bw: float | None = None,
    ):
        """
        Compute received charging power and FSO link capacity.
        Args:
            tx_coords (Coords3d): Transmitter coordinates.
            rx_coords (Coords3d): Receiver coordinates.
            cfg (BHFsoChannelCfg | None): FSO channel parameters.
            get_gain (bool): If True, return gain components only.
            fixed_bw (float | None): Override beamwidth.
        Returns:
            tuple:
                If get_gain=False:
                    (received_charge_power, capacity)
                If get_gain=True:
                    (gain_dB, gml, atmospheric_loss, turbulence, responsivity)
        """
        cfg = BHFsoChannelCfg() if cfg is None else cfg
        wavelength = cfg.wavelength_m
        responsivity = cfg.rx_responsivity
        lens_radius = cfg.rx_diameter_m / 2.0
        noise_power = cfg.noise_variance_w
        bandwidth = cfg.bandwidth_hz
        power_split_ratio = cfg.power_split_ratio
        beamwaist_radius = cfg.beamwaist_radius_m
        transmit_power = cfg.tx_power_w

        distance = tx_coords.get_distance_to(rx_coords)
        atmospheric_loss = StatisticalModel.get_atmospheric_loss(
            distance,
            cfg.weather_coeff_per_m,
        )
        atm_turb_induced_fading = 1  # for completeness
        c_n = StatisticalModel.get_refraction_index(tx_coords, rx_coords)
        rho = StatisticalModel.get_coherence_length(
            c_n,
            wavelength,
            distance,
        )  # (2.11) from [1]
        beam_width = StatisticalModel.get_beamwidth(
            distance,
            rho,
            beamwaist_radius,
            wavelength,
        )
        beam_width = fixed_bw or beam_width
        v1 = StatisticalModel.get_v1(beam_width, lens_radius)
        phi, theta = (
            StatisticalModel.get_orientation_angles(tx_coords, rx_coords)
            if not cfg.fixed_orientation
            else (np.pi / 2, np.pi)
        )
        v2 = StatisticalModel.get_v2(v1, phi, theta)
        a0 = StatisticalModel.get_max_fraction_a0(v1, v2)
        lambda_1, lambda_2 = StatisticalModel.get_ig_fluctuations_eigen_values(
            tx_coords,
            rx_coords,
            phi,
            theta,
            distance,
            lens_radius,
        )
        try:
            t1 = StatisticalModel.get_t1(v1)
            t2 = StatisticalModel.get_t2(v2, phi, theta)
            t = (t1 + t2) / 2
            gml = StatisticalModel.get_gml(
                a0,
                t,
                beam_width,
                misalignment=np.sqrt(lambda_1 + lambda_2),
            )
        except Exception:
            gml = 1
            t = np.inf
        gain = gml * atmospheric_loss * atm_turb_induced_fading * responsivity
        received_charge_power = (1 - power_split_ratio) * gain * transmit_power
        if get_gain:
            return (
                lin2db(gain),
                gml,
                atmospheric_loss,
                atm_turb_induced_fading,
                responsivity,
            )

        # (41)
        c = (
            np.exp(1)
            / (2 * np.pi)
            * responsivity**2
            * atmospheric_loss**2
            * (power_split_ratio * transmit_power) ** 2
            / noise_power
        )
        r_max = 0.5 * np.log2(c * a0**2)
        r_delta = 2 / (t * beam_width**2 * np.log(2)) * (lambda_1 + lambda_2)
        capacity = (r_max - r_delta) * bandwidth

        gain_db = lin2db(gain)
        return gain_db, received_charge_power, capacity

    """Refraction index."""

    @staticmethod
    def get_refraction_index(tx_coords: Coords3d, rx_coords: Coords3d):
        """
        Compute refractive index structure parameter Cn².
        Args:
            tx_coords (Coords3d): Transmitter coordinates.
            rx_coords (Coords3d): Receiver coordinates.
        Returns:
            float: Refractive index parameter Cn².
        """
        h_d = (tx_coords.z + rx_coords.z) / 2
        c_0_2 = 1.7 * 10 ** (-14)  # Nominal refractive index on the ground SQUARED
        return c_0_2 * np.exp(-h_d / 100)

    """Coherence length."""

    @staticmethod
    def get_coherence_length(refraction_index, wavelength, distance):
        """
        Compute atmospheric coherence length ρ(d).
        Args:
            refraction_index (float): Cn² value.
            wavelength (float): Optical wavelength.
            distance (float): Link distance.
        Returns:
            float: Coherence length.
        """
        return (0.55 * refraction_index * (2 * np.pi / wavelength) ** 2 * distance) ** (
            -3 / 5
        )  # (4) from [1]

    """Beam width."""

    @staticmethod
    def get_beamwidth(distance, coherence_length, beamwaist_radius, wavelength):
        """
        Compute the FSO beam width at a given propagation distance.
        Implements equation (3) from [1].
        Args:
            distance (float): Link distance d in meters.
            coherence_length (float): Atmospheric coherence length ρ(d) in meters.
            beamwaist_radius (float, optional): Initial beam waist radius ω0 in meters.
            wavelength (float, optional): Optical wavelength λ in meters.
        Returns:
            float: Beam width ω_d at distance d (in meters).
        """
        return beamwaist_radius * np.sqrt(
            1
            + (1 + (2 * beamwaist_radius**2) / coherence_length**2)
            * (wavelength * distance / (np.pi * beamwaist_radius**2)) ** 2,
        )

    """Atmospheric attenuation."""

    @staticmethod
    def get_atmospheric_loss(distance, weather_coefficient):
        """
        Compute atmospheric attenuation loss for an FSO link.
        Implements equation (2) from [1].
        Args:
            distance (float): Link distance d in meters.
            weather_coefficient (float, optional): Weather-dependent attenuation coefficient κ (in dB/m).
        Returns:
            float: Atmospheric attenuation factor h_p (unitless).
        """
        return 10 ** (-weather_coefficient * distance / 10)

    """Max captured power."""

    @staticmethod
    def get_max_fraction_a0(v1, v2):
        """
        Compute the maximum fraction of optical power captured by the receiver.
        Implements equation (8) from [1].
        Args:
            v1 (float): Normalized aperture parameter along the first axis.
            v2 (float): Normalized aperture parameter along the second axis.
        Returns:
            float: Maximum captured power fraction A0 (unitless).
        """
        a0 = erf(v1) * erf(v2)
        return a0

    """Normalized aperture ratio."""

    @staticmethod
    def get_v1(beam_width, lens_radius):
        """
        Compute the normalized aperture parameter v1.
        Implements equation (7) from [1].
        Args:
            lens_radius (float): Receiver lens radius r0 (in meters).
            beam_width (float): Beam width ω_d at distance d (in meters).
        Returns:
            float: Normalized aperture parameter v1 (unitless).
        """
        v1 = lens_radius / beam_width * np.sqrt(np.pi / 2)
        return v1

    """Orientation scaling."""

    @staticmethod
    def get_v2(v1, phi, theta):
        """
        Compute orientation-scaled aperture parameter v2.
        Args:
            v1 (float): Base aperture parameter.
            phi (float): Elevation angle (rad).
            theta (float): Azimuth angle (rad).
        Returns:
            float: Scaled aperture parameter v2.
        """
        v2 = v1 * abs(np.sin(phi) * np.cos(theta))
        return v2

    """Geometric loss."""

    @staticmethod
    def get_gml(a0, t, beam_width, misalignment=0):
        """
        Compute geometric and misalignment loss (GML).
        Implements equation (9) from [1].
        Args:
            a0 (float): Maximum captured power fraction A0.
            t (float): Geometric correction parameter ζ.
            beam_width (float): Beam width ω_d at distance d (in meters).
            misalignment (float, optional): Radial misalignment distance u between beam center and receiver lens (in meters).
        Returns:
            float: Geometric and misalignment loss h_g (unitless).
        """
        gml = a0 * np.exp(-2 * misalignment**2 / (t * beam_width**2))
        return gml

    """Auxiliary parameter t1."""

    @staticmethod
    def get_t1(v1):
        """
        Compute auxiliary parameter t1 used in GML statistical modeling.
        Args:
            v1 (float): Normalized aperture parameter v1 (unitless).
        Returns:
            float: Auxiliary parameter t1 (unitless).
        """
        v1_abs = abs(float(v1))
        if v1_abs <= np.finfo(float).eps:
            return 1.0

        exp_term = np.exp(-(v1_abs**2))
        if exp_term <= np.finfo(float).tiny:
            return np.inf

        return float(np.sqrt(np.pi) * erf(v1_abs) / (2 * v1_abs * exp_term))

    """Auxiliary parameter t2."""

    @staticmethod
    def get_t2(v2, phi, theta):
        """
        Compute auxiliary parameter t2 for the generalized
        misalignment statistical model.
        Args:
            v2 (float): Normalized aperture parameter v2 (unitless).
            phi (float): Elevation orientation angle (radians).
            theta (float): Azimuth orientation angle (radians).
        Returns:
            float: Auxiliary parameter t2 (unitless).
        """
        v2_abs = abs(float(v2))
        if v2_abs <= np.finfo(float).eps:
            return 1.0

        orientation_scale = float(np.sin(phi) ** 2 * np.cos(theta) ** 2)
        if orientation_scale <= np.finfo(float).eps:
            return np.inf

        exp_term = np.exp(-(v2_abs**2))
        if exp_term <= np.finfo(float).tiny:
            return np.inf

        return float(
            np.sqrt(np.pi) * erf(v2_abs) / (2 * v2_abs * exp_term * orientation_scale)
        )

    """Relative coordinates."""

    @staticmethod
    def get_relative_location(tx_coords: Coords3d, rx_coords: Coords3d):
        """
        Compute relative Cartesian coordinates between transmitter and receiver.
        Args:
            tx_coords (Coords3d): 3D coordinates of transmitter.
            rx_coords (Coords3d): 3D coordinates of receiver.
        Returns:
            tuple: Relative coordinates (Δx, Δy, Δz) in meters.
        """
        return (
            tx_coords.x - rx_coords.x,
            tx_coords.y - rx_coords.y,
            tx_coords.z - rx_coords.z,
        )

    """Link orientation angles."""

    @staticmethod
    def get_orientation_angles(tx_coords: Coords3d, rx_coords: Coords3d):
        """
        Compute link orientation angles between transmitter and receiver (misalignment model of [1]).
        Args:
            tx_coords (Coords3d): 3D transmitter coordinates.
            rx_coords (Coords3d): 3D receiver coordinates.
        Returns:
            tuple:
                phi (float): Elevation angle in radians.
                theta (float): Azimuth angle in radians.
        """
        _x, _y, _z = StatisticalModel.get_relative_location(tx_coords, rx_coords)
        phi = np.pi - np.arccos(_z / (np.sqrt(_x**2 + _y**2 + _z**2)))
        if _x > 0:
            theta = np.pi + np.arctan(_y / _x)
        else:
            theta = np.arctan(_y / _x)
        return phi, theta

    """Fluctuation variances."""

    @staticmethod
    def get_fluctuations_variances(distance, lens_radius, sigma=0.2):
        """
        Compute variances of position and orientation fluctuations.
        Args:
            distance (float): Link distance in meters.
            lens_radius (float): Receiver lens radius r0 (in meters).
            sigma (float, optional): Scaling factor controlling fluctuation intensity.
        Returns:
            tuple:
                sigma_x (float): Variance in x-direction.
                sigma_y (float): Variance in y-direction.
                sigma_z (float): Variance in z-direction.
                sigma_phi (float): Variance of elevation angle.
                sigma_theta (float): Variance of azimuth angle.
        """
        sigma_x = sigma * lens_radius * 0.8
        sigma_y = sigma * lens_radius * 0.27
        sigma_z = sigma * lens_radius * 0.53
        sigma_phi = sigma * lens_radius / distance * 0.44
        sigma_theta = sigma * lens_radius / distance * 0.9
        return sigma_x, sigma_y, sigma_z, sigma_phi, sigma_theta

    """Fluctuation constants."""

    @staticmethod
    def get_fluctuations_constants(_x, _y, _z, phi, theta):
        """
        Compute geometric constants used in the covariance matrixderivation of misalignment fluctuations.
        Args:
            _x (float): Relative x-coordinate.
            _y (float): Relative y-coordinate.
            _z (float): Relative z-coordinate.
            phi (float): Elevation angle (radians).
            theta (float): Azimuth angle (radians).
        Returns:
            tuple: Constants (c1, c2, c3, c4, c5).
        """
        """ (21)"""
        c1 = -np.tan(theta)
        c2 = -_x / (np.cos(theta) ** 2)
        c3 = _x / (np.sin(phi) ** 2 * np.cos(theta))
        c4 = -(_x * (1 / np.tan(phi)) * np.tan(theta)) / np.cos(theta)
        c5 = -1 / (np.cos(theta) * np.tan(phi))
        return c1, c2, c3, c4, c5

    """Misalignment eigenvalues."""

    @staticmethod
    def get_ig_fluctuations_eigen_values(
        tx_coords: Coords3d,
        rx_coords: Coords3d,
        phi,
        theta,
        distance,
        lens_radius,
    ):
        """
        Compute eigenvalues of the 2×2 misalignment covariance matrix.
        Args:
            tx_coords (Coords3d): Transmitter coordinates.
            rx_coords (Coords3d): Receiver coordinates.
            phi (float): Elevation angle (rad).
            theta (float): Azimuth angle (rad).
            distance (float): Link distance.
            lens_radius (float): Receiver lens radius.
        Returns:
            np.ndarray: Two eigenvalues of the misalignment covariance matrix.
        """
        x, _, z = StatisticalModel.get_relative_location(tx_coords, rx_coords)
        sigma_x, sigma_y, sigma_z, sigma_phi, sigma_theta = (
            StatisticalModel.get_fluctuations_variances(distance, lens_radius)
        )

        c1, c2, c3, c4, c5 = StatisticalModel.get_fluctuations_constants(
            x,
            0,
            z,
            phi,
            theta,
        )

        sx2 = sigma_x**2
        sy2 = sigma_y**2
        sz2 = sigma_z**2
        sp2 = sigma_phi**2
        st2 = sigma_theta**2
        m11 = sy2 + (c1**2) * sx2 + (c2**2) * st2
        m22 = sz2 + (c3**2) * sp2 + (c4**2) * st2 + (c5**2) * sx2
        m12 = c1 * c5 * sx2 + c2 * c4 * st2

        cov = np.array([[m11, m12], [m12, m22]])

        return np.linalg.eigvalsh(cov)

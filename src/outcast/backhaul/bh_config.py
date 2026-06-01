"""Backhaul Layer Configuration Parameters.

Defines structural configuration dataclasses for Free Space Optics (FSO)
and millimeter-Wave (mmWave) hybrid backhaul channel communication models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.uavnetsim.utils.math_tools import db2lin


@dataclass(slots=True)
class BHFsoChannelCfg:
    """Configuration mapping to Section 1: FSO Channel Model."""

    rx_diameter_m: float = 0.2  # Controls rx_radius r_0 used in Eq (7) & (8)
    divergence_angle_rad: float = 0.06
    rx_responsivity: float = 0.5  # Constant eta (η) used in Eq (1) & (12)
    avg_gml: float = 3.0  # Eq (9): Geometric and misalignment loss baseline
    weather_coeff_per_m: float = 4.3e-4  # Constant kappa (κ) used in Eq (2)
    power_split_ratio: float = 0.005  # Power split for simultaneous data and EH
    energy_harvesting_efficiency: float = 0.2
    tx_power_w: float = 380.0  # Transmit power P_FSO used in Eq (13)
    bandwidth_hz: float = 1e9
    noise_variance_w: float = 0.8e-9
    noise_power_w: float = 1e-6  # Receiver noise power sigma_n^2 used in Eq (13)
    empirical_snr_losses_lin: float = field(default_factory=lambda: float(db2lin(15.0)))
    beamwaist_radius_m: float = 0.25e-3 * 10.0  # Constant omega_0 used in Eq (3)
    wavelength_m: float = 1550e-9  # Laser wavelength lambda (λ) used in Eq (3), (5)
    fixed_orientation: bool = True  # Flag for position/orientation variances in Sec 1.7


@dataclass(slots=True)
class BHMmWaveChannelCfg:
    """Configuration mapping to Section 2: mmWave Channel Modeling."""

    frequency_hz: float = (
        38e9  # Carrier frequency f (in Hz, converted to GHz for Eq 17)
    )
    reference_distance_m: float = 5.0  # Custom close-in reference distance (cf. Eq 18)
    path_loss_exponent: float = 2.13  # Alpha_mm constant specified in Eq (17) & (19)
    efficiency: float = 1.0  # Throughput efficiency eta^B used in Eq (21)
    bandwidth_hz: float = 400e6  # Effective bandwidth B^eff used in Eq (21)
    tx_power_w: float = 1.0  # Transmission power P^TX used in Eq (20) & (21)
    noise_power_w: float = 1e-9  # Noise power sigma^2 used in Eq (21)


@dataclass(slots=True)
class BHLayerCfg:
    """Configuration mapping to Section 2.3: Model Selection."""

    channel_model: int = 0  # Param 'c' in Eq (22): 0 = M_FSO, 1 = M_mmWave
    fso: BHFsoChannelCfg = field(default_factory=BHFsoChannelCfg)
    mmwave: BHMmWaveChannelCfg = field(default_factory=BHMmWaveChannelCfg)

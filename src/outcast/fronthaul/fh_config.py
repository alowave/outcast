"""Fronthaul channel configuration dataclasses and default profile factories."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.outcast.utils.math_tools import db2lin, lin2db, w_to_dbm


def default_env_profiles() -> dict[str, list[float]]:
    """Return default PLOS environment profiles (alpha, beta, gamma) per environment type."""
    return {
        "Suburban": [0.1, 750.0, 8.0],
        "Urban": [0.3, 500.0, 15.0],
        "Dense Urban": [0.5, 300.0, 20.0],
        "Highrise Urban": [0.5, 300.0, 50.0],
    }


def default_avg_loss_profiles() -> dict[str, list[list[float]]]:
    """Return default average loss profiles [center_freq, tolerance, eta_LOS, eta_NLOS] per environment."""
    return {
        "Suburban": [
            [700e6, 100e6, 0.0, 18.0],
            [2e9, 250e6, 0.1, 21.0],
            [5.8e9, 1e9, 0.2, 24.0],
        ],
        "Urban": [
            [700e6, 100e6, 0.6, 17.0],
            [2e9, 250e6, 1.0, 20.0],
            [5.8e9, 1e9, 1.2, 23.0],
        ],
        "Dense Urban": [
            [700e6, 100e6, 1.0, 20.0],
            [2e9, 250e6, 1.6, 23.0],
            [5.8e9, 1e9, 1.8, 23.0],
        ],
        "Highrise Urban": [
            [700e6, 100e6, 1.5, 29.0],
            [2e9, 250e6, 2.4, 34.0],
            [5.8e9, 1e9, 2.5, 41.0],
        ],
    }


@dataclass(slots=True)
class PlosChannelCfg:
    """Configuration for the Probabilistic Line-of-Sight (PLOS) channel model."""

    env_profiles: dict[str, list[float]] = field(default_factory=default_env_profiles)
    avg_loss_profiles: dict[str, list[list[float]]] = field(
        default_factory=default_avg_loss_profiles
    )

    def resolve_env_params(self, env_type: str) -> tuple[float, float, float]:
        """Resolve and return (alpha, beta, gamma) ITU parameters for the given environment type."""
        if env_type not in self.env_profiles:
            raise ValueError(
                f"Undefined environment type: {env_type}. "
                "Choose one of: Suburban, Urban, Dense Urban, Highrise Urban."
            )
        return tuple(self.env_profiles[env_type])

    def resolve_avg_loss(
        self, env_type: str, frequency_hz: float
    ) -> tuple[float, float]:
        """Resolve and return (eta_LOS, eta_NLOS) average loss values for the given environment and frequency."""
        env_rows = self.avg_loss_profiles.get(env_type)
        if not env_rows:
            raise ValueError(
                f"Undefined environment type: {env_type}. "
                "Choose one of: Suburban, Urban, Dense Urban, Highrise Urban."
            )

        for center_freq, tolerance_hz, los_loss_db, nlos_loss_db in env_rows:
            if abs(frequency_hz - center_freq) <= tolerance_hz:
                return los_loss_db, nlos_loss_db

        fallback = min(env_rows, key=lambda row: abs(frequency_hz - row[0]))
        return fallback[2], fallback[3]


@dataclass(slots=True)
class UmaCellularCfg:
    """Configuration for the Urban Macro (UMa) cellular channel model (TR 38.901)."""

    effective_env_height_mode: str = "expected"
    enable_shadowing: bool = False
    shadowing_los_db: float = 4.0
    shadowing_nlos_db: float = 6.0


@dataclass(slots=True)
class FHChannelCfg:
    """Top-level fronthaul channel configuration combining PLOS and UMa sub-configs."""

    env_type: str = "Urban"
    default_ue_height_m: float = 1.5
    default_bs_height_m: float = 25.0
    default_uav_height_m: float = 50.0
    plos: PlosChannelCfg = field(default_factory=PlosChannelCfg)
    uma: UmaCellularCfg = field(default_factory=UmaCellularCfg)


@dataclass(slots=True)
class FHLayerCfg:
    """Configuration for the fronthaul layer."""

    channel_model_a2g: int = 0
    channel_model_g2g: int = 0
    frequency_hz_a2g: float = 2.02e9
    frequency_hz_g2g: float = 2.0e9
    tx_power_dbm_a2g: float = field(default_factory=lambda: w_to_dbm(0.2))
    tx_power_dbm_g2g: float = 46.0
    user_bandwidth_hz: float = 500e3
    drone_bandwidth_hz: float = 20e6
    noise_spectral_density_dbm_per_hz: float = -174.0
    sinr_threshold_db: float = 10.0
    snr_threshold_db: float = 5.0
    coverage_history_window: int = 100
    channel: FHChannelCfg = field(default_factory=FHChannelCfg)

    @property
    def noise_power_rf_w(self) -> float:
        """Compute the thermal noise power in watts for the configured user bandwidth."""
        # TODO: noise should be adapted to assigned bandwidth which might be varying
        noise_power_dbm = self.noise_spectral_density_dbm_per_hz + float(
            lin2db(self.user_bandwidth_hz)
        )
        return float(db2lin(noise_power_dbm - 30.0))

"""
Centralized configuration for Overheating Classifier.

All default parameters are defined here for consistency.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlicingDefaults:
    """Default parameters for STL slicing."""
    voxel_size: float = 0.1  # mm
    layer_thickness: float = 0.04  # mm (40 microns)
    layer_grouping: int = 1


@dataclass
class EnergyModelDefaults:
    """Default parameters for energy accumulation model."""
    dissipation_factor: float = 0.5  # 0.0-1.0, downward heat escape rate
    convection_factor: float = 0.05  # 0.0-0.5, top surface gas cooling
    use_geometry_multiplier: bool = False  # Mode A (False) vs Mode B (True)
    sigma_mm: float = 1.0  # Gaussian sigma for geometry multiplier
    G_max: float = 2.0  # Maximum geometry multiplier value


@dataclass
class RiskThresholds:
    """Default thresholds for risk classification."""
    threshold_medium: float = 0.3  # Below this = LOW
    threshold_high: float = 0.6  # Above this = HIGH


@dataclass
class AppConfig:
    """Application configuration."""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True

    # File handling
    upload_folder: str = "uploads"
    results_folder: str = "results"
    max_file_size_mb: int = 50
    allowed_extensions: tuple = (".stl",)

    # Session management
    session_timeout_seconds: int = 3600  # 1 hour
    max_concurrent_sessions: int = 10

    # Computation limits
    max_voxels_per_dimension: int = 500
    max_layers: int = 10000

    # Defaults
    slicing: SlicingDefaults = field(default_factory=SlicingDefaults)
    energy_model: EnergyModelDefaults = field(default_factory=EnergyModelDefaults)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

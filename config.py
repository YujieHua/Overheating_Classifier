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
    # Laser parameters for Joule calculation
    laser_power: float = 200.0  # W (typical: 100-400W)
    scan_speed: float = 800.0  # mm/s (typical: 500-1500 mm/s)
    hatch_distance: float = 0.1  # mm (typical: 0.08-0.15 mm)


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


# ---------------------------------------------------------------------------
# Thermal simulation additions
# ---------------------------------------------------------------------------

from enum import Enum


class AnalysisMode(Enum):
    ENERGY = "energy"
    THERMAL = "thermal"


@dataclass
class MaterialProperties:
    name: str
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float         # J/(kg·K)
    density: float               # kg/m³
    melting_point: float         # °C
    absorptivity: float          # 0-1


MATERIAL_DATABASE = {
    'SS316L': MaterialProperties('Stainless Steel 316L', 16.2, 500, 7990, 1400, 0.35),
    'Ti64':   MaterialProperties('Ti-6Al-4V',            6.7,  526, 4430, 1660, 0.40),
    'IN718':  MaterialProperties('Inconel 718',          11.4, 435, 8190, 1336, 0.38),
    'AlSi10Mg': MaterialProperties('AlSi10Mg',           147,  963, 2670,  660, 0.30),
}


@dataclass
class ProcessDefaults:
    laser_power: float = 280.0
    scan_velocity: float = 960.0
    hatch_spacing: float = 0.1
    recoat_time: float = 8.0
    scan_efficiency: float = 0.1


@dataclass
class ThermalSimDefaults:
    convection_coefficient: float = 10.0
    substrate_influence: float = 0.3
    warning_fraction: float = 0.8
    critical_fraction: float = 1.0
    preheat_temp: float = 80.0


THERMAL_PARAMETER_RULES = {
    'laser_power':           {'min': 50,   'max': 1000,  'unit': 'W'},
    'scan_velocity':         {'min': 100,  'max': 5000,  'unit': 'mm/s'},
    'hatch_spacing':         {'min': 0.01, 'max': 1.0,   'unit': 'mm'},
    'recoat_time':           {'min': 1,    'max': 60,    'unit': 's'},
    'scan_efficiency':       {'min': 0.01, 'max': 1.0},
    'convection_coefficient':{'min': 0,    'max': 200,   'unit': 'W/m²K'},
    'substrate_influence':   {'min': 0.0,  'max': 1.0},
    'preheat_temp':          {'min': 20,   'max': 500,   'unit': '°C'},
    'thermal_conductivity':  {'min': 1,    'max': 500,   'unit': 'W/mK'},
    'specific_heat':         {'min': 100,  'max': 2000,  'unit': 'J/kgK'},
    'density':               {'min': 1000, 'max': 20000, 'unit': 'kg/m³'},
    'melting_point':         {'min': 200,  'max': 3500,  'unit': '°C'},
    'absorptivity':          {'min': 0.05, 'max': 0.95},
}

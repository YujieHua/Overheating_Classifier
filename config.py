"""
Centralized configuration for Overheating Classifier.

All default parameters are defined here for consistency.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


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


# ---------------------------------------------------------------------------
# Phase 1: Thermal Simulation Infrastructure
# ---------------------------------------------------------------------------

class AnalysisMode(Enum):
    """Analysis mode selector."""
    ENERGY = "energy"
    THERMAL = "thermal"


@dataclass
class MaterialProperties:
    """Physical material properties for thermal simulation."""
    name: str
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float         # J/(kg·K)
    density: float               # kg/m³
    melting_point: float         # °C
    absorptivity: float          # 0-1 (effective absorptivity for LPBF)


# Built-in material database
MATERIAL_DATABASE: Dict[str, MaterialProperties] = {
    'SS316L': MaterialProperties(
        name='Stainless Steel 316L',
        thermal_conductivity=16.2,
        specific_heat=500,
        density=7990,
        melting_point=1400,
        absorptivity=0.35,
    ),
    'Ti64': MaterialProperties(
        name='Ti-6Al-4V',
        thermal_conductivity=6.7,
        specific_heat=526,
        density=4430,
        melting_point=1660,
        absorptivity=0.40,
    ),
    'IN718': MaterialProperties(
        name='Inconel 718',
        thermal_conductivity=11.4,
        specific_heat=435,
        density=8190,
        melting_point=1336,
        absorptivity=0.38,
    ),
    'AlSi10Mg': MaterialProperties(
        name='AlSi10Mg',
        thermal_conductivity=147,
        specific_heat=963,
        density=2670,
        melting_point=660,
        absorptivity=0.30,
    ),
}


@dataclass
class ProcessDefaults:
    """Default LPBF process parameters."""
    laser_power: float = 280.0       # W
    scan_velocity: float = 960.0     # mm/s
    hatch_spacing: float = 0.1       # mm
    recoat_time: float = 8.0         # s
    scan_efficiency: float = 0.1     # calibration factor (fraction of area actually scanned)


@dataclass
class ThermalSimDefaults:
    """Default thermal simulation parameters."""
    convection_coefficient: float = 10.0   # W/(m²·K) — stagnant argon atmosphere
    substrate_influence: float = 0.3       # 0-1: fraction of substrate temp coupling
    warning_fraction: float = 0.8          # fraction of T_melt for WARNING threshold
    critical_fraction: float = 1.0         # fraction of T_melt for CRITICAL threshold
    preheat_temp: float = 80.0             # °C — build chamber preheat temperature


# Parameter validation rules for thermal mode inputs
THERMAL_PARAMETER_RULES: Dict[str, dict] = {
    'laser_power':          {'min': 50,    'max': 1000,   'unit': 'W'},
    'scan_velocity':        {'min': 100,   'max': 5000,   'unit': 'mm/s'},
    'hatch_spacing':        {'min': 0.01,  'max': 1.0,    'unit': 'mm'},
    'recoat_time':          {'min': 1,     'max': 60,     'unit': 's'},
    'scan_efficiency':      {'min': 0.01,  'max': 1.0},
    'convection_coefficient': {'min': 0,   'max': 200,    'unit': 'W/m²K'},
    'substrate_influence':  {'min': 0.0,   'max': 1.0},
    'preheat_temp':         {'min': 20,    'max': 500,    'unit': '°C'},
    'thermal_conductivity': {'min': 1,     'max': 500,    'unit': 'W/mK'},
    'specific_heat':        {'min': 100,   'max': 2000,   'unit': 'J/kgK'},
    'density':              {'min': 1000,  'max': 20000,  'unit': 'kg/m³'},
    'melting_point':        {'min': 200,   'max': 3500,   'unit': '°C'},
    'absorptivity':         {'min': 0.05,  'max': 0.95},
}

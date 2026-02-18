"""
Tests for Phase 1 config additions:
  - MaterialProperties dataclass
  - MATERIAL_DATABASE contents
  - ProcessDefaults / ThermalSimDefaults dataclasses
  - AnalysisMode enum
  - THERMAL_PARAMETER_RULES
"""

import pytest
from config import (
    AnalysisMode,
    MaterialProperties,
    MATERIAL_DATABASE,
    ProcessDefaults,
    ThermalSimDefaults,
    THERMAL_PARAMETER_RULES,
)


# ---------------------------------------------------------------------------
# AnalysisMode enum
# ---------------------------------------------------------------------------

class TestAnalysisMode:
    def test_energy_value(self):
        assert AnalysisMode.ENERGY.value == "energy"

    def test_thermal_value(self):
        assert AnalysisMode.THERMAL.value == "thermal"

    def test_enum_members_count(self):
        assert len(AnalysisMode) == 2

    def test_from_value(self):
        assert AnalysisMode("energy") is AnalysisMode.ENERGY
        assert AnalysisMode("thermal") is AnalysisMode.THERMAL


# ---------------------------------------------------------------------------
# MaterialProperties dataclass
# ---------------------------------------------------------------------------

class TestMaterialProperties:
    def test_instantiation(self):
        mp = MaterialProperties(
            name="Test Material",
            thermal_conductivity=20.0,
            specific_heat=500.0,
            density=8000.0,
            melting_point=1400.0,
            absorptivity=0.35,
        )
        assert mp.name == "Test Material"
        assert mp.thermal_conductivity == 20.0
        assert mp.specific_heat == 500.0
        assert mp.density == 8000.0
        assert mp.melting_point == 1400.0
        assert mp.absorptivity == 0.35

    def test_fields_are_floats(self):
        mp = MaterialProperties("M", 1.0, 2.0, 3.0, 4.0, 0.5)
        assert isinstance(mp.thermal_conductivity, float)
        assert isinstance(mp.absorptivity, float)


# ---------------------------------------------------------------------------
# MATERIAL_DATABASE
# ---------------------------------------------------------------------------

class TestMaterialDatabase:
    EXPECTED_KEYS = {'SS316L', 'Ti64', 'IN718', 'AlSi10Mg'}

    def test_all_materials_present(self):
        assert self.EXPECTED_KEYS.issubset(set(MATERIAL_DATABASE.keys()))

    def test_ss316l_properties(self):
        mat = MATERIAL_DATABASE['SS316L']
        assert mat.thermal_conductivity == pytest.approx(16.2)
        assert mat.specific_heat == pytest.approx(500.0)
        assert mat.density == pytest.approx(7990.0)
        assert mat.melting_point == pytest.approx(1400.0)
        assert mat.absorptivity == pytest.approx(0.35)

    def test_ti64_properties(self):
        mat = MATERIAL_DATABASE['Ti64']
        assert mat.thermal_conductivity == pytest.approx(6.7)
        assert mat.specific_heat == pytest.approx(526.0)
        assert mat.density == pytest.approx(4430.0)
        assert mat.melting_point == pytest.approx(1660.0)
        assert mat.absorptivity == pytest.approx(0.40)

    def test_in718_properties(self):
        mat = MATERIAL_DATABASE['IN718']
        assert mat.thermal_conductivity == pytest.approx(11.4)
        assert mat.density == pytest.approx(8190.0)
        assert mat.melting_point == pytest.approx(1336.0)
        assert mat.absorptivity == pytest.approx(0.38)

    def test_alsi10mg_properties(self):
        mat = MATERIAL_DATABASE['AlSi10Mg']
        assert mat.thermal_conductivity == pytest.approx(147.0)
        assert mat.density == pytest.approx(2670.0)
        assert mat.melting_point == pytest.approx(660.0)
        assert mat.absorptivity == pytest.approx(0.30)

    def test_all_absorptivities_in_range(self):
        for key, mat in MATERIAL_DATABASE.items():
            assert 0.0 < mat.absorptivity < 1.0, f"{key} absorptivity out of range"

    def test_all_values_are_MaterialProperties(self):
        for key, val in MATERIAL_DATABASE.items():
            assert isinstance(val, MaterialProperties), f"{key} is not MaterialProperties"


# ---------------------------------------------------------------------------
# ProcessDefaults dataclass
# ---------------------------------------------------------------------------

class TestProcessDefaults:
    def test_defaults(self):
        p = ProcessDefaults()
        assert p.laser_power == pytest.approx(280.0)
        assert p.scan_velocity == pytest.approx(960.0)
        assert p.hatch_spacing == pytest.approx(0.1)
        assert p.recoat_time == pytest.approx(8.0)
        assert p.scan_efficiency == pytest.approx(0.1)

    def test_custom_values(self):
        p = ProcessDefaults(laser_power=400.0, scan_velocity=1200.0)
        assert p.laser_power == pytest.approx(400.0)
        assert p.scan_velocity == pytest.approx(1200.0)
        # Other fields still at default
        assert p.hatch_spacing == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# ThermalSimDefaults dataclass
# ---------------------------------------------------------------------------

class TestThermalSimDefaults:
    def test_defaults(self):
        t = ThermalSimDefaults()
        assert t.convection_coefficient == pytest.approx(10.0)
        assert t.substrate_influence == pytest.approx(0.3)
        assert t.warning_fraction == pytest.approx(0.8)
        assert t.critical_fraction == pytest.approx(1.0)
        assert t.preheat_temp == pytest.approx(80.0)

    def test_custom_values(self):
        t = ThermalSimDefaults(preheat_temp=200.0, warning_fraction=0.7)
        assert t.preheat_temp == pytest.approx(200.0)
        assert t.warning_fraction == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# THERMAL_PARAMETER_RULES
# ---------------------------------------------------------------------------

class TestThermalParameterRules:
    REQUIRED_KEYS = {
        'laser_power', 'scan_velocity', 'hatch_spacing', 'recoat_time',
        'scan_efficiency', 'convection_coefficient', 'substrate_influence',
        'preheat_temp', 'thermal_conductivity', 'specific_heat',
        'density', 'melting_point', 'absorptivity',
    }

    def test_all_required_keys_present(self):
        assert self.REQUIRED_KEYS.issubset(set(THERMAL_PARAMETER_RULES.keys()))

    def test_each_rule_has_min_and_max(self):
        for key, rule in THERMAL_PARAMETER_RULES.items():
            assert 'min' in rule, f"'{key}' missing 'min'"
            assert 'max' in rule, f"'{key}' missing 'max'"

    def test_min_less_than_max(self):
        for key, rule in THERMAL_PARAMETER_RULES.items():
            assert rule['min'] < rule['max'], (
                f"'{key}': min={rule['min']} >= max={rule['max']}"
            )

    def test_default_process_values_in_range(self):
        p = ProcessDefaults()
        for attr, param_key in [
            ('laser_power',   'laser_power'),
            ('scan_velocity', 'scan_velocity'),
            ('hatch_spacing', 'hatch_spacing'),
            ('recoat_time',   'recoat_time'),
            ('scan_efficiency', 'scan_efficiency'),
        ]:
            val = getattr(p, attr)
            rule = THERMAL_PARAMETER_RULES[param_key]
            assert rule['min'] <= val <= rule['max'], (
                f"{param_key} default {val} not in [{rule['min']}, {rule['max']}]"
            )

    def test_default_thermal_values_in_range(self):
        t = ThermalSimDefaults()
        for attr, param_key in [
            ('convection_coefficient', 'convection_coefficient'),
            ('substrate_influence',    'substrate_influence'),
            ('preheat_temp',           'preheat_temp'),
        ]:
            val = getattr(t, attr)
            rule = THERMAL_PARAMETER_RULES[param_key]
            assert rule['min'] <= val <= rule['max'], (
                f"{param_key} default {val} not in [{rule['min']}, {rule['max']}]"
            )

    def test_material_defaults_in_range(self):
        mat = MATERIAL_DATABASE['SS316L']
        checks = [
            ('thermal_conductivity', mat.thermal_conductivity),
            ('specific_heat',        mat.specific_heat),
            ('density',              mat.density),
            ('melting_point',        mat.melting_point),
            ('absorptivity',         mat.absorptivity),
        ]
        for param_key, val in checks:
            rule = THERMAL_PARAMETER_RULES[param_key]
            assert rule['min'] <= val <= rule['max'], (
                f"{param_key} SS316L value {val} not in [{rule['min']}, {rule['max']}]"
            )

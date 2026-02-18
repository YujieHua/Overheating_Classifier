"""
Energy-Backbone Thermal Simulation Engine.

State variable: E_region [Joules] — excess energy above powder (preheat) temperature.
Temperature is DERIVED from energy:
    T = T_powder + E / (m × cp)

Heat transfer:
    - Inter-layer conduction via Fourier's law (Z-direction only)
    - Convective cooling via Newton's law (exposed surfaces only)
    - Build plate treated as constant-temperature boundary at T_powder

Physical model notes:
    - T_powder (preheat) is the energy reference (E=0 ↔ T_powder)
    - Layer grouping scales all physics correctly: mass, dz, dt, scan time
    - CFL stability: dt = min(0.001, 0.5 × dz²/(2α)), floor at 1e-5 s
    - Safety clamp: |dT/step| ≤ 50°C; energy cannot fall below 0
    - Equilibrium-skip: regions with |dE| < 1e-6 J for 10 steps are skipped

References:
    INTEGRATION_PLAN.md sections 2.1–2.6
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import importlib.util as _ilu
import os as _os

# Import from root config.py directly (avoid src/config/__init__.py shadowing)
_config_path = _os.path.join(_os.path.dirname(__file__), '..', '..', 'config.py')
_spec = _ilu.spec_from_file_location('root_config', _os.path.abspath(_config_path))
_root_config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_root_config)
MaterialProperties = _root_config.MaterialProperties
ProcessDefaults = _root_config.ProcessDefaults
ThermalSimDefaults = _root_config.ThermalSimDefaults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT_MAX = 0.001          # s — maximum allowed time step
DT_MIN = 1e-5           # s — safety floor (AlSi10Mg has very high diffusivity)
DT_CLAMP_DEG = 50.0     # °C — maximum temperature change per time step
MIN_MASS_KG = 1e-15     # kg — mass floor to prevent division by zero
EQ_SKIP_THRESHOLD = 1e-6  # J — |dE| below which a region is considered equilibrated
EQ_SKIP_STEPS = 10      # number of consecutive steps below threshold before skipping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_energy_input(
    regions_per_layer: Dict,
    material_props: MaterialProperties,
    process_params: ProcessDefaults,
    layer_grouping: int,
    voxel_size: float,       # mm (unused here but kept for API consistency)
) -> Dict[Tuple[int, int], float]:
    """
    Compute laser energy deposited into each region per grouped layer.

    The laser scans the region area; scan time scales linearly with
    layer_grouping because each "grouped layer" represents multiple physical
    layers being processed together.

    Parameters
    ----------
    regions_per_layer : dict
        Output of detect_all_layer_regions()
    material_props : MaterialProperties
        Material physical properties (absorptivity used)
    process_params : ProcessDefaults
        Laser and process settings
    layer_grouping : int
        Number of physical layers grouped into one simulation layer
    voxel_size : float
        Voxel size in mm

    Returns
    -------
    dict
        {(layer, region_id): E_in_joules}
    """
    eta = material_props.absorptivity       # dimensionless
    P = process_params.laser_power          # W
    h = process_params.hatch_spacing        # mm
    v = process_params.scan_velocity        # mm/s
    eff = process_params.scan_efficiency    # dimensionless

    E_in_per_region: Dict[Tuple[int, int], float] = {}

    for layer, ldata in regions_per_layer.items():
        for rid, rinfo in ldata['regions'].items():
            A_mm2 = rinfo['area_mm2']
            # scan_time ∝ area / (h × v), scaled by grouping and efficiency
            t_scan = eff * A_mm2 * layer_grouping / (h * v)   # seconds
            E_in = eta * P * t_scan                             # Joules
            E_in_per_region[(layer, rid)] = E_in

    return E_in_per_region


def simulate_thermal_evolution(
    masks: Dict[int, np.ndarray],
    regions_per_layer: Dict,
    conn_below_lookup: Dict,
    conn_above_lookup: Dict,
    E_in_per_region: Dict[Tuple[int, int], float],
    material_props: MaterialProperties,
    process_params: ProcessDefaults,
    thermal_params: ThermalSimDefaults,
    layer_grouping: int,
    voxel_size: float,       # mm
    layer_thickness: float,  # mm (single physical layer)
    snapshot_interval: float = 2.5,  # s — time between stored snapshots
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict:
    """
    Energy-backbone time-stepped thermal simulation.

    Processes layers in build order (1 → n_layers).  For each new layer:
      1. Compute initial energy from weighted blend of substrate and powder temp.
      2. Add laser energy input.
      3. Time-step until recoat time expires (conduction + convection cooling).
      4. Store peak and final temperatures.
      5. Store time-stepped snapshots for playback visualization.

    Parameters
    ----------
    masks : dict
        Binary masks per layer (not used directly in physics, kept for API)
    regions_per_layer : dict
        Per-layer region metadata (pixel_count, area_mm2, …)
    conn_below_lookup : dict
        {(layer, rid): [conn_dict, …]} — connections to layer below
    conn_above_lookup : dict
        {(layer, rid): [conn_dict, …]} — connections to layer above
    E_in_per_region : dict
        Pre-computed laser energy per region from calculate_energy_input()
    material_props : MaterialProperties
        Thermal and physical properties
    process_params : ProcessDefaults
        Process parameters (laser, scan, recoat)
    thermal_params : ThermalSimDefaults
        Thermal simulation settings (h_conv, substrate_influence, …)
    layer_grouping : int
        Grouped-layer count
    voxel_size : float
        Voxel size in mm
    layer_thickness : float
        Single physical layer thickness in mm
    snapshot_interval : float
        Time between stored snapshots in seconds (default 2.5 s)
    progress_callback : callable, optional
        Called with fraction complete [0, 1] after each layer

    Returns
    -------
    dict with keys:
        temperature_per_layer  : {layer: {rid: T_peak_°C}}
        temperature_end        : {layer: {rid: T_end_°C}}
        energy_conservation    : {total_E_in, total_E_dissipated, final_E_stored}
        layer_times            : {layer: total_time_s}
        scan_times             : {layer: scan_time_s}
        melting_detected       : [(layer, rid), …]
        temperature            : {viewing_layer: {t_snap: {actual_layer: max_T}}}
        temperature_regions    : {viewing_layer: {t_snap: {actual_layer: {rid: T}}}}
        heat_flow              : {viewing_layer: {t_snap: {actual_layer: Q_down_W}}}
        cumulative_heat        : {viewing_layer: {t_snap: {actual_layer: cum_J}}}
        contact_areas          : {layer: area_mm2}
        time_steps             : [t0, t1, …]
    """
    # ------------------------------------------------------------------
    # Material & process constants
    # ------------------------------------------------------------------
    k   = material_props.thermal_conductivity   # W/(m·K)
    cp  = material_props.specific_heat          # J/(kg·K)
    rho = material_props.density                # kg/m³
    T_melt   = material_props.melting_point     # °C
    T_powder = thermal_params.preheat_temp      # °C — energy reference
    w        = thermal_params.substrate_influence
    h_conv   = thermal_params.convection_coefficient  # W/(m²·K)

    # Effective layer thickness [m] — accounts for grouping
    dz_eff = layer_thickness * layer_grouping / 1000.0   # mm → m

    # Thermal diffusivity [m²/s]
    alpha = k / (rho * cp)

    # CFL-stable time step
    dt_cfl = 0.5 * dz_eff ** 2 / (2.0 * alpha)
    dt = float(np.clip(dt_cfl, DT_MIN, DT_MAX))
    logger.debug(f"Thermal dt={dt:.2e} s  (CFL={dt_cfl:.2e}, dz_eff={dz_eff:.4f} m)")

    # ------------------------------------------------------------------
    # Pre-compute per-region mass [kg]
    # ------------------------------------------------------------------
    # Volume = pixel_count × voxel_size² × dz_eff  (all in m)
    voxel_m = voxel_size / 1000.0   # mm → m
    region_mass: Dict[Tuple[int, int], float] = {}

    for layer, ldata in regions_per_layer.items():
        for rid, rinfo in ldata['regions'].items():
            vol_m3 = rinfo['pixel_count'] * (voxel_m ** 2) * dz_eff
            mass = max(rho * vol_m3, MIN_MASS_KG)
            region_mass[(layer, rid)] = mass

    # ------------------------------------------------------------------
    # Layer timing
    # ------------------------------------------------------------------
    n_layers = len(regions_per_layer)
    layer_times: Dict[int, float] = {}
    scan_times: Dict[int, float] = {}

    h_sp = process_params.hatch_spacing   # mm
    v_sc = process_params.scan_velocity   # mm/s
    eff  = process_params.scan_efficiency

    for layer in sorted(regions_per_layer.keys()):
        A_total = sum(
            r['area_mm2'] for r in regions_per_layer[layer]['regions'].values()
        )
        t_scan = eff * A_total * layer_grouping / (h_sp * v_sc)
        t_recoat = process_params.recoat_time * layer_grouping
        scan_times[layer] = t_scan
        layer_times[layer] = t_scan + t_recoat

    # ------------------------------------------------------------------
    # Snapshot time steps (shared across all layers for consistent UI slider)
    # ------------------------------------------------------------------
    max_layer_time = max(layer_times.values()) if layer_times else snapshot_interval
    n_snaps = max(1, int(math.ceil(max_layer_time / snapshot_interval)))
    local_time_steps = [round(i * snapshot_interval, 2) for i in range(n_snaps + 1)]
    if local_time_steps[-1] < max_layer_time:
        local_time_steps.append(round(max_layer_time, 2))

    logger.info(
        f"Snapshots: {len(local_time_steps)} per layer, "
        f"interval={snapshot_interval}s, max_time={max_layer_time:.2f}s"
    )

    # ------------------------------------------------------------------
    # Scan / contact areas per layer (mm² — used by evolution graphs)
    # ------------------------------------------------------------------
    contact_areas: Dict[int, float] = {}
    for layer in sorted(regions_per_layer.keys()):
        A_total = sum(
            r['area_mm2'] for r in regions_per_layer[layer]['regions'].values()
        )
        contact_areas[layer] = A_total

    # ------------------------------------------------------------------
    # State: E_region[(layer, rid)] = excess energy [J] above T_powder
    #        E=0 → T = T_powder
    # ------------------------------------------------------------------
    E_region: Dict[Tuple[int, int], float] = {}

    # Helper: derive temperature from stored energy
    def get_temp(l: int, rid: int) -> float:
        E = E_region.get((l, rid), 0.0)
        m = region_mass.get((l, rid), MIN_MASS_KG)
        return T_powder + E / (m * cp)

    # ------------------------------------------------------------------
    # Output storage
    # ------------------------------------------------------------------
    temperature_per_layer: Dict[int, Dict[int, float]] = {}   # peak (t=0 after laser)
    temperature_end: Dict[int, Dict[int, float]] = {}          # after cooling period
    melting_detected: List[Tuple[int, int]] = []

    total_E_in = 0.0
    total_E_dissipated = 0.0  # estimated from convection + build-plate conduction

    # Snapshot storage (mirrors predictor's data structures for API compatibility)
    # temperature[viewing_layer][t_snap] = {actual_layer: max_T}
    temperature_snaps: Dict[int, Dict[float, Dict[int, float]]] = {}
    # temperature_regions[viewing_layer][t_snap] = {actual_layer: {rid: T}}
    temperature_regions_snaps: Dict[int, Dict[float, Dict[int, Dict[int, float]]]] = {}
    # heat_flow[viewing_layer][t_snap] = {actual_layer: Q_down_W}
    heat_flow_snaps: Dict[int, Dict[float, Dict[int, float]]] = {}
    # cumulative_heat[viewing_layer][t_snap] = {actual_layer: cum_J}
    cumulative_heat_snaps: Dict[int, Dict[float, Dict[int, float]]] = {}

    # Per-layer cumulative heat dissipated (running totals, reset each layer processing)
    # We accumulate per-layer Q_net dissipated during each viewing_layer's cooling phase
    # cum_heat[actual_layer] = cumulative J dissipated during current viewing_layer's cooling
    cum_heat_for_layer: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Main loop — process one grouped layer at a time
    # ------------------------------------------------------------------
    layers_sorted = sorted(regions_per_layer.keys())

    def store_snapshot(viewing_layer: int, t_snap: float, active_layers: list) -> None:
        """Capture temperature & heat-flow snapshot for all active layers."""
        ts = temperature_snaps.setdefault(viewing_layer, {})
        tr = temperature_regions_snaps.setdefault(viewing_layer, {})
        hf = heat_flow_snaps.setdefault(viewing_layer, {})
        ch = cumulative_heat_snaps.setdefault(viewing_layer, {})

        ts[t_snap] = {}
        tr[t_snap] = {}
        hf[t_snap] = {}
        ch[t_snap] = {}

        for al in active_layers:
            rids_al = list(regions_per_layer[al]['regions'].keys())
            # Per-region temperatures
            region_temps_al = {rid: get_temp(al, rid) for rid in rids_al}
            max_t = max(region_temps_al.values()) if region_temps_al else T_powder

            ts[t_snap][al] = max_t
            tr[t_snap][al] = region_temps_al

            # Instantaneous downward heat flow (from this layer to below)
            q_total = 0.0
            for rid in rids_al:
                key = (al, rid)
                E_curr = E_region.get(key, 0.0)
                m = region_mass[key]
                T_curr = T_powder + E_curr / (m * cp)
                for c in conn_below_lookup.get(key, []):
                    if al == 1:
                        T_lower = T_powder
                    else:
                        lower_key = (al - 1, c['lower_region'])
                        E_lower = E_region.get(lower_key, 0.0)
                        m_lower = region_mass.get(lower_key, MIN_MASS_KG)
                        T_lower = T_powder + E_lower / (m_lower * cp)
                    A_m2 = c['overlap_area_mm2'] / 1e6
                    q_total += k * A_m2 * (T_curr - T_lower) / dz_eff
            hf[t_snap][al] = q_total

            # Cumulative heat dissipated for this layer so far
            ch[t_snap][al] = cum_heat_for_layer.get(al, 0.0)

    for layer_idx, layer in enumerate(layers_sorted):
        rids = list(regions_per_layer[layer]['regions'].keys())

        # ---- 1. Set initial energy for new layer ----
        for rid in rids:
            connections_below = conn_below_lookup.get((layer, rid), [])
            total_overlap = sum(c['overlap_area_mm2'] for c in connections_below)

            if total_overlap > 0:
                T_below_weighted = sum(
                    get_temp(layer - 1, c['lower_region']) * c['overlap_area_mm2']
                    for c in connections_below
                ) / total_overlap
            else:
                T_below_weighted = T_powder  # floating region or layer 1

            T_initial = (1.0 - w) * T_powder + w * T_below_weighted
            E_initial = region_mass[(layer, rid)] * cp * (T_initial - T_powder)

            E_laser = E_in_per_region.get((layer, rid), 0.0)
            E_region[(layer, rid)] = E_initial + E_laser
            total_E_in += E_laser

        # ---- 2. Store peak snapshot (immediately after laser, t=0) ----
        temperature_per_layer[layer] = {}
        for rid in rids:
            T_peak = get_temp(layer, rid)
            # Clamp and flag melting
            if T_peak >= T_melt:
                T_peak = T_melt
                if (layer, rid) not in melting_detected:
                    melting_detected.append((layer, rid))
            temperature_per_layer[layer][rid] = T_peak

        # ---- 3. Time-stepping: cool all active layers until recoat done ----
        total_time = layer_times[layer]
        n_steps = max(1, int(math.ceil(total_time / dt)))
        actual_dt = total_time / n_steps  # uniform steps fitting exactly

        # Equilibrium tracking per region: (layer, rid) → consecutive_eq_count
        eq_count: Dict[Tuple[int, int], int] = {}
        eq_skip: Dict[Tuple[int, int], bool] = {}

        active_layers = layers_sorted[: layer_idx + 1]  # layers 1..current

        # Reset cumulative heat counters for snapshot tracking
        cum_heat_for_layer.update({al: 0.0 for al in active_layers})

        # Store t=0 snapshot (immediately after laser)
        temperature_snaps.setdefault(layer, {})
        temperature_regions_snaps.setdefault(layer, {})
        heat_flow_snaps.setdefault(layer, {})
        cumulative_heat_snaps.setdefault(layer, {})
        store_snapshot(layer, 0.0, active_layers)

        # Snapshot schedule — clipped to this layer's actual total_time
        layer_snap_times = [t for t in local_time_steps if t <= total_time + 1e-9]
        if not layer_snap_times or layer_snap_times[-1] < total_time - 1e-9:
            layer_snap_times.append(round(total_time, 2))
        # Remove t=0 (already stored); we'll step through the rest
        next_snap_idx = 1  # index into layer_snap_times

        t_local = 0.0  # elapsed local time within this layer

        for _step in range(n_steps):
            new_E: Dict[Tuple[int, int], float] = {}
            step_heat: Dict[int, float] = {}  # heat dissipated this step per layer

            for al in active_layers:
                step_heat[al] = 0.0
                for rid in regions_per_layer[al]['regions']:
                    key = (al, rid)

                    # Equilibrium skip optimisation
                    if eq_skip.get(key, False):
                        new_E[key] = E_region.get(key, 0.0)
                        continue

                    E_curr = E_region.get(key, 0.0)
                    m = region_mass[key]
                    T_curr = T_powder + E_curr / (m * cp)
                    Q_net = 0.0

                    # ---- Conduction downward (to layer below) ----
                    for c in conn_below_lookup.get(key, []):
                        lower_key = (al - 1, c['lower_region'])
                        if al == 1:
                            # Build plate at T_powder → E=0
                            T_lower = T_powder
                        else:
                            E_lower = E_region.get(lower_key, 0.0)
                            m_lower = region_mass.get(lower_key, MIN_MASS_KG)
                            T_lower = T_powder + E_lower / (m_lower * cp)
                        A_m2 = c['overlap_area_mm2'] / 1e6   # mm² → m²
                        Q_down = k * A_m2 * (T_curr - T_lower) / dz_eff
                        Q_net -= Q_down
                        # Track energy lost to build plate for conservation
                        if al == 1 and Q_down > 0:
                            total_E_dissipated += Q_down * actual_dt
                            step_heat[al] += Q_down * actual_dt

                    # ---- Conduction upward (to layer above) ----
                    for c in conn_above_lookup.get(key, []):
                        upper_key = (al + 1, c['upper_region'])
                        E_upper = E_region.get(upper_key, None)
                        # Only if the upper layer has already been activated
                        if E_upper is not None:
                            m_upper = region_mass.get(upper_key, MIN_MASS_KG)
                            T_upper = T_powder + E_upper / (m_upper * cp)
                            A_m2 = c['overlap_area_mm2'] / 1e6
                            Q_net += k * A_m2 * (T_upper - T_curr) / dz_eff

                    # ---- Convective cooling (exposed surfaces only) ----
                    # A region is exposed if it is the current top layer OR has
                    # no covering region on the layer above that is active.
                    is_top = (al == layer)
                    has_active_above = any(
                        E_region.get((al + 1, c['upper_region'])) is not None
                        for c in conn_above_lookup.get(key, [])
                    )
                    is_exposed = is_top or (not has_active_above)

                    if is_exposed:
                        A_exp = regions_per_layer[al]['regions'][rid]['area_mm2'] / 1e6
                        Q_conv = h_conv * A_exp * (T_curr - T_powder)
                        Q_net -= Q_conv
                        total_E_dissipated += Q_conv * actual_dt
                        step_heat[al] += Q_conv * actual_dt

                    # ---- Energy update with safety clamp ----
                    dE = Q_net * actual_dt
                    dT = dE / (m * cp)
                    dT_clamped = float(np.clip(dT, -DT_CLAMP_DEG, DT_CLAMP_DEG))
                    dE_clamped = dT_clamped * m * cp

                    new_E_val = max(0.0, E_curr + dE_clamped)

                    # NaN / Inf guard
                    if not math.isfinite(new_E_val):
                        new_E_val = 0.0

                    new_E[key] = new_E_val

                    # ---- Equilibrium tracking ----
                    if abs(dE_clamped) < EQ_SKIP_THRESHOLD:
                        eq_count[key] = eq_count.get(key, 0) + 1
                        if eq_count[key] >= EQ_SKIP_STEPS:
                            eq_skip[key] = True
                    else:
                        eq_count[key] = 0
                        eq_skip[key] = False

            E_region.update(new_E)

            # Accumulate heat dissipated this step
            for al in active_layers:
                cum_heat_for_layer[al] = cum_heat_for_layer.get(al, 0.0) + step_heat.get(al, 0.0)

            t_local += actual_dt

            # ---- Take snapshot if we've reached the next target time ----
            if next_snap_idx < len(layer_snap_times):
                next_target = layer_snap_times[next_snap_idx]
                if t_local >= next_target - actual_dt / 2:
                    store_snapshot(layer, next_target, active_layers)
                    next_snap_idx += 1

        # Ensure we always store the final snapshot
        final_t = layer_snap_times[-1]
        if final_t not in temperature_snaps.get(layer, {}):
            store_snapshot(layer, final_t, active_layers)

        # ---- 4. Store final (end-of-cooling) temperatures ----
        temperature_end[layer] = {}
        for rid in rids:
            T_end = get_temp(layer, rid)
            T_end = max(T_powder, min(T_melt, T_end))
            temperature_end[layer][rid] = T_end

        # ---- 5. Clamp stored energies at melting point ----
        for rid in rids:
            T = get_temp(layer, rid)
            if T > T_melt:
                E_region[(layer, rid)] = region_mass[(layer, rid)] * cp * (T_melt - T_powder)

        # Progress callback
        if progress_callback is not None:
            progress_callback((layer_idx + 1) / n_layers)

        if temperature_per_layer[layer]:
            logger.debug(
                f"Layer {layer}: peak={max(temperature_per_layer[layer].values()):.1f}°C  "
                f"end={max(temperature_end[layer].values()):.1f}°C  "
                f"t_total={total_time:.2f}s  n_steps={n_steps}"
            )
        else:
            logger.debug(f"Layer {layer}: no regions (empty layer), skipped")

    # ------------------------------------------------------------------
    # Energy conservation summary
    # ------------------------------------------------------------------
    final_E_stored = sum(E_region.values())
    energy_conservation = {
        'total_E_in': total_E_in,
        'total_E_dissipated': total_E_dissipated,
        'final_E_stored': final_E_stored,
        'balance_error_pct': (
            abs(total_E_in - total_E_dissipated - final_E_stored) / max(total_E_in, 1e-9) * 100
        ),
    }
    logger.info(
        f"Energy conservation: in={total_E_in:.4f} J  "
        f"dissipated={total_E_dissipated:.4f} J  "
        f"stored={final_E_stored:.4f} J  "
        f"error={energy_conservation['balance_error_pct']:.3f}%"
    )

    return {
        'temperature_per_layer': temperature_per_layer,
        'temperature_end': temperature_end,
        'energy_conservation': energy_conservation,
        'layer_times': layer_times,
        'scan_times': scan_times,
        'melting_detected': melting_detected,
        # Snapshot data for T Evolution playback
        'temperature': temperature_snaps,
        'temperature_regions': temperature_regions_snaps,
        'heat_flow': heat_flow_snaps,
        'cumulative_heat': cumulative_heat_snaps,
        'contact_areas': contact_areas,
        'time_steps': local_time_steps,
    }


def classify_thermal_risk(
    temperature_per_layer: Dict[int, Dict[int, float]],
    melting_point: float,
    warning_fraction: float = 0.8,
    critical_fraction: float = 1.0,
) -> Dict[int, str]:
    """
    Classify each layer's thermal risk based on peak temperature.

    Parameters
    ----------
    temperature_per_layer : dict
        {layer: {rid: T_peak_°C}}
    melting_point : float
        Material melting point in °C
    warning_fraction : float
        Fraction of T_melt above which risk = WARNING
    critical_fraction : float
        Fraction of T_melt above which risk = CRITICAL (melting)

    Returns
    -------
    dict
        {layer: 'SAFE' | 'WARNING' | 'CRITICAL'}
    """
    T_warning  = melting_point * warning_fraction
    T_critical = melting_point * critical_fraction

    risk: Dict[int, str] = {}
    for layer, region_temps in temperature_per_layer.items():
        if not region_temps:
            risk[layer] = 'SAFE'
            continue
        T_peak = max(region_temps.values())
        if T_peak >= T_critical:
            risk[layer] = 'CRITICAL'
        elif T_peak >= T_warning:
            risk[layer] = 'WARNING'
        else:
            risk[layer] = 'SAFE'

    return risk


def run_thermal_analysis(
    masks: Dict[int, np.ndarray],
    regions_per_layer: Dict,
    conn_below_lookup: Dict,
    conn_above_lookup: Dict,
    material_props: MaterialProperties,
    process_params: ProcessDefaults,
    thermal_params: ThermalSimDefaults,
    layer_grouping: int,
    voxel_size: float,
    layer_thickness: float,
    snapshot_interval: float = 2.5,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict:
    """
    Top-level entry point for thermal analysis.

    Chains: calculate_energy_input → simulate_thermal_evolution → classify_thermal_risk

    Parameters
    ----------
    (see sub-function docstrings)

    Returns
    -------
    dict with all simulation outputs plus 'risk_per_layer' key.
    """
    # Step 1: compute laser energy input per region
    E_in_per_region = calculate_energy_input(
        regions_per_layer=regions_per_layer,
        material_props=material_props,
        process_params=process_params,
        layer_grouping=layer_grouping,
        voxel_size=voxel_size,
    )

    # Step 2: run thermal simulation
    results = simulate_thermal_evolution(
        masks=masks,
        regions_per_layer=regions_per_layer,
        conn_below_lookup=conn_below_lookup,
        conn_above_lookup=conn_above_lookup,
        E_in_per_region=E_in_per_region,
        material_props=material_props,
        process_params=process_params,
        thermal_params=thermal_params,
        layer_grouping=layer_grouping,
        voxel_size=voxel_size,
        layer_thickness=layer_thickness,
        snapshot_interval=snapshot_interval,
        progress_callback=progress_callback,
    )

    # Step 3: classify risk
    risk_per_layer = classify_thermal_risk(
        temperature_per_layer=results['temperature_per_layer'],
        melting_point=material_props.melting_point,
        warning_fraction=thermal_params.warning_fraction,
        critical_fraction=thermal_params.critical_fraction,
    )
    results['risk_per_layer'] = risk_per_layer
    results['E_in_per_region'] = E_in_per_region

    return results

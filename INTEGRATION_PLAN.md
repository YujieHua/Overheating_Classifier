# Integration Plan: Energy-Backbone Thermal Simulation
## Overheating Classifier Enhancement

**Version:** 3.0
**Date:** 2026-02-18
**Author:** Thursday (Deep Work Agent)
**Approved by:** Yujie Hua
**Status:** Under Review
**Review:** Comprehensive self-review completed 2026-02-18 (see Appendix A)

---

## 1. Executive Summary

Integrate a physics-based thermal simulation mode into the Overheating Classifier using **energy as the backbone state variable** with temperature derived from energy. This replaces the need for a separate "constant delta_T" mode and creates a unified energy-tracking framework where heat flow is computed via Fourier's law using derived temperatures.

The existing Energy Balance mode (unitless, normalized 0-1) remains as a fast screening tool. The new Thermal Simulation mode adds physical units (Joules, °C) and time-stepped heat evolution.

**Target users:** LPBF process engineers, material scientists, researchers. Commercial-grade quality.

---

## 2. Architecture

### 2.1 Dual-Mode Framework

```
┌─── Energy Balance Mode (existing, fast) ─────────────────┐
│ E_in = A_region (unitless, proportional to area)         │
│ E_acc = (E_inherited + E_in) × (1 - R_total)            │
│ Geometry factor: area ratio or Gaussian G                │
│ Output: Normalized 0-1 risk scores (LOW/MEDIUM/HIGH)     │
│ Use case: Quick screening, no material data needed       │
└──────────────────────────────────────────────────────────┘

┌─── Thermal Simulation Mode (new, energy-backbone) ──────┐
│ E_in = η × P × t_scan_region  [Joules]                  │
│ T_initial = w × T_below + (1-w) × T_powder              │
│ E_region = m × cp × (T_initial - T_powder) + E_in        │
│                                                          │
│ Time-stepping loop:                                      │
│   T_i = T_powder + E_i / (m_i × cp)  [derive temp]     │
│   Q_cond = k × A_overlap × (Ti-Tj)/dz [Fourier]        │
│   Q_conv = h × A_exposed × (T-T_amb)  [Newton cooling] │
│   dE = Q_net × dt                     [energy update]   │
│   E_i += dE                                             │
│   Clamp: if T_i >= T_melt → flag "melting detected"     │
│                                                          │
│ Output: Temperature in °C, risk vs melting point         │
│ Use case: Physics-based prediction, validation against   │
│           OT (optical tomography) data                   │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Shared Components

Both modes share:
- STL slicer (vectorized, from Predictor)
- Region detection (8-connectivity, from Predictor)
- Cross-layer connectivity computation
- Geometry G computation (optional, for future Rosenthal integration)
- Flask web UI framework
- 3D Plotly visualization

### 2.3 Energy Input Calculation (Thermal Mode)

The laser deposits energy into each region proportional to scan time:

```
E_in = η × P × t_scan_region

Where:
  η = absorptivity (material-dependent, 0.3-0.4 typically)
  P = laser power [W]
  t_scan_region = scan_efficiency × A_region_mm2 / (h_spacing × v_scan)

  scan_efficiency = configurable (default 0.1)
  A_region_mm2 = region area in mm²
  h_spacing = hatch spacing [mm]
  v_scan = scan velocity [mm/s]
```

### 2.4 Initial Temperature on New Layer

When a new layer is deposited, each region's initial temperature is a weighted average:

```
T_initial = (1 - w) × T_powder + w × T_below_weighted

Where:
  w = substrate_influence (default 0.3, adjustable 0.0-1.0)
  T_powder = preheat temperature (build chamber temp)
  T_below_weighted = overlap-area-weighted average of connected regions below

For regions with no parent (floating/orphan):
  T_initial = T_powder (no substrate coupling)
```

**Physical justification for w=0.3:**
- Powder thermal conductivity is ~0.1-1 W/mK (100-1000× lower than solid metal)
- Only the melt pool zone (depth ~2-3 layer thicknesses) couples thermally
- The bulk of the new powder layer starts at chamber temperature
- 30% substrate coupling accounts for melt pool penetration into previous layer

### 2.5 Time-Stepping Physics

After laser energy is deposited, regions cool via:

**1. Inter-layer conduction (Fourier's law):**
```
Q_cond = k × A_overlap_m2 × (T_current - T_neighbor) / dz_eff
dE_cond = Q_cond × dt
```

**2. Convective cooling (Newton's law, exposed surfaces only):**
```
Q_conv = h_conv × A_exposed_m2 × (T_current - T_ambient)
dE_conv = Q_conv × dt
```

A region is "exposed" if it is the topmost active layer OR has no covering region above it.

**3. Time step (CFL stability):**
```
dt = min(dt_max, 0.5 × dz² / (2 × α))

Where:
  α = k / (ρ × cp) [thermal diffusivity, m²/s]
  dz = effective layer thickness [m] (accounts for grouping)
  dt_max = 0.001 s
  dt_min = 1e-5 s (safety floor for high-conductivity materials like AlSi10Mg)
```

**4. Safety clamp:**
```
|dT_per_step| ≤ 50°C (prevents numerical instability for tiny regions)
T_region ≥ T_powder (can't cool below ambient)
T_region ≤ T_melt (clamp at melting, flag region)
E_region = max(0, E_region) (energy can't go negative relative to T_powder)
```

**5. Temperature derivation (reference = T_powder):**
```
T = T_powder + E / (m × cp)
```
Using T_powder as reference keeps energy values small and meaningful:
- E = 0 means region is at powder temperature
- E > 0 means region is hotter than powder
- Avoids large-number subtraction issues

### 2.6 Layer Grouping in Thermal Mode

When `layer_grouping > 1`, the physics must account for the grouped thickness:

| Parameter | Formula with Grouping | Effect |
|-----------|----------------------|--------|
| Effective layer thickness | `dz_eff = layer_thickness × layer_grouping` | Thicker layers |
| Thermal mass per region | `m = ρ × A_region × voxel² × dz_eff` | 25× more mass for grouping=25 |
| CFL time step | `dt ∝ dz_eff²` | 625× larger dt for grouping=25 |
| Conduction flux | `Q = k × A × ΔT / dz_eff` | 25× weaker per step |
| Scan time per layer | `t_scan = scan_eff × A / (h × v) × layer_grouping` | Proportional to grouped layers |
| Recoat time per group | `t_recoat = recoat_time × layer_grouping` | More cooling time between groups |
| Energy input per group | `E_in = η × P × t_scan_group` | Proportional to scan time |

**Design decision:** All physics parameters automatically scale with grouping. The user does not need to adjust anything manually. Higher grouping = faster but smoother (less resolution in Z). The general trend is preserved.

### 2.7 Material Database

Built-in materials with editable properties:

| Material | k (W/mK) | cp (J/kgK) | ρ (kg/m³) | T_melt (°C) | η |
|----------|----------|-----------|----------|------------|---|
| SS316L | 16.2 | 500 | 7990 | 1400 | 0.35 |
| Ti-6Al-4V | 6.7 | 526 | 4430 | 1660 | 0.40 |
| IN718 | 11.4 | 435 | 8190 | 1336 | 0.38 |
| AlSi10Mg | 147 | 963 | 2670 | 660 | 0.30 |
| Custom | user | user | user | user | user |

---

## 3. Risk Classification

### Energy Mode (unchanged)
- LOW: score < threshold_medium (default 0.3)
- MEDIUM: threshold_medium ≤ score < threshold_high (default 0.6)
- HIGH: score ≥ threshold_high
- Scale: 0-1 (normalized)

### Thermal Mode (new)
- Safe: T_peak < warning_fraction × T_melt (default 0.8)
- Warning: warning_fraction × T_melt ≤ T_peak < critical_fraction × T_melt (default 1.0)
- Critical: T_peak ≥ critical_fraction × T_melt (melting detected)
- Scale: °C (absolute)

---

## 4. UI Design

### 4.1 Mode Selector
Top of sidebar, prominent, not inside an accordion:
- ⚡ Energy Balance (fast screening)
- 🌡️ Thermal Simulation (physics-based)

Switching modes hides/shows relevant parameter sections with smooth transitions.

### 4.2 Parameter Sections

**Shared (always visible):**
- STL Input: file upload, test STL buttons
- Slicing: voxel_size, layer_thickness, layer_grouping, build_direction

**Energy Mode only:**
- Energy Model: dissipation_factor, convection_factor, Mode A/B toggle
  - Mode A: area_ratio_power
  - Mode B: sigma, G_max, gaussian_ratio_power
- Risk Thresholds: medium (0-1), high (0-1)

**Thermal Mode only:**
- Material: dropdown selector + editable fields (k, cp, ρ, T_melt, η)
- Process: laser_power (W), scan_velocity (mm/s), hatch_spacing (mm), recoat_time (s), scan_efficiency
- Thermal: convection_coefficient (W/m²K), substrate_influence (0-1)
- Risk Thresholds: warning_fraction (0-1 of T_melt), critical_fraction (0-1 of T_melt)

### 4.3 Tabs (7 total, mode-aware)

1. **STL Preview** - unchanged, both modes
2. **Sliced Layers** - unchanged, both modes
3. **Geometry** - Energy mode: Area Ratio or Gaussian G depending on Mode A/B. Thermal mode: hidden (G not used in energy-backbone model; will show when Rosenthal added later)
4. **Regions** - both modes, shows connected components per layer
5. **Results** - Mode-dependent content:
   - Energy mode: energy accumulation heatmap (current)
   - Thermal mode: temperature heatmap with time scrubber for evolution playback
6. **Risk Map** - both modes, different color scales and legends
7. **Summary** - both modes, content adapts (energy stats vs temperature stats)

### 4.4 Professional UI Standards

- Clean parameter organization matching commercial simulation tools (Simufact, Amphyon, Netfabb)
- Consistent color scheme and typography
- Input validation with clear error messages
- Progress bar with estimated runtime
- Export to CSV/JSON with all parameters documented
- Session management (cancel, re-run with new params)

---

## 5. Implementation Phases

### Phase 0: Code Cleanup (No Visible Change)
**Goal:** Prepare the Classifier codebase for integration.

**Tasks:**
1. Port vectorized slicer from Predictor (replace loop-based version in stl_loader.py)
2. Create `src/compute/region_detect.py` from Predictor (8-connectivity)
3. Refactor `energy_model.py` to import from `region_detect.py` instead of inline `_label_regions`/`_build_overlap_map`
4. Add mask memory cleanup after connectivity computation
5. Verify Classifier produces identical results after refactor

**Tests:** Run existing tests, compare before/after outputs on test STLs.
**Estimated effort:** 2-3 hours.

### Phase 1: Infrastructure
**Goal:** Add configuration and material support.

**Tasks:**
1. Add to `config.py`: MaterialDefaults, ThermalSimDefaults, ProcessDefaults dataclasses
2. Add material database with SS316L, Ti64, IN718, AlSi10Mg
3. Add AnalysisMode enum (ENERGY / THERMAL)
4. Add parameter validation rules for thermal mode parameters
5. Add mode-aware parameter collection in app.py

**Tests:** Config validation, material property lookup, mode enum.
**Estimated effort:** 1-2 hours.

### Phase 2: Core Algorithm - Energy-Backbone Thermal Model
**Goal:** Implement the time-stepped energy-backbone simulation.

**Tasks:**
1. Create `src/compute/thermal_model.py` with:
   - `calculate_energy_input()` - computes E_in per region from process params
   - `simulate_thermal_evolution()` - main time-stepping loop (energy state variable)
   - `classify_thermal_risk()` - risk classification vs melting point
2. Wire into `run_analysis_worker()` with mode branching
3. Implement layer-grouping-aware physics (all parameters scale correctly)
4. Add energy conservation verification (debug logging)
5. Port cross-layer connectivity from Predictor's region_detect.py
6. Add equilibrium-skip optimization (skip regions where |dE| < tolerance)
7. Optimize snapshot storage (peak temp + cooling curve for current layer only)

**Key design details for thermal_model.py:**
```python
def simulate_thermal_evolution(
    masks, regions_per_layer, conn_below_lookup, conn_above_lookup,
    material_props, process_params, thermal_params,
    layer_grouping, progress_callback
) -> dict:
    """
    Energy-backbone thermal simulation.
    
    State variable: E_region [Joules] per (layer, region_id)
    Temperature derived: T = T_ref + E / (m × cp)
    Heat flow: Fourier + Newton cooling using derived temperatures
    """
```

**Tests:** Unit tests for energy input, time-stepping correctness, energy conservation, melting clamp, layer grouping scaling, edge cases (empty layers, single-pixel regions, floating islands).
**Estimated effort:** 4-6 hours.

### Phase 3: UI Integration
**Goal:** Add mode selector and thermal mode visualization.

**Tasks:**
1. Add mode selector UI (top of sidebar)
2. Implement conditional parameter panels (show/hide by mode)
3. Add material dropdown with auto-fill + editable fields
4. Add process parameters section
5. Adapt run_analysis_worker for mode dispatch
6. Add temperature results API endpoint
7. Add time scrubber for temperature evolution playback (thermal mode)
8. Adapt risk legend and color scales per mode
9. Adapt summary tab content per mode
10. Add runtime estimation display

**Tests:** API endpoint tests for both modes, UI state transitions.
**Estimated effort:** 6-8 hours.

### Phase 4: Polish & Validation
**Goal:** Commercial-quality finish.

**Tasks:**
1. Export CSV for thermal mode (layer, region, time, temperature, energy)
2. Performance benchmarking (energy vs thermal, various layer counts)
3. Edge case testing suite
4. Energy conservation assertion in debug mode
5. Error handling for all parameter combinations
6. Memory profiling for large models
7. Code documentation (docstrings, inline comments)

**Tests:** Full integration tests, stress tests.
**Estimated effort:** 3-4 hours.

---

## 6. File Structure After Integration

```
Overheating_Classifier/
├── app.py                      # Mode-aware web UI (~4000 lines)
├── config.py                   # + MaterialDefaults, ThermalSimDefaults
├── src/
│   ├── data/
│   │   └── stl_loader.py       # Vectorized (ported from Predictor)
│   ├── compute/
│   │   ├── energy_model.py     # Refactored: uses region_detect imports
│   │   ├── thermal_model.py    # NEW: energy-backbone simulation
│   │   ├── region_detect.py    # NEW: 8-connectivity, cross-layer
│   │   └── geometry_score.py   # Unchanged (FFT version)
├── tests/
│   ├── test_energy_model.py    # Existing
│   ├── test_thermal_model.py   # NEW
│   ├── test_region_detect.py   # NEW
│   ├── test_geometry_score.py  # Existing
│   └── test_stl_loader.py      # Updated for vectorized version
```

---

## 7. Known Limitations & Future Work

### Current Limitations (accepted for v1)
1. **Constant material properties** - cp, k do not vary with temperature
2. **No latent heat modeling** - temperature clamped at melting point
3. **No radiation cooling** - only conduction + convection (conservative: overpredicts temps)
4. **Simplified scan time** - uses scan_efficiency factor, not actual scan path
5. **No lateral (XY) heat conduction** - only vertical (Z) between layers
6. **No powder bed conduction** - only solid-to-solid heat flow

### Future Enhancements (not in this plan)
1. Rosenthal-based energy input (replaces constant η×P×t)
2. Temperature-dependent material properties
3. Radiation cooling term
4. Actual scan path integration for energy deposition
5. Lateral heat spreading within layers
6. Export to VTK/ParaView for advanced visualization

---

## 8. Risk Matrix

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|---------|-----------|-----------|
| 1 | Region connectivity mismatch (4 vs 8) | 🔴 Critical | Certain | Phase 0: standardize to 8-connectivity |
| 2 | Memory explosion (>500 layers) | 🔴 High | Likely | Phase 0: mask cleanup after connectivity |
| 3 | Energy conservation violation (bug) | 🔴 High | Possible | Phase 2: conservation check in debug mode |
| 4 | CFL instability with small dz | 🟡 Medium | Unlikely | Auto-calculated dt with safety floor |
| 5 | Slicer performance gap | 🟡 Medium | Certain | Phase 0: vectorized slicer port |
| 6 | Temperature overshoot at melting | 🟡 Medium | Common | Clamp at T_melt, flag region |
| 7 | Confusing dual-mode UI | 🟡 Medium | Possible | Clean show/hide, mode descriptions |
| 8 | Slow thermal sim (>1min) | 🟡 Medium | Likely for ungrouped | Show estimated runtime pre-run |
| 9 | Tiny region numerical instability | 🟢 Low | Rare | MAX_DT_PER_STEP clamp + min mass floor |
| 10 | Float precision loss | 🟢 Low | Very unlikely | Use float64 throughout |
| 11 | 8-connectivity changes energy mode results | 🟡 Medium | Certain | Add connectivity param; document as improvement |
| 12 | Time-step loop O(layers²) performance | 🟡 Medium | Likely | Equilibrium-skip + numpy vectorization |
| 13 | Snapshot memory for large models | 🟡 Medium | Likely | Store only peak + current cooling curve |
| 14 | Absorptivity uncertainty across materials | 🟡 Medium | Inherent | Document as calibration parameter |
| 15 | Instant-flash heating approximation | 🟢 Low | Acceptable | Document; sub-interval splitting for future |

---

## 9. Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State variable | Energy (Joules) | Enables conservation checks, natural material coupling |
| Temperature display | °C (absolute) | More direct for engineers, not normalized |
| Substrate influence default | 0.3 | Literature-supported, powder insulation dominant |
| Phase change handling | Clamp at T_melt | Screening tool, not melt pool simulator |
| Layer grouping physics | All params scale with grouped thickness | Physically correct without manual adjustment |
| Constant delta_T mode | Removed | Replaced by energy-based input (more physical) |
| 3D group detection | Not included | Not needed; 2D regions + cross-layer suffices |
| Connectivity | 8-connectivity | Diagonal pixels = same region in laser scan |
| Radiation cooling | Not included v1 | Conservative (overpredicts), add later |
| Scan time model | scan_efficiency × area-based | Simple, configurable, matches Predictor |

---

## 10. Acceptance Criteria

- [ ] Energy mode produces identical results to current Classifier
- [ ] Thermal mode runs successfully on test STLs
- [ ] Energy conservation holds within 0.1% tolerance (debug check)
- [ ] Temperature output matches expected physical range
- [ ] Layer grouping correctly scales all physics parameters
- [ ] Both modes accessible from single UI with clean switching
- [ ] All existing tests pass after refactor
- [ ] New thermal model has >90% test coverage
- [ ] Memory usage stays within 2× of current for same model
- [ ] No numerical instability (NaN, Inf) on any test case

---

## Appendix A: Comprehensive Review Findings

### A.1 Physics Issues Found & Fixed

**Issue P1: Absorptivity values are uncertain (IMPORTANT)**
The material database lists η values (0.30-0.40) which are "base absorptivity" estimates. In reality, absorptivity during LPBF varies significantly:
- Ti-6Al-4V: measured 0.27±0.03 (Trapp et al., in-situ calorimetry) to 0.40 (powder bed estimates)
- SS316L: calibrated 0.35-0.52 depending on methodology
- Absorptivity changes with temperature, phase (powder/liquid/solid), and keyhole formation

**Resolution:** Keep current values as defaults but add a clear note in the UI that absorptivity is an effective/calibrated parameter, not a fundamental material constant. Users should calibrate against experimental data for their specific machine.

**Issue P2: E_initial formula has a reference temperature ambiguity**
Section 2.1 shows `E_region = m × cp × (T_initial - T_ref) + E_in`. If T_ref = 0°C, then a region at preheat=80°C already has substantial "energy" before the laser even fires. This is physically correct (internal energy relative to 0°C) but means E values are large numbers with small relative changes.

**Resolution:** Use T_ref = T_powder (preheat temperature) instead of 0°C. This way:
- E = 0 means "at powder temperature" (no excess energy)
- E > 0 means "hotter than powder"
- T = T_powder + E / (m × cp)
This keeps energy values smaller, more meaningful, and avoids subtractive cancellation.

**Issue P3: Convection coefficient is geometry-dependent (MINOR)**
The plan uses a single h_conv value. In reality, natural convection depends on surface orientation, temperature difference, and gas flow in the build chamber. Typical values: 5-25 W/m²K for stagnant argon, up to 100 W/m²K with gas flow.

**Resolution:** Default h=10 W/m²K (stagnant argon atmosphere, conservative). Expose as adjustable parameter. Acceptable for a screening tool.

**Issue P4: No heat conduction to build plate modeled explicitly**
Layer 1 regions connect to a "virtual build plate" but the plan doesn't specify the build plate's thermal behavior. Is it a constant temperature (infinite heat sink) or does it heat up?

**Resolution:** Build plate = constant temperature boundary at T_powder. This is physically correct: real build plates are massive (high thermal mass) and actively heated/cooled, so they stay near preheat temperature. The Predictor handles this the same way.

**Issue P5: Energy input timing within a layer**
The plan says "laser deposits energy" as an instant event. In reality, the laser scans for t_scan seconds, during which some cooling already occurs. This means the actual peak temperature is lower than the "instant deposit" model predicts.

**Resolution:** Acceptable approximation for a layer-averaged model. The scan time is still used for the cooling period. Document that this is an "instant flash" heating model. If higher fidelity is needed later, the scan time could be split into sub-intervals.

### A.2 Implementation Issues Found & Fixed

**Issue I1: Region detection refactor could change energy mode results**
Switching from 4-connectivity to 8-connectivity changes which pixels are grouped. A geometry that has two regions touching at a corner will go from 2 regions → 1 region. This changes the energy model results because energy inheritance depends on region boundaries.

**Resolution:** This is a deliberate improvement, not a bug. 8-connectivity is more physically correct for laser scan paths (the laser can reach diagonal neighbors). The acceptance criterion "Energy mode produces identical results" needs to be weakened to: "Energy mode produces results identical to the current version when using 4-connectivity, and improved results with 8-connectivity (default)." Add a `connectivity` parameter (default 8) for backward compatibility testing.

**Issue I2: Time-stepping loop iterates over ALL regions on ALL active layers each step**
In the Predictor, the inner loop is `for l in range(1, layer+1): for rid in rids:`. For layer 200 with 1-3 regions each, that's ~200-600 region updates per dt step. With dt=1e-3s and recoat_time=10s, that's 10,000 steps × 600 regions = 6 million updates per layer. Over 200 layers, 1.2 billion updates total.

**Resolution:** This is the main performance bottleneck. Mitigations:
1. Most regions stabilize quickly (within ~100 steps). Add an "equilibrium check": if |dE| < tolerance for a region, skip it. This could reduce work by 90%+.
2. Use numpy vectorization: store all region energies in a single array, compute all updates in one vectorized operation per time step.
3. Layer grouping naturally reduces total layers (200 layers → 8 groups of 25).

Add this optimization to Phase 2.

**Issue I3: Snapshot storage for thermal mode could be memory-heavy**
If we store temperature for every region at every snapshot time for every layer-being-viewed, the data structure grows as O(layers² × snapshots × regions). For 200 layers, 10 snapshots, 3 regions average: 200 × 200 × 10 × 3 = 1.2M entries.

**Resolution:** Only store snapshots for the CURRENT layer being simulated (not the full history of all previous layers at each viewing step). The Predictor's dual-format storage is excessive for a screening tool. Store: peak temperature per region per layer, and optionally the cooling curve for the current layer only.

**Issue I4: CSV export needs clear column definitions**
For a commercial tool, the exported data must be self-documenting.

**Resolution:** Export format:
```
# Overheating Classifier - Thermal Simulation Results
# Material: Ti-6Al-4V, Mode: Thermal Simulation
# Generated: 2026-02-18 01:00:00
# Parameters: P=280W, v=960mm/s, h=0.1mm, ...
layer,region_id,area_mm2,T_peak_C,T_end_C,risk_level,E_in_J,E_final_J
1,1,12.5,580.2,95.3,Safe,0.42,0.003
```

### A.3 UX Issues Found & Fixed

**Issue U1: Mode switching should preserve shared parameters**
If a user sets voxel_size=0.05 in energy mode, switches to thermal mode, that value should persist.

**Resolution:** Shared parameters are stored independently of mode. Mode switch only shows/hides mode-specific sections.

**Issue U2: Material selector needs "last used" memory**
Engineers typically work with one material for extended periods.

**Resolution:** Store last selected material in localStorage. Auto-select on page load.

**Issue U3: Runtime estimation formula needed**
The plan says "show estimated runtime" but doesn't specify how.

**Resolution:** Estimate formula:
```
n_groups = n_layers / layer_grouping
steps_per_group = (recoat_time × layer_grouping) / dt
total_steps = n_groups × steps_per_group × n_groups/2 (average active layers)
estimated_seconds = total_steps × time_per_step (benchmark: ~1μs per region update)
```
Show before run: "Estimated runtime: ~30s" with a note that actual time may vary.

### A.4 Edge Cases Verified

| Edge Case | Expected Behavior | Handled? |
|-----------|-------------------|----------|
| Empty layer (no solid voxels) | Skip, carry forward previous temps | ✅ Yes (from Predictor logic) |
| Single-pixel region | Minimum mass floor prevents /0 | ✅ Yes (MIN_MASS_KG) |
| Floating region (no overlap below) | T_initial = T_powder only | ✅ Yes |
| Region splits into 2+ children | Energy distributed by overlap fraction | ✅ Yes |
| Multiple regions merge into 1 | Energy summed from all parents | ✅ Yes |
| Very high laser power (>500W) | T may exceed T_melt, clamped + flagged | ✅ Yes |
| Very low power (<50W) | Minimal temperature rise, all Safe | ✅ Yes |
| layer_grouping = 1 | No grouping, maximum resolution | ✅ Yes |
| layer_grouping = n_layers | Entire part = one layer | ⚠️ Edge case: needs testing |
| AlSi10Mg (very high k=147) | Extremely fast cooling, very small dt | ⚠️ CFL dt could be tiny; dt_min floor protects |
| All regions at T_melt | All flagged Critical, simulation continues | ✅ Yes |

### A.5 Remaining Open Questions (Low Priority)

1. Should the time scrubber show LOCAL time (within each layer) or GLOBAL build time?
   - Recommendation: Local time (matches Predictor), with global time shown in summary
2. Should we support re-running thermal mode on a subset of layers?
   - Recommendation: Not in v1. Full run only.
3. Should energy conservation check run in production or debug only?
   - Recommendation: Debug only (adds overhead). Log a warning if violation detected.

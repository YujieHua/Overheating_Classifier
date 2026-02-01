# Overheating Classifier for LPBF

## Simplified Energy-Based Overheating Indicator

A simplified model that tracks **cumulative energy (Joules) per layer** to predict overheating risk, based on Ali's recommendations from the Jan 27, 2026 SRG meeting.

---

## 1. Overview

### Problem Statement
Same as the temperature-based model: Predict overheating risk in LPBF before printing.

### This Approach (Simplified)
Instead of calculating actual temperatures, track **cumulative Joules per layer** as an indicator of overheating risk.

> "Let's not make it complicated and let's not figure out the temperature - let's just simply say cumulation of joules cause that would be a good indication" - Ali (Jan 27, 2026)

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What to track | Cumulative Joules | Simpler than temperature |
| Energy input | J/mm³ × Volume | Known from process parameters |
| Energy output | Dissipation rate × Time | Function of geometry, material |
| Geometry effect | Via dissipation rate | Previous layer area matters |
| Accuracy target | ~60-70% | "Good enough" for trend detection |

### Relationship to Temperature-Based Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Temperature Model (Overheating_Predictor)                                   │
│  ─────────────────────────────────────────                                   │
│  ΔT = Rosenthal(P, v, A) × (1 + G)                                          │
│  T_layer = T_previous + ΔT - cooling                                         │
│                                                                              │
│  THIS Model (Overheating_Classifier)                                         │
│  ────────────────────────────────────                                        │
│  E_in = energy_density × Volume                                              │
│  E_out = dissipation_rate × dt                                               │
│  E_accumulated += E_in - E_out                                               │
│                                                                              │
│  Connection: E_accumulated ∝ ΔT (approximately)                              │
│  Both use Geometry Multiplier (G) from 3D Gaussian convolution               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Physics Basis

### 2.1 Energy Input (Per Layer)

From Ali's explanation:

```
Energy_input = energy_density × layer_volume

where:
  energy_density = P / (v × h × t)    [J/mm³]

  P = laser power [W]
  v = scan velocity [mm/s]
  h = hatch spacing [mm]
  t = layer thickness [mm]

  layer_volume = cross_sectional_area × layer_thickness [mm³]
```

**Key insight:** Energy density (J/mm³) is a known, accurate input from process parameters.

### 2.2 Energy Consumption

Energy is consumed by two mechanisms:

#### A. Latent Heat of Fusion
```
E_latent = ρ × V × L_f

where:
  ρ = density [kg/mm³]
  V = melted volume [mm³]
  L_f = latent heat of fusion [J/kg]
```

**Note:** For layer-averaged models, latent heat absorbed = latent heat released during solidification. Net effect ≈ 0 for cumulative tracking.

#### B. Heat Dissipation
```
E_dissipated = Q̇_dissipation × dt

where:
  Q̇_dissipation = heat dissipation rate [W]
  dt = time available for cooling [s]
```

### 2.3 Heat Dissipation Rate

This is where geometry matters. From Ali:

> "Your previous layer cross-sectional area would actually play a big role... that's basically the majority of the consumption"

```
Q̇_dissipation = k_eff × A_contact × ΔT / Δz

where:
  k_eff = effective thermal conductivity [W/(mm·K)]
  A_contact = contact area to previous layer [mm²]
  ΔT = temperature difference (approximated from accumulated energy)
  Δz = layer thickness [mm]
```

**Bottleneck effect:** If current layer area > previous layer area, heat can only escape through `A_contact = min(A_current, A_previous)`.

### 2.4 Geometry Multiplier (From Temperature Model)

Use the same 3D Gaussian convolution to calculate geometry quality:

```
G = geometry_multiplier (0 = bulk, 2 = severe overhang)

Q̇_effective = Q̇_base / (1 + G)

where:
  G = 0 → full dissipation (bulk solid below)
  G = 2 → 1/3 dissipation (severe overhang, powder below)
```

### 2.5 Energy Accumulation

```
E_accumulated[layer] = E_accumulated[layer-1] + E_in - E_out

where:
  E_in = energy_density × A_layer × Δz
  E_out = Q̇_effective × (t_scan + t_dwell)
```

**Risk indicator:**
- E_accumulated LOW → Heat is dissipating well → LOW risk
- E_accumulated HIGH → Heat is building up → HIGH risk

---

## 3. Algorithm

### 3.1 Inputs

| Parameter | Symbol | Source |
|-----------|--------|--------|
| Laser power | P | Process settings |
| Scan velocity | v | Process settings |
| Hatch spacing | h | Process settings |
| Layer thickness | Δz | Process settings |
| Layer area per slice | A[z] | From STL slicing |
| Geometry multiplier | G[z] | From 3D Gaussian convolution |
| Dwell time | t_dwell | Process settings |
| Gas flow velocity | v_gas | From Ali (convection) |

### 3.2 Material Properties

| Property | Symbol | Value (IN718) | Units |
|----------|--------|---------------|-------|
| Density | ρ | 8.19 | g/cm³ |
| Thermal conductivity | k | 11.4 | W/(m·K) |
| Specific heat | c_p | 435 | J/(kg·K) |
| Latent heat of fusion | L_f | 290 | kJ/kg |

### 3.3 Pseudocode

```python
def calculate_energy_accumulation(slices, process_params, material):
    """
    Calculate cumulative energy per layer.

    Parameters:
    -----------
    slices : list of 2D arrays
        Binary masks for each layer
    process_params : dict
        P, v, h, dz, t_dwell, v_gas
    material : dict
        rho, k, c_p, L_f

    Returns:
    --------
    list : Accumulated energy per layer [J]
    """
    # Calculate energy density
    energy_density = process_params['P'] / (
        process_params['v'] * process_params['h'] * process_params['dz']
    )  # J/mm³

    # Initialize
    E_accumulated = 0
    results = []

    for z, mask in enumerate(slices):
        # Layer area
        A_layer = np.sum(mask) * pixel_area  # mm²

        # Geometry multiplier (from 3D Gaussian)
        G = calculate_geometry_multiplier(slices, z)

        # Energy input
        E_in = energy_density * A_layer * process_params['dz']  # J

        # Contact area (bottleneck)
        if z > 0:
            A_contact = min(A_layer, A_previous)
        else:
            A_contact = A_layer  # Connected to baseplate

        # Dissipation rate (simplified)
        # Assume ΔT ∝ E_accumulated / (mass × c_p)
        T_effective = E_accumulated / (material['rho'] * A_layer * process_params['dz'] * material['c_p'])
        Q_dot = material['k'] * A_contact * T_effective / process_params['dz']
        Q_dot_effective = Q_dot / (1 + G)

        # Time for cooling
        t_scan = A_layer / (process_params['v'] * process_params['h'])  # s
        t_total = t_scan + process_params['t_dwell']

        # Energy output
        E_out = Q_dot_effective * t_total  # J

        # Accumulate
        E_accumulated = max(0, E_accumulated + E_in - E_out)

        results.append({
            'z': z,
            'A_layer': A_layer,
            'G': G,
            'E_in': E_in,
            'E_out': E_out,
            'E_accumulated': E_accumulated
        })

        A_previous = A_layer

    return results
```

---

## 4. Output Interpretation

### 4.1 Risk Levels

| E_accumulated | Risk Level | Interpretation |
|---------------|------------|----------------|
| < E_threshold_1 | LOW | Normal operation |
| E_threshold_1 to E_threshold_2 | MEDIUM | Monitor, possible issues |
| > E_threshold_2 | HIGH | Likely overheating |

### 4.2 Threshold Calibration

Thresholds must be calibrated against OT data from Ali:
- Compare E_accumulated trends with OT intensity
- Find thresholds that match observed defects

### 4.3 Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│  E_accumulated vs Z Height                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  E_acc                                                           │
│    │                                    ╭─────╮                  │
│    │                                   ╱       ╲  ← HIGH RISK    │
│    │                          ╭───────╯         ╲                │
│    │              ╭──────────╯                   │ ← bottleneck  │
│    │         ╭────╯                              │                │
│    │    ╭────╯                                   │                │
│    │────╯                                        ╰────            │
│    └────────────────────────────────────────────────────→ Z     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Validation Plan

### 5.1 Data from Ali (Pending)

- [ ] Manifolds geometry STL file
- [ ] OT results for that geometry (intensity vs height)
- [ ] Gas flow velocity value

### 5.2 Validation Approach

1. Run energy accumulation model on manifolds geometry
2. Plot E_accumulated vs Z height
3. Compare TRENDS with OT data (not absolute values)
4. Adjust thresholds to match

---

## 6. Comparison with Temperature Model

| Aspect | Energy Model (This) | Temperature Model |
|--------|---------------------|-------------------|
| **Complexity** | Low | Medium-High |
| **Physics** | Energy balance | Rosenthal + Fourier |
| **Output** | Relative risk (Joules) | Absolute temp (°C) |
| **Accuracy** | ~60-70% | ~80-90% |
| **Speed** | Very fast | Fast |
| **Use case** | Quick screening | Detailed prediction |
| **Calibration** | OT trend matching | OT + FEM |

---

## 7. Project Structure

```
Overheating_Classifier/
├── src/
│   ├── __init__.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── slicer.py           # STL → 2D masks (shared with temp model)
│   │   └── geometry_score.py   # 3D Gaussian convolution (shared)
│   ├── energy/
│   │   ├── __init__.py
│   │   ├── accumulation.py     # Core energy tracking
│   │   └── parameters.py       # Material/process parameters
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # E_accumulated vs Z plots
├── tests/
│   └── test_*.py
├── validation/
│   ├── manifolds_geometry.stl  # From Ali
│   └── OT_data/                # From Ali
├── PLAN.md                     # This file
├── CLAUDE.md                   # Development guidelines
└── requirements.txt
```

---

## 8. Dependencies

```
# requirements.txt
numpy>=1.20
trimesh>=3.10
scipy>=1.7
matplotlib>=3.5
```

---

## 9. Meeting Notes Reference

See `MEETING_NOTES_2026-01-27.md` for the full context of Ali's recommendations.

Key quotes:

> "We just go simple as this energy input... the unit is joule per um like a unit in volume"

> "Let's not make it complicated and let's not figure out the temperature let's just simply say cumulation of joules"

> "Your previous layer cross sectional area would actually play a big role on that"

---

## 10. Next Steps

1. [ ] **Receive validation data from Ali**
   - Manifolds geometry STL
   - OT results
   - Gas flow velocity

2. [ ] **Implement core algorithm**
   - STL slicer (can reuse from temp model)
   - Geometry multiplier (can reuse from temp model)
   - Energy accumulation calculation

3. [ ] **Validate against OT data**
   - Compare trends
   - Calibrate thresholds

4. [ ] **Create simple visualization**
   - E_accumulated vs Z plot
   - 3D view with risk coloring

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01-27 | Initial draft based on Ali's meeting recommendations |

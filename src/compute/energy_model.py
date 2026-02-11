"""
Energy accumulation model for overheating risk prediction.

This module implements the core algorithm for tracking energy accumulation
per layer in LPBF (Laser Powder Bed Fusion) processes. Instead of calculating
actual temperatures, it uses geometric ratios to estimate relative heat buildup.

Region-aware: tracks energy per connected component (island) within each layer,
handling split/merge events when geometry branches or converges.

Based on Ali's recommendations from Jan 30, 2026 SRG meeting.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def _label_regions(mask: np.ndarray):
    """Label connected regions (islands) in a 2D binary mask.

    Returns (labeled_array, n_regions).
    """
    return ndimage.label(mask > 0)


def _build_overlap_map(prev_labeled: np.ndarray, curr_labeled: np.ndarray):
    """Build parent-child overlap maps between two labeled arrays.

    Returns (child_parents, parent_children) where:
      child_parents[child_rid] = {parent_rid: overlap_count, ...}
      parent_children[parent_rid] = {child_rid: overlap_count, ...}
    """
    overlap_mask = (curr_labeled > 0) & (prev_labeled > 0)
    if not np.any(overlap_mask):
        return {}, {}

    prev_ids = prev_labeled[overlap_mask]
    curr_ids = curr_labeled[overlap_mask]
    pairs = np.stack([prev_ids, curr_ids], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

    child_parents = {}   # child_rid -> {parent_rid: count}
    parent_children = {} # parent_rid -> {child_rid: count}

    for (parent_rid, child_rid), count in zip(unique_pairs, counts):
        parent_rid = int(parent_rid)
        child_rid = int(child_rid)
        count = int(count)

        if child_rid not in child_parents:
            child_parents[child_rid] = {}
        child_parents[child_rid][parent_rid] = count

        if parent_rid not in parent_children:
            parent_children[parent_rid] = {}
        parent_children[parent_rid][child_rid] = count

    return child_parents, parent_children


def calculate_energy_accumulation(
    masks: Dict[int, np.ndarray],
    G_layers: Optional[Dict[int, Union[float, np.ndarray]]] = None,
    dissipation_factor: float = 0.5,
    convection_factor: float = 0.05,
    use_geometry_multiplier: bool = False,
    area_ratio_power: float = 1.0,
    gaussian_ratio_power: float = 0.15,
    laser_power: float = 200.0,
    scan_speed: float = 800.0,
    hatch_distance: float = 0.1,
    layer_thickness: float = 0.03,
    voxel_size: float = 0.1,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict[int, float], Dict[int, float], Dict]:
    """
    Calculate energy accumulation per layer in Joules using region-aware tracking.

    Each layer is decomposed into connected components (regions/islands).
    Energy is tracked per-region with proper split/merge handling.
    The per-layer score is the MAX across all regions in that layer.

    Parameters
    ----------
    masks : dict
        Layer masks from STL slicer. Maps layer_number -> 2D numpy array.
    G_layers : dict, optional
        Geometry multiplier per layer (Mode B only).
    dissipation_factor : float
        Downward dissipation strength (0.0-1.0, default 0.5)
    convection_factor : float
        Top surface loss strength (0.0-0.5, default 0.05)
    use_geometry_multiplier : bool
        If True, use Mode B (geometry multiplier). If False, use Mode A (area-only).
    area_ratio_power : float
        Power exponent for area ratio in Mode A (1.0-3.0, default 3.0).
        Higher values amplify sensitivity to contact area changes.
    gaussian_ratio_power : float
        Power exponent for Gaussian multiplier in Mode B (0.01-1.0, default 0.15).
        Values < 1 dampen the effect, bringing extreme values closer to 1.
    laser_power : float
        Laser power in Watts (W = J/s). Default 200W.
    scan_speed : float
        Scan speed in mm/s. Default 800 mm/s.
    hatch_distance : float
        Hatch distance in mm. Default 0.1 mm.
    layer_thickness : float
        Layer thickness in mm. Default 0.03 mm (30 microns).
    voxel_size : float
        Voxel size in mm. Default 0.1 mm.
    progress_callback : callable, optional
        Progress callback function(percent, message)

    Returns
    -------
    tuple of (risk_scores, raw_energy_scores, region_data)
        risk_scores: dict mapping layer_number -> risk_score (0-1, normalized)
        raw_energy_scores: dict mapping layer_number -> energy in Joules
        region_data: dict mapping layer_number -> region info for visualization
    """
    n_layers = len(masks)
    if n_layers == 0:
        return {}, {}, {}

    # Validate laser parameters to prevent division by zero
    if scan_speed <= 0:
        raise ValueError(f"scan_speed must be positive, got {scan_speed}")
    if hatch_distance <= 0:
        raise ValueError(f"hatch_distance must be positive, got {hatch_distance}")
    if layer_thickness <= 0:
        raise ValueError(f"layer_thickness must be positive, got {layer_thickness}")

    # Calculate Energy Density: ED = P / (v · h · Δz)  [J/mm³]
    energy_density = laser_power / (scan_speed * hatch_distance * layer_thickness)
    logger.info(f"Energy density: {energy_density:.2f} J/mm³ "
                f"(P={laser_power}W, v={scan_speed}mm/s, h={hatch_distance}mm, Δz={layer_thickness}mm)")

    logger.info(f"Calculating region-aware energy accumulation for {n_layers} layers")
    logger.info(f"Parameters: dissipation_factor={dissipation_factor}, "
                f"convection_factor={convection_factor}, "
                f"use_geometry_multiplier={use_geometry_multiplier}, "
                f"area_ratio_power={area_ratio_power}, "
                f"gaussian_ratio_power={gaussian_ratio_power}, "
                f"laser_power={laser_power}W, scan_speed={scan_speed}mm/s, "
                f"hatch_distance={hatch_distance}mm")

    # Per-layer results
    E_layer_scores = {}  # layer -> raw energy (max across regions)
    region_data = {}     # layer -> region visualization data

    # State carried between layers
    prev_labeled = None
    prev_region_energies = {}    # rid -> energy
    prev_region_geo_factors = {} # rid -> geometry_factor (for empty layer decay)

    for n in range(1, n_layers + 1):
        if progress_callback and n % 50 == 0:
            progress_callback(int(n / n_layers * 100), f"Processing layer {n}/{n_layers}")

        mask = masks[n]
        A_total = np.sum(mask > 0)

        # --- Handle empty layers ---
        if A_total == 0:
            # Apply both conduction and convection decay to all previous regions
            if prev_region_energies:
                for rid in list(prev_region_energies.keys()):
                    geo_f = prev_region_geo_factors.get(rid, 0.0)
                    R_total = min(dissipation_factor * geo_f + convection_factor, 1.0)
                    prev_region_energies[rid] = max(0, prev_region_energies[rid] * (1 - R_total))
                E_layer_scores[n] = max(prev_region_energies.values()) if prev_region_energies else 0.0
            else:
                E_layer_scores[n] = 0.0

            region_data[n] = {
                'n_regions': 0,
                'labeled': np.zeros_like(mask, dtype=np.int32),
                'region_energies': {},
                'region_areas': {},
            }
            # prev_labeled stays the same (carry forward across gap)
            logger.debug(f"Layer {n}: EMPTY, E={E_layer_scores[n]:.2f}")
            continue

        # --- Label regions in current layer ---
        curr_labeled, n_regions = _label_regions(mask)
        logger.debug(f"Layer {n}: {n_regions} region(s)")

        # --- Build overlap map with previous layer ---
        if prev_labeled is not None:
            child_parents, parent_children = _build_overlap_map(prev_labeled, curr_labeled)
        else:
            child_parents, parent_children = {}, {}

        # --- Per-region energy calculation ---
        curr_region_energies = {}
        curr_region_areas = {}
        curr_region_contact_areas = {}
        curr_region_geo_factors = {}

        for rid in range(1, n_regions + 1):
            region_mask = (curr_labeled == rid)
            A_region = int(np.sum(region_mask))
            curr_region_areas[rid] = A_region

            if A_region == 0:
                continue

            # 1. Inherited energy from parent regions
            parents = child_parents.get(rid, {})
            E_inherited = 0.0

            if parents:
                for parent_rid, overlap_count in parents.items():
                    parent_energy = prev_region_energies.get(parent_rid, 0.0)
                    # Distribute parent energy proportionally across its children
                    parent_total_to_children = sum(parent_children.get(parent_rid, {}).values())
                    if parent_total_to_children > 0:
                        fraction = overlap_count / parent_total_to_children
                        E_inherited += parent_energy * fraction

            # 2. Energy input in Joules: E_in = ED × A_region_mm² × Δz
            A_region_mm2 = A_region * (voxel_size ** 2)  # Convert voxels to mm²
            E_in = energy_density * A_region_mm2 * layer_thickness  # Joules

            # Compute contact area (overlap with previous layer) for every region
            if prev_labeled is not None:
                A_contact_region = int(np.sum(region_mask & (prev_labeled > 0)))
            else:
                A_contact_region = A_region  # First layer: full contact with baseplate
            curr_region_contact_areas[rid] = A_contact_region

            # 3. Geometry factor (per-region)
            if n == 1:
                # Layer 1: baseplate = perfect heat sink
                geometry_factor = 1.0
            elif not parents:
                # Orphan region (not layer 1, no parent overlap)
                # Floating on powder = near-zero conduction downward
                geometry_factor = 0.0
            elif use_geometry_multiplier and G_layers is not None and n in G_layers:
                # Mode B: 1/(1+G_avg) per region (no area ratio)
                G_data = G_layers[n]
                if isinstance(G_data, np.ndarray):
                    G_vals = G_data[region_mask]
                    G_avg = float(np.mean(G_vals)) if len(G_vals) > 0 else 0.0
                else:
                    G_avg = float(G_data)
                geometry_factor = (1.0 / (1.0 + G_avg)) ** gaussian_ratio_power
            else:
                # Mode A: (A_contact_region / A_region) ^ area_ratio_power
                ratio = A_contact_region / A_region if A_region > 0 else 1.0
                ratio = min(ratio, 1.0)
                geometry_factor = ratio ** area_ratio_power

            curr_region_geo_factors[rid] = geometry_factor

            # 4. Dissipation
            R_total = min(dissipation_factor * geometry_factor + convection_factor, 1.0)

            # 5. Energy balance
            E_pool = E_inherited + E_in
            E_acc = max(0, E_pool * (1 - R_total))
            curr_region_energies[rid] = E_acc

        # --- Aggregate to per-layer scalar: USE MAX ---
        if curr_region_energies:
            E_layer_scores[n] = max(curr_region_energies.values())
        else:
            E_layer_scores[n] = 0.0

        # --- Store region data for visualization ---
        region_data[n] = {
            'n_regions': n_regions,
            'labeled': curr_labeled,
            'region_energies': dict(curr_region_energies),
            'region_areas': dict(curr_region_areas),
            'region_contact_areas': dict(curr_region_contact_areas),
        }

        # --- Update previous-layer state ---
        prev_labeled = curr_labeled
        prev_region_energies = curr_region_energies
        prev_region_geo_factors = curr_region_geo_factors

        logger.debug(f"Layer {n}: regions={n_regions}, "
                     f"E_max={E_layer_scores[n]:.2f}, "
                     f"energies={[f'{e:.1f}' for e in curr_region_energies.values()]}")

    # --- Normalize to 0-1 range ---
    E_max = max(E_layer_scores.values()) if E_layer_scores else 0
    if E_max > 0:
        risk_scores = {n: E / E_max for n, E in E_layer_scores.items()}
    else:
        risk_scores = {n: 0.0 for n in range(1, n_layers + 1)}
        logger.warning("All energy dissipated - returning uniform zero risk scores")

    logger.info(f"Energy accumulation complete. Max raw E={E_max:.2f} J, "
                f"Max risk score={max(risk_scores.values()) if risk_scores else 0:.3f}")

    # Return both normalized risk_scores and raw energy in Joules
    return risk_scores, E_layer_scores, region_data


def classify_risk_levels(
    risk_scores: Dict[int, float],
    threshold_medium: float = 0.3,
    threshold_high: float = 0.6
) -> Dict[int, str]:
    """
    Classify each layer's risk score into LOW/MEDIUM/HIGH categories.

    Parameters
    ----------
    risk_scores : dict
        Maps layer_number -> risk_score (0-1)
    threshold_medium : float
        Scores below this are LOW (default 0.3)
    threshold_high : float
        Scores above this are HIGH (default 0.6)

    Returns
    -------
    dict
        Maps layer_number -> risk_level ("LOW", "MEDIUM", "HIGH")
    """
    risk_levels = {}
    for layer, score in risk_scores.items():
        if score >= threshold_high:
            risk_levels[layer] = "HIGH"
        elif score >= threshold_medium:
            risk_levels[layer] = "MEDIUM"
        else:
            risk_levels[layer] = "LOW"
    return risk_levels


def calculate_layer_areas(
    masks: Dict[int, np.ndarray],
    voxel_size: float = 1.0
) -> Dict[int, float]:
    """
    Calculate cross-sectional area for each layer.

    Parameters
    ----------
    masks : dict
        Layer masks from STL slicer
    voxel_size : float
        Voxel size in mm (default 1.0 for voxel count)

    Returns
    -------
    dict
        Maps layer_number -> area in mm^2 (or voxel count if voxel_size=1.0)
    """
    return {
        layer: np.sum(mask > 0) * (voxel_size ** 2)
        for layer, mask in masks.items()
    }


def calculate_energy_density(
    raw_energy_scores: Dict[int, float],
    layer_areas: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate energy density (J/mm²) for each layer.

    Energy density is a more objective metric for overheating risk because
    it normalizes by scan area, preventing larger layers from artificially
    appearing as higher risk simply due to having more total energy.

    Parameters
    ----------
    raw_energy_scores : dict
        Maps layer_number -> accumulated energy in Joules
    layer_areas : dict
        Maps layer_number -> scan area in mm²

    Returns
    -------
    dict
        Maps layer_number -> energy density in J/mm²
    """
    energy_density = {}
    for layer in raw_energy_scores:
        energy = raw_energy_scores.get(layer, 0.0)
        area = layer_areas.get(layer, 0.0)
        if area > 0:
            energy_density[layer] = energy / area
        else:
            energy_density[layer] = 0.0
    return energy_density


def calculate_contact_areas(
    masks: Dict[int, np.ndarray],
    voxel_size: float = 1.0
) -> Dict[int, float]:
    """
    Calculate contact area between consecutive layers using voxel intersection.

    Parameters
    ----------
    masks : dict
        Layer masks from STL slicer
    voxel_size : float
        Voxel size in mm

    Returns
    -------
    dict
        Maps layer_number -> contact area with layer below in mm^2
        Layer 1 returns its own area (baseplate = unlimited contact)
    """
    n_layers = len(masks)
    contact_areas = {}

    for n in range(1, n_layers + 1):
        if n not in masks:
            contact_areas[n] = 0.0
            continue

        if n == 1:
            # Baseplate contact = full layer area
            contact_areas[n] = np.sum(masks[n] > 0) * (voxel_size ** 2)
        else:
            if n - 1 in masks:
                # Voxel intersection with layer below
                intersection = (masks[n] > 0) & (masks[n-1] > 0)
                contact_areas[n] = np.sum(intersection) * (voxel_size ** 2)
            else:
                contact_areas[n] = 0.0

    return contact_areas


def get_energy_statistics(
    risk_scores: Dict[int, float],
    risk_levels: Dict[int, str]
) -> Dict:
    """
    Calculate summary statistics for energy analysis results.

    Parameters
    ----------
    risk_scores : dict
        Maps layer_number -> risk_score (0-1)
    risk_levels : dict
        Maps layer_number -> risk_level

    Returns
    -------
    dict
        Summary statistics including counts, percentiles, max layer, etc.
    """
    if not risk_scores:
        return {
            'n_layers': 0,
            'n_low': 0,
            'n_medium': 0,
            'n_high': 0,
            'max_risk_score': 0.0,
            'max_risk_layer': 0,
            'mean_risk_score': 0.0,
            'risk_score_p90': 0.0,
            'risk_score_p95': 0.0,
        }

    scores = list(risk_scores.values())
    levels = list(risk_levels.values())

    # Find layer with max risk
    max_layer = max(risk_scores, key=risk_scores.get)

    return {
        'n_layers': len(risk_scores),
        'n_low': levels.count('LOW'),
        'n_medium': levels.count('MEDIUM'),
        'n_high': levels.count('HIGH'),
        'max_risk_score': float(max(scores)),
        'max_risk_layer': int(max_layer),
        'mean_risk_score': float(np.mean(scores)),
        'risk_score_p90': float(np.percentile(scores, 90)),
        'risk_score_p95': float(np.percentile(scores, 95)),
    }


def run_energy_analysis(
    masks: Dict[int, np.ndarray],
    G_layers: Optional[Dict[int, Union[float, np.ndarray]]] = None,
    dissipation_factor: float = 0.5,
    convection_factor: float = 0.05,
    use_geometry_multiplier: bool = False,
    area_ratio_power: float = 1.0,
    gaussian_ratio_power: float = 0.15,
    threshold_medium: float = 0.3,
    threshold_high: float = 0.6,
    voxel_size: float = 0.1,
    layer_thickness: float = 0.03,
    laser_power: float = 200.0,
    scan_speed: float = 800.0,
    hatch_distance: float = 0.1,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Run complete energy analysis pipeline.

    This is the main entry point that combines:
    1. Region-aware energy accumulation calculation (in Joules)
    2. Risk classification
    3. Statistics generation

    Parameters
    ----------
    masks : dict
        Layer masks from STL slicer
    G_layers : dict, optional
        Geometry multiplier per layer (Mode B only)
    dissipation_factor : float
        Downward dissipation strength (0.0-1.0)
    convection_factor : float
        Top surface loss strength (0.0-0.5)
    use_geometry_multiplier : bool
        If True, use Mode B. If False, use Mode A.
    area_ratio_power : float
        Power exponent for area ratio in Mode A (1.0-3.0)
    threshold_medium : float
        Risk classification threshold for MEDIUM
    threshold_high : float
        Risk classification threshold for HIGH
    voxel_size : float
        Voxel size in mm
    layer_thickness : float
        Layer thickness in mm
    laser_power : float
        Laser power in Watts
    scan_speed : float
        Scan speed in mm/s
    hatch_distance : float
        Hatch distance in mm
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    dict
        Complete analysis results including:
        - risk_scores: normalized scores per layer (0-1)
        - raw_energy_scores: energy in Joules per layer
        - risk_levels: classification per layer
        - layer_areas: cross-sectional areas
        - contact_areas: contact areas with layer below
        - region_data: per-layer region info for visualization
        - summary: statistics
        - params: parameters used
    """
    logger.info("Starting energy analysis pipeline")

    # Calculate areas
    layer_areas = calculate_layer_areas(masks, voxel_size)
    contact_areas = calculate_contact_areas(masks, voxel_size)

    # Calculate energy accumulation (region-aware) in Joules
    # Note: risk_scores from this function are based on total energy (legacy)
    energy_risk_scores, raw_energy_scores, region_data = calculate_energy_accumulation(
        masks=masks,
        G_layers=G_layers,
        dissipation_factor=dissipation_factor,
        convection_factor=convection_factor,
        use_geometry_multiplier=use_geometry_multiplier,
        area_ratio_power=area_ratio_power,
        gaussian_ratio_power=gaussian_ratio_power,
        laser_power=laser_power,
        scan_speed=scan_speed,
        hatch_distance=hatch_distance,
        layer_thickness=layer_thickness,
        voxel_size=voxel_size,
        progress_callback=progress_callback
    )

    # Calculate energy density (J/mm²) - more objective metric
    energy_density_scores = calculate_energy_density(raw_energy_scores, layer_areas)

    # Normalize energy density to 0-1 range for risk classification
    max_density = max(energy_density_scores.values()) if energy_density_scores else 0
    if max_density > 0:
        risk_scores = {n: d / max_density for n, d in energy_density_scores.items()}
    else:
        risk_scores = {n: 0.0 for n in energy_density_scores}
        logger.warning("All energy density is zero - returning uniform zero risk scores")

    logger.info(f"Energy density: max={max_density:.4f} J/mm², "
                f"mean={np.mean(list(energy_density_scores.values())):.4f} J/mm²")

    # Classify risk levels based on energy density (not total energy)
    risk_levels = classify_risk_levels(
        risk_scores=risk_scores,
        threshold_medium=threshold_medium,
        threshold_high=threshold_high
    )

    # Get statistics (now based on energy density)
    summary = get_energy_statistics(risk_scores, risk_levels)

    logger.info(f"Energy analysis complete (density-based): {summary['n_high']} HIGH, "
                f"{summary['n_medium']} MEDIUM, {summary['n_low']} LOW")

    return {
        'risk_scores': risk_scores,  # Normalized energy density (0-1) - used for risk classification
        'raw_energy_scores': raw_energy_scores,  # Total energy in Joules
        'energy_density_scores': energy_density_scores,  # Energy density in J/mm²
        'risk_levels': risk_levels,  # Classification based on energy density
        'layer_areas': layer_areas,
        'contact_areas': contact_areas,
        'region_data': region_data,
        'summary': summary,
        'params': {
            'dissipation_factor': dissipation_factor,
            'convection_factor': convection_factor,
            'use_geometry_multiplier': use_geometry_multiplier,
            'area_ratio_power': area_ratio_power,
            'gaussian_ratio_power': gaussian_ratio_power,
            'threshold_medium': threshold_medium,
            'threshold_high': threshold_high,
            'mode': 'geometry_multiplier' if use_geometry_multiplier else 'area_only',
            'laser_power': laser_power,
            'scan_speed': scan_speed,
            'hatch_distance': hatch_distance,
        }
    }

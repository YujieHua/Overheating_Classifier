"""
Energy accumulation model for overheating risk prediction.

This module implements the core algorithm for tracking energy accumulation
per layer in LPBF (Laser Powder Bed Fusion) processes. Instead of calculating
actual temperatures, it uses geometric ratios to estimate relative heat buildup.

Based on Ali's recommendations from Jan 30, 2026 SRG meeting.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_energy_accumulation(
    masks: Dict[int, np.ndarray],
    G_layers: Optional[Dict[int, Union[float, np.ndarray]]] = None,
    dissipation_factor: float = 0.5,
    convection_factor: float = 0.05,
    use_geometry_multiplier: bool = False,
    progress_callback: Optional[callable] = None
) -> Dict[int, float]:
    """
    Calculate normalized energy accumulation per layer.

    The algorithm tracks a dimensionless "energy proxy" that represents
    relative heat buildup. Higher values indicate higher overheating risk.

    Energy Balance:
        E_pool = E_accumulated[n-1] + E_in
        E_out = E_pool * R_total
        E_accumulated[n] = max(0, E_pool - E_out)

    Where:
        E_in = A_layer[n] (area-based input)
        R_total = R_down + R_top (combined dissipation)
        R_down = dissipation_factor * geometry_factor
        R_top = convection_factor

    Parameters
    ----------
    masks : dict
        Layer masks from STL slicer. Maps layer_number -> 2D numpy array.
        Layer numbers are 1-indexed.
    G_layers : dict, optional
        Geometry multiplier per layer (Mode B only).
        Can be scalar float or 2D array per layer.
    dissipation_factor : float
        Downward dissipation strength (0.0-1.0, default 0.5)
    convection_factor : float
        Top surface loss strength (0.0-0.5, default 0.05)
    use_geometry_multiplier : bool
        If True, use Mode B (geometry multiplier). If False, use Mode A (area-only).
    progress_callback : callable, optional
        Progress callback function(percent, message)

    Returns
    -------
    dict
        Maps layer_number -> risk_score (0-1, normalized)
    """
    n_layers = len(masks)
    if n_layers == 0:
        return {}

    E_accumulated = {0: 0.0}  # Baseplate = perfect heat sink
    E_accumulated_raw = {}  # Store raw values before normalization

    logger.info(f"Calculating energy accumulation for {n_layers} layers")
    logger.info(f"Parameters: dissipation_factor={dissipation_factor}, "
                f"convection_factor={convection_factor}, "
                f"use_geometry_multiplier={use_geometry_multiplier}")

    for n in range(1, n_layers + 1):
        if progress_callback and n % 50 == 0:
            progress_callback(int(n / n_layers * 100), f"Processing layer {n}/{n_layers}")

        # Current layer area (voxel count)
        A_layer = np.sum(masks[n] > 0)

        # Handle empty layers (still apply convection loss from previous)
        if A_layer == 0:
            E_out_top = E_accumulated[n-1] * convection_factor
            E_accumulated[n] = max(0, E_accumulated[n-1] - E_out_top)
            E_accumulated_raw[n] = E_accumulated[n]
            logger.debug(f"Layer {n}: EMPTY, E={E_accumulated[n]:.2f}")
            continue

        # Contact area calculation
        if n == 1:
            # First layer: baseplate provides unlimited contact (perfect heat sink)
            # No thermal bottleneck - full layer area can dissipate
            A_contact = A_layer
        else:
            # Voxel-based intersection: count solid voxels present in BOTH layers
            # This correctly handles offset/non-aligned geometry
            A_contact = np.sum((masks[n] > 0) & (masks[n-1] > 0))
            # Fallback: if no intersection detected (floating layer), use min area
            if A_contact == 0:
                A_contact = min(A_layer, np.sum(masks[n-1] > 0))
                if A_contact == 0:
                    A_contact = A_layer  # Completely disconnected - treat as first layer

        # Energy input (proportional to area)
        E_in = float(A_layer)

        # Geometry factor for downward dissipation
        geometry_factor = A_contact / A_layer if A_layer > 0 else 1.0

        if use_geometry_multiplier and G_layers is not None and n in G_layers:
            # Calculate average G for solid voxels in this layer
            G_data = G_layers[n]
            if isinstance(G_data, np.ndarray):
                solid_mask = masks[n] > 0
                if solid_mask.sum() > 0:
                    G_avg = float(np.mean(G_data[solid_mask]))
                else:
                    G_avg = 0.0
            else:
                G_avg = float(G_data)  # Already a scalar
            geometry_factor *= 1.0 / (1.0 + G_avg)
            logger.debug(f"Layer {n}: G_avg={G_avg:.3f}, geometry_factor={geometry_factor:.3f}")

        # Combined dissipation ratio
        R_down = dissipation_factor * geometry_factor
        R_top = convection_factor
        R_total = R_down + R_top

        # Clamp R_total to prevent negative energy
        R_total = min(R_total, 1.0)

        # Energy balance: apply dissipation to TOTAL pool (existing + new)
        E_pool = E_accumulated[n-1] + E_in
        E_out = E_pool * R_total
        E_accumulated[n] = max(0, E_pool - E_out)
        E_accumulated_raw[n] = E_accumulated[n]

        logger.debug(f"Layer {n}: A={A_layer}, A_contact={A_contact}, "
                     f"E_in={E_in:.2f}, E_out={E_out:.2f}, E_acc={E_accumulated[n]:.2f}")

    # Normalize to 0-1 range
    E_max = max(E_accumulated.values()) if E_accumulated else 0
    if E_max > 0:
        risk_scores = {n: E / E_max for n, E in E_accumulated.items() if n > 0}
    else:
        # All energy dissipated - report uniform low risk
        risk_scores = {n: 0.0 for n in range(1, n_layers + 1)}
        logger.warning("All energy dissipated - returning uniform zero risk scores")

    logger.info(f"Energy accumulation complete. Max raw E={E_max:.2f}, "
                f"Max risk score={max(risk_scores.values()) if risk_scores else 0:.3f}")

    return risk_scores


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
        Maps layer_number -> area in mm² (or voxel count if voxel_size=1.0)
    """
    return {
        layer: np.sum(mask > 0) * (voxel_size ** 2)
        for layer, mask in masks.items()
    }


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
        Maps layer_number -> contact area with layer below in mm²
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
    threshold_medium: float = 0.3,
    threshold_high: float = 0.6,
    voxel_size: float = 1.0,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Run complete energy analysis pipeline.

    This is the main entry point that combines:
    1. Energy accumulation calculation
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
    threshold_medium : float
        Risk classification threshold for MEDIUM
    threshold_high : float
        Risk classification threshold for HIGH
    voxel_size : float
        Voxel size in mm
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    dict
        Complete analysis results including:
        - risk_scores: normalized scores per layer
        - risk_levels: classification per layer
        - layer_areas: cross-sectional areas
        - contact_areas: contact areas with layer below
        - summary: statistics
        - params: parameters used
    """
    logger.info("Starting energy analysis pipeline")

    # Calculate areas
    layer_areas = calculate_layer_areas(masks, voxel_size)
    contact_areas = calculate_contact_areas(masks, voxel_size)

    # Calculate energy accumulation
    risk_scores = calculate_energy_accumulation(
        masks=masks,
        G_layers=G_layers,
        dissipation_factor=dissipation_factor,
        convection_factor=convection_factor,
        use_geometry_multiplier=use_geometry_multiplier,
        progress_callback=progress_callback
    )

    # Classify risk levels
    risk_levels = classify_risk_levels(
        risk_scores=risk_scores,
        threshold_medium=threshold_medium,
        threshold_high=threshold_high
    )

    # Get statistics
    summary = get_energy_statistics(risk_scores, risk_levels)

    logger.info(f"Energy analysis complete: {summary['n_high']} HIGH, "
                f"{summary['n_medium']} MEDIUM, {summary['n_low']} LOW")

    return {
        'risk_scores': risk_scores,
        'risk_levels': risk_levels,
        'layer_areas': layer_areas,
        'contact_areas': contact_areas,
        'summary': summary,
        'params': {
            'dissipation_factor': dissipation_factor,
            'convection_factor': convection_factor,
            'use_geometry_multiplier': use_geometry_multiplier,
            'threshold_medium': threshold_medium,
            'threshold_high': threshold_high,
            'mode': 'geometry_multiplier' if use_geometry_multiplier else 'area_only',
        }
    }

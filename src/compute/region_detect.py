"""
2D per-layer region detection and cross-layer connectivity.

Detects disconnected 2D regions within each layer mask and computes
overlap-based connectivity between regions on adjacent layers. This
enables per-region temperature tracking where heat only flows between
physically overlapping regions.
"""

import numpy as np
from scipy.ndimage import label
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# Virtual build plate entity key: (layer=0, region_id=1).
# Used in conn_above_lookup to represent heat flow from layer 1 → build plate.
BUILD_PLATE_KEY = (0, 1)


def detect_2d_regions(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Detect disconnected 2D regions in a single layer mask.

    Uses 8-connectivity (including diagonals) because diagonal pixels
    touching at corners are still part of the same region in a laser
    scan path.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask (nonzero = solid)

    Returns
    -------
    tuple
        (label_map, n_regions) where label_map has 0=background,
        1..N=region IDs
    """
    binary = (mask > 0).astype(np.int32)

    # 8-connectivity structure (includes diagonals)
    structure = np.ones((3, 3), dtype=np.int32)

    label_map, n_regions = label(binary, structure=structure)

    return label_map, n_regions


def detect_all_layer_regions(
    masks: Dict[int, np.ndarray],
    voxel_size: float,
    layer_thickness: float,
    G_layers: Optional[Dict[int, np.ndarray]] = None
) -> Dict:
    """
    Detect 2D regions for all layers with per-region metadata.

    Parameters
    ----------
    masks : dict
        Binary masks for each layer (1-indexed)
    voxel_size : float
        Voxel size in mm
    layer_thickness : float
        Layer thickness in mm
    G_layers : dict, optional
        Geometry multiplier G for each layer (2D arrays)

    Returns
    -------
    dict
        regions_per_layer[layer] = {
            'n_regions': int,
            'label_map': np.ndarray,
            'regions': {
                region_id: {
                    'pixel_count': int,
                    'area_mm2': float,
                    'G_avg': float,
                    'mask': np.ndarray (boolean 2D)
                }
            }
        }
    """
    n_layers = len(masks)
    regions_per_layer = {}

    for layer in range(1, n_layers + 1):
        mask = masks[layer]
        label_map, n_regions = detect_2d_regions(mask)

        regions = {}
        for rid in range(1, n_regions + 1):
            region_mask = label_map == rid
            pixel_count = int(region_mask.sum())
            area_mm2 = pixel_count * (voxel_size ** 2)

            # Compute average G for this region
            G_avg = 0.0
            if G_layers is not None and layer in G_layers:
                G = G_layers[layer]
                if region_mask.any():
                    G_avg = float(G[region_mask].mean())

            regions[rid] = {
                'pixel_count': pixel_count,
                'area_mm2': area_mm2,
                'G_avg': G_avg,
                'mask': region_mask,
            }

        regions_per_layer[layer] = {
            'n_regions': n_regions,
            'label_map': label_map,
            'regions': regions,
        }

        if n_regions > 1:
            logger.info(f"  Layer {layer}: {n_regions} regions detected "
                        f"(areas: {[r['area_mm2'] for r in regions.values()]})")

    # Summary
    multi_region_layers = sum(
        1 for v in regions_per_layer.values() if v['n_regions'] > 1
    )
    total_regions = sum(v['n_regions'] for v in regions_per_layer.values())
    logger.info(f"Region detection: {total_regions} total regions across {n_layers} layers "
                f"({multi_region_layers} layers with multiple regions)")

    return regions_per_layer


def compute_cross_layer_connectivity(
    regions_per_layer: Dict,
    voxel_size: float
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute pixel overlap between regions on adjacent layers.

    For each pair of adjacent layers (K, K-1), finds which regions
    overlap and by how much. Also handles layer 1 -> build plate
    connectivity.

    Parameters
    ----------
    regions_per_layer : dict
        Output from detect_all_layer_regions()
    voxel_size : float
        Voxel size in mm

    Returns
    -------
    tuple
        (region_connections, conn_below_lookup, conn_above_lookup)

        region_connections[layer] = list of connection dicts
        conn_below_lookup[(layer, rid)] = list of connections where upper_region == rid
        conn_above_lookup[(layer, rid)] = list of connections from layer+1 where lower_region == rid
    """
    layers = sorted(regions_per_layer.keys())
    region_connections = {}
    conn_below_lookup = {}
    conn_above_lookup = {}

    # Initialize empty lookups for all (layer, region) pairs
    for layer in layers:
        for rid in regions_per_layer[layer]['regions']:
            conn_below_lookup[(layer, rid)] = []
            conn_above_lookup[(layer, rid)] = []

    # Layer 1 -> build plate connections
    if 1 in regions_per_layer:
        region_connections[1] = []
        for rid, rinfo in regions_per_layer[1]['regions'].items():
            conn = {
                'upper_region': rid,
                'lower_region': 1,  # Virtual build plate region
                'overlap_pixels': rinfo['pixel_count'],
                'overlap_area_mm2': rinfo['area_mm2'],
            }
            region_connections[1].append(conn)
            conn_below_lookup[(1, rid)].append(conn)

    # Build plate above-lookup (build plate connects up to layer 1 regions)
    conn_above_lookup[BUILD_PLATE_KEY] = []

    # Adjacent layer connections
    for i in range(1, len(layers)):
        layer_above = layers[i]
        layer_below = layers[i - 1]

        connections = []
        regions_above = regions_per_layer[layer_above]['regions']
        regions_below = regions_per_layer[layer_below]['regions']

        for rid_above, rinfo_above in regions_above.items():
            mask_above = rinfo_above['mask']

            for rid_below, rinfo_below in regions_below.items():
                mask_below = rinfo_below['mask']

                # Compute overlap
                overlap = np.logical_and(mask_above, mask_below)
                overlap_pixels = int(overlap.sum())

                if overlap_pixels > 0:
                    overlap_area = overlap_pixels * (voxel_size ** 2)
                    conn = {
                        'upper_region': rid_above,
                        'lower_region': rid_below,
                        'overlap_pixels': overlap_pixels,
                        'overlap_area_mm2': overlap_area,
                    }
                    connections.append(conn)
                    conn_below_lookup[(layer_above, rid_above)].append(conn)
                    conn_above_lookup[(layer_below, rid_below)].append(conn)

        region_connections[layer_above] = connections

        # Log multi-connection layers
        if len(connections) > 1:
            logger.debug(f"  Layer {layer_above}->{layer_below}: "
                         f"{len(connections)} region connections")

    # Log floating regions (no connection below)
    for layer in layers:
        for rid in regions_per_layer[layer]['regions']:
            if not conn_below_lookup.get((layer, rid)):
                if layer > 1:
                    logger.warning(f"  Floating region: layer {layer}, region {rid} "
                                   f"(no overlap with layer below)")

    return region_connections, conn_below_lookup, conn_above_lookup

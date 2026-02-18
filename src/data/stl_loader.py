"""
STL file loading and slicing module.

Provides functions to load STL files and slice them into binary masks
for thermal simulation.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def load_stl(filepath: str) -> Dict:
    """
    Load an STL file and extract mesh information.

    Parameters
    ----------
    filepath : str
        Path to the STL file

    Returns
    -------
    dict
        Mesh information including vertices, faces, bounds, dimensions
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"STL file not found: {filepath}")

    # Check file size (max 50 MB)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 50:
        raise ValueError(f"STL file too large: {file_size_mb:.1f} MB (max 50 MB)")

    logger.info(f"Loading STL: {filepath}")

    # Load mesh
    mesh = trimesh.load(filepath)

    # Get bounds
    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    dimensions = bounds[1] - bounds[0]

    # Validate
    if not mesh.is_watertight:
        logger.warning("Mesh is not watertight - results may have gaps")

    info = {
        'vertices': np.array(mesh.vertices),
        'faces': np.array(mesh.faces),
        'bounds': bounds,
        'dimensions': dimensions,
        'n_triangles': len(mesh.faces),
        'n_vertices': len(mesh.vertices),
        'is_watertight': mesh.is_watertight,
        'volume': mesh.volume if mesh.is_watertight else None,
        'mesh': mesh,  # Keep reference for slicing
    }

    logger.info(f"Loaded STL: {info['n_triangles']} triangles, "
                f"dimensions {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} mm")

    return info


def slice_stl(mesh_info: Dict,
              layer_thickness: float = 0.04,
              voxel_size: float = 0.1,
              layer_grouping: int = 1,
              progress_callback: Optional[Callable] = None) -> Dict:
    """
    Slice STL mesh into binary layer masks AND extract contours for visualization.

    Uses RAY CASTING for robust inside/outside determination:
    - Cast rays from each XY grid point in +Z direction
    - Count mesh surface intersections above each Z height
    - Odd count = inside, even count = outside

    OPTIMIZED: Uses fully vectorized NumPy operations instead of Python loops
    for significant speedup (10-100x faster on large models).

    Returns both:
    - masks: binary voxel data for thermal simulation and G calculation
    - layer_contours: polygon contours for surface-based visualization

    Parameters
    ----------
    mesh_info : dict
        Output from load_stl()
    layer_thickness : float
        Layer thickness in mm (default 0.04 mm = 40 microns)
    voxel_size : float
        XY voxel size in mm
    layer_grouping : int
        Number of layers to group together (skip layers during slicing for speed)
    progress_callback : callable, optional
        Function to call with (progress_percent, message)

    Returns
    -------
    dict
        Contains:
        - masks: dict[layer_num] -> 2D binary array (for simulation)
        - layer_contours: dict[layer_num] -> list of polygon coords (for visualization)
        - n_layers: number of layers (after grouping)
        - grid_shape: (nx, ny) shape of each mask
        - voxel_size: actual voxel size used
        - layer_thickness: actual layer thickness used (multiplied by grouping)
        - z_heights: z-coordinate of each layer
    """
    import time
    _t0 = time.perf_counter()

    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required")

    mesh = mesh_info['mesh']
    bounds = mesh_info['bounds']

    # Calculate effective layer thickness with grouping
    effective_layer_thickness = layer_thickness * layer_grouping

    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    # Add padding
    padding = voxel_size
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    nx = int(np.ceil((x_max - x_min) / voxel_size))
    ny = int(np.ceil((y_max - y_min) / voxel_size))
    n_layers = int(np.ceil((z_max - z_min) / effective_layer_thickness))

    logger.info(f"Slicing into {n_layers} layers (grouping={layer_grouping}), grid size {nx} x {ny}")

    # Coordinate arrays for output
    x_coords = np.linspace(x_min + voxel_size/2, x_min + (nx - 0.5) * voxel_size, nx)
    y_coords = np.linspace(y_min + voxel_size/2, y_min + (ny - 0.5) * voxel_size, ny)

    # ===== RAY CASTING for inside/outside determination =====
    # This is mathematically robust - no voxel fill needed
    if progress_callback:
        progress_callback(5, "Setting up ray casting grid...")

    # Create grid of XY points
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    n_points = nx * ny

    # Ray origins: start from below the mesh
    ray_origins = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(n_points, z_min - 1.0)  # Start below mesh
    ])

    # Ray directions: +Z (upward)
    ray_directions = np.tile([0, 0, 1], (n_points, 1)).astype(np.float64)

    if progress_callback:
        progress_callback(10, f"Casting {n_points} rays through mesh...")

    _t1 = time.perf_counter()

    # Find all ray-mesh intersections
    # Returns: locations (Nx3), ray_indices, triangle_indices
    try:
        intersector = mesh.ray
        locations, ray_indices, _ = intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        _t2 = time.perf_counter()
        logger.info(f"[TIMING] Ray casting: {_t2 - _t1:.3f}s ({len(locations)} intersections)")

        if progress_callback:
            progress_callback(30, f"Found {len(locations)} intersections, building Z-lists (vectorized)...")

        # ===== VECTORIZED Z-HIT ORGANIZATION =====
        # Instead of Python loops, use NumPy groupby-style operations

        if len(locations) > 0:
            z_values = locations[:, 2]  # Extract Z coordinates
            ray_indices = np.asarray(ray_indices)

            # Sort by ray index, then by z value within each ray
            # This groups all intersections for the same ray together
            sort_order = np.lexsort((z_values, ray_indices))
            sorted_ray_indices = ray_indices[sort_order]
            sorted_z_values = z_values[sort_order]

            # Find boundaries between different rays
            # np.diff finds where ray index changes
            ray_changes = np.diff(sorted_ray_indices, prepend=-1) != 0
            ray_start_positions = np.where(ray_changes)[0]

            # Count hits per ray
            unique_rays, hit_counts = np.unique(sorted_ray_indices, return_counts=True)
            max_hits = hit_counts.max() if len(hit_counts) > 0 else 0

            # Build padded array: shape (n_points, max_hits)
            # Use infinity as padding (will always be "above" any z, so contributes 0 to count)
            z_hits_padded = np.full((n_points, max_hits), np.inf, dtype=np.float64)
            hit_count_per_ray = np.zeros(n_points, dtype=np.int32)

            # Fill in the actual z values for rays that have intersections
            for i, ray_idx in enumerate(unique_rays):
                start = ray_start_positions[np.searchsorted(unique_rays, ray_idx)]
                count = hit_counts[i]
                z_hits_padded[ray_idx, :count] = sorted_z_values[start:start+count]
                hit_count_per_ray[ray_idx] = count

            _t3 = time.perf_counter()
            logger.info(f"[TIMING] Z-hit organization: {_t3 - _t2:.3f}s")
        else:
            # No intersections - all points outside
            max_hits = 0
            z_hits_padded = np.full((n_points, 1), np.inf, dtype=np.float64)
            hit_count_per_ray = np.zeros(n_points, dtype=np.int32)
            _t3 = time.perf_counter()

        ray_casting_success = True

    except Exception as e:
        logger.warning(f"Ray casting failed: {e}, falling back to mesh.contains()")
        ray_casting_success = False
        z_hits_padded = None
        hit_count_per_ray = None
        _t3 = time.perf_counter()

    if progress_callback:
        progress_callback(40, "Extracting layer masks (vectorized)...")

    # ===== VECTORIZED LAYER MASK GENERATION =====
    # Compute ALL layer masks at once using broadcasting

    # Z heights for all layers
    z_heights = z_min + (np.arange(1, n_layers + 1) - 0.5) * effective_layer_thickness

    masks = {}
    layer_contours = {}

    if ray_casting_success and z_hits_padded is not None:
        # Vectorized approach: compute all masks simultaneously
        # For each (point, layer), count intersections above z_height
        # Shape: z_heights is (n_layers,), z_hits_padded is (n_points, max_hits)
        # We want: for each point, for each layer, count how many z_hits > z_height

        # Broadcasting: z_hits_padded[:, :, None] is (n_points, max_hits, 1)
        #               z_heights[None, None, :] is (1, 1, n_layers)
        # Comparison result: (n_points, max_hits, n_layers)
        # Sum over max_hits axis: (n_points, n_layers)

        _t4 = time.perf_counter()

        # Process in chunks to avoid memory explosion for large grids
        chunk_size = min(50, n_layers)  # Process 50 layers at a time
        all_masks_3d = np.zeros((n_points, n_layers), dtype=np.uint8)

        for chunk_start in range(0, n_layers, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_layers)
            z_chunk = z_heights[chunk_start:chunk_end]

            # Count intersections above each z height for this chunk
            # z_hits_padded: (n_points, max_hits)
            # z_chunk: (chunk_size,)
            # Compare: (n_points, max_hits, chunk_size)
            above_mask = z_hits_padded[:, :, None] > z_chunk[None, None, :]
            counts_above = above_mask.sum(axis=1)  # (n_points, chunk_size)

            # Odd count = inside
            all_masks_3d[:, chunk_start:chunk_end] = (counts_above % 2).astype(np.uint8)

            if progress_callback:
                progress = 40 + (chunk_end / n_layers) * 40
                progress_callback(progress, f"Computing masks: layers {chunk_start+1}-{chunk_end}/{n_layers}")

        _t5 = time.perf_counter()
        logger.info(f"[TIMING] Vectorized mask computation: {_t5 - _t4:.3f}s for {n_layers} layers")

        # Reshape and store masks
        for layer_idx in range(n_layers):
            masks[layer_idx + 1] = all_masks_3d[:, layer_idx].reshape(nx, ny)
    else:
        # Fallback to mesh.contains() - slower but works
        _t4 = time.perf_counter()
        for layer in range(1, n_layers + 1):
            if progress_callback:
                progress = 40 + (layer / n_layers) * 40
                progress_callback(progress, f"Processing layer {layer}/{n_layers} (fallback)")

            z = z_heights[layer - 1]
            points_3d = np.column_stack([
                xx.ravel(),
                yy.ravel(),
                np.full(n_points, z)
            ])
            try:
                inside = mesh.contains(points_3d)
            except Exception:
                inside = np.zeros(n_points, dtype=bool)
            masks[layer] = inside.reshape(nx, ny).astype(np.uint8)
        _t5 = time.perf_counter()
        logger.info(f"[TIMING] Fallback mask computation: {_t5 - _t4:.3f}s")

    if progress_callback:
        progress_callback(80, "Extracting contours...")

    # Convert z_heights to list for output
    z_heights = z_heights.tolist()

    # Extract contours for each layer (this part remains sequential as it's I/O bound)
    _t6 = time.perf_counter()
    for layer in range(1, n_layers + 1):
        mask = masks[layer]
        z = z_heights[layer - 1]

        # ===== CONTOURS (for visualization) =====
        # Primary method: Generate contours from voxel mask using marching squares
        # This is more reliable than mesh.section() which can fail at boundaries
        polygons = []

        if mask.sum() > 0:
            try:
                from skimage.measure import find_contours
                # find_contours returns contours in (row, col) = (y_idx, x_idx) format
                contours = find_contours(mask.T, 0.5)  # Transpose to match (y, x) convention

                for contour in contours:
                    if len(contour) >= 3:
                        # Convert pixel indices to world coordinates
                        # contour is in (row, col) format, we need (x, y) in world coords
                        coords = []
                        for pt in contour:
                            # pt[0] is row (y_index), pt[1] is col (x_index)
                            x_world = x_coords[0] + pt[1] * voxel_size
                            y_world = y_coords[0] + pt[0] * voxel_size
                            coords.append((float(x_world), float(y_world)))
                        if len(coords) >= 3:
                            polygons.append(coords)

            except ImportError:
                logger.debug("skimage not available, falling back to mesh.section")
            except Exception as e:
                logger.debug(f"Mask contour extraction failed for layer {layer}: {e}")

        # Fallback: Use mesh.section() if mask-based extraction failed
        if not polygons:
            try:
                section = mesh.section(
                    plane_origin=[0, 0, z],
                    plane_normal=[0, 0, 1]
                )

                if section is not None:
                    path_2d, transform = section.to_planar()

                    # Try multiple methods to extract polygon coordinates
                    if hasattr(path_2d, 'polygons_closed') and path_2d.polygons_closed:
                        for poly in path_2d.polygons_closed:
                            coords = list(poly.exterior.coords)
                            polygons.append([(float(c[0]), float(c[1])) for c in coords])
                    elif hasattr(path_2d, 'polygons_full') and path_2d.polygons_full:
                        for poly in path_2d.polygons_full:
                            coords = list(poly.exterior.coords)
                            polygons.append([(float(c[0]), float(c[1])) for c in coords])
                    elif hasattr(path_2d, 'discrete') and path_2d.discrete is not None:
                        for discrete_path in path_2d.discrete:
                            if len(discrete_path) >= 3:
                                coords = [(float(p[0]), float(p[1])) for p in discrete_path]
                                polygons.append(coords)
                    elif hasattr(path_2d, 'vertices') and len(path_2d.vertices) > 0:
                        if hasattr(path_2d, 'entities'):
                            for entity in path_2d.entities:
                                if hasattr(entity, 'points'):
                                    indices = entity.points
                                    coords = [(float(path_2d.vertices[i][0]), float(path_2d.vertices[i][1])) for i in indices]
                                    if len(coords) >= 3:
                                        polygons.append(coords)
                        else:
                            coords = [(float(v[0]), float(v[1])) for v in path_2d.vertices]
                            if len(coords) >= 3:
                                polygons.append(coords)
            except Exception as e:
                logger.debug(f"mesh.section fallback failed for layer {layer}: {e}")

        layer_contours[layer] = {'polygons': polygons, 'z': z}

    _t7 = time.perf_counter()
    logger.info(f"[TIMING] Contour extraction: {_t7 - _t6:.3f}s")

    if progress_callback:
        progress_callback(95, "Finalizing slice data...")

    result = {
        'masks': masks,
        'layer_contours': layer_contours,
        'n_layers': n_layers,
        'grid_shape': (nx, ny),
        'voxel_size': voxel_size,
        'layer_thickness': effective_layer_thickness,
        'layer_grouping': layer_grouping,
        'z_heights': z_heights,
        'bounds': bounds,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }

    n_solid = sum(m.sum() for m in masks.values())
    n_contours = sum(1 for c in layer_contours.values() if c['polygons'])

    _t_end = time.perf_counter()
    logger.info(f"[TIMING] TOTAL slicing time: {_t_end - _t0:.3f}s")
    logger.info(f"Slicing complete: {n_layers} layers, {n_solid} solid voxels, {n_contours} with contours")

    return result


def create_3d_voxel_grid(slice_result: Dict) -> np.ndarray:
    """
    Stack 2D masks into a 3D voxel grid.

    Parameters
    ----------
    slice_result : dict
        Output from slice_stl()

    Returns
    -------
    np.ndarray
        3D binary array of shape (n_layers, nx, ny)
    """
    n_layers = slice_result['n_layers']
    nx, ny = slice_result['grid_shape']

    voxels = np.zeros((n_layers, nx, ny), dtype=np.uint8)

    for layer in range(1, n_layers + 1):
        voxels[layer - 1] = slice_result['masks'][layer]

    return voxels


def validate_stl_file(filepath: str) -> Tuple[bool, list]:
    """
    Validate an STL file without fully processing it.

    Parameters
    ----------
    filepath : str
        Path to STL file

    Returns
    -------
    tuple
        (is_valid, list_of_errors)
    """
    errors = []

    path = Path(filepath)

    # Check exists
    if not path.exists():
        errors.append(f"File not found: {filepath}")
        return False, errors

    # Check extension
    if path.suffix.lower() != '.stl':
        errors.append(f"File must have .stl extension")
        return False, errors

    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 50:
        errors.append(f"File too large: {file_size_mb:.1f} MB (max 50 MB)")
        return False, errors

    # Try to load and check validity
    try:
        import trimesh
        mesh = trimesh.load(filepath)

        # Check triangle count
        if len(mesh.faces) > 500000:
            errors.append(f"Too many triangles: {len(mesh.faces)} (max 500,000)")

        # Check dimensions
        dims = mesh.bounds[1] - mesh.bounds[0]
        if np.any(dims > 500):
            errors.append(f"Part too large: max dimension {dims.max():.0f} mm (max 500 mm)")
        if np.any(dims < 0.1):
            errors.append(f"Part too small: min dimension {dims.min():.2f} mm (min 0.1 mm)")

        # Warning for non-watertight (not an error)
        if not mesh.is_watertight:
            errors.append("Warning: Mesh is not watertight - may have gaps")

    except Exception as e:
        errors.append(f"Failed to load STL: {str(e)}")
        return False, errors

    # Filter out warnings for validity check
    real_errors = [e for e in errors if not e.startswith("Warning:")]

    return len(real_errors) == 0, errors

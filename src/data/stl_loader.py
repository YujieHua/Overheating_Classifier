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

    # Find all ray-mesh intersections
    # Returns: locations (Nx3), ray_indices, triangle_indices
    try:
        intersector = mesh.ray
        locations, ray_indices, _ = intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if progress_callback:
            progress_callback(30, f"Found {len(locations)} intersections, building Z-hit matrix...")

        # VECTORIZED APPROACH: Build padded 2D array of Z-hits for fast processing
        # First, count hits per ray to determine max_hits
        hit_counts = np.bincount(ray_indices, minlength=n_points)
        max_hits = hit_counts.max() if len(hit_counts) > 0 else 0

        if max_hits == 0:
            # No intersections at all
            z_hits_padded = None
            ray_casting_success = True
        else:
            # Create padded 2D array: shape (n_points, max_hits), padded with inf
            z_hits_padded = np.full((n_points, max_hits), np.inf, dtype=np.float64)

            # Fill in the z-values using advanced indexing
            # First, get the position within each ray's hit list
            ray_order = np.argsort(ray_indices, kind='stable')
            sorted_ray_indices = ray_indices[ray_order]
            sorted_z_values = locations[ray_order, 2]

            # Calculate position within each ray's hits
            hit_positions = np.zeros(len(ray_indices), dtype=np.int32)
            current_ray = -1
            current_pos = 0
            for idx in range(len(ray_order)):
                ray_idx = sorted_ray_indices[idx]
                if ray_idx != current_ray:
                    current_ray = ray_idx
                    current_pos = 0
                hit_positions[idx] = current_pos
                current_pos += 1

            # Assign z-values to the padded array
            z_hits_padded[sorted_ray_indices, hit_positions] = sorted_z_values

            # Sort each row (only the non-inf values matter)
            z_hits_padded.sort(axis=1)

            ray_casting_success = True

        logger.info(f"Ray casting complete: max {max_hits} hits/ray, ready for vectorized layer processing")

    except Exception as e:
        logger.warning(f"Ray casting failed: {e}, falling back to mesh.contains()")
        ray_casting_success = False
        z_hits_padded = None

    if progress_callback:
        progress_callback(40, "Extracting layer masks (vectorized)...")

    masks = {}
    layer_contours = {}
    z_heights = []

    for layer in range(1, n_layers + 1):
        if progress_callback:
            progress = 40 + (layer / n_layers) * 50
            progress_callback(progress, f"Processing layer {layer}/{n_layers}")

        # Z coordinate for this layer (center of layer)
        z = z_min + (layer - 0.5) * effective_layer_thickness
        z_heights.append(z)

        # ===== VECTORIZED INSIDE/OUTSIDE determination =====
        if ray_casting_success and z_hits_padded is not None:
            # Count intersections ABOVE z for all points at once using broadcasting
            # z_hits_padded shape: (n_points, max_hits)
            # Comparison creates boolean array, sum along axis=1 gives count
            count_above = np.sum(z_hits_padded > z, axis=1)  # Shape: (n_points,)

            # Odd count = inside, vectorized
            mask_flat = ((count_above % 2) == 1).astype(np.uint8)
            mask = mask_flat.reshape(nx, ny)
        elif ray_casting_success and z_hits_padded is None:
            # No intersections at all - empty layer
            mask = np.zeros((nx, ny), dtype=np.uint8)
        else:
            # Fallback to mesh.contains() - slower but works
            points_3d = np.column_stack([
                xx.ravel(),
                yy.ravel(),
                np.full(n_points, z)
            ])
            try:
                inside = mesh.contains(points_3d)
            except Exception:
                inside = np.zeros(n_points, dtype=bool)
            mask = inside.reshape(nx, ny).astype(np.uint8)

        masks[layer] = mask

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

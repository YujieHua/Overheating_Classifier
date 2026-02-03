"""
Geometry scoring module for calculating the geometry multiplier G.

The geometry multiplier G captures how local geometry affects heat accumulation:
- Thin walls and overhangs have higher G (less material to conduct heat away)
- Bulk material has lower G (better heat dissipation)

G is calculated using 3D Gaussian convolution on the inverse of solid voxels.
"""

import numpy as np
from scipy import ndimage, signal
from typing import Dict, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_geometry_multiplier(
    voxels_3d: np.ndarray,
    sigma_mm: float = 1.0,
    voxel_size: float = 0.1,
    layer_thickness: float = 0.04,
    G_max: float = 2.0,
    progress_callback: Optional[Callable] = None
) -> np.ndarray:
    """
    Calculate the geometry multiplier G for each voxel using CAUSAL 3D Gaussian convolution.

    IMPORTANT: Uses a HALF-SPHERE kernel that only looks at layers BELOW the current one.
    This is physically correct because when a layer is being printed, only the layers
    below it exist - the layers above haven't been printed yet.

    The algorithm:
    1. Create inverse mask (1 where empty, 0 where solid)
    2. Apply CAUSAL 3D Gaussian filter (half-sphere, only -Z direction)
    3. Normalize to [0, G_max] range
    4. Mask to only solid voxels

    Parameters
    ----------
    voxels_3d : np.ndarray
        3D binary voxel grid, shape (n_layers, nx, ny)
    sigma_mm : float
        Gaussian sigma in mm (heat diffusion characteristic length)
    voxel_size : float
        XY voxel size in mm
    layer_thickness : float
        Z (layer) thickness in mm
    G_max : float
        Maximum geometry multiplier value
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    np.ndarray
        Geometry multiplier G, same shape as voxels_3d, values in [0, G_max]
    """
    if progress_callback:
        progress_callback(32, "Computing geometry score G (causal kernel)...")

    logger.info(f"Calculating geometry multiplier: sigma={sigma_mm}mm, G_max={G_max} (causal half-sphere)")

    n_layers, nx, ny = voxels_3d.shape

    # Convert sigma from mm to voxels
    sigma_xy = sigma_mm / voxel_size  # in voxels
    sigma_z = sigma_mm / layer_thickness  # in layers

    logger.debug(f"Sigma in voxels: XY={sigma_xy:.2f}, Z={sigma_z:.2f}")

    # Create inverse mask (1 = empty space, 0 = solid)
    inverse_mask = 1 - voxels_3d.astype(np.float32)

    if progress_callback:
        progress_callback(34, "Building causal half-sphere kernel...")

    # Build a CAUSAL (half-sphere) Gaussian kernel
    # Only includes negative Z direction (layers below)
    truncate = 3.0  # Reduced from 4.0 for memory efficiency
    radius_z = min(int(truncate * sigma_z + 0.5), min(n_layers - 1, 20))  # Cap at 20 layers
    radius_xy = min(int(truncate * sigma_xy + 0.5), min(nx // 2, ny // 2, 15))  # Cap at 15 voxels

    logger.info(f"Kernel radius: Z={radius_z} layers, XY={radius_xy} voxels")

    # Create coordinate grids for kernel
    # Z: only negative values (layers below) and zero
    z_range = np.arange(-radius_z, 1)  # -radius_z to 0 (inclusive)
    x_range = np.arange(-radius_xy, radius_xy + 1)
    y_range = np.arange(-radius_xy, radius_xy + 1)

    zz, xx, yy = np.meshgrid(z_range, x_range, y_range, indexing='ij')

    # Gaussian kernel with anisotropic sigma
    kernel = np.exp(-0.5 * ((zz / sigma_z) ** 2 + (xx / sigma_xy) ** 2 + (yy / sigma_xy) ** 2))

    # Normalize kernel to sum to 1
    kernel = kernel / kernel.sum()

    if progress_callback:
        progress_callback(36, "Applying causal convolution (FFT)...")

    # Apply convolution with the causal kernel using FFT (300x faster than direct convolution)
    # Pad with 1.0 to treat outside as empty space (max G for overhangs at bottom)
    # Padding: Z needs radius_z at top only (causal), XY needs symmetric padding
    padded = np.pad(
        inverse_mask,
        [(radius_z, 0), (radius_xy, radius_xy), (radius_xy, radius_xy)],
        mode='constant',
        constant_values=1.0
    )
    smoothed = signal.fftconvolve(padded, kernel, mode='valid')

    if progress_callback:
        progress_callback(40, "Normalizing geometry score...")

    # Normalize to [0, G_max]
    # Higher smoothed value = more empty space below = higher G (overhang)
    G = smoothed * G_max

    # Only keep values for solid voxels
    G = G * voxels_3d

    # Statistics
    solid_mask = voxels_3d > 0
    if solid_mask.sum() > 0:
        G_solid = G[solid_mask]
        logger.info(f"Geometry G stats: min={G_solid.min():.3f}, max={G_solid.max():.3f}, "
                    f"mean={G_solid.mean():.3f}")

    if progress_callback:
        progress_callback(42, "Geometry scoring complete")

    return G


def calculate_geometry_multiplier_per_layer(
    slice_result: Dict,
    sigma_mm: float = 1.0,
    G_max: float = 2.0,
    progress_callback: Optional[Callable] = None
) -> Dict[int, np.ndarray]:
    """
    Calculate geometry multiplier G for each layer.

    This is a convenience wrapper that works with slice_result dict format.

    Parameters
    ----------
    slice_result : dict
        Output from slice_stl()
    sigma_mm : float
        Gaussian sigma in mm
    G_max : float
        Maximum geometry multiplier
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    dict
        Maps layer number (1-indexed) to 2D G array
    """
    from .geometry_score import calculate_geometry_multiplier

    # Stack masks into 3D array
    n_layers = slice_result['n_layers']
    nx, ny = slice_result['grid_shape']
    voxel_size = slice_result['voxel_size']
    layer_thickness = slice_result['layer_thickness']

    voxels_3d = np.zeros((n_layers, nx, ny), dtype=np.uint8)
    for layer in range(1, n_layers + 1):
        voxels_3d[layer - 1] = slice_result['masks'][layer]

    # Calculate 3D geometry multiplier
    G_3d = calculate_geometry_multiplier(
        voxels_3d,
        sigma_mm=sigma_mm,
        voxel_size=voxel_size,
        layer_thickness=layer_thickness,
        G_max=G_max,
        progress_callback=progress_callback
    )

    # Convert back to per-layer dict
    G_layers = {}
    for layer in range(1, n_layers + 1):
        G_layers[layer] = G_3d[layer - 1]

    return G_layers


def get_geometry_statistics(G_3d: np.ndarray, voxels_3d: np.ndarray) -> Dict:
    """
    Calculate statistics about the geometry multiplier distribution.

    Parameters
    ----------
    G_3d : np.ndarray
        3D geometry multiplier array
    voxels_3d : np.ndarray
        3D binary voxel array

    Returns
    -------
    dict
        Statistics including mean, std, percentiles, etc.
    """
    solid_mask = voxels_3d > 0
    G_solid = G_3d[solid_mask]

    if len(G_solid) == 0:
        return {
            'n_solid_voxels': 0,
            'G_mean': 0,
            'G_std': 0,
            'G_min': 0,
            'G_max': 0,
            'G_median': 0,
            'G_p90': 0,
            'G_p95': 0,
            'G_p99': 0,
        }

    return {
        'n_solid_voxels': int(solid_mask.sum()),
        'G_mean': float(G_solid.mean()),
        'G_std': float(G_solid.std()),
        'G_min': float(G_solid.min()),
        'G_max': float(G_solid.max()),
        'G_median': float(np.median(G_solid)),
        'G_p90': float(np.percentile(G_solid, 90)),
        'G_p95': float(np.percentile(G_solid, 95)),
        'G_p99': float(np.percentile(G_solid, 99)),
    }


def calculate_layer_averaged_G(
    G_layers: Dict[int, np.ndarray],
    masks: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """
    Calculate layer-averaged geometry multiplier.

    For the energy model (Mode B), we need a single G value per layer
    rather than per-voxel values. This function computes the average G
    over all solid voxels in each layer.

    Parameters
    ----------
    G_layers : dict
        Per-voxel G values from calculate_geometry_multiplier_per_layer()
        Maps layer_number -> 2D numpy array of G values
    masks : dict
        Binary masks for each layer
        Maps layer_number -> 2D numpy array (1 = solid, 0 = empty)

    Returns
    -------
    dict
        Maps layer_number -> scalar G_avg value (float)
    """
    G_avg = {}
    for layer, G_2d in G_layers.items():
        mask = masks.get(layer)
        if mask is None:
            G_avg[layer] = 0.0
            continue

        solid_voxels = mask > 0
        if solid_voxels.sum() > 0:
            G_avg[layer] = float(np.mean(G_2d[solid_voxels]))
        else:
            G_avg[layer] = 0.0

    return G_avg

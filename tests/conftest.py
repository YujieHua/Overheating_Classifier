"""Pytest fixtures for Overheating Classifier tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_cylinder_masks():
    """
    Simple cylinder geometry: 10 layers, uniform 10x10 cross-section.
    All layers have same area - should show gradual energy accumulation.
    """
    masks = {}
    for layer in range(1, 11):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1  # 10x10 square centered
        masks[layer] = mask
    return masks


@pytest.fixture
def cone_tip_down_masks():
    """
    Cone with tip pointing down (inverted cone).
    Small base → large top. Creates thermal bottleneck.
    Should show HIGH energy accumulation.
    """
    masks = {}
    for layer in range(1, 11):
        mask = np.zeros((20, 20), dtype=np.uint8)
        # Radius grows with layer: 1, 2, 3, ... 10
        radius = layer
        center = 10
        for i in range(20):
            for j in range(20):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    mask[i, j] = 1
        masks[layer] = mask
    return masks


@pytest.fixture
def cone_base_down_masks():
    """
    Cone with base pointing down (normal cone).
    Large base → small top. Good thermal dissipation.
    Should show LOW energy accumulation.
    """
    masks = {}
    for layer in range(1, 11):
        mask = np.zeros((20, 20), dtype=np.uint8)
        # Radius shrinks with layer: 10, 9, 8, ... 1
        radius = 11 - layer
        center = 10
        for i in range(20):
            for j in range(20):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    mask[i, j] = 1
        masks[layer] = mask
    return masks


@pytest.fixture
def single_layer_masks():
    """Single layer geometry for edge case testing."""
    masks = {1: np.ones((10, 10), dtype=np.uint8)}
    return masks


@pytest.fixture
def empty_middle_layer_masks():
    """Geometry with an empty layer in the middle."""
    masks = {}
    for layer in range(1, 6):
        mask = np.zeros((10, 10), dtype=np.uint8)
        if layer != 3:  # Layer 3 is empty
            mask[2:8, 2:8] = 1
        masks[layer] = mask
    return masks


@pytest.fixture
def offset_layers_masks():
    """
    Two layers with partial overlap (offset geometry).
    Tests voxel-intersection contact area calculation.
    """
    masks = {}
    # Layer 1: left side
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask1[2:8, 0:6] = 1
    masks[1] = mask1

    # Layer 2: right side (partial overlap with layer 1)
    mask2 = np.zeros((10, 10), dtype=np.uint8)
    mask2[2:8, 4:10] = 1
    masks[2] = mask2

    return masks


@pytest.fixture
def default_params():
    """Default energy model parameters."""
    return {
        "dissipation_factor": 0.5,
        "convection_factor": 0.05,
        "use_geometry_multiplier": False,
        "threshold_medium": 0.3,
        "threshold_high": 0.6,
    }

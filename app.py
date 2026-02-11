"""
Overheating Classifier - Web Interface
Flask application with embedded HTML/CSS/JS
Energy-based overheating risk prediction for LPBF

Based on Ali's recommendations from Jan 30, 2026 SRG meeting.
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import argparse
from pathlib import Path
from queue import Queue
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union

import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory, make_response
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(log_dir: str = "logs"):
    """Configure structured logging with rotation."""
    from logging.handlers import RotatingFileHandler

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Use RotatingFileHandler to prevent unbounded log growth
    # Max 5MB per file, keep 3 backup files
    file_handler = RotatingFileHandler(
        log_path / 'app.log',
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Less verbose console output

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, stream_handler]
    )

setup_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class CancelledException(Exception):
    """Raised when an operation is cancelled by user."""
    pass

# =============================================================================
# SESSION REGISTRY (Thread-Safe)
# =============================================================================
@dataclass
class AnalysisSession:
    """Thread-safe analysis session state."""
    session_id: str
    status: str = 'initializing'
    progress: float = 0.0
    current_step: str = ''
    error_message: Optional[str] = None
    results_path: Optional[str] = None
    results: Optional[Dict] = None
    slice_data: Optional[Dict] = None
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False
    computation_time: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _progress_queue: Queue = field(default_factory=Queue)

    def cancel(self):
        with self._lock:
            self.cancelled = True
            self.status = 'cancelled'
            logger.info(f"Session {self.session_id} cancelled")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self.cancelled

    def update_progress(self, progress: float, step: str = ''):
        with self._lock:
            self.progress = progress
            self.current_step = step
            self._progress_queue.put({
                'progress': progress,
                'step': step,
                'status': self.status,
                'timestamp': time.time()
            })

    def get_progress_update(self, timeout: float = 1.0):
        try:
            return self._progress_queue.get(timeout=timeout)
        except Exception:  # Queue.Empty or timeout
            return None

    def set_complete(self, results: Dict, computation_time: float = 0.0):
        with self._lock:
            self.status = 'complete'
            self.progress = 100.0
            self.results = results
            self.computation_time = computation_time
            self._progress_queue.put({
                'progress': 100.0,
                'step': 'Complete',
                'status': 'complete',
                'timestamp': time.time()
            })

    def set_error(self, message: str):
        with self._lock:
            self.status = 'error'
            self.error_message = message
            self._progress_queue.put({
                'progress': self.progress,
                'step': 'Error',
                'status': 'error',
                'error': message,
                'timestamp': time.time()
            })


class SessionRegistry:
    def __init__(self):
        self._sessions: Dict[str, AnalysisSession] = {}
        self._lock = threading.RLock()

    def create_session(self) -> AnalysisSession:
        with self._lock:
            session_id = str(uuid.uuid4())[:8]
            session = AnalysisSession(session_id=session_id)
            self._sessions[session_id] = session
            logger.info(f"Created session: {session_id}")
            return session

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")

    def cleanup_all_except(self, keep_session_id: str = None):
        with self._lock:
            to_delete = [sid for sid in self._sessions.keys() if sid != keep_session_id]
            for session_id in to_delete:
                session = self._sessions[session_id]
                session.results = None
                session.slice_data = None
                del self._sessions[session_id]
            if to_delete:
                logger.info(f"Cleared {len(to_delete)} old sessions")


session_registry = SessionRegistry()

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================
PARAMETER_RULES = {
    'voxel_size': {'min': 0.02, 'max': 0.5, 'unit': 'mm'},
    'layer_thickness': {'min': 0.02, 'max': 0.1, 'unit': 'mm'},
    'dissipation_factor': {'min': 0.0, 'max': 1.0},
    'convection_factor': {'min': 0.0, 'max': 0.5},
    'sigma_mm': {'min': 0.1, 'max': 5.0, 'unit': 'mm'},
    'G_max': {'min': 0.5, 'max': 100.0},
    'threshold_medium': {'min': 0.0, 'max': 1.0},
    'threshold_high': {'min': 0.0, 'max': 1.0},
    'area_ratio_power': {'min': 0.1},
    # Laser parameters for Joule calculation (literature-based ranges)
    'laser_power': {'min': 50, 'max': 500, 'unit': 'W'},
    'scan_speed': {'min': 100, 'max': 2000, 'unit': 'mm/s'},
    'hatch_distance': {'min': 0.05, 'max': 0.15, 'unit': 'mm'},
}

def safe_float(value, default=0.0):
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default

def validate_parameters(params: dict) -> tuple:
    errors = []
    for param, rule in PARAMETER_RULES.items():
        if param not in params:
            continue
        value = params[param]
        if not isinstance(value, (int, float)):
            errors.append(f"{param} must be numeric")
            continue
        if rule.get('positive') and value <= 0:
            errors.append(f"{param} must be positive")
        if 'min' in rule and value < rule['min']:
            errors.append(f"{param} must be >= {rule['min']}")
        if 'max' in rule and value > rule['max']:
            errors.append(f"{param} must be <= {rule['max']}")
    return len(errors) == 0, errors


# =============================================================================
# FLASK APPLICATION
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'results'
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

current_state = {
    'stl_loaded': False,
    'stl_path': None,
    'stl_info': None,
    'cached_masks': None,
    'cached_slice_params': None,
    'cached_slice_info': None,
    'cached_region_data': None,  # Region detection results from slicing
    'lock': threading.RLock()
}


# =============================================================================
# SHARED GEOMETRY UTILITIES
# =============================================================================
def apply_build_direction_rotation(mesh, build_direction: str):
    """Apply rotation to mesh based on build direction.

    This is a shared function used by both slice_worker and run_analysis_worker
    to ensure consistent rotation behavior.

    Args:
        mesh: trimesh mesh object
        build_direction: 'Z' (default), 'Y', 'Y-', or 'X'

    Returns:
        Rotated mesh (copy of original)
    """
    import trimesh

    if build_direction == 'Z':
        return mesh  # No rotation needed

    rotated_mesh = mesh.copy()

    if build_direction == 'Y':
        # Y-down: rotate so Y becomes Z, then flip Z
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        rotated_mesh.apply_transform(rotation)
        flip = trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])
        rotated_mesh.apply_transform(flip)
        logger.debug(f"Applied Y→Z rotation + Z flip (Y-down)")
    elif build_direction == 'Y-':
        # Y-up: rotate so Y becomes Z (no flip)
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        rotated_mesh.apply_transform(rotation)
        logger.debug(f"Applied Y→Z rotation (Y-up)")
    elif build_direction == 'X':
        # X-up: rotate so X becomes Z
        rotation = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
        rotated_mesh.apply_transform(rotation)
        logger.debug(f"Applied X→Z rotation")
    else:
        logger.warning(f"Unknown build_direction '{build_direction}', no rotation applied")

    return rotated_mesh


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'version': '1.0.0',
        'project': 'Overheating Classifier'
    })


@app.route('/api/upload_stl', methods=['POST'])
def upload_stl():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not file.filename.lower().endswith('.stl'):
        return jsonify({'status': 'error', 'message': 'File must be an STL'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        from src.data.stl_loader import load_stl, validate_stl_file

        is_valid, validation_msg = validate_stl_file(str(filepath))
        if not is_valid:
            filepath.unlink()
            return jsonify({'status': 'error', 'message': validation_msg}), 400

        mesh_info = load_stl(str(filepath))

        with current_state['lock']:
            current_state['stl_loaded'] = True
            current_state['stl_path'] = str(filepath)
            current_state['stl_info'] = mesh_info
            current_state['cached_masks'] = None
            current_state['cached_slice_params'] = None
            current_state['cached_slice_info'] = None

        logger.info(f"STL uploaded: {filename}, {mesh_info['n_triangles']} triangles")

        # Convert numpy arrays to lists for JSON serialization
        bounds = mesh_info['bounds']
        dimensions = mesh_info['dimensions']
        if hasattr(bounds, 'tolist'):
            bounds = bounds.tolist()
        if hasattr(dimensions, 'tolist'):
            dimensions = dimensions.tolist()

        return jsonify({
            'status': 'success',
            'filename': filename,
            'info': {
                'n_triangles': mesh_info['n_triangles'],
                'bounds': bounds,
                'dimensions': dimensions
            }
        })

    except Exception as e:
        logger.error(f"STL upload error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/load_test_stl', methods=['POST'])
def load_test_stl():
    # Support multiple test STL files via index parameter
    stl_index = request.args.get('index', '1')

    test_stl_paths = {
        '1': os.getenv('TEST_STL_PATH',
            r"C:\Users\huayu\Local\Desktop\Overheating_Classifier\CAD\SmartFusion_Calibration_Square.stl"),
        '2': os.getenv('TEST_STL_PATH_2',
            r"C:\Users\huayu\Local\Desktop\Overheating_Classifier\CAD\SF_3_overhanges_less_triangles.stl"),
        '3': os.getenv('TEST_STL_PATH_3',
            r"C:\Users\huayu\Local\Desktop\Overheating_Classifier\CAD\Korper1173.stl"),
        '4': os.getenv('TEST_STL_PATH_4',
            r"C:\Users\huayu\Local\Desktop\Overheating_Classifier\CAD\Test_Geo_Group_20260210.STL"),
    }

    default_path = test_stl_paths.get(stl_index, test_stl_paths['1'])

    if not os.path.exists(default_path):
        return jsonify({
            'status': 'error',
            'message': f'Test STL {stl_index} not found at: {default_path}'
        }), 404

    try:
        from src.data.stl_loader import load_stl, validate_stl_file

        is_valid, validation_msg = validate_stl_file(default_path)
        if not is_valid:
            return jsonify({'status': 'error', 'message': validation_msg}), 400

        mesh_info = load_stl(default_path)

        with current_state['lock']:
            current_state['stl_loaded'] = True
            current_state['stl_path'] = default_path
            current_state['stl_info'] = mesh_info
            current_state['cached_masks'] = None
            current_state['cached_slice_params'] = None
            current_state['cached_slice_info'] = None

        logger.info(f"Test STL loaded: {default_path}")

        # Convert numpy arrays to lists for JSON serialization
        bounds = mesh_info['bounds']
        dimensions = mesh_info['dimensions']
        if hasattr(bounds, 'tolist'):
            bounds = bounds.tolist()
        if hasattr(dimensions, 'tolist'):
            dimensions = dimensions.tolist()

        return jsonify({
            'status': 'success',
            'filename': os.path.basename(default_path),
            'info': {
                'n_triangles': mesh_info['n_triangles'],
                'bounds': bounds,
                'dimensions': dimensions
            }
        })

    except Exception as e:
        logger.error(f"Test STL load error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/slice', methods=['POST'])
def api_slice():
    """Slice-only endpoint: slices STL and detects regions without running energy analysis."""
    params = request.json or {}

    try:
        voxel_size = float(params.get('voxel_size', 0.1))
        layer_thickness = float(params.get('layer_thickness', 0.04))
        layer_grouping = int(params.get('layer_grouping', 25))
        build_direction = params.get('build_direction', 'Z')
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid parameter format: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid parameter format'}), 400

    is_valid, errors = validate_parameters({
        'voxel_size': voxel_size,
        'layer_thickness': layer_thickness,
        'layer_grouping': layer_grouping
    })
    if not is_valid:
        return jsonify({'status': 'error', 'message': str(errors)}), 400

    with current_state['lock']:
        if not current_state['stl_loaded']:
            return jsonify({'status': 'error', 'message': 'No STL loaded'}), 400
        stl_path = current_state['stl_path']

    session_registry.cleanup_all_except(None)
    session = session_registry.create_session()
    session.status = 'slicing'

    thread = threading.Thread(
        target=slice_worker,
        args=(session, stl_path, voxel_size, layer_thickness, layer_grouping, build_direction)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'session_id': session.session_id
    })


def slice_worker(session: AnalysisSession, stl_path: str, voxel_size: float, layer_thickness: float,
                  layer_grouping: int = 25, build_direction: str = 'Z'):
    """Slice STL and detect connected regions (islands) per layer.

    This worker handles the slicing pipeline that is shared between the Slice button
    and the Run Analysis button. Results are cached for reuse.
    """
    try:
        start_time = time.time()
        from src.data.stl_loader import load_stl, slice_stl
        from scipy import ndimage

        def progress_with_cancel(progress, step):
            if session.is_cancelled():
                raise CancelledException("Slicing cancelled by user")
            session.update_progress(progress, step)

        session.update_progress(5, "[STAGE] Loading STL mesh...")

        # Load mesh first (slice_stl expects mesh_info dict, not path)
        mesh_info = load_stl(stl_path)

        # Apply build direction transformation using shared function
        if build_direction != 'Z':
            session.update_progress(6, f"[STAGE] Rotating mesh ({build_direction}→Z)...")
            mesh = mesh_info['mesh']
            rotated_mesh = apply_build_direction_rotation(mesh, build_direction)
            # Update mesh_info with transformed mesh
            mesh_info['mesh'] = rotated_mesh
            mesh_info['bounds'] = rotated_mesh.bounds
            mesh_info['dimensions'] = rotated_mesh.bounds[1] - rotated_mesh.bounds[0]
            session.update_progress(8, f"[INFO] Rotated dimensions: {mesh_info['dimensions'][0]:.1f} x {mesh_info['dimensions'][1]:.1f} x {mesh_info['dimensions'][2]:.1f} mm")

        session.update_progress(10, "[STAGE] Slicing geometry into layers...")
        slice_result = slice_stl(
            mesh_info=mesh_info,
            voxel_size=voxel_size,
            layer_thickness=layer_thickness,
            layer_grouping=layer_grouping,
            progress_callback=lambda p, s: progress_with_cancel(10 + p * 0.80, s)
        )

        # Detect regions (connected components) per layer
        session.update_progress(92, "[STAGE] Detecting regions per layer...")
        masks = slice_result['masks']
        region_data = {}
        for layer_num, mask in masks.items():
            labeled, n_regions = ndimage.label(mask > 0)
            region_areas = {}
            for rid in range(1, n_regions + 1):
                region_areas[rid] = int(np.sum(labeled == rid))
            region_data[layer_num] = {
                'n_regions': n_regions,
                'labeled': labeled,
                'region_areas': region_areas,
            }

        slice_result['region_data'] = region_data
        n_layers = len(masks)
        session.update_progress(95, f"[INFO] {n_layers} layers sliced, regions detected")

        if not session.is_cancelled():
            with current_state['lock']:
                current_state['cached_masks'] = slice_result['masks']
                current_state['cached_slice_params'] = {
                    'voxel_size': voxel_size,
                    'layer_thickness': layer_thickness,
                    'layer_grouping': layer_grouping,
                    'build_direction': build_direction,  # Now cached!
                }
                current_state['cached_slice_info'] = {
                    'n_layers': slice_result['n_layers'],
                    'grid_shape': slice_result['grid_shape'],
                }
                current_state['cached_region_data'] = region_data  # Cache region data!
                logger.info(f"Cached slice results: {n_layers} layers, build_direction={build_direction}")

            session.slice_data = slice_result
            computation_time = time.time() - start_time
            session.set_complete({
                'n_layers': slice_result['n_layers'],
                'grid_shape': slice_result['grid_shape'],
                'voxel_size': voxel_size,
                'layer_thickness': layer_thickness,
                'layer_grouping': layer_grouping,
                'build_direction': build_direction,
            }, computation_time)

    except CancelledException:
        logger.info(f"Slicing cancelled for session {session.session_id}")
    except Exception as e:
        logger.error(f"Slicing error: {e}", exc_info=True)
        session.set_error(str(e))


@app.route('/api/slice_results/<session_id>', methods=['GET'])
def get_slice_results(session_id: str):
    """Get slice results including region data for visualization."""
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404

    if session.status != 'complete':
        return jsonify({'status': 'error', 'message': f'Session status: {session.status}'}), 400

    if not session.slice_data:
        return jsonify({'status': 'error', 'message': 'No slice data available'}), 400

    slice_data = session.slice_data

    # Prepare region summary for each layer (without sending full labeled arrays)
    region_summary = {}
    region_data = slice_data.get('region_data', {})
    for layer_num, rd in region_data.items():
        region_summary[layer_num] = {
            'n_regions': rd['n_regions'],
            'region_areas': rd['region_areas'],
        }

    return jsonify({
        'status': 'success',
        'results': {
            'n_layers': slice_data['n_layers'],
            'grid_shape': slice_data['grid_shape'],
            'voxel_size': slice_data['voxel_size'],
            'layer_thickness': slice_data['layer_thickness'],
            'region_summary': region_summary,
        }
    })


@app.route('/api/run', methods=['POST'])
def run_analysis():
    params = request.json

    is_valid, errors = validate_parameters(params)
    if not is_valid:
        logger.warning(f"Parameter validation failed: {errors}")
        return jsonify({
            'status': 'error',
            'message': f'Parameter validation failed: {errors}',
            'errors': errors
        }), 400

    with current_state['lock']:
        if not current_state['stl_loaded']:
            return jsonify({'status': 'error', 'message': 'No STL loaded'}), 400
        stl_path = current_state['stl_path']
        cached_masks = current_state.get('cached_masks')
        cached_slice_params = current_state.get('cached_slice_params')
        cached_slice_info = current_state.get('cached_slice_info')
        cached_region_data = current_state.get('cached_region_data')

    session_registry.cleanup_all_except(None)
    session = session_registry.create_session()

    slice_cache = None
    if cached_masks is not None and cached_slice_params is not None:
        slice_cache = {
            'masks': cached_masks,
            'params': cached_slice_params,
            'info': cached_slice_info,
            'region_data': cached_region_data,  # Include cached region data
        }

    thread = threading.Thread(
        target=run_analysis_worker,
        args=(session, stl_path, params, slice_cache)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'session_id': session.session_id
    })


def run_analysis_worker(session: AnalysisSession, stl_path: str, params: dict, slice_cache: dict = None):
    try:
        session.status = 'running'
        start_time = time.time()

        if session.is_cancelled():
            return

        from src.data.stl_loader import load_stl, slice_stl
        from src.compute.energy_model import run_energy_analysis
        from src.compute.geometry_score import calculate_geometry_multiplier_per_layer

        def progress_with_cancel(progress, step):
            if session.is_cancelled():
                raise CancelledException("Analysis cancelled by user")
            session.update_progress(progress, step)

        voxel_size = params.get('voxel_size', 0.1)
        layer_thickness = params.get('layer_thickness', 0.04)
        layer_grouping = params.get('layer_grouping', 25)
        build_direction = params.get('build_direction', 'Z')
        dissipation_factor = params.get('dissipation_factor', 0.5)
        convection_factor = params.get('convection_factor', 0.05)
        use_geometry_multiplier = params.get('use_geometry_multiplier', False)
        sigma_mm = params.get('sigma_mm', 1.0)
        G_max = params.get('G_max', 99.0)
        threshold_medium = params.get('threshold_medium', 0.3)
        threshold_high = params.get('threshold_high', 0.6)
        area_ratio_power = params.get('area_ratio_power', 3.0)
        gaussian_ratio_power = params.get('gaussian_ratio_power', 0.15)
        # Laser parameters for Joule calculation
        laser_power = params.get('laser_power', 200.0)
        scan_speed = params.get('scan_speed', 800.0)
        hatch_distance = params.get('hatch_distance', 0.1)

        need_reslice = True
        masks = None
        slice_info = None

        if slice_cache is not None:
            cached_params = slice_cache.get('params', {})
            if (cached_params.get('voxel_size') == voxel_size and
                cached_params.get('layer_thickness') == layer_thickness and
                cached_params.get('layer_grouping') == layer_grouping and
                cached_params.get('build_direction') == build_direction):
                masks = slice_cache['masks']
                slice_info = slice_cache['info']
                need_reslice = False
                progress_with_cancel(20, "[INFO] Using cached slice data")

        if need_reslice:
            progress_with_cancel(5, "[STAGE] Loading STL mesh...")
            mesh_info = load_stl(stl_path)

            # Apply rotation using shared function (same as slice_worker)
            if build_direction != 'Z':
                progress_with_cancel(6, f"[STAGE] Rotating mesh ({build_direction}→Z)...")
                mesh = mesh_info['mesh']
                rotated_mesh = apply_build_direction_rotation(mesh, build_direction)
                # Update mesh_info with transformed mesh
                mesh_info['mesh'] = rotated_mesh
                mesh_info['bounds'] = rotated_mesh.bounds
                mesh_info['dimensions'] = rotated_mesh.bounds[1] - rotated_mesh.bounds[0]
                progress_with_cancel(8, f"[INFO] Rotated dimensions: {mesh_info['dimensions'][0]:.1f} x {mesh_info['dimensions'][1]:.1f} x {mesh_info['dimensions'][2]:.1f} mm")

            progress_with_cancel(10, "[STAGE] Slicing geometry into layers...")
            slice_result = slice_stl(
                mesh_info=mesh_info,
                voxel_size=voxel_size,
                layer_thickness=layer_thickness,
                layer_grouping=layer_grouping,
                progress_callback=lambda p, s: progress_with_cancel(10 + p * 0.10, s)
            )
            masks = slice_result['masks']
            slice_info = {
                'n_layers': slice_result['n_layers'],
                'grid_shape': slice_result['grid_shape'],
            }
            logger.info(f"run_analysis_worker: Re-sliced geometry ({len(masks)} layers)")
        else:
            logger.info(f"run_analysis_worker: Using cached slice data ({len(masks)} layers)")

        n_layers = len(masks)
        progress_with_cancel(22, f"[INFO] {n_layers} layers ready")

        G_layers = None
        if use_geometry_multiplier:
            progress_with_cancel(25, "[STAGE] Computing geometry multiplier G...")

            slice_result_for_G = {
                'masks': masks,
                'n_layers': n_layers,
                'grid_shape': slice_info.get('grid_shape', masks[1].shape),
                'voxel_size': voxel_size,
                'layer_thickness': layer_thickness,
            }

            G_layers_2d = calculate_geometry_multiplier_per_layer(
                slice_result_for_G,
                sigma_mm=sigma_mm,
                G_max=G_max,
                progress_callback=lambda p, s: progress_with_cancel(25 + p * 0.15, s)
            )

            G_layers = G_layers_2d  # Pass per-voxel 2D arrays (not pre-averaged scalars)
            progress_with_cancel(42, "[INFO] Geometry G computed")
        else:
            progress_with_cancel(42, "[INFO] Using Area-Only mode (no geometry G)")

        progress_with_cancel(45, "[STAGE] Running energy accumulation analysis...")

        # Effective layer thickness with grouping
        effective_layer_thickness = layer_thickness * layer_grouping

        energy_results = run_energy_analysis(
            masks=masks,
            G_layers=G_layers,
            dissipation_factor=dissipation_factor,
            convection_factor=convection_factor,
            use_geometry_multiplier=use_geometry_multiplier,
            area_ratio_power=area_ratio_power,
            gaussian_ratio_power=gaussian_ratio_power,
            threshold_medium=threshold_medium,
            threshold_high=threshold_high,
            voxel_size=voxel_size,
            layer_thickness=effective_layer_thickness,
            laser_power=laser_power,
            scan_speed=scan_speed,
            hatch_distance=hatch_distance,
            progress_callback=lambda p, s: progress_with_cancel(45 + p * 0.45, s)
        )

        progress_with_cancel(92, "[STAGE] Preparing results...")

        computation_time = time.time() - start_time

        results = {
            'n_layers': n_layers,
            'risk_scores': energy_results['risk_scores'],
            'raw_energy_scores': energy_results['raw_energy_scores'],
            'energy_density_scores': energy_results['energy_density_scores'],
            'risk_levels': energy_results['risk_levels'],
            'layer_areas': energy_results['layer_areas'],
            'contact_areas': energy_results['contact_areas'],
            'summary': energy_results['summary'],
            'params_used': {
                'voxel_size': voxel_size,
                'layer_thickness': layer_thickness,
                'layer_grouping': layer_grouping,
                'build_direction': build_direction,
                'effective_layer_thickness': effective_layer_thickness,
                'dissipation_factor': dissipation_factor,
                'convection_factor': convection_factor,
                'use_geometry_multiplier': use_geometry_multiplier,
                'sigma_mm': sigma_mm if use_geometry_multiplier else None,
                'G_max': G_max if use_geometry_multiplier else None,
                'area_ratio_power': area_ratio_power,
                'gaussian_ratio_power': gaussian_ratio_power if use_geometry_multiplier else None,
                'threshold_medium': threshold_medium,
                'threshold_high': threshold_high,
                'mode': energy_results['params']['mode'],
                'laser_power': laser_power,
                'scan_speed': scan_speed,
                'hatch_distance': hatch_distance,
            },
            'computation_time_seconds': computation_time,
            'masks': masks,
            'G_layers': G_layers if use_geometry_multiplier else None,
            'region_data': energy_results.get('region_data', {}),
        }

        progress_with_cancel(98, f"[INFO] Analysis complete in {computation_time:.1f}s")

        if not session.is_cancelled():
            session.set_complete(results, computation_time)

    except CancelledException:
        logger.info(f"Analysis cancelled for session {session.session_id}")
    except Exception as e:
        if not session.is_cancelled():
            logger.error(f"Analysis error: {e}", exc_info=True)
            session.set_error(str(e))


@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 404

    def generate():
        while True:
            update = session.get_progress_update(timeout=1.0)
            if update:
                yield f"data: {json.dumps(update)}\n\n"
                if update.get('status') in ['complete', 'error', 'cancelled']:
                    break
            else:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/api/status/<session_id>')
def get_session_status(session_id):
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 404

    return jsonify({
        'status': session.status,
        'progress': session.progress,
        'current_step': session.current_step,
        'error_message': session.error_message,
    })


@app.route('/api/cancel/<session_id>', methods=['POST'])
def cancel_session(session_id):
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 404

    session.cancel()
    return jsonify({'status': 'cancelled', 'session_id': session_id})


@app.route('/api/results/<session_id>')
def get_results(session_id):
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 404

    if session.status != 'complete':
        return jsonify({
            'status': session.status,
            'message': 'Analysis not complete'
        }), 400

    results = session.results.copy()
    results.pop('masks', None)
    results.pop('G_layers', None)
    results.pop('region_data', None)

    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'results': results
    })


@app.route('/api/layer_data/<session_id>/<int:layer>')
def get_layer_data(session_id, layer):
    session = session_registry.get_session(session_id)
    if not session or not session.results:
        return jsonify({'status': 'error', 'message': 'No results available'}), 404

    results = session.results
    n_layers = results['n_layers']

    if layer < 1 or layer > n_layers:
        return jsonify({'status': 'error', 'message': f'Layer must be between 1 and {n_layers}'}), 400

    mask = results['masks'].get(layer)
    mask_list = mask.tolist() if mask is not None else None

    G_value = None
    if results.get('G_layers') and layer in results['G_layers']:
        G_value = results['G_layers'][layer]

    return jsonify({
        'status': 'success',
        'layer': layer,
        'data': {
            'mask': mask_list,
            'risk_score': results['risk_scores'].get(layer, 0),
            'risk_level': results['risk_levels'].get(layer, 'LOW'),
            'layer_area': results['layer_areas'].get(layer, 0),
            'contact_area': results['contact_areas'].get(layer, 0),
            'G_value': G_value,
        }
    })


@app.route('/api/export/<session_id>')
def export_results(session_id):
    session = session_registry.get_session(session_id)
    if not session or not session.results:
        return jsonify({'status': 'error', 'message': 'No results available'}), 404

    format_type = request.args.get('format', 'json')
    results = session.results

    if format_type == 'csv':
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['layer', 'risk_score', 'risk_level', 'layer_area_mm2', 'contact_area_mm2'])

        n_layers = results['n_layers']
        for layer in range(1, n_layers + 1):
            writer.writerow([
                layer,
                results['risk_scores'].get(layer, 0),
                results['risk_levels'].get(layer, 'LOW'),
                results['layer_areas'].get(layer, 0),
                results['contact_areas'].get(layer, 0),
            ])

        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=energy_results_{session_id}.csv'
        return response
    else:
        export_data = {
            'session_id': session_id,
            'n_layers': results['n_layers'],
            'risk_scores': results['risk_scores'],
            'risk_levels': results['risk_levels'],
            'layer_areas': results['layer_areas'],
            'contact_areas': results['contact_areas'],
            'summary': results['summary'],
            'params_used': results['params_used'],
            'computation_time_seconds': results['computation_time_seconds'],
        }

        response = make_response(json.dumps(export_data, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=energy_results_{session_id}.json'
        return response


@app.route('/api/layer_surfaces/<session_id>/<data_type>')
def get_layer_surfaces(session_id, data_type):
    """Get 3D layer surfaces with per-layer values for different data types.

    Supports: energy (risk scores), risk (risk levels), slices, regions

    Each layer is rendered as a surface with color based on its value.
    Also supports slice-only mode where only masks and regions are available.
    """
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404

    # Support both full analysis results and slice-only data
    # Check slice_data first since set_complete may set session.results to metadata without masks
    results = session.results
    slice_data = session.slice_data

    # Prefer slice_data if it has masks (slice-only mode or slice from full analysis)
    if slice_data and slice_data.get('masks'):
        # Slice-only mode: use slice_data for slices/regions visualization
        masks = slice_data.get('masks', {})
        params = {
            'voxel_size': slice_data.get('voxel_size', 0.1),
            'layer_thickness': slice_data.get('layer_thickness', 0.04),
        }
        # For slice-only mode, only slices and regions are available
        if data_type not in ['slices', 'regions']:
            return jsonify({'status': 'error',
                           'message': f'{data_type} requires full analysis (Run Analysis), not just slicing'}), 400
    elif results and results.get('masks'):
        # Full analysis mode with masks in results
        masks = results.get('masks', {})
        params = results.get('params_used', {})
    else:
        return jsonify({'status': 'error', 'message': 'No results available'}), 404

    if not masks:
        return jsonify({'status': 'error', 'message': 'No mask data available'}), 404

    voxel_size = params.get('voxel_size', 0.1)
    layer_thickness = params.get('effective_layer_thickness', params.get('layer_thickness', 0.04))

    # Get layer values based on data type
    if data_type == 'energy':
        # Use raw energy in Joules (not normalized risk scores)
        layer_values = {int(k): float(v) for k, v in results.get('raw_energy_scores', {}).items()}
        value_label = 'Energy (J)'
        # Use dynamic min/max for actual Joule values
        if layer_values:
            min_val = min(layer_values.values())
            max_val = max(layer_values.values())
        else:
            min_val = 0.0
            max_val = 1.0
    elif data_type == 'density':
        # Per-region energy density visualization in J/mm²
        # Each region displays its OWN density, not aggregated per layer
        from scipy import ndimage

        region_data_all = results.get('region_data', {})
        voxel_size_sq = voxel_size ** 2  # For area conversion

        if not region_data_all:
            return jsonify({'status': 'error', 'message': 'No region data available for density visualization'}), 400

        # Collect all region densities to find global min/max
        # Calculate density on-the-fly: energy / (area_voxels * voxel_size²)
        all_densities = []
        for layer_idx, rd in region_data_all.items():
            region_energies = rd.get('region_energies', {})
            region_areas = rd.get('region_areas', {})
            for rid, energy in region_energies.items():
                area_voxels = region_areas.get(rid, 0)
                if area_voxels > 0:
                    density = energy / (area_voxels * voxel_size_sq)
                    all_densities.append(density)

        if all_densities:
            min_val = min(all_densities)
            max_val = max(all_densities)
        else:
            min_val = 0.0
            max_val = 1.0

        # Generate per-region surfaces with density-based coloring
        layers_data = []
        sorted_layers = sorted(masks.keys())

        for layer in sorted_layers:
            mask = masks.get(layer)
            if mask is None:
                continue
            mask_arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
            if mask_arr.sum() == 0:
                continue

            z = layer * layer_thickness

            # Get region data for this layer
            rd = region_data_all.get(layer)
            if rd and rd.get('n_regions', 0) > 0:
                labeled = rd['labeled']
                n_regions = rd['n_regions']
                region_energies = rd.get('region_energies', {})
                region_areas = rd.get('region_areas', {})
            else:
                labeled, n_regions = ndimage.label(mask_arr > 0)
                region_energies = {}
                region_areas = {}

            if n_regions == 0:
                continue

            # Generate one surface per region with density-based color
            for rid in range(1, n_regions + 1):
                region_mask = (labeled == rid).astype(np.uint8)
                if region_mask.sum() == 0:
                    continue

                # Calculate density on-the-fly
                energy = region_energies.get(rid, 0.0)
                area_voxels = region_areas.get(rid, 0)
                if area_voxels > 0:
                    density_value = energy / (area_voxels * voxel_size_sq)
                else:
                    density_value = 0.0

                vertices, faces = _generate_layer_surface(region_mask, voxel_size, z,
                                                          skip_garbage_filter=True)
                if vertices and faces:
                    layers_data.append({
                        'layer': int(layer),
                        'z': float(z),
                        'value': float(density_value),
                        'region_id': int(rid),
                        'n_regions': int(n_regions),
                        'vertices': vertices,
                        'faces': faces
                    })

        return jsonify({
            'status': 'success',
            'layers': layers_data,
            'n_layers': len(sorted_layers),
            'n_valid_layers': len(layers_data),
            'min_val': float(min_val),
            'max_val': float(max_val),
            'value_label': 'Energy Density (J/mm²)',
            'layer_thickness': layer_thickness,
            'voxel_size': voxel_size,
            'per_region': True  # Flag indicating per-region visualization
        })
    elif data_type == 'risk':
        # Map risk levels to numeric values: LOW=0, MEDIUM=1, HIGH=2
        risk_levels = results.get('risk_levels', {})
        layer_values = {}
        for k, level in risk_levels.items():
            if level == 'LOW':
                layer_values[int(k)] = 0
            elif level == 'MEDIUM':
                layer_values[int(k)] = 1
            else:  # HIGH
                layer_values[int(k)] = 2
        value_label = 'Risk Level'
        min_val = 0
        max_val = 2
    elif data_type == 'area_ratio':
        layer_areas = results.get('layer_areas', {})
        contact_areas = results.get('contact_areas', {})
        area_ratio_power = results.get('params_used', {}).get('area_ratio_power',
                           results.get('params', {}).get('area_ratio_power', 3.0))
        layer_values = {}
        for k in layer_areas:
            a_layer = float(layer_areas[k])
            a_contact = float(contact_areas.get(k, 0))
            ratio = min(1.0, a_contact / a_layer) if a_layer > 0 else 0.0
            layer_values[int(k)] = ratio ** area_ratio_power
        power_label = f'^{area_ratio_power}' if area_ratio_power != 1.0 else ''
        value_label = f'Area Ratio (A_contact / A_layer){power_label}'
        min_val = 0.0
        max_val = 1.0
    elif data_type == 'gaussian_factor':
        G_layers = results.get('G_layers')
        if G_layers is None:
            return jsonify({'status': 'error', 'message': 'Gaussian factor only available in Mode B'}), 400
        grp = results.get('params_used', {}).get('gaussian_ratio_power', 0.15)
        layer_values = {}
        for k, g_val in G_layers.items():
            if hasattr(g_val, '__len__'):
                # 2D array: average multiplier over solid voxels only
                solid = masks[k] > 0
                if np.any(solid):
                    G_solid = g_val[solid]
                    g_avg = float(np.mean(G_solid))
                else:
                    g_avg = 0.0
            else:
                g_avg = float(g_val)
            layer_values[int(k)] = (1.0 / (1.0 + g_avg)) ** grp
        power_label = f'^{grp}' if grp != 1.0 else ''
        value_label = f'Gaussian Multiplier (1/(1+G)){power_label}'
        min_val = 0.0
        max_val = 1.0
    elif data_type == 'regions':
        # Regions visualization - supports two view modes
        from scipy import ndimage
        view_mode = request.args.get('view_mode', 'branches_3d')  # 'per_layer' or 'branches_3d'
        # Get region_data from either full results or slice-only data
        if results:
            region_data_all = results.get('region_data', {})
        elif slice_data:
            region_data_all = slice_data.get('region_data', {})
        else:
            region_data_all = {}

        # Color palette for per-layer view (distinct colors)
        REGION_COLORS = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
        ]

        if view_mode == 'per_layer':
            # Per-layer view: use 3D branch tracking to ensure consistent coloring
            # across layers (same physical object = same color, even if scipy
            # assigns different region IDs on different layers)
            branch_info = _build_3d_branches(masks, region_data_all)
            layer_regions = branch_info['layer_regions']

            layers_data = []
            sorted_layers = sorted(masks.keys())
            max_branches = len(branch_info['branches'])

            for layer in sorted_layers:
                mask = masks.get(layer)
                if mask is None:
                    continue
                mask_arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
                if mask_arr.sum() == 0:
                    continue

                z = layer * layer_thickness

                # Use region_data if available, otherwise label on the fly
                rd = region_data_all.get(layer)
                if rd and rd.get('n_regions', 0) > 0:
                    labeled = rd['labeled']
                    n_regions = rd['n_regions']
                else:
                    labeled, n_regions = ndimage.label(mask_arr > 0)

                if n_regions == 0:
                    continue

                # Generate one surface per region with color based on BRANCH ID
                # (tracked across layers for consistency)
                for rid in range(1, n_regions + 1):
                    region_mask = (labeled == rid).astype(np.uint8)
                    if region_mask.sum() == 0:
                        continue

                    # Get branch color from layer_regions (consistent tracking)
                    region_info = layer_regions.get(layer, {}).get(rid, {})
                    branch_id = region_info.get('branch_id', rid)
                    # Use REGION_COLORS palette but indexed by branch_id
                    color = REGION_COLORS[(branch_id - 1) % len(REGION_COLORS)]

                    vertices, faces = _generate_layer_surface(region_mask, voxel_size, z,
                                                              skip_garbage_filter=True)
                    if vertices and faces:
                        layers_data.append({
                            'layer': int(layer),
                            'z': float(z),
                            'value': int(branch_id),
                            'region_id': int(rid),
                            'branch_id': int(branch_id),
                            'n_regions': int(n_regions),
                            'color': color,
                            'vertices': vertices,
                            'faces': faces
                        })

            return jsonify({
                'status': 'success',
                'layers': layers_data,
                'n_layers': len(sorted_layers),
                'n_valid_layers': len(layers_data),
                'min_val': 1,
                'max_val': max_branches,
                'value_label': 'Region ID (tracked)',
                'layer_thickness': layer_thickness,
                'voxel_size': voxel_size,
                'is_categorical': True
            })

        # Default: 3D Branch visualization - connected regions across layers colored consistently
        # Build 3D branch tracking
        branch_info = _build_3d_branches(masks, region_data_all)
        layer_regions = branch_info['layer_regions']

        layers_data = []
        sorted_layers = sorted(masks.keys())
        max_branches = len(branch_info['branches'])

        for layer in sorted_layers:
            mask = masks.get(layer)
            if mask is None:
                continue
            mask_arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
            if mask_arr.sum() == 0:
                continue

            z = layer * layer_thickness

            # Use region_data if available, otherwise label on the fly
            rd = region_data_all.get(layer)
            if rd and rd.get('n_regions', 0) > 0:
                labeled = rd['labeled']
                n_regions = rd['n_regions']
            else:
                labeled, n_regions = ndimage.label(mask_arr > 0)

            if n_regions == 0:
                continue

            # Calculate region sizes for filtering out garbage (small fragments from messy STL)
            region_sizes = ndimage.sum(mask_arr > 0, labeled, range(1, n_regions + 1))
            max_region_size = np.max(region_sizes) if len(region_sizes) > 0 else 0

            # Overlap-based garbage detection:
            # - Small regions whose BOUNDING BOX OVERLAPS with larger regions = garbage
            # - Small regions that are standalone (no bbox overlap) = legitimate features
            # Threshold for "small": regions < 20% of largest region
            SMALL_REGION_THRESHOLD = 0.20
            small_region_size_limit = max_region_size * SMALL_REGION_THRESHOLD

            # Pre-compute bounding boxes for all regions
            region_bboxes = {}
            for rid in range(1, n_regions + 1):
                rows, cols = np.where(labeled == rid)
                if len(rows) > 0:
                    region_bboxes[rid] = (rows.min(), rows.max(), cols.min(), cols.max())

            # Helper function to check if two bounding boxes overlap
            def bboxes_overlap(bbox1, bbox2):
                r1_min, r1_max, c1_min, c1_max = bbox1
                r2_min, r2_max, c2_min, c2_max = bbox2
                # Check if boxes overlap (share any space)
                return (r1_min <= r2_max and r1_max >= r2_min and
                        c1_min <= c2_max and c1_max >= c2_min)

            # Pre-compute which regions are "garbage" (small AND bbox overlaps with larger)
            garbage_regions = set()
            for rid in range(1, n_regions + 1):
                region_size = region_sizes[rid - 1]
                if region_size >= small_region_size_limit:
                    continue  # Not small, definitely not garbage

                if rid not in region_bboxes:
                    continue

                small_bbox = region_bboxes[rid]

                # Check if this small region's bbox overlaps with any larger region's bbox
                for other_rid in range(1, n_regions + 1):
                    if other_rid == rid:
                        continue
                    other_size = region_sizes[other_rid - 1]
                    if other_size <= region_size:
                        continue  # Only check against larger regions

                    if other_rid not in region_bboxes:
                        continue

                    if bboxes_overlap(small_bbox, region_bboxes[other_rid]):
                        garbage_regions.add(rid)
                        break

            # Generate one surface per region with its 3D branch color (skip garbage)
            for rid in range(1, n_regions + 1):
                # Skip garbage regions (small AND near larger regions)
                if rid in garbage_regions:
                    continue

                region_mask = (labeled == rid).astype(np.uint8)
                if region_mask.sum() == 0:
                    continue

                # Get 3D branch color for this region
                region_info = layer_regions.get(layer, {}).get(rid, {})
                branch_id = region_info.get('branch_id', rid)
                color = region_info.get('color', '#888888')

                # Skip garbage filter since we already filtered at region level
                vertices, faces = _generate_layer_surface(region_mask, voxel_size, z,
                                                          skip_garbage_filter=True)
                if vertices and faces:
                    layers_data.append({
                        'layer': int(layer),
                        'z': float(z),
                        'value': int(branch_id),
                        'region_id': int(rid),
                        'branch_id': int(branch_id),
                        'n_regions': int(n_regions),
                        'color': color,
                        'vertices': vertices,
                        'faces': faces
                    })

        return jsonify({
            'status': 'success',
            'layers': layers_data,
            'n_layers': len(sorted_layers),
            'n_valid_layers': len(layers_data),
            'min_val': 1,
            'max_val': max_branches,
            'value_label': '3D Branch ID',
            'layer_thickness': layer_thickness,
            'voxel_size': voxel_size,
            'is_categorical': True
        })
    else:
        return jsonify({'status': 'error', 'message': f'Unknown data type: {data_type}'}), 400

    # Generate layer surfaces from masks
    layers_data = []
    sorted_layers = sorted(masks.keys())

    for layer in sorted_layers:
        mask = masks.get(layer)
        if mask is None:
            continue

        mask_arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        if mask_arr.sum() == 0:
            continue

        # Get the value for this layer
        value = layer_values.get(layer, 0)

        # Calculate Z height for this layer
        z = layer * layer_thickness

        # Find contour of the mask using simple boundary detection
        # Get bounding box and create vertices from mask outline
        rows, cols = np.where(mask_arr > 0)
        if len(rows) == 0:
            continue

        # Create a simplified polygon from the mask using convex hull approach
        # or generate a grid-based surface
        vertices, faces = _generate_layer_surface(mask_arr, voxel_size, z)

        if vertices and faces:
            layers_data.append({
                'layer': int(layer),
                'z': float(z),
                'value': float(value) if isinstance(value, (int, float, np.number)) else value,
                'vertices': vertices,
                'faces': faces
            })

    return jsonify({
        'status': 'success',
        'layers': layers_data,
        'n_layers': len(sorted_layers),
        'n_valid_layers': len(layers_data),
        'min_val': float(min_val),
        'max_val': float(max_val),
        'value_label': value_label,
        'layer_thickness': layer_thickness,
        'voxel_size': voxel_size
    })


def _mix_colors(hex_colors: list) -> str:
    """
    Mix multiple hex colors by averaging their RGB values.

    Parameters
    ----------
    hex_colors : list of str
        List of hex color strings (e.g., ['#3b82f6', '#f97316'])

    Returns
    -------
    str
        Mixed color as hex string
    """
    if not hex_colors:
        return '#888888'  # Default gray
    if len(hex_colors) == 1:
        return hex_colors[0]

    # Convert hex to RGB
    rgb_values = []
    for hex_color in hex_colors:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        rgb_values.append((r, g, b))

    # Average RGB values
    avg_r = int(sum(rgb[0] for rgb in rgb_values) / len(rgb_values))
    avg_g = int(sum(rgb[1] for rgb in rgb_values) / len(rgb_values))
    avg_b = int(sum(rgb[2] for rgb in rgb_values) / len(rgb_values))

    # Convert back to hex
    return f'#{avg_r:02x}{avg_g:02x}{avg_b:02x}'


def _build_3d_branches(masks: Dict[int, np.ndarray], region_data_all: Dict) -> Dict:
    """
    Build 3D connected component (branch) tracking across layers.

    Each 2D region on a layer is assigned to a 3D branch ID based on overlap
    with parent regions. Branches are colored consistently, with color variations
    when branches split (Y-branching) or merge (multiple branches joining).

    Parameters
    ----------
    masks : dict
        Layer masks {layer_idx: 2D numpy array}
    region_data_all : dict
        Region data from energy calculation {layer_idx: {'labeled': array, 'n_regions': int}}

    Returns
    -------
    dict with keys:
        - branches: {branch_id: {'base_color': str, 'child_colors': [str], 'layers': [int]}}
        - layer_regions: {layer_idx: {region_id: {'branch_id': int, 'color': str}}}
    """
    from scipy import ndimage

    # Base color palette (10 distinct colors)
    BASE_COLORS = [
        '#3b82f6',  # blue
        '#f97316',  # orange
        '#22c55e',  # green
        '#ef4444',  # red
        '#a855f7',  # purple
        '#eab308',  # yellow
        '#06b6d4',  # cyan
        '#ec4899',  # pink
        '#84cc16',  # lime
        '#f59e0b',  # amber
    ]

    branches = {}  # branch_id -> {color, layers}
    layer_regions = {}  # layer_idx -> {region_id: {branch_id, color}}

    next_branch_id = 1
    prev_labeled = None
    prev_layer_branches = {}  # region_id -> branch_id from previous layer
    prev_layer_idx = None  # Track previous layer index for debugging

    sorted_layers = sorted(masks.keys())

    # Debug: Track branch assignments for debugging first/last layer issues
    branch_debug_info = {}  # branch_id -> {'first_layer': int, 'last_layer': int, 'layers': []}

    for layer_idx in sorted_layers:
        mask = masks[layer_idx]
        if mask is None or np.sum(mask > 0) == 0:
            layer_regions[layer_idx] = {}
            continue

        # Get labeled regions
        rd = region_data_all.get(layer_idx)
        if rd and rd.get('n_regions', 0) > 0:
            labeled = rd['labeled']
            n_regions = rd['n_regions']
        else:
            labeled, n_regions = ndimage.label(mask > 0)

        # Filter out garbage regions BEFORE branch tracking
        # Garbage = small regions (< 20% of largest) whose BOUNDING BOX OVERLAPS with larger regions
        garbage_regions = set()
        if n_regions > 1:
            region_sizes = ndimage.sum(np.array(mask) > 0, labeled, range(1, n_regions + 1))
            max_region_size = np.max(region_sizes) if len(region_sizes) > 0 else 0
            SMALL_REGION_THRESHOLD = 0.20

            # Pre-compute bounding boxes
            region_bboxes = {}
            for rid in range(1, n_regions + 1):
                rows, cols = np.where(labeled == rid)
                if len(rows) > 0:
                    region_bboxes[rid] = (rows.min(), rows.max(), cols.min(), cols.max())

            def bboxes_overlap(bbox1, bbox2):
                r1_min, r1_max, c1_min, c1_max = bbox1
                r2_min, r2_max, c2_min, c2_max = bbox2
                return (r1_min <= r2_max and r1_max >= r2_min and
                        c1_min <= c2_max and c1_max >= c2_min)

            for rid in range(1, n_regions + 1):
                region_size = region_sizes[rid - 1]
                if region_size >= max_region_size * SMALL_REGION_THRESHOLD:
                    continue  # Not small

                if rid not in region_bboxes:
                    continue

                small_bbox = region_bboxes[rid]

                for other_rid in range(1, n_regions + 1):
                    if other_rid == rid:
                        continue
                    other_size = region_sizes[other_rid - 1]
                    if other_size <= region_size:
                        continue
                    if other_rid not in region_bboxes:
                        continue
                    if bboxes_overlap(small_bbox, region_bboxes[other_rid]):
                        garbage_regions.add(rid)
                        break

        curr_layer_branches = {}
        layer_regions[layer_idx] = {}

        # FIRST: Build parent-to-children mapping to detect splits
        parent_to_children = {}  # parent_branch_id -> [list of child region ids]
        region_parent_branches = {}  # region_id -> {parent_branch_id: overlap_count}

        for rid in range(1, n_regions + 1):
            # Skip garbage regions during branch tracking
            if rid in garbage_regions:
                continue

            region_mask = (labeled == rid)

            # Find overlapping parent regions
            parent_branches = {}
            if prev_labeled is not None:
                # First check for direct overlap
                overlap = region_mask & (prev_labeled > 0)
                if np.any(overlap):
                    parent_rids = np.unique(prev_labeled[overlap])
                    parent_rids = parent_rids[parent_rids > 0]

                    for parent_rid in parent_rids:
                        overlap_count = np.sum((labeled == rid) & (prev_labeled == parent_rid))
                        if overlap_count > 0:
                            parent_branch_id = prev_layer_branches.get(int(parent_rid))
                            if parent_branch_id:
                                parent_branches[parent_branch_id] = parent_branches.get(parent_branch_id, 0) + overlap_count

                # If no overlap found, check for proximity using dilation (for thin bridges/overhangs)
                if not parent_branches:
                    from scipy.ndimage import binary_dilation

                    # Dilate current region by 5 voxels to check for nearby parent regions
                    # This handles thin structures that shift position between layers
                    dilation_iterations = 5
                    dilated_region = binary_dilation(region_mask, iterations=dilation_iterations)

                    # Check which parent regions overlap with dilated current region
                    overlap_with_dilated = dilated_region & (prev_labeled > 0)
                    if np.any(overlap_with_dilated):
                        parent_rids = np.unique(prev_labeled[overlap_with_dilated])
                        parent_rids = parent_rids[parent_rids > 0]

                        for parent_rid in parent_rids:
                            # Calculate overlap between dilated current region and parent
                            proximity_overlap = np.sum(dilated_region & (prev_labeled == parent_rid))
                            if proximity_overlap > 0:
                                parent_branch_id = prev_layer_branches.get(int(parent_rid))
                                if parent_branch_id:
                                    # Score based on overlap area after dilation
                                    parent_branches[parent_branch_id] = parent_branches.get(parent_branch_id, 0) + proximity_overlap

            region_parent_branches[rid] = parent_branches

            # Build parent -> children mapping
            if parent_branches:
                # Child belongs to largest parent
                parent_branch_id = max(parent_branches, key=parent_branches.get)
                if parent_branch_id not in parent_to_children:
                    parent_to_children[parent_branch_id] = []
                parent_to_children[parent_branch_id].append(rid)

        # SECOND: Assign branch IDs with split and merge detection
        for rid in range(1, n_regions + 1):
            # Skip garbage regions
            if rid in garbage_regions:
                continue

            parent_branches = region_parent_branches.get(rid, {})

            if not parent_branches:
                # No parent - create new branch
                branch_id = next_branch_id
                next_branch_id += 1
                color = BASE_COLORS[(branch_id - 1) % len(BASE_COLORS)]
                branches[branch_id] = {
                    'color': color,
                    'layers': [layer_idx]
                }
            elif len(parent_branches) > 1:
                # MERGE DETECTED - multiple branches merging into one
                # Create new branch with mixed color from all parent branches
                branch_id = next_branch_id
                next_branch_id += 1

                # Mix colors from all parent branches
                parent_colors = [branches[pb_id]['color'] for pb_id in parent_branches.keys()]
                color = _mix_colors(parent_colors)

                branches[branch_id] = {
                    'color': color,
                    'layers': [layer_idx],
                    'is_merge': True,
                    'parent_branches': list(parent_branches.keys())
                }
            else:
                # Single parent
                parent_branch_id = max(parent_branches, key=parent_branches.get)
                children_of_parent = parent_to_children.get(parent_branch_id, [])

                if len(children_of_parent) == 1:
                    # No split - inherit parent branch ID
                    branch_id = parent_branch_id
                    if layer_idx not in branches[branch_id]['layers']:
                        branches[branch_id]['layers'].append(layer_idx)
                    color = branches[branch_id]['color']
                else:
                    # Split detected - create NEW branch for this child
                    branch_id = next_branch_id
                    next_branch_id += 1
                    color = BASE_COLORS[(branch_id - 1) % len(BASE_COLORS)]
                    branches[branch_id] = {
                        'color': color,
                        'layers': [layer_idx]
                    }

            curr_layer_branches[rid] = branch_id
            layer_regions[layer_idx][rid] = {
                'branch_id': branch_id,
                'color': color
            }

            # Track branch first/last layer info
            if branch_id not in branch_debug_info:
                # Calculate centroid for this region
                rows, cols = np.where(labeled == rid)
                centroid_r, centroid_c = np.mean(rows), np.mean(cols)
                branch_debug_info[branch_id] = {
                    'first_layer': layer_idx,
                    'first_layer_centroid': (centroid_r, centroid_c),
                    'last_layer': layer_idx,
                    'layers': [layer_idx]
                }
            else:
                branch_debug_info[branch_id]['last_layer'] = layer_idx
                branch_debug_info[branch_id]['layers'].append(layer_idx)

        prev_labeled = labeled
        prev_layer_branches = curr_layer_branches
        prev_layer_idx = layer_idx

    # Log branch debug info
    logger.debug(f"Branch tracking complete: {len(branches)} branches")
    for bid in sorted(branch_debug_info.keys())[:10]:  # Log first 10 branches
        info = branch_debug_info[bid]
        logger.debug(f"  Branch {bid}: layers {info['first_layer']}-{info['last_layer']} "
                    f"(centroid at first: {info.get('first_layer_centroid', 'N/A')})")

    return {
        'branches': branches,
        'layer_regions': layer_regions,
        'branch_debug_info': branch_debug_info
    }


def _generate_layer_surface(mask: np.ndarray, voxel_size: float, z: float,
                            min_area_fraction: float = 0.01,
                            min_area_voxels: int = 25,
                            skip_garbage_filter: bool = False) -> tuple:
    """Generate triangulated surface from a 2D mask with overlap-based garbage filtering.

    Uses marching squares-like approach to find contours and triangulate them.
    Returns (vertices, faces) for mesh3d rendering.

    Garbage filtering logic:
    - Small regions (< 20% of largest) whose BOUNDING BOX OVERLAPS with larger regions are filtered
    - Small regions that are standalone (no bbox overlap) are KEPT (legitimate features)

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask of the layer
    voxel_size : float
        Physical size of each voxel in mm
    z : float
        Z height of this layer in mm
    min_area_fraction : float
        (Deprecated) Not used with overlap-based filtering
    min_area_voxels : int
        (Deprecated) Not used with overlap-based filtering
    skip_garbage_filter : bool
        If True, skip garbage filtering (for pre-filtered single-region masks)
    """
    from scipy import ndimage

    # Find connected regions and their boundaries
    labeled, n_features = ndimage.label(mask)

    if n_features == 0:
        return [], []

    # Calculate area of each region
    region_sizes = ndimage.sum(mask > 0, labeled, range(1, n_features + 1))
    if len(region_sizes) == 0:
        return [], []

    # Overlap-based garbage detection (unless filtering is skipped)
    garbage_regions = set()
    if not skip_garbage_filter and n_features > 1:
        max_region_size = np.max(region_sizes)
        SMALL_REGION_THRESHOLD = 0.20  # 20% of largest
        small_region_size_limit = max_region_size * SMALL_REGION_THRESHOLD

        # Pre-compute bounding boxes
        region_bboxes = {}
        for region_id in range(1, n_features + 1):
            rows, cols = np.where(labeled == region_id)
            if len(rows) > 0:
                region_bboxes[region_id] = (rows.min(), rows.max(), cols.min(), cols.max())

        def bboxes_overlap(bbox1, bbox2):
            r1_min, r1_max, c1_min, c1_max = bbox1
            r2_min, r2_max, c2_min, c2_max = bbox2
            return (r1_min <= r2_max and r1_max >= r2_min and
                    c1_min <= c2_max and c1_max >= c2_min)

        for region_id in range(1, n_features + 1):
            region_size = region_sizes[region_id - 1]
            if region_size >= small_region_size_limit:
                continue  # Not small, not garbage

            if region_id not in region_bboxes:
                continue

            small_bbox = region_bboxes[region_id]

            for other_id in range(1, n_features + 1):
                if other_id == region_id:
                    continue
                other_size = region_sizes[other_id - 1]
                if other_size <= region_size:
                    continue  # Only check against larger regions

                if other_id not in region_bboxes:
                    continue

                if bboxes_overlap(small_bbox, region_bboxes[other_id]):
                    garbage_regions.add(region_id)
                    break

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for region_id in range(1, n_features + 1):
        # Skip garbage regions (small AND near larger regions)
        if region_id in garbage_regions:
            continue

        region_mask = (labeled == region_id)

        # Find boundary pixels using erosion
        eroded = ndimage.binary_erosion(region_mask)
        boundary = region_mask & ~eroded

        # Get boundary coordinates
        rows, cols = np.where(boundary)
        if len(rows) < 3:
            # If boundary too small, use all region pixels
            rows, cols = np.where(region_mask)
            if len(rows) < 3:
                continue

        # Convert to physical coordinates
        points = np.column_stack([cols * voxel_size, rows * voxel_size])

        # Create convex hull for triangulation
        try:
            from scipy.spatial import ConvexHull, Delaunay
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            if len(hull_points) < 3:
                continue

            # Calculate centroid
            cx = np.mean(hull_points[:, 0])
            cy = np.mean(hull_points[:, 1])

            # Add centroid as first vertex
            all_vertices.append([float(cx), float(cy), float(z)])
            centroid_idx = vertex_offset

            # Add hull vertices
            for pt in hull_points:
                all_vertices.append([float(pt[0]), float(pt[1]), float(z)])

            # Create fan triangulation from centroid
            n_pts = len(hull_points)
            for i in range(n_pts):
                v1 = vertex_offset + 1 + i
                v2 = vertex_offset + 1 + ((i + 1) % n_pts)
                all_faces.append([centroid_idx, v1, v2])

            vertex_offset = len(all_vertices)

        except Exception:
            # Fallback: create simple rectangular approximation
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Create rectangle corners
            x1, y1 = min_col * voxel_size, min_row * voxel_size
            x2, y2 = (max_col + 1) * voxel_size, (max_row + 1) * voxel_size

            # Add 4 vertices
            all_vertices.extend([
                [float(x1), float(y1), float(z)],
                [float(x2), float(y1), float(z)],
                [float(x2), float(y2), float(z)],
                [float(x1), float(y2), float(z)]
            ])

            # Add 2 triangles for rectangle
            all_faces.extend([
                [vertex_offset, vertex_offset + 1, vertex_offset + 2],
                [vertex_offset, vertex_offset + 2, vertex_offset + 3]
            ])

            vertex_offset = len(all_vertices)

    return all_vertices, all_faces


@app.route('/api/slice_visualization/<session_id>')
def get_slice_visualization(session_id):
    """Get sliced layer visualization data with edge points for kernel display.

    Returns layer surfaces and edge points where kernels can be placed.
    The frontend handles kernel rendering, updating when sigma changes.
    Also supports slice-only mode (from Slice button).
    """
    session = session_registry.get_session(session_id)
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404

    # Support both full analysis results and slice-only data
    # Check slice_data first (for slice-only mode) since set_complete may set
    # session.results to metadata without masks
    results = session.results
    slice_data = session.slice_data

    # Prefer slice_data if it has masks, otherwise use results
    if slice_data and slice_data.get('masks'):
        # Slice-only mode (or slice_data available from full analysis)
        masks = slice_data.get('masks', {})
        params = {
            'voxel_size': slice_data.get('voxel_size', 0.1),
            'layer_thickness': slice_data.get('layer_thickness', 0.04),
        }
    elif results and results.get('masks'):
        # Full analysis mode with masks in results
        masks = results.get('masks', {})
        params = results.get('params_used', {})
    else:
        return jsonify({'status': 'error', 'message': 'No results available'}), 404

    if not masks:
        return jsonify({'status': 'error', 'message': 'No mask data available'}), 404

    voxel_size = params.get('voxel_size', 0.1)
    layer_thickness = params.get('effective_layer_thickness', params.get('layer_thickness', 0.04))

    layers_data = []
    all_edge_points = []
    sorted_layers = sorted(masks.keys())

    for layer in sorted_layers:
        mask = masks.get(layer)
        if mask is None:
            continue

        mask_arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        if mask_arr.sum() == 0:
            continue

        z = layer * layer_thickness

        # Generate surface for this layer
        vertices, faces = _generate_layer_surface(mask_arr, voxel_size, z)

        if vertices and faces:
            layers_data.append({
                'layer': int(layer),
                'z': float(z),
                'vertices': vertices,
                'faces': faces
            })

            # Extract edge points from this layer's mask for kernel placement
            edge_points = _extract_edge_points(mask_arr, voxel_size, z)
            all_edge_points.extend(edge_points)

    return jsonify({
        'status': 'success',
        'layers': layers_data,
        'edge_points': all_edge_points,  # Random subset will be used by frontend
        'n_layers': len(sorted_layers),
        'n_valid_layers': len(layers_data),
        'layer_thickness': layer_thickness,
        'voxel_size': voxel_size
    })


def _extract_edge_points(mask: np.ndarray, voxel_size: float, z: float, sample_rate: int = 5) -> list:
    """Extract edge points from a 2D mask for kernel placement.

    Returns a list of [x, y, z] points on the outer edges.
    """
    from scipy import ndimage

    # Find boundary pixels
    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded

    rows, cols = np.where(boundary)
    if len(rows) == 0:
        return []

    # Sample every N points to reduce density
    indices = range(0, len(rows), sample_rate)
    edge_points = []

    for i in indices:
        x = cols[i] * voxel_size
        y = rows[i] * voxel_size
        edge_points.append([float(x), float(y), float(z)])

    return edge_points


@app.route('/api/stl_preview')
def get_stl_preview():
    """Get STL mesh data for 3D preview."""
    with current_state['lock']:
        if not current_state['stl_loaded']:
            return jsonify({'status': 'error', 'message': 'No STL loaded'}), 404
        stl_path = current_state['stl_path']
        stl_info = current_state['stl_info']

    try:
        import trimesh

        mesh = trimesh.load(stl_path)

        # Sample vertices if too many (for performance)
        max_triangles = 50000
        if len(mesh.faces) > max_triangles:
            # Simplify mesh
            mesh = mesh.simplify_quadric_decimation(max_triangles)

        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()

        return jsonify({
            'status': 'success',
            'vertices': vertices,
            'faces': faces,
            'n_triangles': len(faces),
            'bounds': mesh.bounds.tolist() if hasattr(mesh.bounds, 'tolist') else mesh.bounds
        })

    except Exception as e:
        logger.error(f"STL preview error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# MAIN PAGE - EMBEDDED HTML/CSS/JS
# =============================================================================
PRIMARY_COLOR = '#1E6537'
PRIMARY_DARK = '#154a28'
PRIMARY_LIGHT = '#2a804a'
ACCENT_COLOR = '#00A35F'
BG_PAGE = '#080e0c'
BG_HEADER = '#0c1810'
BG_SIDEBAR = '#141c1a'
BG_TABS = '#101a16'
BG_CARD = '#1a3426'
BG_MAIN = '#1a1a1a'
BG_INPUT = '#0e1a14'

HTML_TEMPLATE = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Overheating Classifier for LPBF</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary: {PRIMARY_COLOR};
            --primary-dark: {PRIMARY_DARK};
            --primary-light: {PRIMARY_LIGHT};
            --accent: {ACCENT_COLOR};
            --bg-page: {BG_PAGE};
            --bg-sidebar: {BG_SIDEBAR};
            --bg-card: {BG_CARD};
            --bg-main: {BG_MAIN};
            --bg-tabs: {BG_TABS};
            --bg-input: {BG_INPUT};
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --border-color: rgba(30, 101, 55, 0.4);
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-page);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .header {{
            background: {BG_HEADER};
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            font-size: 1.4rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header h1::before {{
            content: '';
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        .sidebar {{
            width: 320px;
            background: var(--bg-sidebar);
            overflow-y: auto;
            padding: 16px;
        }}
        .panel-title {{
            color: #fff;
            font-size: 15px;
            font-weight: bold;
            margin: 0 0 12px 0;
            padding-bottom: 8px;
        }}
        .sidebar-section {{
            margin-bottom: 6px;
            background: var(--bg-card);
            border-radius: 6px;
            border: 1px solid rgba(45, 95, 142, 0.15);
            transition: all 0.3s ease;
        }}
        .sidebar-section.expanded {{
            background: #1f422d;
            box-shadow: 0 4px 20px rgba(0, 127, 163, 0.25), 0 0 0 1px rgba(0, 127, 163, 0.3);
            border-color: rgba(0, 127, 163, 0.4);
        }}
        .section-header {{
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
            font-size: 0.8rem;
            border-radius: 6px;
        }}
        .section-header:hover {{ background: rgba(45, 95, 142, 0.2); }}
        .section-header .arrow {{ font-size: 0.75rem; color: var(--text-secondary); transition: transform 0.3s; }}
        .section-header.collapsed .arrow {{ transform: rotate(-90deg); }}
        .section-content {{
            padding: 0 12px 6px 12px;
            max-height: 800px;
            opacity: 1;
            transition: all 0.3s ease;
        }}
        .section-content.collapsed {{ max-height: 0; opacity: 0; padding: 0 12px; overflow: hidden; }}
        .param-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .param-label {{ font-size: 0.8rem; color: var(--text-secondary); }}
        .param-input {{
            width: 100px;
            padding: 6px 10px;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 0.85rem;
            text-align: right;
        }}
        .param-input:focus {{ outline: none; border-color: var(--accent); }}
        .param-unit {{ font-size: 0.75rem; color: var(--text-secondary); margin-left: 6px; min-width: 35px; }}
        .radio-option {{
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            cursor: pointer;
            padding: 2px 0;
        }}
        .radio-option:hover {{ color: var(--text-primary); }}
        .radio-option input {{ accent-color: var(--accent); width: 16px; height: 16px; }}
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            color: white;
            width: 100%;
            margin-top: 8px;
        }}
        .btn-primary:hover {{ background: linear-gradient(135deg, var(--primary-light), var(--accent)); }}
        .btn-primary:disabled {{ background: var(--border-color); cursor: not-allowed; }}
        .btn-primary.running {{
            background: linear-gradient(to right, var(--accent) var(--progress, 0%), var(--bg-card) var(--progress, 0%));
        }}
        .file-upload {{
            border: 2px dashed var(--border-color);
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 6px;
        }}
        .file-upload:hover {{ border-color: var(--accent); background: rgba(91, 217, 163, 0.1); }}
        .file-upload.loaded {{ border-color: var(--success); background: rgba(74, 222, 128, 0.1); }}
        .content {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .tab-nav {{
            display: flex;
            background: var(--bg-tabs);
            overflow-x: auto;
            padding: 0 16px;
        }}
        .tab-btn {{
            padding: 12px 20px;
            background: transparent;
            border: none;
            color: #888;
            font-size: 0.85rem;
            cursor: pointer;
            white-space: nowrap;
            border-top: 2px solid transparent;
            transition: all 0.15s;
        }}
        .tab-btn:hover:not(.active) {{ color: #bbb; background: rgba(255,255,255,0.03); }}
        .tab-btn.active {{ background: var(--bg-main); color: #fff; border-top-color: var(--accent); }}
        .tab-content {{ flex: 1; padding: 0; overflow: hidden; display: flex; flex-direction: column; }}
        .tab-panel {{ display: none; height: 100%; flex: 1; flex-direction: column; overflow: hidden; position: relative; }}
        .tab-panel.active {{ display: flex; }}
        .viz-container {{
            flex: 1;
            background: var(--bg-main);
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            min-height: 0;
        }}
        .viz-container .plotly-graph-div {{ width: 100% !important; height: 100% !important; position: absolute !important; top: 0 !important; left: 0 !important; }}
        .viz-container .js-plotly-plot {{ width: 100% !important; height: 100% !important; }}
        .viz-container .svg-container {{ width: 100% !important; height: 100% !important; }}
        .progress-bar {{
            height: 8px;
            background: var(--bg-input);
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            border-radius: 4px;
            transition: width 0.3s;
            width: 0%;
        }}
        /* Floating Console - positioned at top-left of main area */
        .floating-console {{
            position: absolute;
            top: 8px;
            left: 12px;
            z-index: 50;
            max-width: 500px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
            display: flex;
            flex-direction: column;
            pointer-events: none;
        }}

        .console-header-row {{
            pointer-events: auto;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 4px;
        }}

        .console-toggle-btn {{
            background: transparent;
            border: 1px solid rgba(255,255,255,0.3);
            color: #ccc;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 5px;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9);
        }}

        .console-toggle-btn:hover {{
            background: rgba(45, 95, 142, 0.3);
            color: #fff;
            border-color: var(--accent);
        }}

        .console-toggle-btn.active {{
            background: rgba(45, 95, 142, 0.4);
            color: #fff;
            border-color: var(--primary);
        }}

        .console-toggle-btn .chevron {{
            font-size: 9px;
            transition: transform 0.2s;
        }}

        .console-toggle-btn.active .chevron {{
            transform: rotate(90deg);
        }}

        #console-wrapper {{
            display: none;
            flex-direction: column;
            max-height: 600px;
            background: transparent;
            border: none;
            overflow: hidden;
            box-shadow: none;
            pointer-events: none;
        }}

        #console-wrapper.visible {{
            display: flex;
        }}

        .floating-console-content {{
            flex: 1;
            overflow-y: scroll;
            overflow-x: hidden;
            padding: 10px 4px 10px 12px;
            color: #d0f0d0;
            line-height: 1.6;
            background: transparent;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9), 0 0 8px rgba(0, 0, 0, 0.7);
            pointer-events: auto;
            user-select: none;
            direction: rtl;
            scrollbar-width: thin;
            scrollbar-color: rgba(45, 95, 142, 0.6) transparent;
            max-height: 550px;
            font-size: 11px;
            border: none;
        }}

        .floating-console-content::-webkit-scrollbar {{
            width: 6px;
        }}

        .floating-console-content::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .floating-console-content::-webkit-scrollbar-thumb {{
            background: rgba(45, 95, 142, 0.6);
            border-radius: 3px;
        }}

        .floating-console-content::-webkit-scrollbar-thumb:hover {{
            background: rgba(45, 95, 142, 0.8);
        }}

        .floating-console-content > * {{
            direction: ltr;
            pointer-events: none;
        }}

        .floating-console-content .error {{ color: #ff6b6b; font-weight: 500; }}
        .floating-console-content .info {{ color: #74ffb9; }}
        .floating-console-content .warn {{ color: #ffeaa7; }}
        .floating-console-content .success {{ color: #55efc4; font-weight: 500; }}
        .floating-console-content .progress {{ color: #a29bfe; }}
        .floating-console-content .stage {{ color: #5BD9A3; font-weight: 600; }}

        /* Floating Run Analysis button */
        .floating-simulate-btn {{
            pointer-events: auto;
            width: 300px;
            padding: 10px 24px;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
            overflow: hidden;
            margin-bottom: 6px;
        }}

        .floating-simulate-btn:hover:not(.running) {{
            background: linear-gradient(135deg, var(--primary-light), var(--accent));
            transform: translateY(-1px);
        }}

        .floating-simulate-btn.running {{
            background: linear-gradient(to right, var(--accent) var(--progress, 0%), var(--bg-input) var(--progress, 0%));
            cursor: default;
        }}

        .floating-simulate-btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}
        .info-card {{
            background: var(--bg-input);
            border-radius: 8px;
            padding: 12px;
            margin-top: 12px;
        }}
        .info-card h4 {{ font-size: 0.9rem; margin-bottom: 8px; color: var(--accent); }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 0.85rem;
            border-bottom: 1px solid var(--border-color);
        }}
        .info-row:last-child {{ border-bottom: none; }}
        .risk-legend {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(10, 15, 30, 0.9);
            padding: 10px;
            border-radius: 6px;
            border: 1px solid rgba(100, 200, 150, 0.2);
            z-index: 50;
        }}
        .risk-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.75rem; margin: 4px 0; }}
        .risk-color {{ width: 12px; height: 12px; border-radius: 3px; }}
        .risk-color.low {{ background: var(--success); }}
        .risk-color.medium {{ background: var(--warning); }}
        .risk-color.high {{ background: var(--danger); }}
        .sidebar-right {{
            width: 260px;
            background: var(--bg-sidebar);
            padding: 16px;
            overflow-y: auto;
        }}
        .control-group {{ margin-bottom: 12px; }}
        .control-group label {{ display: block; margin-bottom: 4px; color: var(--text-secondary); font-size: 11px; }}
        .control-group .value {{ color: #fff; font-weight: bold; float: right; }}
        .control-group input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
        .viz-block {{
            background: var(--bg-card);
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }}
        .viz-block .block-title {{ color: #fff; font-size: 12px; font-weight: 600; margin-bottom: 10px; }}
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-page); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 4px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            padding: 20px;
            padding-top: 70px;  /* Space for floating buttons */
        }}
        .summary-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 16px;
            border: 1px solid var(--border-color);
        }}
        .summary-card h4 {{ color: var(--accent); margin-bottom: 12px; font-size: 0.9rem; }}
        .summary-stat {{ display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.85rem; }}
        .summary-stat .label {{ color: var(--text-secondary); }}
        .summary-stat .value {{ color: var(--text-primary); font-weight: 500; }}
        /* Loading overlay for visualization tabs */
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(10, 15, 25, 0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }}
        .loading-overlay.hidden {{
            display: none;
        }}
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .loading-text {{
            margin-top: 12px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <header class="header">
        <h1>Overheating Classifier for LPBF</h1>
        <span style="color: var(--text-secondary); font-size: 0.85rem;">Energy-based risk prediction</span>
    </header>

    <div class="main-container">
        <!-- Left Sidebar -->
        <aside class="sidebar">
            <div class="panel-title">Parameters</div>

            <!-- STL Input Section -->
            <div class="sidebar-section expanded">
                <div class="section-header" onclick="toggleSection(this)">
                    <span>STL Input</span>
                    <span class="arrow">&#9660;</span>
                </div>
                <div class="section-content">
                    <div class="file-upload" id="fileUpload" onclick="document.getElementById('fileInput').click()">
                        <div style="font-size: 1.2rem; margin-bottom: 2px;">📁</div>
                        <div id="fileUploadText" style="font-size: 0.8rem;">Click or drag STL file here</div>
                    </div>
                    <input type="file" id="fileInput" accept=".stl" style="display: none;" onchange="handleFileUpload(this)">
                    <div style="display: flex; gap: 4px; margin-top: 4px;">
                        <button class="btn btn-secondary" style="flex: 1; padding: 4px 8px; font-size: 0.75rem;" onclick="loadTestSTL(1)">Test STL 1</button>
                        <button class="btn btn-secondary" style="flex: 1; padding: 4px 8px; font-size: 0.75rem;" onclick="loadTestSTL(2)">Test STL 2</button>
                        <button class="btn btn-secondary" style="flex: 1; padding: 4px 8px; font-size: 0.75rem;" onclick="loadTestSTL(3)">Test STL 3</button>
                        <button class="btn btn-secondary" style="flex: 1; padding: 4px 8px; font-size: 0.75rem;" onclick="loadTestSTL(4)">Test STL 4</button>
                    </div>

                    <!-- STL Info (shown when loaded) -->
                    <div class="stl-info-inline" id="stlInfo" style="display: none; margin-top: 6px; padding: 4px 8px; background: rgba(0,127,163,0.1); border-radius: 4px; border: 1px solid rgba(0,127,163,0.2);">
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.7rem; color: var(--text-secondary);">
                            <span><span id="infoTriangles">-</span> triangles</span>
                            <span id="infoDimensions">-</span>
                        </div>
                    </div>

                    <div style="margin-top: 4px; padding-top: 4px; border-top: 1px solid var(--border-color);">
                        <div class="param-row">
                            <span class="param-label">Voxel Size</span>
                            <input type="number" class="param-input" id="voxelSize" value="0.1" step="0.01">
                            <span class="param-unit">mm</span>
                        </div>
                        <div class="param-row">
                            <span class="param-label">Layer Thickness</span>
                            <input type="number" class="param-input" id="layerThickness" value="0.04" step="0.01">
                            <span class="param-unit">mm</span>
                        </div>
                        <div class="param-row" style="margin-top: 4px;">
                            <span class="param-label">Layer Grouping</span>
                            <input type="range" id="layerGrouping" min="1" max="100" value="25" step="1" style="flex: 1; margin: 0 6px;" oninput="updateLayerGrouping()">
                            <input type="number" class="param-input" id="layerGroupingValue" value="25" min="1" max="500" step="1" style="width: 60px;" onchange="syncLayerGrouping()">
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-secondary); margin-top: 2px;">
                            <span id="layerGroupingDim">= 0.04 mm effective</span>
                            <span id="layerGroupingLayers"></span>
                        </div>
                        <div style="margin-top: 2px;">
                            <button class="btn btn-secondary" style="padding: 2px 6px; font-size: 0.65rem;" onclick="autoLayerGrouping()">Auto (~200 layers)</button>
                        </div>
                        <div class="param-row" style="margin-top: 6px;">
                            <span class="param-label">Build Direction</span>
                            <select class="param-input" id="buildDirection" style="flex: 1;" onchange="onBuildDirectionChange()">
                                <option value="Z">Z-up (default)</option>
                                <option value="Y-">Y-up (rotate Y→Z)</option>
                                <option value="Y">Y-down (rotate Y→Z, flip)</option>
                                <option value="X">X-up (rotate X→Z)</option>
                            </select>
                        </div>

                        <!-- Slice Button -->
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color);">
                            <button class="btn btn-primary" id="sliceBtn" style="width: 100%; padding: 6px 12px; font-size: 0.8rem;" onclick="runSliceOnly()" disabled>
                                Slice
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Energy Model Section -->
            <div class="sidebar-section">
                <div class="section-header" onclick="toggleSection(this)">
                    <span>Energy Model</span>
                    <span class="arrow">&#9660;</span>
                </div>
                <div class="section-content">
                    <div style="margin-bottom: 6px;">
                        <label class="radio-option">
                            <input type="radio" name="energyModel" value="area_only" checked onchange="updateModelVisibility()">
                            <span>Area-Only Mode (faster)</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="energyModel" value="geometry_multiplier" onchange="updateModelVisibility()">
                            <span>Geometry Multiplier Mode</span>
                        </label>
                    </div>
                    <!-- Laser Parameters for Energy Calculation -->
                    <div style="margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid var(--border-color);">
                        <div style="font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 4px;">Laser Parameters</div>
                        <div class="param-row">
                            <span class="param-label">Power</span>
                            <input type="number" class="param-input" id="laserPower" value="200" step="10" min="50" max="500">
                            <span class="param-unit">W</span>
                        </div>
                        <div class="param-row">
                            <span class="param-label">Scan Speed</span>
                            <input type="number" class="param-input" id="scanSpeed" value="800" step="50" min="100" max="2000">
                            <span class="param-unit">mm/s</span>
                        </div>
                        <div class="param-row">
                            <span class="param-label">Hatch Distance</span>
                            <input type="number" class="param-input" id="hatchDistance" value="0.1" step="0.01" min="0.05" max="0.15">
                            <span class="param-unit">mm</span>
                        </div>
                    </div>
                    <div class="param-row">
                        <span class="param-label">Dissipation Factor</span>
                        <input type="number" class="param-input" id="dissipationFactor" value="0.5" step="0.05" min="0" max="1">
                        <span class="param-unit"></span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">Convection Factor</span>
                        <input type="number" class="param-input" id="convectionFactor" value="0.05" step="0.01" min="0" max="0.5">
                        <span class="param-unit"></span>
                    </div>
                    <div class="param-row" id="powerParamGroup">
                        <span class="param-label">Area Ratio Power</span>
                        <input type="number" class="param-input" id="areaRatioPower" value="3.0" step="0.1" min="0.1">
                        <span class="param-unit"></span>
                    </div>
                    <div id="geometryParams" style="display: block; margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border-color);">
                        <div class="param-row">
                            <span class="param-label">Sigma (σ)</span>
                            <input type="number" class="param-input" id="sigmaMM" value="1.0" step="0.1" onchange="onSigmaChange()">
                            <span class="param-unit">mm</span>
                        </div>
                        <div class="param-row">
                            <span class="param-label">G Max</span>
                            <input type="number" class="param-input" id="gMax" value="99" step="0.1" max="100">
                            <span class="param-unit"></span>
                        </div>
                        <div class="param-row" id="gaussianPowerGroup">
                            <span class="param-label">Gaussian Ratio Power</span>
                            <input type="number" class="param-input" id="gaussianRatioPower" value="0.15" step="0.01" min="0.01" max="1.0">
                            <span class="param-unit"></span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Risk Thresholds Section -->
            <div class="sidebar-section">
                <div class="section-header collapsed" onclick="toggleSection(this)">
                    <span>Risk Thresholds</span>
                    <span class="arrow">&#9660;</span>
                </div>
                <div class="section-content collapsed">
                    <div class="param-row">
                        <span class="param-label">Medium Threshold</span>
                        <input type="number" class="param-input" id="thresholdMedium" value="0.3" step="0.05" min="0" max="1">
                        <span class="param-unit"></span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">High Threshold</span>
                        <input type="number" class="param-input" id="thresholdHigh" value="0.6" step="0.05" min="0" max="1">
                        <span class="param-unit"></span>
                    </div>
                    <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 4px; line-height: 1.3;">
                        <strong style="color: var(--success);">LOW:</strong> score &lt; medium<br>
                        <strong style="color: var(--warning);">MEDIUM:</strong> medium ≤ score &lt; high<br>
                        <strong style="color: var(--danger);">HIGH:</strong> score ≥ high
                    </div>
                </div>
            </div>

        </aside>

        <!-- Main Content -->
        <main class="content">
            <nav class="tab-nav">
                <button class="tab-btn active" onclick="switchTab('preview', event)">STL Preview</button>
                <button class="tab-btn" onclick="switchTab('slices', event)">Sliced Layers</button>
                <button class="tab-btn" onclick="switchTab('regions', event)">Regions</button>
                <button class="tab-btn" onclick="switchTab('area_ratio', event)">Area Ratio</button>
                <button class="tab-btn" onclick="switchTab('gaussian', event)">Gaussian Multiplier</button>
                <button class="tab-btn" onclick="switchTab('energy', event)">Energy Accumulation</button>
                <button class="tab-btn" onclick="switchTab('density', event)">Energy Density</button>
                <button class="tab-btn" onclick="switchTab('risk', event)">Risk Map</button>
                <button class="tab-btn" onclick="switchTab('summary', event)">Summary</button>
            </nav>

            <div class="tab-content" style="position: relative;">
                <!-- Floating Console & Run Analysis Button -->
                <div class="floating-console">
                    <button class="floating-simulate-btn" id="runBtn" onclick="runAnalysis()" disabled>
                        &#9654; Run Analysis
                    </button>
                    <div class="console-header-row">
                        <button class="console-toggle-btn" id="consoleToggle" onclick="toggleConsole()">
                            <span class="chevron">&#9654;</span> Console
                        </button>
                    </div>
                    <div id="console-wrapper">
                        <div class="floating-console-content" id="console"></div>
                    </div>
                </div>

                <!-- STL Preview Tab -->
                <div class="tab-panel active" id="tab-preview">
                    <div class="viz-container" id="previewPlot">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Load an STL file to preview
                        </div>
                    </div>
                </div>

                <!-- Sliced Layers Tab -->
                <div class="tab-panel" id="tab-slices">
                    <div class="viz-container" id="slicesPlot">
                        <div class="loading-overlay hidden" id="slicesLoading">
                            <div class="loading-spinner"></div>
                            <div class="loading-text">Loading sliced layers...</div>
                        </div>
                        <div id="slicesPlaceholder" style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Click "Slice" or "Run Analysis" to see sliced layers
                        </div>
                    </div>
                </div>

                <!-- Regions Tab (moved after Sliced Layers) -->
                <div class="tab-panel" id="tab-regions">
                    <div class="viz-container" id="regionsPlot">
                        <div class="loading-overlay hidden" id="regionsLoading">
                            <div class="loading-spinner"></div>
                            <div class="loading-text">Loading regions...</div>
                        </div>
                        <div id="regionsPlaceholder" style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Click "Slice" or "Run Analysis" to see connected regions (islands)
                        </div>
                    </div>
                </div>

                <!-- Area Ratio Tab -->
                <div class="tab-panel" id="tab-area_ratio">
                    <div class="viz-container" id="areaRatioPlot">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Run analysis to see area ratio (A_contact / A_layer)
                        </div>
                    </div>
                </div>

                <!-- Gaussian Multiplier Tab -->
                <div class="tab-panel" id="tab-gaussian">
                    <div class="viz-container" id="gaussianPlot">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Run analysis (Mode B) to see Gaussian multiplier 1/(1+G)
                        </div>
                    </div>
                </div>

                <!-- Energy Accumulation Tab -->
                <div class="tab-panel" id="tab-energy">
                    <div class="viz-container" id="energyPlot">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Run analysis to see energy accumulation
                        </div>
                    </div>
                </div>

                <!-- Energy Density Tab -->
                <div class="tab-panel" id="tab-density">
                    <div class="viz-container" id="densityPlot">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Run analysis to see energy density (J/mm²)
                        </div>
                    </div>
                </div>

                <!-- Risk Map Tab -->
                <div class="tab-panel" id="tab-risk">
                    <div class="viz-container" id="riskPlot">
                        <div class="risk-legend" style="display: none;" id="riskLegend">
                            <div class="risk-item"><div class="risk-color low"></div> LOW (safe)</div>
                            <div class="risk-item"><div class="risk-color medium"></div> MEDIUM (caution)</div>
                            <div class="risk-item"><div class="risk-color high"></div> HIGH (risk)</div>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Run analysis to see risk classification
                        </div>
                    </div>
                </div>

                <!-- Summary Tab -->
                <div class="tab-panel" id="tab-summary">
                    <div class="summary-grid" id="summaryContent">
                        <div style="grid-column: 1 / -1; text-align: center; color: var(--text-secondary); padding: 40px;">
                            Run analysis to see summary
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Right Sidebar -->
        <aside class="sidebar-right">
            <div class="panel-title">Visualization</div>

            <div class="viz-block">
                <div class="block-title">Layer Control</div>
                <div class="control-group">
                    <label>Layer <span class="value" id="layerVal">1 / 1</span></label>
                    <input type="range" id="layerSlider" min="1" max="1" value="1" onchange="updateLayerViz(this.value)">
                </div>
            </div>

            <div class="viz-block">
                <div class="block-title">Display Options</div>
                <div class="control-group">
                    <label>Marker Size <span class="value" id="markerSizeVal">4</span></label>
                    <input type="range" id="markerSize" min="1" max="10" value="4" onchange="updateMarkerSize(this.value)">
                </div>
            </div>

            <div class="viz-block" id="colorScaleBlock" style="display: none;">
                <div class="block-title">Color Scale</div>
                <div class="control-group">
                    <label>Min Value</label>
                    <input type="number" id="colorScaleMin" step="any" style="width: 100%; padding: 4px 8px; background: {BG_CARD}; border: 1px solid #444; border-radius: 4px; color: #e0e0e0;" onchange="updateColorScaleMin(this.value)">
                </div>
                <div class="control-group">
                    <label>Max Value</label>
                    <input type="number" id="colorScaleMax" step="any" style="width: 100%; padding: 4px 8px; background: {BG_CARD}; border: 1px solid #444; border-radius: 4px; color: #e0e0e0;" onchange="updateColorScaleMax(this.value)">
                </div>
                <button id="colorScaleResetBtn" style="display: none; width: 100%; padding: 6px 12px; margin-top: 8px; background: #4a5568; border: none; border-radius: 4px; color: #e0e0e0; cursor: pointer; font-size: 12px;" onclick="resetColorScale()">
                    Reset to Adaptive
                </button>
            </div>

            <div class="viz-block" id="regionsViewBlock" style="display: none;">
                <div class="block-title">View Mode</div>
                <div class="control-group" style="margin-bottom: 0;">
                    <label class="radio-option" style="margin-bottom: 6px;">
                        <input type="radio" name="regionsView" value="per_layer" checked onchange="updateRegionsView()">
                        <span>Per-Layer Colors</span>
                    </label>
                    <label class="radio-option">
                        <input type="radio" name="regionsView" value="branches_3d" onchange="updateRegionsView()">
                        <span>3D Branch Tracking</span>
                    </label>
                </div>
            </div>

            <div class="viz-block" id="kernelVizBlock" style="display: none;">
                <div class="block-title">Kernel Visualization</div>
                <div class="control-group">
                    <label>Kernel Count <span class="value" id="kernelCountVal">10</span></label>
                    <input type="range" id="kernelCount" min="5" max="100" value="10" onchange="updateKernelCount(this.value)">
                </div>
                <div class="control-group">
                    <label>Opacity <span class="value" id="kernelOpacityVal">0.8</span></label>
                    <input type="range" id="kernelOpacity" min="0.1" max="1.0" step="0.1" value="0.8" onchange="updateKernelOpacity(this.value)">
                </div>
            </div>
        </aside>
    </div>

    <script>
        // Global state
        let currentSessionId = null;
        let sliceSessionId = null;  // Session ID for slice-only operations
        let analysisResults = null;
        let sliceResults = null;  // Stores slice-only results (masks, regions)
        let stlLoaded = false;
        let currentEventSource = null;  // Track EventSource for cleanup
        let currentRegionsView = 'per_layer';  // 'per_layer' or 'branches_3d'

        // Color scale state (null = use adaptive from data)
        let manualColorMin = null;
        let manualColorMax = null;
        let lastAdaptiveMin = null;  // Store adaptive values for reset (current tab)
        let lastAdaptiveMax = null;
        let currentColorDataType = null;  // Track which data type color scale applies to

        // Per-tab cache for adaptive min/max values (for instant switching)
        const tabAdaptiveRanges = {{}};

        // Toggle section expand/collapse (accordion behavior - only one open at a time)
        function toggleSection(header) {{
            const section = header.parentElement;
            const isCurrentlyCollapsed = header.classList.contains('collapsed');

            // Collapse all sections first (accordion behavior)
            document.querySelectorAll('.sidebar-section').forEach(sec => {{
                const h = sec.querySelector('.section-header');
                const c = sec.querySelector('.section-content');
                if (h && c) {{
                    h.classList.add('collapsed');
                    c.classList.add('collapsed');
                    sec.classList.remove('expanded');
                }}
            }});

            // If the clicked section was collapsed, expand it
            if (isCurrentlyCollapsed) {{
                header.classList.remove('collapsed');
                header.nextElementSibling.classList.remove('collapsed');
                section.classList.add('expanded');
            }}
        }}

        // Update geometry params visibility
        function updateModelVisibility() {{
            const isGeometry = document.querySelector('input[name="energyModel"]:checked').value === 'geometry_multiplier';
            document.getElementById('geometryParams').style.display = isGeometry ? 'block' : 'none';
            document.getElementById('powerParamGroup').style.display = isGeometry ? 'none' : '';
        }}

        // Shared 3D camera state across tabs
        let shared3DCamera = null;
        const plotIds3D = {{ 'preview': 'previewPlot', 'slices': 'slicesPlot', 'energy': 'energyPlot', 'density': 'densityPlot', 'risk': 'riskPlot', 'area_ratio': 'areaRatioPlot', 'gaussian': 'gaussianPlot', 'regions': 'regionsPlot' }};

        // Switch tabs
        function switchTab(tabName, evt) {{
            // Capture camera from currently active 3D plot before switching
            const activePanel = document.querySelector('.tab-panel.active');
            if (activePanel) {{
                for (const [tab, pid] of Object.entries(plotIds3D)) {{
                    const plotEl = document.getElementById(pid);
                    if (plotEl && plotEl.layout && plotEl.layout.scene && plotEl.layout.scene.camera &&
                        activePanel.id === 'tab-' + tab) {{
                        shared3DCamera = JSON.parse(JSON.stringify(plotEl.layout.scene.camera));
                        break;
                    }}
                }}
            }}

            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            if (evt && evt.target) {{
                evt.target.classList.add('active');
            }}
            document.getElementById('tab-' + tabName).classList.add('active');

            // Resize and apply shared camera to the new tab's plot
            setTimeout(() => {{
                const plotId = plotIds3D[tabName];
                if (plotId) {{
                    const plotEl = document.getElementById(plotId);
                    if (plotEl && plotEl.data) {{
                        Plotly.Plots.resize(plotEl);
                        if (shared3DCamera) {{
                            Plotly.relayout(plotEl, {{ 'scene.camera': shared3DCamera }});
                        }}
                    }}
                }}
            }}, 50);

            // Show/hide kernel visualization controls
            const kernelBlock = document.getElementById('kernelVizBlock');
            if (kernelBlock) {{
                kernelBlock.style.display = (tabName === 'gaussian') ? 'block' : 'none';
            }}

            // Show/hide regions view mode controls
            const regionsViewBlock = document.getElementById('regionsViewBlock');
            if (regionsViewBlock) {{
                regionsViewBlock.style.display = (tabName === 'regions') ? 'block' : 'none';
            }}

            // Show/hide color scale controls (for tabs with continuous color scales)
            const colorScaleBlock = document.getElementById('colorScaleBlock');
            const colorScaleTabs = ['energy', 'density', 'risk', 'area_ratio', 'gaussian'];
            if (colorScaleBlock) {{
                colorScaleBlock.style.display = colorScaleTabs.includes(tabName) ? 'block' : 'none';
            }}

            // Reset color scale when switching to a different visualization type
            if (colorScaleTabs.includes(tabName) && currentColorDataType !== tabName) {{
                currentColorDataType = tabName;
                // Reset to adaptive when switching tabs
                manualColorMin = null;
                manualColorMax = null;
                updateColorScaleResetButton();

                // Immediately restore cached adaptive values for this tab (responsive switching)
                const cacheKey = tabName === 'gaussian' ? 'gaussian_factor' : tabName;
                if (tabAdaptiveRanges[cacheKey]) {{
                    const cached = tabAdaptiveRanges[cacheKey];
                    lastAdaptiveMin = cached.min;
                    lastAdaptiveMax = cached.max;
                    updateColorScaleInputs(cached.min, cached.max);
                }}
            }}
        }}

        // Resize all visible Plotly plots
        function resizeAllPlots() {{
            ['previewPlot', 'slicesPlot', 'areaRatioPlot', 'gaussianPlot', 'regionsPlot', 'energyPlot', 'riskPlot'].forEach(plotId => {{
                const plotEl = document.getElementById(plotId);
                if (plotEl && plotEl.data) {{
                    Plotly.Plots.resize(plotEl);
                }}
            }});
        }}

        // Window resize handler
        window.addEventListener('resize', () => {{
            setTimeout(resizeAllPlots, 100);
        }});

        // Log to floating console
        function logConsole(message, type = '') {{
            const consoleEl = document.getElementById('console');
            const line = document.createElement('div');
            line.className = type;
            line.textContent = message;
            consoleEl.appendChild(line);
            consoleEl.scrollTop = consoleEl.scrollHeight;
            // Console stays in its current state - doesn't auto-expand
        }}

        // Toggle console visibility
        function toggleConsole() {{
            const wrapper = document.getElementById('console-wrapper');
            const toggleBtn = document.getElementById('consoleToggle');
            wrapper.classList.toggle('visible');
            toggleBtn.classList.toggle('active');
        }}

        // Store STL info for layer grouping calculation
        let stlDimensions = null;

        // Layer grouping functions
        function updateLayerGrouping() {{
            const slider = document.getElementById('layerGrouping');
            const valueInput = document.getElementById('layerGroupingValue');
            valueInput.value = slider.value;
            updateLayerGroupingDisplay();
        }}

        function syncLayerGrouping() {{
            const valueInput = document.getElementById('layerGroupingValue');
            const slider = document.getElementById('layerGrouping');
            let val = parseInt(valueInput.value) || 1;
            val = Math.max(1, Math.min(500, val));
            valueInput.value = val;
            slider.value = Math.min(val, 100);
            updateLayerGroupingDisplay();
        }}

        function updateLayerGroupingDisplay() {{
            const grouping = parseInt(document.getElementById('layerGroupingValue').value) || 1;
            const layerThickness = parseFloat(document.getElementById('layerThickness').value) || 0.04;
            const effectiveThickness = (grouping * layerThickness).toFixed(3);
            document.getElementById('layerGroupingDim').textContent = '= ' + effectiveThickness + ' mm effective';

            if (stlDimensions) {{
                const zHeight = stlDimensions[2];
                const nLayers = Math.ceil(zHeight / (grouping * layerThickness));
                document.getElementById('layerGroupingLayers').textContent = '~' + nLayers + ' layers';
            }} else {{
                document.getElementById('layerGroupingLayers').textContent = '';
            }}
        }}

        function autoLayerGrouping() {{
            if (!stlDimensions) {{
                logConsole('Load an STL first to auto-calculate layer grouping', 'warning');
                return;
            }}
            const layerThickness = parseFloat(document.getElementById('layerThickness').value) || 0.04;
            const zHeight = stlDimensions[2];
            const targetLayers = 200;
            const grouping = Math.max(1, Math.ceil(zHeight / (targetLayers * layerThickness)));
            document.getElementById('layerGroupingValue').value = grouping;
            document.getElementById('layerGrouping').value = Math.min(grouping, 100);
            updateLayerGroupingDisplay();
            logConsole('Auto layer grouping: ' + grouping + ' (~' + Math.ceil(zHeight / (grouping * layerThickness)) + ' layers)', 'info');
        }}

        // Load test STL (stlIndex: 1 or 2)
        async function loadTestSTL(stlIndex = 1) {{
            logConsole('Loading test STL ' + stlIndex + '...', 'info');

            try {{
                const response = await fetch('/api/load_test_stl?index=' + stlIndex, {{ method: 'POST' }});
                const data = await response.json();

                if (data.status === 'success') {{
                    // Clear previous analysis results
                    analysisResults = null;
                    currentSessionId = null;

                    stlLoaded = true;
                    stlDimensions = data.info.dimensions;
                    document.getElementById('runBtn').disabled = false;
                    document.getElementById('sliceBtn').disabled = false;
                    document.getElementById('fileUpload').classList.add('loaded');
                    document.getElementById('fileUploadText').textContent = data.filename || 'Test STL';

                    document.getElementById('stlInfo').style.display = 'block';
                    document.getElementById('infoTriangles').textContent = data.info.n_triangles.toLocaleString();
                    const dims = data.info.dimensions;
                    document.getElementById('infoDimensions').textContent = dims[0].toFixed(1) + ' × ' + dims[1].toFixed(1) + ' × ' + dims[2].toFixed(1) + ' mm';

                    updateLayerGroupingDisplay();

                    // Auto-select build direction based on test STL
                    // STL 1, 3 → Z-up; STL 2, 4 → Y-up
                    const buildDirSelect = document.getElementById('buildDirection');
                    if (stlIndex === 1 || stlIndex === 3) {{
                        buildDirSelect.value = 'Z';
                        logConsole('Build direction auto-set to: Z-up', 'info');
                    }} else if (stlIndex === 2 || stlIndex === 4) {{
                        buildDirSelect.value = 'Y-';
                        logConsole('Build direction auto-set to: Y-up', 'info');
                    }}
                    // Trigger change event to update preview
                    buildDirSelect.dispatchEvent(new Event('change'));

                    logConsole('Test STL loaded: ' + data.info.n_triangles + ' triangles', 'success');

                    // Render 3D STL preview
                    renderSTLPreview();
                }} else {{
                    logConsole('Error: ' + data.message, 'error');
                }}
            }} catch (err) {{
                logConsole('Failed to load test STL: ' + err.message, 'error');
            }}
        }}

        // Handle file upload
        async function handleFileUpload(input) {{
            if (!input.files || !input.files[0]) return;

            const file = input.files[0];
            const formData = new FormData();
            formData.append('file', file);

            logConsole('Uploading ' + file.name + '...', 'info');

            try {{
                const response = await fetch('/api/upload_stl', {{
                    method: 'POST',
                    body: formData
                }});

                const data = await response.json();

                if (data.status === 'success') {{
                    // Clear previous analysis results
                    analysisResults = null;
                    currentSessionId = null;

                    stlLoaded = true;
                    stlDimensions = data.info.dimensions;
                    document.getElementById('runBtn').disabled = false;
                    document.getElementById('sliceBtn').disabled = false;
                    document.getElementById('fileUpload').classList.add('loaded');
                    document.getElementById('fileUploadText').textContent = file.name;

                    document.getElementById('stlInfo').style.display = 'block';
                    document.getElementById('infoTriangles').textContent = data.info.n_triangles.toLocaleString();
                    const dims = data.info.dimensions;
                    document.getElementById('infoDimensions').textContent = dims[0].toFixed(1) + ' × ' + dims[1].toFixed(1) + ' × ' + dims[2].toFixed(1) + ' mm';

                    updateLayerGroupingDisplay();
                    logConsole('STL loaded: ' + data.info.n_triangles + ' triangles', 'success');

                    // Render 3D STL preview
                    renderSTLPreview();

                    // Clear file input to allow reselecting same file
                    input.value = '';
                }} else {{
                    logConsole('Error: ' + data.message, 'error');
                    input.value = '';  // Clear on error too
                }}
            }} catch (err) {{
                logConsole('Upload failed: ' + err.message, 'error');
            }}
        }}

        // Run slice-only (no energy analysis, just slicing and region detection)
        async function runSliceOnly() {{
            if (!stlLoaded) {{
                logConsole('Please upload an STL file first', 'warning');
                return;
            }}

            const params = {{
                voxel_size: parseFloat(document.getElementById('voxelSize').value),
                layer_thickness: parseFloat(document.getElementById('layerThickness').value),
                layer_grouping: parseInt(document.getElementById('layerGroupingValue').value) || 1,
                build_direction: document.getElementById('buildDirection').value
            }};

            const sliceBtn = document.getElementById('sliceBtn');
            sliceBtn.disabled = true;
            sliceBtn.textContent = 'Slicing...';

            try {{
                logConsole('Starting slice operation (layer grouping: ' + params.layer_grouping + 'x)...', 'info');

                const response = await fetch('/api/slice', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(params)
                }});

                const data = await response.json();

                if (data.status === 'started') {{
                    monitorSliceProgress(data.session_id);
                }} else {{
                    logConsole('Error: ' + data.message, 'error');
                    sliceBtn.disabled = false;
                    sliceBtn.textContent = 'Slice';
                }}
            }} catch (err) {{
                logConsole('Slice failed: ' + err.message, 'error');
                sliceBtn.disabled = false;
                sliceBtn.textContent = 'Slice';
            }}
        }}

        // Monitor slice progress via SSE
        function monitorSliceProgress(sessionId) {{
            const evtSource = new EventSource('/api/progress/' + sessionId);

            evtSource.onmessage = function(event) {{
                const data = JSON.parse(event.data);
                const sliceBtn = document.getElementById('sliceBtn');

                if (data.heartbeat) return;

                if (data.progress !== undefined) {{
                    logConsole('[' + data.progress + '%] ' + (data.message || ''), 'info');
                }}

                if (data.status === 'complete') {{
                    evtSource.close();
                    sliceBtn.disabled = false;
                    sliceBtn.textContent = 'Slice';
                    logConsole('Slice completed! Fetching results...', 'success');
                    // Store session ID for visualization
                    sliceSessionId = sessionId;
                    fetchSliceResults(sessionId);
                }}

                if (data.status === 'error' || data.status === 'cancelled') {{
                    evtSource.close();
                    sliceBtn.disabled = false;
                    sliceBtn.textContent = 'Slice';
                    logConsole('Slice failed: ' + (data.message || 'Unknown error'), 'error');
                }}
            }};

            evtSource.onerror = function() {{
                evtSource.close();
                document.getElementById('sliceBtn').disabled = false;
                document.getElementById('sliceBtn').textContent = 'Slice';
            }};
        }}

        // Fetch slice results after completion
        async function fetchSliceResults(sessionId) {{
            try {{
                const response = await fetch('/api/slice_results/' + sessionId);
                const data = await response.json();

                if (data.status === 'success') {{
                    sliceResults = data.results;
                    logConsole('Slice data ready: ' + sliceResults.n_layers + ' layers', 'success');

                    // Update layer slider
                    const slider = document.getElementById('layerSlider');
                    slider.max = sliceResults.n_layers;
                    slider.value = 1;
                    document.getElementById('layerVal').textContent = '1 / ' + sliceResults.n_layers;

                    // Render sliced layers and regions
                    updateSlicesVisualization();
                    updateRegionsVisualization();

                    // Switch to Regions tab
                    switchTab('regions');
                }} else {{
                    logConsole('Error fetching slice results: ' + data.message, 'error');
                }}
            }} catch (err) {{
                logConsole('Error fetching slice results: ' + err.message, 'error');
            }}
        }}

        // Update regions view toggle
        function updateRegionsView() {{
            currentRegionsView = document.querySelector('input[name="regionsView"]:checked').value;
            updateRegionsVisualization();
        }}

        // Render slices visualization (used by both Slice and Run Analysis)
        function updateSlicesVisualization() {{
            const results = analysisResults || sliceResults;
            if (!results) return;

            // Re-render the slices tab
            fetchAndRenderVisualization('slices', document.getElementById('layerSlider').value);
        }}

        // Render regions visualization based on current view mode
        function updateRegionsVisualization() {{
            const results = analysisResults || sliceResults;
            if (!results) return;

            const layer = parseInt(document.getElementById('layerSlider').value);
            fetchAndRenderVisualization('regions', layer, currentRegionsView);
        }}

        // Run analysis (can be called during existing run to restart with new params)
        async function runAnalysis() {{
            if (!stlLoaded) {{
                logConsole('Please upload an STL file first', 'warning');
                return;
            }}

            // If already running, cancel the previous session first
            if (currentSessionId && currentEventSource) {{
                logConsole('Cancelling previous analysis...', 'warning');
                try {{
                    await fetch('/api/cancel/' + currentSessionId, {{ method: 'POST' }});
                }} catch (e) {{
                    // Ignore cancel errors
                }}
                if (currentEventSource) {{
                    currentEventSource.close();
                    currentEventSource = null;
                }}
            }}

            const params = {{
                voxel_size: parseFloat(document.getElementById('voxelSize').value),
                layer_thickness: parseFloat(document.getElementById('layerThickness').value),
                layer_grouping: parseInt(document.getElementById('layerGroupingValue').value) || 1,
                build_direction: document.getElementById('buildDirection').value,
                dissipation_factor: parseFloat(document.getElementById('dissipationFactor').value),
                convection_factor: parseFloat(document.getElementById('convectionFactor').value),
                use_geometry_multiplier: document.querySelector('input[name="energyModel"]:checked').value === 'geometry_multiplier',
                sigma_mm: parseFloat(document.getElementById('sigmaMM').value),
                G_max: parseFloat(document.getElementById('gMax').value),
                threshold_medium: parseFloat(document.getElementById('thresholdMedium').value),
                threshold_high: parseFloat(document.getElementById('thresholdHigh').value),
                area_ratio_power: parseFloat(document.getElementById('areaRatioPower').value) || 3.0,
                gaussian_ratio_power: parseFloat(document.getElementById('gaussianRatioPower').value) || 0.15,
                laser_power: parseFloat(document.getElementById('laserPower').value) || 200,
                scan_speed: parseFloat(document.getElementById('scanSpeed').value) || 800,
                hatch_distance: parseFloat(document.getElementById('hatchDistance').value) || 0.1
            }};

            const runBtn = document.getElementById('runBtn');
            // Don't disable - allow rerun
            runBtn.classList.add('running');
            runBtn.innerHTML = '&#9632; Running...';
            runBtn.style.setProperty('--progress', '0%');

            try {{
                const grouping = params.layer_grouping;
                if (grouping > 1) {{
                    logConsole('Starting analysis (layer grouping: ' + grouping + 'x)...', 'info');
                }} else {{
                    logConsole('Starting analysis...', 'info');
                }}

                const response = await fetch('/api/run', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(params)
                }});

                const data = await response.json();

                if (data.status === 'started') {{
                    currentSessionId = data.session_id;
                    monitorProgress(currentSessionId);
                }} else {{
                    logConsole('Error: ' + data.message, 'error');
                    runBtn.classList.remove('running');
                    runBtn.innerHTML = '&#9654; Run Analysis';
                }}
            }} catch (err) {{
                logConsole('Analysis failed: ' + err.message, 'error');
                runBtn.classList.remove('running');
                runBtn.innerHTML = '&#9654; Run Analysis';
            }}
        }}

        // Monitor progress via SSE
        function monitorProgress(sessionId) {{
            // Close previous EventSource to prevent memory leak
            if (currentEventSource) {{
                currentEventSource.close();
                currentEventSource = null;
            }}
            const evtSource = new EventSource('/api/progress/' + sessionId);
            currentEventSource = evtSource;  // Track for cleanup

            evtSource.onmessage = function(event) {{
                const data = JSON.parse(event.data);
                const runBtn = document.getElementById('runBtn');

                if (data.heartbeat) return;

                const progress = data.progress || 0;
                runBtn.style.setProperty('--progress', progress + '%');
                runBtn.innerHTML = '&#9632; ' + Math.round(progress) + '%';

                if (data.step && !data.step.includes('[HEARTBEAT]')) {{
                    logConsole(data.step, 'info');
                }}

                if (data.status === 'complete') {{
                    evtSource.close();
                    currentEventSource = null;
                    // Keep button in running state - loadResults will finish it
                    runBtn.style.setProperty('--progress', '90%');
                    runBtn.innerHTML = '&#9632; Loading results...';
                    logConsole('Analysis complete! Loading visualizations...', 'success');
                    loadResults(sessionId);
                }} else if (data.status === 'error') {{
                    evtSource.close();
                    currentEventSource = null;
                    runBtn.classList.remove('running');
                    runBtn.innerHTML = '&#9654; Run Analysis';
                    logConsole('Error: ' + (data.error || 'Unknown error'), 'error');
                }}
            }};

            evtSource.onerror = function() {{
                evtSource.close();
                currentEventSource = null;
                const runBtn = document.getElementById('runBtn');
                runBtn.classList.remove('running');
                runBtn.innerHTML = '&#9654; Run Analysis';
                logConsole('Connection lost', 'error');
            }};
        }}

        // Load results
        async function loadResults(sessionId) {{
            const runBtn = document.getElementById('runBtn');
            try {{
                const response = await fetch('/api/results/' + sessionId);
                const data = await response.json();

                if (data.status === 'success') {{
                    analysisResults = data.results;
                    await updateVisualizations();
                }}
            }} catch (err) {{
                logConsole('Failed to load results: ' + err.message, 'error');
            }} finally {{
                // Now that all visualizations are rendered, mark as complete
                runBtn.classList.remove('running');
                runBtn.innerHTML = '&#9654; Run Analysis';
                runBtn.style.setProperty('--progress', '0%');
            }}
        }}

        // Update all visualizations (sequential to avoid UI freeze)
        async function updateVisualizations() {{
            if (!analysisResults) return;

            const runBtn = document.getElementById('runBtn');
            const nLayers = analysisResults.n_layers;
            document.getElementById('layerSlider').max = nLayers;
            document.getElementById('layerVal').textContent = '1 / ' + nLayers;

            const isModeB = analysisResults.params_used && analysisResults.params_used.mode === 'geometry_multiplier';
            const vizSteps = [
                {{ label: 'Sliced Layers', fn: () => renderSlicesPlot() }},
                {{ label: 'Energy', fn: () => renderLayerSurfaces('energy') }},
                {{ label: 'Density', fn: () => renderLayerSurfaces('density') }},
                {{ label: 'Risk', fn: () => renderLayerSurfaces('risk') }},
                {{ label: 'Area Ratio', fn: () => renderLayerSurfaces('area_ratio') }},
                ...(isModeB ? [
                    {{ label: 'Gaussian Multiplier', fn: () => renderLayerSurfaces('gaussian_factor') }},
                ] : []),
                {{ label: 'Regions', fn: () => renderLayerSurfaces('regions') }},
                {{ label: 'Summary', fn: () => updateSummary() }}
            ];

            for (let i = 0; i < vizSteps.length; i++) {{
                const step = vizSteps[i];
                const pct = 90 + Math.round((i / vizSteps.length) * 10);
                runBtn.style.setProperty('--progress', pct + '%');
                runBtn.innerHTML = '&#9632; ' + step.label + '... ' + (i + 1) + '/' + vizSteps.length;
                // Yield to browser to update UI before rendering
                await new Promise(r => setTimeout(r, 20));
                await step.fn();
            }}

            document.getElementById('riskLegend').style.display = 'block';
            logConsole('All visualizations loaded', 'success');
        }}

        // Helper to fetch and render visualization with optional view mode
        async function fetchAndRenderVisualization(dataType, layer, viewMode) {{
            if (dataType === 'slices') {{
                renderSlicesPlot();
            }} else if (dataType === 'regions') {{
                renderLayerSurfaces('regions', viewMode || currentRegionsView);
            }} else {{
                renderLayerSurfaces(dataType);
            }}
        }}

        // Unified layer surface rendering for 3D visualization
        async function renderLayerSurfaces(dataType, viewMode) {{
            const sessionId = currentSessionId || sliceSessionId;
            if (!sessionId) return;

            const plotConfig = {{
                'energy': {{ plotId: 'energyPlot', title: 'Energy Accumulation (3D)', loadingId: null, placeholderId: null }},
                'density': {{ plotId: 'densityPlot', title: 'Energy Density (J/mm²)', loadingId: null, placeholderId: null }},
                'risk': {{ plotId: 'riskPlot', title: 'Risk Classification (3D)', loadingId: null, placeholderId: null }},
                'area_ratio': {{ plotId: 'areaRatioPlot', title: 'Area Ratio (A_contact / A_layer)', loadingId: null, placeholderId: null }},
                'gaussian_factor': {{ plotId: 'gaussianPlot', title: 'Gaussian Multiplier 1/(1+G)', loadingId: null, placeholderId: null }},
                'regions': {{ plotId: 'regionsPlot', title: 'Connected Regions (Islands)', loadingId: 'regionsLoading', placeholderId: 'regionsPlaceholder' }}
            }};

            const config = plotConfig[dataType];
            if (!config) {{
                logConsole('Unknown data type: ' + dataType, 'error');
                return;
            }}

            // Show loading overlay if available
            if (config.loadingId) {{
                const loadingEl = document.getElementById(config.loadingId);
                if (loadingEl) loadingEl.classList.remove('hidden');
            }}
            // Hide placeholder
            if (config.placeholderId) {{
                const placeholderEl = document.getElementById(config.placeholderId);
                if (placeholderEl) placeholderEl.style.display = 'none';
            }}

            logConsole('Loading ' + config.title + ' surfaces...', 'info');

            try {{
                let url = '/api/layer_surfaces/' + sessionId + '/' + dataType;
                if (viewMode) {{
                    url += '?view_mode=' + viewMode;
                }}
                const response = await fetch(url);
                if (!response.ok) {{
                    logConsole('Failed to load ' + config.title + ': HTTP ' + response.status, 'error');
                    return;
                }}

                const data = await response.json();
                if (data.status !== 'success') {{
                    logConsole(config.title + ' error: ' + (data.message || 'Unknown'), 'error');
                    return;
                }}

                if (data.n_valid_layers === 0) {{
                    logConsole('No layer geometry available', 'warning');
                    return;
                }}

                // Create mesh3d traces for each layer
                const traces = [];
                const allX = [], allY = [], allZ = [];

                // Store adaptive values and update color scale inputs (with dataType for caching)
                updateColorScaleInputs(data.min_val, data.max_val, dataType);

                // Use manual values if set, otherwise use adaptive from data
                const minVal = manualColorMin !== null ? manualColorMin : data.min_val;
                const maxVal = manualColorMax !== null ? manualColorMax : data.max_val;
                const valueRange = maxVal - minVal || 1;

                const layerValues = [];

                data.layers.forEach((layerData) => {{
                    if (!layerData.vertices || layerData.vertices.length === 0) return;

                    const vertices = layerData.vertices;
                    const faces = layerData.faces;
                    const value = layerData.value;
                    const layer = layerData.layer;
                    const z = layerData.z;

                    layerValues.push(value);

                    const x = vertices.map(v => v[0]);
                    const y = vertices.map(v => v[1]);
                    const zArr = vertices.map(v => v[2]);

                    allX.push(...x);
                    allY.push(...y);
                    allZ.push(...zArr);

                    const i = faces.map(f => f[0]);
                    const j = faces.map(f => f[1]);
                    const k = faces.map(f => f[2]);

                    // Guard against division by zero when all values are identical
                    let normalizedValue;
                    if (valueRange === 0) {{
                        normalizedValue = 0.5;  // Middle of scale when all values equal
                    }} else {{
                        normalizedValue = (value - minVal) / valueRange;
                        // Clamp to [0, 1] when using manual color scale
                        normalizedValue = Math.max(0, Math.min(1, normalizedValue));
                    }}

                    // Get color based on data type
                    let color;
                    if (dataType === 'regions') {{
                        // Regions: use categorical color from backend
                        color = layerData.color || '#888';
                    }} else if (dataType === 'risk') {{
                        // Risk uses categorical colors: LOW=green, MEDIUM=yellow, HIGH=red
                        if (value >= 2) color = '#f87171';  // HIGH - red
                        else if (value >= 1) color = '#fbbf24';  // MEDIUM - yellow
                        else color = '#4ade80';  // LOW - green
                    }} else if (dataType === 'area_ratio' || dataType === 'gaussian_factor') {{
                        // Geometry factors: REVERSED - high=green (good), low=red (bad)
                        const rv = 1.0 - normalizedValue;  // Reverse: 0→1, 1→0
                        if (rv < 0.3) {{
                            const t = rv / 0.3;
                            color = `rgb(${{Math.round(74 + t * 107)}}, ${{Math.round(222 - t * 33)}}, ${{Math.round(128 - t * 92)}})`;
                        }} else if (rv < 0.6) {{
                            const t = (rv - 0.3) / 0.3;
                            color = `rgb(${{Math.round(181 + t * 70)}}, ${{Math.round(189 - t * 0)}}, ${{Math.round(36 - t * 0)}})`;
                        }} else {{
                            const t = (rv - 0.6) / 0.4;
                            color = `rgb(${{Math.round(251 - t * 3)}}, ${{Math.round(189 - t * 76)}}, ${{Math.round(36 + t * 77)}})`;
                        }}
                    }} else {{
                        // Energy: Jet colormap (blue → cyan → green → yellow → red)
                        let r, g, b;
                        if (normalizedValue < 0.125) {{
                            const t = normalizedValue / 0.125;
                            r = 0; g = 0; b = Math.round(128 + t * 127);
                        }} else if (normalizedValue < 0.375) {{
                            const t = (normalizedValue - 0.125) / 0.25;
                            r = 0; g = Math.round(t * 255); b = 255;
                        }} else if (normalizedValue < 0.625) {{
                            const t = (normalizedValue - 0.375) / 0.25;
                            r = Math.round(t * 255); g = 255; b = Math.round(255 * (1 - t));
                        }} else if (normalizedValue < 0.875) {{
                            const t = (normalizedValue - 0.625) / 0.25;
                            r = 255; g = Math.round(255 * (1 - t)); b = 0;
                        }} else {{
                            const t = (normalizedValue - 0.875) / 0.125;
                            r = Math.round(255 - t * 127); g = 0; b = 0;
                        }}
                        color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                    }}

                    let hoverText;
                    if (dataType === 'regions') {{
                        hoverText = 'Layer ' + layer + '<br>Z: ' + z.toFixed(2) + ' mm<br>Region: ' + value + ' / ' + (layerData.n_regions || '?') + '<extra></extra>';
                    }} else if (dataType === 'energy') {{
                        // Energy: show Joules with 2 decimal places
                        hoverText = 'Layer ' + layer + '<br>Z: ' + z.toFixed(2) + ' mm<br>Energy: ' + (typeof value === 'number' ? value.toFixed(2) : value) + ' J<extra></extra>';
                    }} else {{
                        hoverText = 'Layer ' + layer + '<br>Z: ' + z.toFixed(2) + ' mm<br>' + data.value_label + ': ' + (typeof value === 'number' ? value.toFixed(3) : value) + '<extra></extra>';
                    }}

                    traces.push({{
                        type: 'mesh3d',
                        x: x,
                        y: y,
                        z: zArr,
                        i: i,
                        j: j,
                        k: k,
                        color: color,
                        opacity: 1.0,  // Full opacity for proper depth sorting
                        flatshading: true,
                        lighting: {{ ambient: 0.8, diffuse: 0.5, specular: 0.1, roughness: 0.5 }},
                        lightposition: {{ x: 1000, y: 1000, z: 1000 }},
                        hovertemplate: hoverText,
                        showscale: false
                    }});
                }});

                if (traces.length === 0) {{
                    logConsole('No valid layer geometry to display', 'warning');
                    return;
                }}

                // Add colorbar via invisible scatter3d trace (skip for categorical regions)
                if (dataType !== 'regions') {{
                    let colorscale;
                    if (dataType === 'risk') {{
                        colorscale = [[0, '#4ade80'], [0.5, '#fbbf24'], [1.0, '#f87171']];
                    }} else if (dataType === 'area_ratio' || dataType === 'gaussian_factor') {{
                        // Reversed: red at 0 (bad), green at 1 (good)
                        colorscale = [[0, '#f87171'], [0.4, '#fbbd24'], [0.7, '#b5bd24'], [1.0, '#4ade80']];
                    }} else {{
                        // Energy: Jet colorscale (blue → cyan → green → yellow → red)
                        colorscale = [
                            [0.0, '#00007F'],    // Dark blue
                            [0.125, '#0000FF'],  // Blue
                            [0.375, '#00FFFF'],  // Cyan
                            [0.5, '#00FF00'],    // Green
                            [0.625, '#FFFF00'],  // Yellow
                            [0.875, '#FF0000'],  // Red
                            [1.0, '#7F0000']     // Dark red
                        ];
                    }}

                    const colorbarTrace = {{
                        type: 'scatter3d',
                        mode: 'markers',
                        x: [null],
                        y: [null],
                        z: [null],
                        marker: {{
                            size: 0.001,
                            color: layerValues,
                            colorscale: colorscale,
                            cmin: minVal,
                            cmax: maxVal,
                            colorbar: {{
                                title: {{ text: data.value_label, font: {{ color: '#e0e0e0', size: 11 }} }},
                                tickfont: {{ color: '#c0c0c0', size: 10 }},
                                x: 0.98,
                                xanchor: 'right',
                                y: 0.5,
                                yanchor: 'middle',
                                len: 0.5,
                                thickness: 12,
                                bgcolor: 'rgba(30,30,30,0.85)',
                                bordercolor: '#555',
                                borderwidth: 1
                            }},
                            showscale: true
                        }},
                        hoverinfo: 'skip',
                        showlegend: false
                    }};
                    traces.push(colorbarTrace);
                }}

                // Add kernel visualization for gaussian_factor tab
                if (dataType === 'gaussian_factor' && sliceVizData && sliceVizData.edge_points && sliceVizData.edge_points.length > 0) {{
                    const sigmaMM = parseFloat(document.getElementById('sigmaMM').value) || 1.0;
                    const kernelCount = parseInt(document.getElementById('kernelCount').value) || 10;
                    const kernelTraces = generateKernelTraces(sliceVizData.edge_points, sigmaMM, kernelCount);
                    traces.push(...kernelTraces);
                }}

                // Calculate scene bounds with padding
                const padding = 0.1;
                const xRange = [Math.min(...allX), Math.max(...allX)];
                const yRange = [Math.min(...allY), Math.max(...allY)];
                const zRange = [Math.min(...allZ), Math.max(...allZ)];
                const xPad = (xRange[1] - xRange[0]) * padding;
                const yPad = (yRange[1] - yRange[0]) * padding;
                const zPad = (zRange[1] - zRange[0]) * padding;

                const layout = {{
                    title: {{ text: config.title + ' (' + data.n_valid_layers + ' layers)', font: {{ color: '#fff', size: 14 }}, x: 0.5, y: 0.98 }},
                    paper_bgcolor: '{BG_MAIN}',
                    plot_bgcolor: '{BG_MAIN}',
                    scene: {{
                        xaxis: {{ title: 'X (mm)', color: '#aaa', gridcolor: '#333', range: [xRange[0] - xPad, xRange[1] + xPad] }},
                        yaxis: {{ title: 'Y (mm)', color: '#aaa', gridcolor: '#333', range: [yRange[0] - yPad, yRange[1] + yPad] }},
                        zaxis: {{ title: 'Z (mm)', color: '#aaa', gridcolor: '#333', range: [zRange[0] - zPad, zRange[1] + zPad] }},
                        bgcolor: '{BG_MAIN}',
                        aspectmode: 'data',
                        domain: {{ x: [0, 1], y: [0, 1] }}
                    }},
                    margin: {{ l: 0, r: 0, t: 30, b: 0 }},
                    autosize: true
                }};

                Plotly.newPlot(config.plotId, traces, layout, {{ responsive: true, displayModeBar: false }});

                // Force resize after plot is created (fixes sizing when tab is hidden)
                setTimeout(() => {{
                    const plotEl = document.getElementById(config.plotId);
                    if (plotEl && plotEl.data) {{
                        Plotly.Plots.resize(plotEl);
                    }}
                }}, 100);

                // Hide loading overlay
                if (config.loadingId) {{
                    const loadingEl = document.getElementById(config.loadingId);
                    if (loadingEl) loadingEl.classList.add('hidden');
                }}

                logConsole(config.title + ' loaded: ' + data.n_valid_layers + ' layers', 'success');

            }} catch (e) {{
                console.error(e);
                logConsole('Error loading ' + config.title + ': ' + e.message, 'error');
                // Hide loading overlay on error too
                if (config.loadingId) {{
                    const loadingEl = document.getElementById(config.loadingId);
                    if (loadingEl) loadingEl.classList.add('hidden');
                }}
            }}
        }}

        // Global storage for slice visualization data (to update kernels without re-fetching)
        let sliceVizData = null;

        // Render sliced layers with kernel visualization
        async function renderSlicesPlot() {{
            const sessionId = currentSessionId || sliceSessionId;
            if (!sessionId) return;

            // Show loading overlay, hide placeholder
            const slicesLoading = document.getElementById('slicesLoading');
            const slicesPlaceholder = document.getElementById('slicesPlaceholder');
            if (slicesLoading) slicesLoading.classList.remove('hidden');
            if (slicesPlaceholder) slicesPlaceholder.style.display = 'none';

            logConsole('Loading sliced layers visualization...', 'info');

            try {{
                const response = await fetch('/api/slice_visualization/' + sessionId);
                if (!response.ok) {{
                    logConsole('Failed to load slices: HTTP ' + response.status, 'error');
                    if (slicesLoading) slicesLoading.classList.add('hidden');
                    return;
                }}

                const data = await response.json();
                if (data.status !== 'success') {{
                    logConsole('Slices error: ' + (data.message || 'Unknown'), 'error');
                    if (slicesLoading) slicesLoading.classList.add('hidden');
                    return;
                }}

                // Store globally for kernel updates
                sliceVizData = data;

                // Actually render the plot
                updateSlicesPlotWithKernels();

                // Hide loading overlay
                if (slicesLoading) slicesLoading.classList.add('hidden');

                logConsole('Sliced layers loaded: ' + data.n_valid_layers + ' layers', 'success');

            }} catch (e) {{
                console.error(e);
                logConsole('Error loading slices: ' + e.message, 'error');
                if (slicesLoading) slicesLoading.classList.add('hidden');
            }}
        }}

        // Update slices plot (no re-fetch)
        function updateSlicesPlotWithKernels() {{
            if (!sliceVizData) return;

            const data = sliceVizData;

            // Create mesh3d traces for layers (neutral gray color)
            const traces = [];
            const allX = [], allY = [], allZ = [];

            // Layer surfaces
            data.layers.forEach((layerData, idx) => {{
                if (!layerData.vertices || layerData.vertices.length === 0) return;

                const vertices = layerData.vertices;
                const faces = layerData.faces;
                const x = vertices.map(v => v[0]);
                const y = vertices.map(v => v[1]);
                const z = vertices.map(v => v[2]);

                allX.push(...x);
                allY.push(...y);
                allZ.push(...z);

                // Alternate colors for layer visibility (solid colors for proper depth sorting)
                const color = (idx % 2 === 0) ? 'rgb(100, 200, 150)' : 'rgb(80, 180, 130)';

                traces.push({{
                    type: 'mesh3d',
                    x: x,
                    y: y,
                    z: z,
                    i: faces.map(f => f[0]),
                    j: faces.map(f => f[1]),
                    k: faces.map(f => f[2]),
                    color: color,
                    opacity: 1.0,  // Full opacity for proper depth sorting
                    flatshading: true,
                    lighting: {{ ambient: 0.8, diffuse: 0.5, specular: 0.1, roughness: 0.5 }},
                    lightposition: {{ x: 1000, y: 1000, z: 1000 }},
                    hovertemplate: 'Layer ' + layerData.layer + '<br>Z: %{{z:.2f}} mm<extra></extra>',
                    name: 'Layer ' + layerData.layer,
                    showlegend: false
                }});
            }});

            // Calculate scene bounds with padding
            if (allX.length === 0) return;

            const padding = 0.15;
            const xRange = [Math.min(...allX), Math.max(...allX)];
            const yRange = [Math.min(...allY), Math.max(...allY)];
            const zRange = [Math.min(...allZ), Math.max(...allZ)];
            const xPad = (xRange[1] - xRange[0]) * padding;
            const yPad = (yRange[1] - yRange[0]) * padding;
            const zPad = (zRange[1] - zRange[0]) * padding;

            const layout = {{
                title: {{ text: 'Sliced Layers (' + data.n_valid_layers + ' layers)', font: {{ color: '#fff', size: 14 }}, x: 0.5, y: 0.98 }},
                paper_bgcolor: '{BG_MAIN}',
                plot_bgcolor: '{BG_MAIN}',
                scene: {{
                    xaxis: {{ title: 'X (mm)', color: '#aaa', gridcolor: '#333', range: [xRange[0] - xPad, xRange[1] + xPad] }},
                    yaxis: {{ title: 'Y (mm)', color: '#aaa', gridcolor: '#333', range: [yRange[0] - yPad, yRange[1] + yPad] }},
                    zaxis: {{ title: 'Z (mm)', color: '#aaa', gridcolor: '#333', range: [zRange[0] - zPad, zRange[1] + zPad] }},
                    bgcolor: '{BG_MAIN}',
                    aspectmode: 'data',
                    domain: {{ x: [0, 1], y: [0, 1] }}
                }},
                margin: {{ l: 0, r: 0, t: 30, b: 0 }},
                autosize: true
            }};

            Plotly.newPlot('slicesPlot', traces, layout, {{ responsive: true, displayModeBar: false }});

            // Force resize after plot is created
            setTimeout(() => {{
                const plotEl = document.getElementById('slicesPlot');
                if (plotEl && plotEl.data) {{
                    Plotly.Plots.resize(plotEl);
                }}
            }}, 100);
        }}

        // Generate half-sphere kernel traces with gradient shells
        function generateKernelTraces(edgePoints, sigmaMM, count) {{
            const opacity = parseFloat(document.getElementById('kernelOpacity').value) || 0.8;
            const radius = sigmaMM * 2;  // Visual radius based on sigma (2σ covers ~95% of Gaussian)

            // Randomly select 'count' edge points
            const shuffled = [...edgePoints].sort(() => Math.random() - 0.5);
            const selected = shuffled.slice(0, Math.min(count, edgePoints.length));

            const traces = [];
            const numShells = 4;  // Number of concentric shells
            const resolution = 12;

            // Color gradient: bright red at center -> fading to transparent at edge
            const colors = [
                `rgba(255, 50, 50, ${{opacity}})`,           // Inner - bright red
                `rgba(255, 100, 50, ${{opacity * 0.7}})`,    // Orange-red
                `rgba(255, 150, 80, ${{opacity * 0.4}})`,    // Light orange
                `rgba(255, 200, 150, ${{opacity * 0.2}})`    // Pale, nearly transparent
            ];

            selected.forEach((point) => {{
                const [px, py, pz] = point;

                // Generate concentric shells from inside out
                for (let shell = 0; shell < numShells; shell++) {{
                    const shellRadius = radius * (shell + 1) / numShells;
                    const halfSphere = generateHalfSphereMesh(shellRadius, resolution);

                    traces.push({{
                        type: 'mesh3d',
                        x: halfSphere.x.map(v => v + px),
                        y: halfSphere.y.map(v => v + py),
                        z: halfSphere.z.map(v => v + pz),
                        i: halfSphere.i,
                        j: halfSphere.j,
                        k: halfSphere.k,
                        color: colors[shell],
                        flatshading: false,
                        opacity: 1,  // Using color alpha instead
                        hovertemplate: `Kernel (σ=${{sigmaMM.toFixed(1)}}mm, shell ${{shell+1}}/${{numShells}})<extra></extra>`,
                        name: 'Kernel Shell',
                        showlegend: false
                    }});
                }}
            }});

            return traces;
        }}

        // Helper: Generate half-sphere mesh pointing downward
        function generateHalfSphereMesh(radius, segments) {{
            const vertices = {{ x: [], y: [], z: [] }};
            const faces = {{ i: [], j: [], k: [] }};

            for (let lat = 0; lat <= segments / 2; lat++) {{
                const theta = (lat / segments) * Math.PI;
                const sinTheta = Math.sin(theta);
                const cosTheta = Math.cos(theta);

                for (let lon = 0; lon <= segments; lon++) {{
                    const phi = (lon / segments) * 2 * Math.PI;
                    vertices.x.push(radius * sinTheta * Math.cos(phi));
                    vertices.y.push(radius * sinTheta * Math.sin(phi));
                    vertices.z.push(-radius * cosTheta);  // Downward
                }}
            }}

            const latCount = Math.floor(segments / 2) + 1;
            const lonCount = segments + 1;

            for (let lat = 0; lat < latCount - 1; lat++) {{
                for (let lon = 0; lon < segments; lon++) {{
                    const first = lat * lonCount + lon;
                    const second = first + lonCount;
                    faces.i.push(first, second);
                    faces.j.push(first + 1, second + 1);
                    faces.k.push(second, first + 1);
                }}
            }}

            return {{ x: vertices.x, y: vertices.y, z: vertices.z, i: faces.i, j: faces.j, k: faces.k }};
        }}

        // Update kernel count from slider
        function updateKernelCount(value) {{
            document.getElementById('kernelCountVal').textContent = value;
            renderLayerSurfaces('gaussian_factor');
        }}

        // Update kernel opacity
        function updateKernelOpacity(value) {{
            document.getElementById('kernelOpacityVal').textContent = value;
            renderLayerSurfaces('gaussian_factor');
        }}

        // Handle sigma parameter change - update kernel visualization
        function onSigmaChange() {{
            if (sliceVizData) {{
                logConsole('Sigma changed - updating kernel visualization', 'info');
                updateSlicesPlotWithKernels();
            }}
            renderLayerSurfaces('gaussian_factor');
        }}

        // Apply build direction rotation to coordinates (for preview)
        // Must match backend trimesh rotation: -90° around X for Y-up, etc.
        function applyBuildDirectionRotation(x, y, z, buildDir) {{
            if (buildDir === 'Y') {{
                // Backend: -90° around X axis: (x,y,z) -> (x, z, -y)
                return {{ x: x, y: z, z: y.map(v => -v) }};
            }} else if (buildDir === 'Y-') {{
                // Backend: -90° around X then flip Z: (x,y,z) -> (x, z, y)
                return {{ x: x, y: z, z: y }};
            }} else if (buildDir === 'X') {{
                // Backend: 90° around Y axis: (x,y,z) -> (z, y, -x)
                return {{ x: z, y: y, z: x.map(v => -v) }};
            }}
            // Z-up (default): no rotation
            return {{ x, y, z }};
        }}

        // Handle build direction change
        function onBuildDirectionChange() {{
            const buildDir = document.getElementById('buildDirection').value;
            logConsole('Build direction changed to: ' + buildDir + '-up', 'info');
            if (stlLoaded) {{
                renderSTLPreview();
            }}
        }}

        // Render STL preview
        async function renderSTLPreview() {{
            logConsole('Loading STL preview...', 'info');

            try {{
                const buildDir = document.getElementById('buildDirection').value;
                const response = await fetch('/api/stl_preview?build_direction=' + buildDir);
                if (!response.ok) {{
                    logConsole('Failed to load STL preview: HTTP ' + response.status, 'error');
                    return;
                }}

                const data = await response.json();
                if (data.status !== 'success') {{
                    logConsole('STL preview error: ' + (data.message || 'Unknown'), 'error');
                    return;
                }}

                const vertices = data.vertices;
                const faces = data.faces;

                let x = vertices.map(v => v[0]);
                let y = vertices.map(v => v[1]);
                let z = vertices.map(v => v[2]);

                // Apply rotation for preview
                const rotated = applyBuildDirectionRotation(x, y, z, buildDir);
                x = rotated.x;
                y = rotated.y;
                z = rotated.z;

                const i = faces.map(f => f[0]);
                const j = faces.map(f => f[1]);
                const k = faces.map(f => f[2]);

                const trace = {{
                    type: 'mesh3d',
                    x: x,
                    y: y,
                    z: z,
                    i: i,
                    j: j,
                    k: k,
                    color: '#5BD9A3',
                    opacity: 1.0,
                    flatshading: true,
                    hovertemplate: 'X: %{{x:.2f}} mm<br>Y: %{{y:.2f}} mm<br>Z: %{{z:.2f}} mm<extra></extra>'
                }};

                // Calculate bounds
                const padding = 0.1;
                const xRange = [Math.min(...x), Math.max(...x)];
                const yRange = [Math.min(...y), Math.max(...y)];
                const zRange = [Math.min(...z), Math.max(...z)];
                const xPad = (xRange[1] - xRange[0]) * padding;
                const yPad = (yRange[1] - yRange[0]) * padding;
                const zPad = (zRange[1] - zRange[0]) * padding;

                const dirLabel = buildDir === 'Z' ? '' : ' (' + buildDir + '-up)';
                const layout = {{
                    title: {{ text: 'STL Preview (' + data.n_triangles + ' triangles)' + dirLabel, font: {{ color: '#fff', size: 14 }}, x: 0.5, y: 0.98 }},
                    paper_bgcolor: '{BG_MAIN}',
                    plot_bgcolor: '{BG_MAIN}',
                    scene: {{
                        xaxis: {{ title: 'X (mm)', color: '#aaa', gridcolor: '#333', range: [xRange[0] - xPad, xRange[1] + xPad] }},
                        yaxis: {{ title: 'Y (mm)', color: '#aaa', gridcolor: '#333', range: [yRange[0] - yPad, yRange[1] + yPad] }},
                        zaxis: {{ title: 'Z (mm)', color: '#aaa', gridcolor: '#333', range: [zRange[0] - zPad, zRange[1] + zPad] }},
                        bgcolor: '{BG_MAIN}',
                        aspectmode: 'data',
                        domain: {{ x: [0, 1], y: [0, 1] }}
                    }},
                    margin: {{ l: 0, r: 0, t: 30, b: 0 }},
                    autosize: true
                }};

                Plotly.newPlot('previewPlot', [trace], layout, {{ responsive: true, displayModeBar: false }});

                // Force resize after plot is created
                setTimeout(() => {{
                    const plotEl = document.getElementById('previewPlot');
                    if (plotEl && plotEl.data) {{
                        Plotly.Plots.resize(plotEl);
                    }}
                }}, 100);

                logConsole('STL preview loaded: ' + data.n_triangles + ' triangles' + (buildDir !== 'Z' ? ' (' + buildDir + '-up)' : ''), 'success');

            }} catch (e) {{
                console.error(e);
                logConsole('Error loading STL preview: ' + e.message, 'error');
            }}
        }}

        // Update summary tab
        function updateSummary() {{
            const r = analysisResults;
            const s = r.summary;
            const p = r.params_used;

            // Calculate max energy density for display
            const densityScores = r.energy_density_scores || {{}};
            const densityValues = Object.values(densityScores);
            const maxDensity = densityValues.length > 0 ? Math.max(...densityValues) : 0;
            const meanDensity = densityValues.length > 0 ? densityValues.reduce((a, b) => a + b, 0) / densityValues.length : 0;

            document.getElementById('summaryContent').innerHTML = `
                <div class="summary-card">
                    <h4>Analysis Results (Density-Based)</h4>
                    <div class="summary-stat"><span class="label">Total Layers</span><span class="value">${{s.n_layers}}</span></div>
                    <div class="summary-stat"><span class="label">Max Risk Score</span><span class="value">${{s.max_risk_score.toFixed(3)}}</span></div>
                    <div class="summary-stat"><span class="label">Max Risk Layer</span><span class="value">${{s.max_risk_layer}}</span></div>
                    <div class="summary-stat"><span class="label">Max Density</span><span class="value">${{maxDensity.toFixed(4)}} J/mm²</span></div>
                    <div class="summary-stat"><span class="label">Mean Density</span><span class="value">${{meanDensity.toFixed(4)}} J/mm²</span></div>
                </div>
                <div class="summary-card">
                    <h4>Risk Distribution</h4>
                    <div class="summary-stat"><span class="label" style="color: var(--success)">LOW</span><span class="value">${{s.n_low}} layers</span></div>
                    <div class="summary-stat"><span class="label" style="color: var(--warning)">MEDIUM</span><span class="value">${{s.n_medium}} layers</span></div>
                    <div class="summary-stat"><span class="label" style="color: var(--danger)">HIGH</span><span class="value">${{s.n_high}} layers</span></div>
                </div>
                <div class="summary-card">
                    <h4>Parameters Used</h4>
                    <div class="summary-stat"><span class="label">Mode</span><span class="value">${{p.mode}}</span></div>
                    <div class="summary-stat"><span class="label">Build Direction</span><span class="value">${{p.build_direction || 'Z'}}-up</span></div>
                    <div class="summary-stat"><span class="label">Voxel Size</span><span class="value">${{p.voxel_size}} mm</span></div>
                    <div class="summary-stat"><span class="label">Layer Grouping</span><span class="value">${{p.layer_grouping}}x (${{p.effective_layer_thickness.toFixed(3)}} mm)</span></div>
                    <div class="summary-stat"><span class="label">Dissipation Factor</span><span class="value">${{p.dissipation_factor}}</span></div>
                    <div class="summary-stat"><span class="label">Convection Factor</span><span class="value">${{p.convection_factor}}</span></div>
                    ${{p.mode === 'area_only' ? '<div class="summary-stat"><span class="label">Area Ratio Power</span><span class="value">' + (p.area_ratio_power || 3.0) + '</span></div>' : '<div class="summary-stat"><span class="label">Gaussian Ratio Power</span><span class="value">' + (p.gaussian_ratio_power || 0.15) + '</span></div>'}}
                </div>
                <div class="summary-card">
                    <h4>Thresholds</h4>
                    <div class="summary-stat"><span class="label">Medium</span><span class="value">${{p.threshold_medium}}</span></div>
                    <div class="summary-stat"><span class="label">High</span><span class="value">${{p.threshold_high}}</span></div>
                    <div class="summary-stat"><span class="label">Computation Time</span><span class="value">${{r.computation_time_seconds.toFixed(1)}} s</span></div>
                </div>
            `;
        }}

        // Update layer visualization
        function updateLayerViz(layer) {{
            if (!analysisResults) return;
            const nLayers = analysisResults.n_layers;
            document.getElementById('layerVal').textContent = layer + ' / ' + nLayers;
        }}

        // Update marker size
        function updateMarkerSize(size) {{
            document.getElementById('markerSizeVal').textContent = size;
        }}

        // Color scale control functions
        function updateColorScaleMin(value) {{
            const numVal = parseFloat(value);
            if (!isNaN(numVal)) {{
                manualColorMin = numVal;
                updateColorScaleResetButton();
                refreshCurrentVisualization();
            }}
        }}

        function updateColorScaleMax(value) {{
            const numVal = parseFloat(value);
            if (!isNaN(numVal)) {{
                manualColorMax = numVal;
                updateColorScaleResetButton();
                refreshCurrentVisualization();
            }}
        }}

        function resetColorScale() {{
            manualColorMin = null;
            manualColorMax = null;
            // Restore adaptive values to input fields
            if (lastAdaptiveMin !== null) {{
                document.getElementById('colorScaleMin').value = lastAdaptiveMin.toFixed(4);
            }}
            if (lastAdaptiveMax !== null) {{
                document.getElementById('colorScaleMax').value = lastAdaptiveMax.toFixed(4);
            }}
            updateColorScaleResetButton();
            refreshCurrentVisualization();
        }}

        function updateColorScaleResetButton() {{
            const resetBtn = document.getElementById('colorScaleResetBtn');
            if (resetBtn) {{
                // Show button if either min or max is manually set
                resetBtn.style.display = (manualColorMin !== null || manualColorMax !== null) ? 'block' : 'none';
            }}
        }}

        function updateColorScaleInputs(minVal, maxVal, dataType) {{
            // Store adaptive values for reset
            lastAdaptiveMin = minVal;
            lastAdaptiveMax = maxVal;

            // Cache adaptive values per data type for instant tab switching
            if (dataType) {{
                tabAdaptiveRanges[dataType] = {{ min: minVal, max: maxVal }};
            }}

            // Update input fields with current effective values
            const minInput = document.getElementById('colorScaleMin');
            const maxInput = document.getElementById('colorScaleMax');
            if (minInput) {{
                minInput.value = (manualColorMin !== null ? manualColorMin : minVal).toFixed(4);
            }}
            if (maxInput) {{
                maxInput.value = (manualColorMax !== null ? manualColorMax : maxVal).toFixed(4);
            }}
        }}

        function refreshCurrentVisualization() {{
            // Find the currently active tab and refresh its visualization
            const activePanel = document.querySelector('.tab-panel.active');
            if (!activePanel) return;

            const tabId = activePanel.id.replace('tab-', '');
            // Only refresh tabs that use color scale (energy, density, risk, area_ratio, gaussian)
            const colorScaleTabs = ['energy', 'density', 'risk', 'area_ratio', 'gaussian'];
            if (colorScaleTabs.includes(tabId)) {{
                const dataType = tabId === 'gaussian' ? 'gaussian_factor' : tabId;
                renderLayerSurfaces(dataType);
            }}
        }}

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            logConsole('Auto-loading Test STL 4...', 'info');
            updateModelVisibility();
            // Auto-load Test STL 4 by default
            loadTestSTL(4);
        }});
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_TEMPLATE


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overheating Classifier Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    logger.info(f"Starting Overheating Classifier server on {{args.host}}:{{args.port}}")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

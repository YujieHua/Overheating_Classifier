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
    'G_max': {'min': 0.5, 'max': 5.0},
    'threshold_medium': {'min': 0.0, 'max': 1.0},
    'threshold_high': {'min': 0.0, 'max': 1.0},
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
    'lock': threading.RLock()
}


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
    default_path = os.getenv('TEST_STL_PATH',
        r"C:\Users\huayu\Local\Desktop\Overheating_Classifier\CAD\SF_3_overhanges_less_triangles.stl")

    if not os.path.exists(default_path):
        return jsonify({
            'status': 'error',
            'message': f'Test STL not found at: {default_path}'
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
    try:
        voxel_size = float(request.form.get('voxel_size', 0.1))
        layer_thickness = float(request.form.get('layer_thickness', 0.04))
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid parameter format: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid parameter format'}), 400

    is_valid, errors = validate_parameters({
        'voxel_size': voxel_size,
        'layer_thickness': layer_thickness
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
        args=(session, stl_path, voxel_size, layer_thickness)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'session_id': session.session_id
    })


def slice_worker(session: AnalysisSession, stl_path: str, voxel_size: float, layer_thickness: float):
    try:
        start_time = time.time()
        from src.data.stl_loader import load_stl, slice_stl

        def progress_with_cancel(progress, step):
            if session.is_cancelled():
                raise CancelledException("Slicing cancelled by user")
            session.update_progress(progress, step)

        session.update_progress(5, "[STAGE] Loading STL mesh...")

        # Load mesh first (slice_stl expects mesh_info dict, not path)
        mesh_info = load_stl(stl_path)

        slice_result = slice_stl(
            mesh_info=mesh_info,
            voxel_size=voxel_size,
            layer_thickness=layer_thickness,
            layer_grouping=1,  # Default grouping for preview slicing
            progress_callback=progress_with_cancel
        )

        if not session.is_cancelled():
            with current_state['lock']:
                current_state['cached_masks'] = slice_result['masks']
                current_state['cached_slice_params'] = {
                    'voxel_size': voxel_size,
                    'layer_thickness': layer_thickness,
                    'layer_grouping': 1  # Default for preview slicing
                }
                current_state['cached_slice_info'] = {
                    'n_layers': slice_result['n_layers'],
                    'grid_shape': slice_result['grid_shape'],
                }

            session.slice_data = slice_result
            computation_time = time.time() - start_time
            session.set_complete({
                'n_layers': slice_result['n_layers'],
                'grid_shape': slice_result['grid_shape'],
                'voxel_size': voxel_size,
                'layer_thickness': layer_thickness,
            }, computation_time)

    except CancelledException:
        logger.info(f"Slicing cancelled for session {session.session_id}")
    except Exception as e:
        logger.error(f"Slicing error: {e}", exc_info=True)
        session.set_error(str(e))


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

    session_registry.cleanup_all_except(None)
    session = session_registry.create_session()

    slice_cache = None
    if cached_masks is not None and cached_slice_params is not None:
        slice_cache = {
            'masks': cached_masks,
            'params': cached_slice_params,
            'info': cached_slice_info
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
        from src.compute.geometry_score import (
            calculate_geometry_multiplier_per_layer,
            calculate_layer_averaged_G
        )

        def progress_with_cancel(progress, step):
            if session.is_cancelled():
                raise CancelledException("Analysis cancelled by user")
            session.update_progress(progress, step)

        voxel_size = params.get('voxel_size', 0.1)
        layer_thickness = params.get('layer_thickness', 0.04)
        layer_grouping = params.get('layer_grouping', 1)
        build_direction = params.get('build_direction', 'Z')
        dissipation_factor = params.get('dissipation_factor', 0.5)
        convection_factor = params.get('convection_factor', 0.05)
        use_geometry_multiplier = params.get('use_geometry_multiplier', False)
        sigma_mm = params.get('sigma_mm', 1.0)
        G_max = params.get('G_max', 2.0)
        threshold_medium = params.get('threshold_medium', 0.3)
        threshold_high = params.get('threshold_high', 0.6)

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
            import trimesh

            progress_with_cancel(5, "[STAGE] Loading STL mesh...")
            mesh_info = load_stl(stl_path)

            # Apply rotation if build direction is not Z
            if build_direction != 'Z':
                progress_with_cancel(6, f"[STAGE] Rotating mesh ({build_direction}→Z)...")

                mesh = mesh_info['mesh']

                if build_direction == 'Y':
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.radians(-90), [1, 0, 0]
                    )
                    progress_with_cancel(7, "[INFO] Applying Y→Z rotation (-90 deg around X)")
                elif build_direction == 'Y-':
                    # Y-down: rotate Y→Z then flip Z
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.radians(-90), [1, 0, 0]
                    )
                    flip_matrix = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]
                    ])
                    rotation_matrix = flip_matrix @ rotation_matrix
                    progress_with_cancel(7, "[INFO] Applying Y→Z rotation + Z flip (Y-down)")
                elif build_direction == 'X':
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.radians(90), [0, 1, 0]
                    )
                    progress_with_cancel(7, "[INFO] Applying X→Z rotation (90 deg around Y)")
                else:
                    rotation_matrix = np.eye(4)

                rotated_mesh = mesh.copy()
                rotated_mesh.apply_transform(rotation_matrix)

                bounds = rotated_mesh.bounds
                mesh_info = {
                    'vertices': np.array(rotated_mesh.vertices),
                    'faces': np.array(rotated_mesh.faces),
                    'bounds': bounds,
                    'dimensions': bounds[1] - bounds[0],
                    'n_triangles': len(rotated_mesh.faces),
                    'n_vertices': len(rotated_mesh.vertices),
                    'is_watertight': rotated_mesh.is_watertight,
                    'volume': rotated_mesh.volume if rotated_mesh.is_watertight else None,
                    'mesh': rotated_mesh,
                }
                progress_with_cancel(8, f"[INFO] Rotated dimensions: {mesh_info['dimensions'][0]:.1f} x {mesh_info['dimensions'][1]:.1f} x {mesh_info['dimensions'][2]:.1f} mm")

            progress_with_cancel(8, "[STAGE] Slicing geometry into layers...")
            slice_result = slice_stl(
                mesh_info=mesh_info,
                voxel_size=voxel_size,
                layer_thickness=layer_thickness,
                layer_grouping=layer_grouping,
                progress_callback=lambda p, s: progress_with_cancel(8 + p * 0.12, s)
            )
            masks = slice_result['masks']
            slice_info = {
                'n_layers': slice_result['n_layers'],
                'grid_shape': slice_result['grid_shape'],
            }

        n_layers = len(masks)
        progress_with_cancel(22, f"[INFO] {n_layers} layers sliced")

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

            G_layers = calculate_layer_averaged_G(G_layers_2d, masks)
            progress_with_cancel(42, "[INFO] Geometry G computed")
        else:
            progress_with_cancel(42, "[INFO] Using Area-Only mode (no geometry G)")

        progress_with_cancel(45, "[STAGE] Running energy accumulation analysis...")

        energy_results = run_energy_analysis(
            masks=masks,
            G_layers=G_layers,
            dissipation_factor=dissipation_factor,
            convection_factor=convection_factor,
            use_geometry_multiplier=use_geometry_multiplier,
            threshold_medium=threshold_medium,
            threshold_high=threshold_high,
            voxel_size=voxel_size,
            progress_callback=lambda p, s: progress_with_cancel(45 + p * 0.45, s)
        )

        progress_with_cancel(92, "[STAGE] Preparing results...")

        computation_time = time.time() - start_time

        results = {
            'n_layers': n_layers,
            'risk_scores': energy_results['risk_scores'],
            'risk_levels': energy_results['risk_levels'],
            'layer_areas': energy_results['layer_areas'],
            'contact_areas': energy_results['contact_areas'],
            'summary': energy_results['summary'],
            'params_used': {
                'voxel_size': voxel_size,
                'layer_thickness': layer_thickness,
                'layer_grouping': layer_grouping,
                'build_direction': build_direction,
                'effective_layer_thickness': layer_thickness * layer_grouping,
                'dissipation_factor': dissipation_factor,
                'convection_factor': convection_factor,
                'use_geometry_multiplier': use_geometry_multiplier,
                'sigma_mm': sigma_mm if use_geometry_multiplier else None,
                'G_max': G_max if use_geometry_multiplier else None,
                'threshold_medium': threshold_medium,
                'threshold_high': threshold_high,
                'mode': energy_results['params']['mode'],
            },
            'computation_time_seconds': computation_time,
            'masks': masks,
            'G_layers': G_layers if use_geometry_multiplier else None,
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

    Supports: energy (risk scores), risk (risk levels)

    Each layer is rendered as a surface with color based on its value.
    """
    session = session_registry.get_session(session_id)
    if not session or not session.results:
        return jsonify({'status': 'error', 'message': 'No results available'}), 404

    results = session.results
    masks = results.get('masks', {})
    params = results.get('params_used', {})

    if not masks:
        return jsonify({'status': 'error', 'message': 'No mask data available'}), 404

    voxel_size = params.get('voxel_size', 0.1)
    layer_thickness = params.get('effective_layer_thickness', params.get('layer_thickness', 0.04))

    # Get layer values based on data type
    if data_type == 'energy':
        layer_values = {int(k): float(v) for k, v in results.get('risk_scores', {}).items()}
        value_label = 'Risk Score'
        min_val = 0.0
        max_val = 1.0
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


def _generate_layer_surface(mask: np.ndarray, voxel_size: float, z: float) -> tuple:
    """Generate triangulated surface from a 2D mask.

    Uses marching squares-like approach to find contours and triangulate them.
    Returns (vertices, faces) for mesh3d rendering.
    """
    from scipy import ndimage

    # Find connected regions and their boundaries
    labeled, n_features = ndimage.label(mask)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for region_id in range(1, n_features + 1):
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
PRIMARY_COLOR = '#1E3765'
PRIMARY_DARK = '#152850'
PRIMARY_LIGHT = '#2a4a80'
ACCENT_COLOR = '#007FA3'
BG_PAGE = '#080c14'
BG_HEADER = '#0c1018'
BG_SIDEBAR = '#141c28'
BG_TABS = '#101620'
BG_CARD = '#1a2436'
BG_MAIN = '#1a1a1a'
BG_INPUT = '#0e1420'

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
            --border-color: rgba(30, 55, 101, 0.4);
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
            margin-bottom: 10px;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid rgba(45, 95, 142, 0.15);
        }}
        .section-header {{
            padding: 10px 14px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
            font-size: 0.85rem;
        }}
        .section-header:hover {{ background: rgba(45, 95, 142, 0.2); border-radius: 8px; }}
        .section-header .arrow {{ font-size: 0.75rem; color: var(--text-secondary); transition: transform 0.3s; }}
        .section-header.collapsed .arrow {{ transform: rotate(-90deg); }}
        .section-content {{
            padding: 0 14px 12px 14px;
            max-height: 800px;
            opacity: 1;
            transition: all 0.3s ease;
        }}
        .section-content.collapsed {{ max-height: 0; opacity: 0; padding: 0 14px; overflow: hidden; }}
        .param-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .param-label {{ font-size: 0.85rem; color: var(--text-secondary); }}
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
            gap: 8px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            cursor: pointer;
            padding: 4px 0;
        }}
        .radio-option:hover {{ color: var(--text-primary); }}
        .radio-option input {{ accent-color: var(--accent); width: 16px; height: 16px; }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            color: white;
            width: 100%;
            margin-top: 12px;
        }}
        .btn-primary:hover {{ background: linear-gradient(135deg, var(--primary-light), var(--accent)); }}
        .btn-primary:disabled {{ background: var(--border-color); cursor: not-allowed; }}
        .btn-primary.running {{
            background: linear-gradient(to right, var(--accent) var(--progress, 0%), var(--bg-card) var(--progress, 0%));
        }}
        .file-upload {{
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 12px;
        }}
        .file-upload:hover {{ border-color: var(--accent); background: rgba(91, 163, 217, 0.1); }}
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
        .tab-panel {{ display: none; height: 100%; flex: 1; flex-direction: column; }}
        .tab-panel.active {{ display: flex; }}
        .viz-container {{
            flex: 1;
            background: var(--bg-main);
            position: relative;
            min-height: 400px;
        }}
        .viz-container .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
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
        .floating-console-content .info {{ color: #74b9ff; }}
        .floating-console-content .warn {{ color: #ffeaa7; }}
        .floating-console-content .success {{ color: #55efc4; font-weight: 500; }}
        .floating-console-content .progress {{ color: #a29bfe; }}
        .floating-console-content .stage {{ color: #5BA3D9; font-weight: 600; }}

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
            border: 1px solid rgba(100, 150, 200, 0.2);
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
            <div class="sidebar-section">
                <div class="section-header" onclick="toggleSection(this)">
                    <span>STL Input</span>
                    <span class="arrow">&#9660;</span>
                </div>
                <div class="section-content">
                    <div class="file-upload" id="fileUpload" onclick="document.getElementById('fileInput').click()">
                        <div style="font-size: 1.5rem; margin-bottom: 8px;">📁</div>
                        <div id="fileUploadText">Click to upload STL file</div>
                    </div>
                    <input type="file" id="fileInput" accept=".stl" style="display: none;" onchange="handleFileUpload(this)">
                    <button class="btn btn-secondary" style="width: 100%; margin-top: 8px;" onclick="loadTestSTL()">Load Test STL</button>

                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color);">
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
                        <div class="param-row" style="margin-top: 8px;">
                            <span class="param-label">Layer Grouping</span>
                            <input type="range" id="layerGrouping" min="1" max="100" value="1" step="1" style="flex: 1; margin: 0 8px;" oninput="updateLayerGrouping()">
                            <input type="number" class="param-input" id="layerGroupingValue" value="1" min="1" max="500" step="1" style="width: 65px;" onchange="syncLayerGrouping()">
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-secondary); margin-top: 4px;">
                            <span id="layerGroupingDim">= 0.04 mm effective</span>
                            <span id="layerGroupingLayers"></span>
                        </div>
                        <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 4px;">
                            <button class="btn btn-secondary" style="padding: 2px 8px; font-size: 0.7rem;" onclick="autoLayerGrouping()">Auto (~200 layers)</button>
                        </div>
                        <div class="param-row" style="margin-top: 12px;">
                            <span class="param-label">Build Direction</span>
                            <select class="param-input" id="buildDirection" style="flex: 1;" onchange="onBuildDirectionChange()">
                                <option value="Z">Z-up (default)</option>
                                <option value="Y">Y-up (rotate Y→Z)</option>
                                <option value="Y-">Y-down (rotate Y→Z, flip)</option>
                                <option value="X">X-up (rotate X→Z)</option>
                            </select>
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
                    <div style="margin-bottom: 12px;">
                        <label class="radio-option">
                            <input type="radio" name="energyModel" value="area_only" checked onchange="updateModelVisibility()">
                            <span>Area-Only Mode (faster)</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="energyModel" value="geometry_multiplier" onchange="updateModelVisibility()">
                            <span>Geometry Multiplier Mode</span>
                        </label>
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

                    <div id="geometryParams" style="display: none; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">
                        <div class="param-row">
                            <span class="param-label">Sigma (σ)</span>
                            <input type="number" class="param-input" id="sigmaMM" value="1.0" step="0.1">
                            <span class="param-unit">mm</span>
                        </div>
                        <div class="param-row">
                            <span class="param-label">G Max</span>
                            <input type="number" class="param-input" id="gMax" value="2.0" step="0.1">
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
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px;">
                        <strong style="color: var(--success);">LOW:</strong> score &lt; medium<br>
                        <strong style="color: var(--warning);">MEDIUM:</strong> medium ≤ score &lt; high<br>
                        <strong style="color: var(--danger);">HIGH:</strong> score ≥ high
                    </div>
                </div>
            </div>

            <!-- STL Info -->
            <div class="info-card" id="stlInfo" style="display: none;">
                <h4>STL Information</h4>
                <div class="info-row"><span>File:</span><span id="infoFilename">-</span></div>
                <div class="info-row"><span>Triangles:</span><span id="infoTriangles">-</span></div>
                <div class="info-row"><span>Dimensions:</span><span id="infoDimensions">-</span></div>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="content">
            <nav class="tab-nav">
                <button class="tab-btn active" onclick="switchTab('preview', event)">STL Preview</button>
                <button class="tab-btn" onclick="switchTab('energy', event)">Energy Accumulation</button>
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

                <!-- Energy Accumulation Tab -->
                <div class="tab-panel" id="tab-energy">
                    <div class="viz-container" id="energyPlot">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">
                            Run analysis to see energy accumulation
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
        </aside>
    </div>

    <script>
        // Global state
        let currentSessionId = null;
        let analysisResults = null;
        let stlLoaded = false;
        let currentEventSource = null;  // Track EventSource for cleanup

        // Toggle section expand/collapse
        function toggleSection(header) {{
            header.classList.toggle('collapsed');
            const content = header.nextElementSibling;
            content.classList.toggle('collapsed');
        }}

        // Update geometry params visibility
        function updateModelVisibility() {{
            const isGeometry = document.querySelector('input[name="energyModel"]:checked').value === 'geometry_multiplier';
            document.getElementById('geometryParams').style.display = isGeometry ? 'block' : 'none';
        }}

        // Switch tabs
        function switchTab(tabName, evt) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            if (evt && evt.target) {{
                evt.target.classList.add('active');
            }}
            document.getElementById('tab-' + tabName).classList.add('active');
        }}

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

        // Load test STL
        async function loadTestSTL() {{
            logConsole('Loading test STL...', 'info');

            try {{
                const response = await fetch('/api/load_test_stl', {{ method: 'POST' }});
                const data = await response.json();

                if (data.status === 'success') {{
                    stlLoaded = true;
                    stlDimensions = data.info.dimensions;
                    document.getElementById('runBtn').disabled = false;
                    document.getElementById('fileUpload').classList.add('loaded');
                    document.getElementById('fileUploadText').textContent = data.filename || 'Test STL';

                    document.getElementById('stlInfo').style.display = 'block';
                    document.getElementById('infoFilename').textContent = data.filename || 'Test STL';
                    document.getElementById('infoTriangles').textContent = data.info.n_triangles.toLocaleString();
                    const dims = data.info.dimensions;
                    document.getElementById('infoDimensions').textContent = dims[0].toFixed(1) + ' × ' + dims[1].toFixed(1) + ' × ' + dims[2].toFixed(1) + ' mm';

                    updateLayerGroupingDisplay();
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
                    stlLoaded = true;
                    stlDimensions = data.info.dimensions;
                    document.getElementById('runBtn').disabled = false;
                    document.getElementById('fileUpload').classList.add('loaded');
                    document.getElementById('fileUploadText').textContent = file.name;

                    document.getElementById('stlInfo').style.display = 'block';
                    document.getElementById('infoFilename').textContent = data.filename;
                    document.getElementById('infoTriangles').textContent = data.info.n_triangles.toLocaleString();
                    const dims = data.info.dimensions;
                    document.getElementById('infoDimensions').textContent = dims[0].toFixed(1) + ' × ' + dims[1].toFixed(1) + ' × ' + dims[2].toFixed(1) + ' mm';

                    updateLayerGroupingDisplay();
                    logConsole('STL loaded: ' + data.info.n_triangles + ' triangles', 'success');

                    // Render 3D STL preview
                    renderSTLPreview();
                }} else {{
                    logConsole('Error: ' + data.message, 'error');
                }}
            }} catch (err) {{
                logConsole('Upload failed: ' + err.message, 'error');
            }}
        }}

        // Run analysis
        async function runAnalysis() {{
            if (!stlLoaded) {{
                logConsole('Please upload an STL file first', 'warning');
                return;
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
                threshold_high: parseFloat(document.getElementById('thresholdHigh').value)
            }};

            const runBtn = document.getElementById('runBtn');
            runBtn.disabled = true;
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
                    runBtn.disabled = false;
                }}
            }} catch (err) {{
                logConsole('Analysis failed: ' + err.message, 'error');
                runBtn.classList.remove('running');
                runBtn.innerHTML = '&#9654; Run Analysis';
                runBtn.disabled = false;
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
                    runBtn.classList.remove('running');
                    runBtn.innerHTML = '&#9654; Run Analysis';
                    runBtn.disabled = false;
                    logConsole('Analysis complete!', 'success');
                    loadResults(sessionId);
                }} else if (data.status === 'error') {{
                    evtSource.close();
                    currentEventSource = null;
                    runBtn.classList.remove('running');
                    runBtn.innerHTML = '&#9654; Run Analysis';
                    runBtn.disabled = false;
                    logConsole('Error: ' + (data.error || 'Unknown error'), 'error');
                }}
            }};

            evtSource.onerror = function() {{
                evtSource.close();
                currentEventSource = null;
                const runBtn = document.getElementById('runBtn');
                runBtn.classList.remove('running');
                runBtn.innerHTML = '&#9654; Run Analysis';
                runBtn.disabled = false;
                logConsole('Connection lost', 'error');
            }};
        }}

        // Load results
        async function loadResults(sessionId) {{
            try {{
                const response = await fetch('/api/results/' + sessionId);
                const data = await response.json();

                if (data.status === 'success') {{
                    analysisResults = data.results;
                    updateVisualizations();
                    document.getElementById('runBtn').disabled = false;
                }}
            }} catch (err) {{
                logConsole('Failed to load results: ' + err.message, 'error');
            }}
        }}

        // Update all visualizations
        function updateVisualizations() {{
            if (!analysisResults) return;

            const nLayers = analysisResults.n_layers;
            document.getElementById('layerSlider').max = nLayers;
            document.getElementById('layerVal').textContent = '1 / ' + nLayers;

            // Render 3D layer surfaces for Energy and Risk tabs
            renderLayerSurfaces('energy');
            renderLayerSurfaces('risk');

            document.getElementById('riskLegend').style.display = 'block';

            // Summary
            updateSummary();
        }}

        // Unified layer surface rendering for 3D visualization
        async function renderLayerSurfaces(dataType) {{
            if (!currentSessionId) return;

            const plotConfig = {{
                'energy': {{ plotId: 'energyPlot', title: 'Energy Accumulation (3D)' }},
                'risk': {{ plotId: 'riskPlot', title: 'Risk Classification (3D)' }}
            }};

            const config = plotConfig[dataType];
            if (!config) {{
                logConsole('Unknown data type: ' + dataType, 'error');
                return;
            }}

            logConsole('Loading ' + config.title + ' surfaces...', 'info');

            try {{
                const response = await fetch('/api/layer_surfaces/' + currentSessionId + '/' + dataType);
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
                const minVal = data.min_val;
                const maxVal = data.max_val;
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

                    const normalizedValue = (value - minVal) / valueRange;

                    // Get color based on data type
                    let color;
                    if (dataType === 'risk') {{
                        // Risk uses categorical colors: LOW=green, MEDIUM=yellow, HIGH=red
                        if (value >= 2) color = '#f87171';  // HIGH - red
                        else if (value >= 1) color = '#fbbf24';  // MEDIUM - yellow
                        else color = '#4ade80';  // LOW - green
                    }} else {{
                        // Energy: RdYlGn reversed (green=low, red=high)
                        if (normalizedValue < 0.3) {{
                            // Green range
                            const t = normalizedValue / 0.3;
                            color = `rgb(${{Math.round(74 + t * 107)}}, ${{Math.round(222 - t * 33)}}, ${{Math.round(128 - t * 92)}})`;
                        }} else if (normalizedValue < 0.6) {{
                            // Yellow range
                            const t = (normalizedValue - 0.3) / 0.3;
                            color = `rgb(${{Math.round(181 + t * 70)}}, ${{Math.round(189 - t * 0)}}, ${{Math.round(36 - t * 0)}})`;
                        }} else {{
                            // Red range
                            const t = (normalizedValue - 0.6) / 0.4;
                            color = `rgb(${{Math.round(251 - t * 3)}}, ${{Math.round(189 - t * 76)}}, ${{Math.round(36 + t * 77)}})`;
                        }}
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
                        opacity: 0.9,
                        flatshading: true,
                        hovertemplate: 'Layer ' + layer + '<br>Z: ' + z.toFixed(2) + ' mm<br>' + data.value_label + ': ' + (typeof value === 'number' ? value.toFixed(3) : value) + '<extra></extra>',
                        showscale: false
                    }});
                }});

                if (traces.length === 0) {{
                    logConsole('No valid layer geometry to display', 'warning');
                    return;
                }}

                // Add colorbar via invisible scatter3d trace
                let colorscale;
                if (dataType === 'risk') {{
                    colorscale = [[0, '#4ade80'], [0.5, '#fbbf24'], [1.0, '#f87171']];
                }} else {{
                    colorscale = [[0, '#4ade80'], [0.3, '#b5bd24'], [0.6, '#fbbd24'], [1.0, '#f87171']];
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
                            title: {{ text: data.value_label, font: {{ color: '#e0e0e0', size: 12 }} }},
                            tickfont: {{ color: '#c0c0c0', size: 11 }},
                            x: 1.02,
                            xanchor: 'left',
                            y: 0.5,
                            yanchor: 'middle',
                            len: 0.6,
                            thickness: 15,
                            bgcolor: 'rgba(40,40,40,0.9)',
                            bordercolor: '#666',
                            borderwidth: 1
                        }},
                        showscale: true
                    }},
                    hoverinfo: 'skip',
                    showlegend: false
                }};
                traces.push(colorbarTrace);

                // Calculate scene bounds with padding
                const padding = 0.1;
                const xRange = [Math.min(...allX), Math.max(...allX)];
                const yRange = [Math.min(...allY), Math.max(...allY)];
                const zRange = [Math.min(...allZ), Math.max(...allZ)];
                const xPad = (xRange[1] - xRange[0]) * padding;
                const yPad = (yRange[1] - yRange[0]) * padding;
                const zPad = (zRange[1] - zRange[0]) * padding;

                const layout = {{
                    title: {{ text: config.title + ' (' + data.n_valid_layers + ' layers)', font: {{ color: '#fff', size: 14 }} }},
                    paper_bgcolor: '{BG_MAIN}',
                    plot_bgcolor: '{BG_MAIN}',
                    scene: {{
                        xaxis: {{ title: 'X (mm)', color: '#aaa', gridcolor: '#333', range: [xRange[0] - xPad, xRange[1] + xPad] }},
                        yaxis: {{ title: 'Y (mm)', color: '#aaa', gridcolor: '#333', range: [yRange[0] - yPad, yRange[1] + yPad] }},
                        zaxis: {{ title: 'Z (mm)', color: '#aaa', gridcolor: '#333', range: [zRange[0] - zPad, zRange[1] + zPad] }},
                        bgcolor: '{BG_MAIN}',
                        aspectmode: 'data'
                    }},
                    margin: {{ l: 0, r: 80, t: 40, b: 0 }}
                }};

                Plotly.newPlot(config.plotId, traces, layout, {{ responsive: true, displayModeBar: true }});
                logConsole(config.title + ' loaded: ' + data.n_valid_layers + ' layers', 'success');

            }} catch (e) {{
                console.error(e);
                logConsole('Error loading ' + config.title + ': ' + e.message, 'error');
            }}
        }}

        // Apply build direction rotation to coordinates (for preview)
        function applyBuildDirectionRotation(x, y, z, buildDir) {{
            if (buildDir === 'Y') {{
                // Rotate Y to Z: swap Y and Z, negate new Y
                return {{ x: x, y: z.map(v => -v), z: y }};
            }} else if (buildDir === 'Y-') {{
                // Rotate Y to Z then flip: swap Y and Z, negate both
                return {{ x: x, y: z.map(v => -v), z: y.map(v => -v) }};
            }} else if (buildDir === 'X') {{
                // Rotate X to Z: swap X and Z, negate new X
                return {{ x: z.map(v => -v), y: y, z: x }};
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
                    color: '#5BA3D9',
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
                    title: {{ text: 'STL Preview (' + data.n_triangles + ' triangles)' + dirLabel, font: {{ color: '#fff', size: 14 }} }},
                    paper_bgcolor: '{BG_MAIN}',
                    plot_bgcolor: '{BG_MAIN}',
                    scene: {{
                        xaxis: {{ title: 'X (mm)', color: '#aaa', gridcolor: '#333', range: [xRange[0] - xPad, xRange[1] + xPad] }},
                        yaxis: {{ title: 'Y (mm)', color: '#aaa', gridcolor: '#333', range: [yRange[0] - yPad, yRange[1] + yPad] }},
                        zaxis: {{ title: 'Z (mm)', color: '#aaa', gridcolor: '#333', range: [zRange[0] - zPad, zRange[1] + zPad] }},
                        bgcolor: '{BG_MAIN}',
                        aspectmode: 'data'
                    }},
                    margin: {{ l: 0, r: 20, t: 40, b: 0 }}
                }};

                Plotly.newPlot('previewPlot', [trace], layout, {{ responsive: true, displayModeBar: true }});
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

            document.getElementById('summaryContent').innerHTML = `
                <div class="summary-card">
                    <h4>Analysis Results</h4>
                    <div class="summary-stat"><span class="label">Total Layers</span><span class="value">${{s.n_layers}}</span></div>
                    <div class="summary-stat"><span class="label">Max Risk Score</span><span class="value">${{s.max_risk_score.toFixed(3)}}</span></div>
                    <div class="summary-stat"><span class="label">Max Risk Layer</span><span class="value">${{s.max_risk_layer}}</span></div>
                    <div class="summary-stat"><span class="label">Mean Risk Score</span><span class="value">${{s.mean_risk_score.toFixed(3)}}</span></div>
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

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            logConsole('Ready - Load an STL file or use "Load Test STL"', 'info');
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

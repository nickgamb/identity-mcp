#!/usr/bin/env python3
"""
================================================================================
EEG BRAINWAVE ENROLLMENT
================================================================================

PURPOSE:
    Guides a user through a series of neurofeedback tasks while capturing raw EEG
    from an EMOTIV Epoc X headset. The captured data is processed into a reference
    identity model (centroid-based, same approach as train_identity_model.py).

ENROLLMENT TASKS:
    1. Baseline       - Resting state, eyes open, relaxed focus (20s)
    2. Meditation      - Guided breathing exercise (20s)
    3. Word Focus      - Focus on displayed words one at a time (5 words x 10s)
    4. Actions         - Motor/expression tasks: blink, finger movement,
                         smile, frown, neutral face (6 actions x 10s)

WHAT IT CREATES:
    models/eeg_identity/
    ├── config.json           # Model configuration and statistics
    ├── eeg_centroid.npy      # Identity centroid (mean feature vector)
    ├── spectral_profile.json # Per-channel frequency band power profile
    ├── feature_scaler.json   # Feature normalization parameters
    └── enrollment_log.json   # Enrollment metadata and task log

DEPENDENCIES:
    pip install numpy scipy scikit-learn tqdm
    pip install hidapi pycryptodome   (for hardware)

USAGE:
    python enroll_brainwaves.py --synthetic                       # Test without hardware
    python enroll_brainwaves.py --serial EX0000B29AB0205E         # With EMOTIV dongle
    python enroll_brainwaves.py --synthetic --task-duration 10    # Shorter tasks

CONNECTION:
    Requires the EMOTIV USB wireless dongle (VID 0x1234, PID 0xED02).
    Direct USB cable is charge-only. Bluetooth LE is not supported.

================================================================================
"""

import os
import sys
import io
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows (cp1252 can't handle box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Dashboard event prefix — the backend SSE bridge looks for this marker
_DASHBOARD_MODE = False

def emit_event(event: str, data: Optional[Dict] = None):
    """Emit a structured event for the dashboard SSE bridge."""
    if not _DASHBOARD_MODE:
        return
    payload = {"event": event}
    if data:
        payload.update(data)
    # Flush immediately so the backend SSE polling picks it up
    print(f"@@EEG_EVENT:{json.dumps(payload)}", flush=True)

# Check dependencies
try:
    from scipy import signal as scipy_signal
    from scipy.stats import skew, kurtosis
    from scipy.signal import welch
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)

# Paths (following project conventions)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Support multi-user: read USER_ID from environment
USER_ID = os.environ.get("USER_ID")
def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    """Get user-specific directory if user_id is provided, otherwise base directory."""
    if user_id:
        return base_dir / user_id
    return base_dir

MODELS_DIR = get_user_dir(PROJECT_ROOT / "models" / "eeg_identity", USER_ID)

# Import the EMOTIV reader from the same directory
sys.path.insert(0, str(SCRIPT_DIR))
from emotiv_reader import (
    EmotivReader, EEG_CHANNELS, NUM_CHANNELS, SAMPLE_RATE, FREQ_BANDS,
    preprocess_eeg,
)


def check_dependencies():
    """Verify all required dependencies are installed."""
    if not HAS_DEPS:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print(f"\nError: {MISSING_DEP}")
        print("\nPlease install required packages:")
        print("  pip install numpy scipy scikit-learn tqdm")
        print("\nFor EMOTIV hardware support:")
        print("  pip install hidapi pycryptodome")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment task definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_enrollment_tasks(task_duration: int = 20, word_duration: int = 10, action_duration: int = 10) -> List[Dict]:
    """
    Define the enrollment task sequence.
    
    Based on well-established neurofeedback techniques, each task captures
    distinct neural signatures that together form a unique identity fingerprint.
    """
    tasks = []
    
    # Task 1: Baseline (resting state)
    tasks.append({
        "name": "baseline",
        "label": "baseline",
        "instruction": "Sit comfortably in a quiet environment. Keep your eyes open and "
                       "focus on a fixed point in front of you. Try to relax and minimize "
                       "movement. Breathe naturally.",
        "duration": task_duration,
        "category": "resting",
    })
    
    # Task 2: Meditation (guided breathing)
    tasks.append({
        "name": "meditation",
        "label": "meditation",
        "instruction": "Close your eyes gently. Breathe in slowly for 4 seconds, hold for "
                       "4 seconds, breathe out for 4 seconds. Focus only on your breath. "
                       "Let thoughts pass without engaging them.",
        "duration": task_duration,
        "category": "relaxation",
    })
    
    # Task 3: Word Focus (cognitive engagement)
    words = ["TREE", "BOOK", "RIVER", "CLOUD", "CHAIR"]
    for word in words:
        tasks.append({
            "name": f"word_{word.lower()}",
            "label": f"word_{word.lower()}",
            "instruction": f"Focus on the word: >>> {word} <<<\n"
                           f"Think about what this word means to you. Visualize it. "
                           f"Let associations form naturally.",
            "duration": word_duration,
            "category": "cognitive",
        })
    
    # Task 4: Actions (motor/expression signatures)
    actions = [
        ("blink", "Blink your eyes naturally, about once every 2 seconds."),
        ("fingers_right", "Gently tap your RIGHT hand fingers on the desk, one at a time."),
        ("fingers_left", "Gently tap your LEFT hand fingers on the desk, one at a time."),
        ("face_happy", "Smile genuinely. Think of something that makes you happy."),
        ("face_sad", "Let your face relax into a slight frown. Think of something somber."),
        ("face_neutral", "Keep a completely neutral expression. Relax all facial muscles."),
    ]
    
    for label, instruction in actions:
        tasks.append({
            "name": label,
            "label": label,
            "instruction": instruction,
            "duration": action_duration,
            "category": "motor" if "finger" in label else "expression",
        })
    
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (mirrors the approach in enroll_brainwaves from pieeg)
# ─────────────────────────────────────────────────────────────────────────────

def extract_eeg_features(eeg_data: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a comprehensive feature vector from an EEG segment.
    
    Features extracted per channel:
        - Time-domain: mean, variance, skewness, kurtosis (4 features)
        - Frequency-domain: relative band powers for delta, theta, alpha, beta, gamma (5 features)
    
    Cross-channel features:
        - Connectivity: upper triangle of correlation matrix (91 features for 14 channels)
        - Frontal alpha asymmetry (1 feature)
    
    Total: 14 * (4 + 5) + 91 + 1 = 218 features
    
    Args:
        eeg_data: Preprocessed EEG of shape (channels, samples)
        fs: Sampling rate
    
    Returns:
        1D feature vector
    """
    if eeg_data.ndim != 2 or eeg_data.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Expected shape ({NUM_CHANNELS}, samples), got {eeg_data.shape}")
    
    n_channels, n_samples = eeg_data.shape
    features = []
    
    # ── Time-domain features (per channel) ──
    features.append(np.mean(eeg_data, axis=1))         # 14 values
    features.append(np.var(eeg_data, axis=1))           # 14 values
    features.append(skew(eeg_data, axis=1))             # 14 values
    features.append(kurtosis(eeg_data, axis=1))         # 14 values
    
    # ── Frequency-domain features (per channel) ──
    # Compute PSD using Welch's method
    nperseg = min(fs * 2, n_samples)  # 2-second windows or shorter
    freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, axis=1)
    
    # Total power per channel (for relative power)
    total_power = np.sum(psd, axis=1, keepdims=True)
    total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero
    
    # Relative band powers
    for band_name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[:, mask], axis=1) / total_power.flatten()
        features.append(band_power)                     # 14 values each
    
    # ── Connectivity features ──
    # Correlation matrix upper triangle (unique pairs only)
    corr = np.corrcoef(eeg_data)
    # Handle NaN in correlation (can happen with flat signals)
    corr = np.nan_to_num(corr, nan=0.0)
    upper_tri = corr[np.triu_indices(n_channels, k=1)]
    features.append(upper_tri)                          # 91 values
    
    # ── Asymmetry features ──
    # Frontal alpha asymmetry (F4 - F3 alpha power, log ratio)
    f3_idx = EEG_CHANNELS.index("F3")
    f4_idx = EEG_CHANNELS.index("F4")
    alpha_mask = (freqs >= FREQ_BANDS["alpha"][0]) & (freqs <= FREQ_BANDS["alpha"][1])
    f3_alpha = np.sum(psd[f3_idx, alpha_mask])
    f4_alpha = np.sum(psd[f4_idx, alpha_mask])
    asymmetry = np.log(f4_alpha + 1e-10) - np.log(f3_alpha + 1e-10)
    features.append(np.array([asymmetry]))              # 1 value
    
    return np.concatenate(features)


def extract_spectral_profile(eeg_data: np.ndarray, fs: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Compute detailed spectral profile for each channel.
    
    Returns a dict with per-channel band power distributions,
    useful for profile inspection and visualization.
    """
    nperseg = min(fs * 2, eeg_data.shape[1])
    freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, axis=1)
    
    total_power = np.sum(psd, axis=1)  # shape (n_channels,) — scalar per channel
    total_power = np.maximum(total_power, 1e-10)
    
    profile = {}
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        channel_profile = {}
        for band_name, (low, high) in FREQ_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            abs_power = float(np.sum(psd[ch_idx, mask]))
            rel_power = float(np.sum(psd[ch_idx, mask]) / total_power[ch_idx])
            channel_profile[band_name] = {
                "absolute_power": abs_power,
                "relative_power": rel_power,
            }
        profile[ch_name] = channel_profile
    
    return profile


def check_eeg_quality(eeg_data: np.ndarray, min_variance: float = 0.01, max_amplitude: float = 500.0) -> Tuple[bool, str]:
    """
    Check if EEG data is of sufficient quality.
    
    Returns:
        (is_good, message)
    """
    if eeg_data.size == 0:
        return False, "No data collected"
    
    # Check amplitude range
    max_val = np.max(np.abs(eeg_data))
    if max_val > max_amplitude:
        return False, f"Amplitude too high ({max_val:.1f} uV > {max_amplitude} uV limit)"
    
    # Check variance (flat signal detection)
    variances = np.var(eeg_data, axis=1)
    if np.any(variances < min_variance):
        flat_channels = [EEG_CHANNELS[i] for i in range(NUM_CHANNELS) if variances[i] < min_variance]
        return False, f"Flat signal detected on channels: {', '.join(flat_channels)}"
    
    # Check for NaN/Inf
    if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
        return False, "Data contains NaN or Inf values"
    
    return True, "Quality check passed"


# ─────────────────────────────────────────────────────────────────────────────
# Model training (centroid-based, following train_identity_model.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eeg_centroid(feature_vectors: np.ndarray) -> np.ndarray:
    """Compute the identity centroid (mean of all feature vectors)."""
    return np.mean(feature_vectors, axis=0)


def compute_identity_statistics(feature_vectors: np.ndarray, centroid: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics about the feature vector distribution.
    
    Mirrors compute_identity_statistics() from train_identity_model.py.
    """
    # Distances from centroid
    distances = np.linalg.norm(feature_vectors - centroid, axis=1)
    
    # Cosine similarities to centroid
    norms = np.linalg.norm(feature_vectors, axis=1) * np.linalg.norm(centroid)
    similarities = np.dot(feature_vectors, centroid) / (norms + 1e-8)
    
    return {
        "num_samples": len(feature_vectors),
        "feature_dim": len(centroid),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "threshold_1std": float(np.mean(distances) + np.std(distances)),
        "threshold_2std": float(np.mean(distances) + 2 * np.std(distances)),
        "similarity_threshold_1std": float(np.mean(similarities) - np.std(similarities)),
        "similarity_threshold_2std": float(np.mean(similarities) - 2 * np.std(similarities)),
        "percentiles": {
            "p25": float(np.percentile(similarities, 25)),
            "p50": float(np.percentile(similarities, 50)),
            "p75": float(np.percentile(similarities, 75)),
            "p90": float(np.percentile(similarities, 90)),
            "p95": float(np.percentile(similarities, 95)),
        },
    }


def save_eeg_identity_model(
    centroid: np.ndarray,
    scaler_params: Dict[str, Any],
    spectral_profile: Dict[str, Any],
    statistics: Dict[str, Any],
    config: Dict[str, Any],
    enrollment_log: List[Dict],
    output_dir: Path,
):
    """
    Save the trained EEG identity model.
    
    Follows save_identity_model() pattern from train_identity_model.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save centroid
    np.save(output_dir / "eeg_centroid.npy", centroid)
    
    # Save scaler parameters (for normalizing future feature vectors)
    with open(output_dir / "feature_scaler.json", 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    # Save spectral profile
    with open(output_dir / "spectral_profile.json", 'w') as f:
        json.dump(spectral_profile, f, indent=2)
    
    # Save config with statistics
    config_data = {
        **config,
        "statistics": statistics,
        "created_at": datetime.now().isoformat(),
        "centroid_shape": list(centroid.shape),
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Save enrollment log
    with open(output_dir / "enrollment_log.json", 'w') as f:
        json.dump(enrollment_log, f, indent=2, default=str)
    
    print(f"\n   Model saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment flow
# ─────────────────────────────────────────────────────────────────────────────

def _visual_type_for_task(task: Dict) -> str:
    """Map a task to the visual element the dashboard should show."""
    name = task.get("name", "")
    if name == "baseline":
        return "checkerboard"
    elif name == "meditation":
        return "breathing_circle"
    elif name.startswith("word_"):
        return "word"
    return "instruction"


def _word_from_task(task: Dict) -> Optional[str]:
    """Extract the display word from a word-focus task."""
    name = task.get("name", "")
    if name.startswith("word_"):
        return name.replace("word_", "").upper()
    return None


def run_enrollment_task(
    reader: EmotivReader,
    task: Dict,
    task_num: int,
    total_tasks: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Execute a single enrollment task.
    
    Returns:
        (raw_data, feature_vector, task_log_entry) or (None, None, log) on failure
    """
    label = task["label"]
    duration = task["duration"]
    
    # Dashboard event — tells the modal which visual to show
    emit_event("task_start", {
        "task": task["name"],
        "task_num": task_num,
        "total_tasks": total_tasks,
        "visual": _visual_type_for_task(task),
        "word": _word_from_task(task),
        "instruction": task["instruction"],
        "duration": duration,
        "category": task.get("category", ""),
    })
    
    print(f"\n{'─' * 60}")
    print(f"   Task {task_num}/{total_tasks}: {task['name'].upper()}")
    print(f"{'─' * 60}")
    print(f"\n   {task['instruction']}")
    print(f"\n   Duration: {duration} seconds")
    print(f"   Recording starts in 3 seconds...")
    time.sleep(1)
    print(f"   2...")
    time.sleep(1)
    print(f"   1...")
    time.sleep(1)
    print(f"   >>> RECORDING <<<")
    
    # Record EEG
    start_time = time.time()
    
    def progress_callback(elapsed, total):
        bar_width = 30
        filled = int(bar_width * elapsed / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r   [{bar}] {elapsed:.0f}/{total:.0f}s", end="", flush=True)
        emit_event("recording_progress", {
            "task": task["name"],
            "elapsed": round(elapsed, 1),
            "total": round(total, 1),
        })
    
    raw_data = reader.read_seconds(duration, task_label=label, callback=progress_callback)
    recording_duration = time.time() - start_time
    
    print(f"\n   Recording complete ({raw_data.shape[1]} samples)")
    
    # Preprocess
    print(f"   Preprocessing...", end=" ")
    filtered_data = preprocess_eeg(raw_data)
    
    # Quality check
    is_good, quality_msg = check_eeg_quality(filtered_data)
    if is_good:
        print(f"Quality: PASS")
    else:
        print(f"Quality: WARNING - {quality_msg}")
    
    # Extract features
    print(f"   Extracting features...", end=" ")
    try:
        features = extract_eeg_features(filtered_data)
        print(f"OK ({len(features)} dimensions)")
    except Exception as e:
        print(f"FAILED: {e}")
        features = None
    
    # Build task log entry
    task_log = {
        "task": task["name"],
        "label": label,
        "category": task["category"],
        "duration_requested": duration,
        "duration_actual": recording_duration,
        "samples_collected": raw_data.shape[1],
        "quality_passed": is_good,
        "quality_message": quality_msg,
        "feature_dim": len(features) if features is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Dashboard event — task finished
    emit_event("task_complete", {
        "task": task["name"],
        "task_num": task_num,
        "total_tasks": total_tasks,
        "quality_passed": is_good,
        "quality_message": quality_msg,
        "feature_dim": len(features) if features is not None else 0,
    })
    
    return filtered_data, features, task_log


def main():
    parser = argparse.ArgumentParser(
        description="EEG Brainwave Enrollment - Capture and train identity model"
    )
    parser.add_argument(
        "--serial", type=str, default=None,
        help="EMOTIV device serial number (required for HID mode)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no hardware needed)"
    )
    parser.add_argument(
        "--device-type", choices=["consumer", "research"], default="consumer",
        help="EMOTIV device edition (default: consumer)"
    )
    parser.add_argument(
        "--task-duration", type=int, default=20,
        help="Duration per major task in seconds (default: 20)"
    )
    parser.add_argument(
        "--word-duration", type=int, default=10,
        help="Duration per word focus task in seconds (default: 10)"
    )
    parser.add_argument(
        "--action-duration", type=int, default=10,
        help="Duration per action task in seconds (default: 10)"
    )
    parser.add_argument(
        "--synthetic-seed", type=int, default=42,
        help="Random seed for synthetic identity (synthetic mode only)"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Emit structured @@EEG_EVENT lines for dashboard SSE streaming"
    )
    
    args = parser.parse_args()
    
    # Enable dashboard event mode
    global _DASHBOARD_MODE
    _DASHBOARD_MODE = args.dashboard
    
    # Check dependencies
    check_dependencies()
    
    mode = "synthetic" if args.synthetic else "hid"
    
    print("=" * 60)
    print("EEG BRAINWAVE ENROLLMENT")
    print("=" * 60)
    print(f"\n   Mode: {mode}")
    print(f"   Channels: {NUM_CHANNELS} ({', '.join(EEG_CHANNELS)})")
    print(f"   Sample rate: {SAMPLE_RATE} Hz")
    print(f"   Output: {MODELS_DIR}")
    
    if args.synthetic:
        print(f"   Synthetic seed: {args.synthetic_seed}")
    else:
        print(f"   Device: EPOC X via USB dongle (auto-detect)")
    
    # ── Connect to device ──
    print(f"\n   Connecting to device...")
    reader = EmotivReader(
        mode=mode,
        serial_number=args.serial,
        is_research=(args.device_type == "research"),
        synthetic_seed=args.synthetic_seed,
    )
    
    if not reader.connect():
        print("\n   Failed to connect to EMOTIV device.")
        print("   Ensure the USB wireless dongle is plugged in and the headset is powered on.")
        sys.exit(1)
    
    print(f"   Connected!")
    
    # ── Get enrollment tasks ──
    tasks = get_enrollment_tasks(
        task_duration=args.task_duration,
        word_duration=args.word_duration,
        action_duration=args.action_duration,
    )
    
    total_duration = sum(t["duration"] for t in tasks)
    print(f"\n   Enrollment will include {len(tasks)} tasks (~{total_duration}s total)")
    print(f"\n   Please follow each instruction carefully.")
    print(f"   Minimize movement and stay relaxed between tasks.")
    
    # Dashboard event — enrollment starting
    emit_event("enrollment_start", {
        "num_tasks": len(tasks),
        "total_duration": total_duration,
        "tasks": [{"name": t["name"], "category": t["category"], "duration": t["duration"],
                    "visual": _visual_type_for_task(t), "word": _word_from_task(t)} for t in tasks],
        "channels": EEG_CHANNELS,
        "sample_rate": SAMPLE_RATE,
    })
    
    # Countdown before starting
    print(f"\n   Starting enrollment in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    # ── Run enrollment tasks ──
    all_features = []
    all_spectral_profiles = []
    enrollment_log = []
    
    for i, task in enumerate(tasks, 1):
        task_data, features, task_log = run_enrollment_task(
            reader, task, i, len(tasks)
        )
        
        enrollment_log.append(task_log)
        
        if features is not None:
            all_features.append(features)
            
            # Compute spectral profile for this task
            spectral = extract_spectral_profile(task_data)
            all_spectral_profiles.append({
                "task": task["name"],
                "label": task["label"],
                "profile": spectral,
            })
        
        # Brief pause between tasks
        if i < len(tasks):
            print(f"\n   Rest for 3 seconds before next task...")
            time.sleep(3)
    
    # ── Disconnect device ──
    reader.disconnect()
    
    # ── Process results ──
    print(f"\n\n{'=' * 60}")
    print("PROCESSING ENROLLMENT DATA")
    print("=" * 60)
    
    if not all_features:
        print("\n   No valid features extracted. Enrollment failed.")
        sys.exit(1)
    
    feature_matrix = np.array(all_features)
    print(f"\n   Feature matrix shape: {feature_matrix.shape}")
    print(f"   Valid tasks: {len(all_features)}/{len(tasks)}")
    
    # ── Normalize features (two-step: StandardScaler then L2) ──
    print(f"\n   Normalizing features...")
    
    # Step 1: StandardScaler — equalize feature scales so all dimensions
    # contribute equally (e.g. variance features ~0-1000 vs asymmetry ~0-1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }
    
    # Step 2: L2-normalize — project to unit sphere for cosine similarity
    norms = np.linalg.norm(scaled_features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    normalized_features = scaled_features / norms
    
    # ── Compute centroid ──
    print(f"\n   Computing identity centroid...")
    centroid = compute_eeg_centroid(normalized_features)
    # Re-normalize centroid to unit length (mean of unit vectors is not unit)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    print(f"   Centroid shape: {centroid.shape}")
    
    # ── Compute statistics ──
    print(f"\n   Computing identity statistics...")
    statistics = compute_identity_statistics(normalized_features, centroid)
    print(f"   Mean similarity to centroid: {statistics['mean_similarity']:.4f}")
    print(f"   Std similarity: {statistics['std_similarity']:.4f}")
    print(f"   Verification threshold (1 sigma): {statistics['similarity_threshold_1std']:.4f}")
    print(f"   Verification threshold (2 sigma): {statistics['similarity_threshold_2std']:.4f}")
    
    # ── Aggregate spectral profile ──
    print(f"\n   Computing aggregate spectral profile...")
    # Average spectral profiles across all tasks
    agg_spectral = {}
    for ch_name in EEG_CHANNELS:
        ch_profiles = []
        for sp in all_spectral_profiles:
            if ch_name in sp["profile"]:
                ch_profiles.append(sp["profile"][ch_name])
        
        if ch_profiles:
            agg_spectral[ch_name] = {}
            for band_name in FREQ_BANDS:
                abs_powers = [p[band_name]["absolute_power"] for p in ch_profiles]
                rel_powers = [p[band_name]["relative_power"] for p in ch_profiles]
                agg_spectral[ch_name][band_name] = {
                    "mean_absolute_power": float(np.mean(abs_powers)),
                    "std_absolute_power": float(np.std(abs_powers)),
                    "mean_relative_power": float(np.mean(rel_powers)),
                    "std_relative_power": float(np.std(rel_powers)),
                }
    
    spectral_output = {
        "aggregate": agg_spectral,
        "per_task": all_spectral_profiles,
    }
    
    # ── Save model ──
    print(f"\n   Saving EEG identity model...")
    config = {
        "mode": mode,
        "device_type": args.device_type,
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "channels": EEG_CHANNELS,
        "num_tasks": len(tasks),
        "num_valid_tasks": len(all_features),
        "task_duration": args.task_duration,
        "word_duration": args.word_duration,
        "action_duration": args.action_duration,
        "feature_dim": feature_matrix.shape[1],
        "model_type": "centroid",
        "frequency_bands": {k: list(v) for k, v in FREQ_BANDS.items()},
    }
    
    save_eeg_identity_model(
        centroid=centroid,
        scaler_params=scaler_params,
        spectral_profile=spectral_output,
        statistics=statistics,
        config=config,
        enrollment_log=enrollment_log,
        output_dir=MODELS_DIR,
    )
    
    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("ENROLLMENT COMPLETE")
    print("=" * 60)
    print(f"\n   Model saved to: {MODELS_DIR}")
    print(f"\n   EEG Identity Statistics:")
    print(f"   - Tasks completed: {len(all_features)}/{len(tasks)}")
    print(f"   - Feature dimensions: {feature_matrix.shape[1]}")
    print(f"   - Mean similarity: {statistics['mean_similarity']:.4f}")
    print(f"   - Verification threshold (1 sigma): {statistics['similarity_threshold_1std']:.4f}")
    print(f"   - Verification threshold (2 sigma): {statistics['similarity_threshold_2std']:.4f}")
    print(f"\n   Next steps:")
    print(f"   1. Run authorize_brainwaves.py to test live verification")
    print(f"   2. Re-enroll multiple times to improve the model")
    print(f"   3. Add to dashboard pipeline for UI-based enrollment")
    
    # Dashboard event — enrollment complete
    emit_event("enrollment_complete", {
        "success": True,
        "tasks_completed": len(all_features),
        "tasks_total": len(tasks),
        "feature_dim": int(feature_matrix.shape[1]),
        "mean_similarity": round(float(statistics['mean_similarity']), 4),
        "std_similarity": round(float(statistics['std_similarity']), 4),
        "threshold_1std": round(float(statistics['similarity_threshold_1std']), 4),
        "threshold_2std": round(float(statistics['similarity_threshold_2std']), 4),
    })


if __name__ == "__main__":
    main()

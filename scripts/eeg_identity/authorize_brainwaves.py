#!/usr/bin/env python3
"""
================================================================================
EEG BRAINWAVE AUTHORIZATION
================================================================================

PURPOSE:
    Reads live EEG data from an EMOTIV Epoc X headset and compares it against
    the enrolled identity model to produce an assurance signal. This is NOT
    a binary credential check - it's a continuous confidence score (0.0-1.0)
    that indicates how closely the current brainwave patterns match the
    enrolled identity.

DESIGN:
    Like the Identity MCP's conversation-based continuous evaluation,
    this provides an ongoing neurophysiological assurance signal. At some
    point it could evolve toward credential-like use (similar to FaceID),
    but the primary design is as an assurance layer.

ASSURANCE SIGNAL:
    - 0.0-0.3: Low confidence  (patterns do not match enrollment)
    - 0.3-0.6: Moderate        (some similarity, possible match)
    - 0.6-0.8: High confidence (strong pattern match)
    - 0.8-1.0: Very high       (patterns closely match enrollment)

PREREQUISITES:
    Run enroll_brainwaves.py first to create the reference model.

USAGE:
    python authorize_brainwaves.py --synthetic                       # Test without hardware
    python authorize_brainwaves.py --serial EX0000B29AB0205E         # With EMOTIV dongle
    python authorize_brainwaves.py --synthetic --continuous --interval 5

CONNECTION:
    Requires the EMOTIV USB wireless dongle (VID 0x1234, PID 0xED02).
    Direct USB cable is charge-only. Bluetooth LE is not supported.

OUTPUT:
    JSON assurance result to stdout:
    {
        "assurance_score": 0.82,
        "confidence": "high",
        "similarity": 0.84,
        "distance": 2.31,
        "threshold_used": "1std",
        "timestamp": "2026-02-06T12:00:00"
    }

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
from typing import Dict, Any, Optional
import warnings

# Force UTF-8 output on Windows (cp1252 can't handle box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')

# Dashboard event prefix — the backend SSE bridge looks for this marker
_DASHBOARD_MODE = False

def emit_event(event: str, data: Optional[Dict] = None):
    """Emit a structured event for the dashboard SSE bridge."""
    if not _DASHBOARD_MODE:
        return
    payload = {"event": event}
    if data:
        payload.update(data)
    print(f"@@EEG_EVENT:{json.dumps(payload)}", flush=True)

# Check dependencies
try:
    from scipy.stats import skew, kurtosis
    from scipy.signal import welch
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)

# Paths (following project conventions)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Support multi-user
USER_ID = os.environ.get("USER_ID")
def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    if user_id:
        return base_dir / user_id
    return base_dir

MODELS_DIR = get_user_dir(PROJECT_ROOT / "models" / "eeg_identity", USER_ID)

# Import from sibling modules
sys.path.insert(0, str(SCRIPT_DIR))
from emotiv_reader import (
    EmotivReader, EEG_CHANNELS, NUM_CHANNELS, SAMPLE_RATE, FREQ_BANDS,
    preprocess_eeg,
)
from enroll_brainwaves import extract_eeg_features


def check_dependencies():
    """Verify all required dependencies are installed."""
    if not HAS_DEPS:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print(f"\nError: {MISSING_DEP}")
        print("\nPlease install required packages:")
        print("  pip install numpy scipy scikit-learn tqdm")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_eeg_identity_model(models_dir: Path) -> Dict[str, Any]:
    """
    Load the enrolled EEG identity model.
    
    Returns dict with: centroid, scaler, config, statistics
    """
    # Check model exists
    config_path = models_dir / "config.json"
    centroid_path = models_dir / "eeg_centroid.npy"
    scaler_path = models_dir / "feature_scaler.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"No EEG identity model found at {models_dir}\n"
            f"Run enroll_brainwaves.py first to create the reference model."
        )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load centroid
    centroid = np.load(centroid_path)
    
    # Load scaler parameters
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    return {
        "centroid": centroid,
        "scaler_mean": np.array(scaler_params["mean"]),
        "scaler_scale": np.array(scaler_params["scale"]),
        "config": config,
        "statistics": config.get("statistics", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Assurance computation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(features: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray) -> np.ndarray:
    """Normalize feature vector using the same two-step pipeline as enrollment.
    
    Step 1: StandardScaler z-scoring (equalize feature scales)
    Step 2: L2 normalization (unit vector for cosine similarity)
    """
    # Step 1: StandardScaler z-scoring using saved params
    scaled = (features - scaler_mean) / (scaler_scale + 1e-10)
    # Step 2: L2-normalize to unit vector (matches enrollment centroid space)
    norm = np.linalg.norm(scaled)
    if norm > 0:
        scaled = scaled / norm
    return scaled


def compute_assurance(
    features: np.ndarray,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute assurance signal by comparing live features against enrolled model.
    
    Mirrors the verification approach from identityVerificationTools.ts and
    train_identity_model.py's verify_message().
    
    Args:
        features: Normalized feature vector from live EEG
        model: Loaded model dict (centroid, statistics, etc.)
    
    Returns:
        Assurance result dict
    """
    centroid = model["centroid"]
    stats = model["statistics"]
    
    # Cosine similarity to centroid
    dot_product = np.dot(features, centroid)
    norm_features = np.linalg.norm(features)
    norm_centroid = np.linalg.norm(centroid)
    similarity = float(dot_product / (norm_features * norm_centroid + 1e-8))
    
    # Euclidean distance from centroid
    distance = float(np.linalg.norm(features - centroid))
    
    # Map similarity to assurance score (0.0-1.0)
    # Use the enrollment statistics to calibrate
    mean_sim = stats.get("mean_similarity", 0.8)
    std_sim = stats.get("std_similarity", 0.1)
    sim_threshold_1std = stats.get("similarity_threshold_1std", mean_sim - std_sim)
    sim_threshold_2std = stats.get("similarity_threshold_2std", mean_sim - 2 * std_sim)
    
    # Normalize similarity to 0-1 range based on enrollment distribution
    # Score = 1.0 at mean_sim, drops toward 0 at 2std below mean
    if similarity >= mean_sim:
        assurance_score = min(1.0, 0.8 + 0.2 * (similarity - mean_sim) / (std_sim + 1e-8))
    elif similarity >= sim_threshold_1std:
        # Between mean and 1std: score 0.5-0.8
        t = (similarity - sim_threshold_1std) / (mean_sim - sim_threshold_1std + 1e-8)
        assurance_score = 0.5 + 0.3 * t
    elif similarity >= sim_threshold_2std:
        # Between 1std and 2std: score 0.2-0.5
        t = (similarity - sim_threshold_2std) / (sim_threshold_1std - sim_threshold_2std + 1e-8)
        assurance_score = 0.2 + 0.3 * t
    else:
        # Below 2std: score 0.0-0.2
        assurance_score = max(0.0, 0.2 * (similarity / (sim_threshold_2std + 1e-8)))
    
    assurance_score = float(np.clip(assurance_score, 0.0, 1.0))
    
    # Determine confidence level
    if assurance_score >= 0.8:
        confidence = "very_high"
        threshold_used = "above_mean"
    elif assurance_score >= 0.6:
        confidence = "high"
        threshold_used = "within_1std"
    elif assurance_score >= 0.3:
        confidence = "moderate"
        threshold_used = "within_2std"
    else:
        confidence = "low"
        threshold_used = "below_2std"
    
    return {
        "assurance_score": round(assurance_score, 4),
        "confidence": confidence,
        "similarity": round(similarity, 4),
        "distance": round(distance, 4),
        "threshold_used": threshold_used,
        "reference_thresholds": {
            "mean_similarity": round(mean_sim, 4),
            "threshold_1std": round(sim_threshold_1std, 4),
            "threshold_2std": round(sim_threshold_2std, 4),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Authorization flow
# ─────────────────────────────────────────────────────────────────────────────

def _compute_live_spectral(raw_data: np.ndarray, fs: int = SAMPLE_RATE) -> Optional[Dict]:
    """Compute a quick spectral profile from live data for dashboard overlay."""
    try:
        from scipy.signal import welch as sp_welch
        profile = {}
        for i, ch in enumerate(EEG_CHANNELS):
            if i >= raw_data.shape[0]:
                break
            freqs, psd = sp_welch(raw_data[i], fs=fs, nperseg=min(256, raw_data.shape[1]))
            total_power = float(np.sum(psd)) + 1e-10
            profile[ch] = {}
            for band, (lo, hi) in FREQ_BANDS.items():
                mask = (freqs >= lo) & (freqs < hi)
                profile[ch][band] = round(float(np.sum(psd[mask])) / total_power, 4)
        return profile
    except Exception:
        return None


def authorize_single(
    reader: EmotivReader,
    model: Dict[str, Any],
    window_seconds: float = 10.0,
    verbose: bool = True,
    check_num: int = 1,
) -> Dict[str, Any]:
    """
    Capture a single window of EEG and compute assurance.
    
    Args:
        reader: Connected EmotivReader
        model: Loaded identity model
        window_seconds: Duration of EEG to capture
        verbose: Whether to print progress
        check_num: Reading window number (for dashboard)
    
    Returns:
        Assurance result dict
    """
    emit_event("reading_start", {"window": check_num, "duration": window_seconds})
    
    if verbose:
        print(f"\n   Capturing {window_seconds}s of EEG data...")
    
    def progress_callback(elapsed, total):
        if verbose:
            bar_width = 30
            filled = int(bar_width * elapsed / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r   [{bar}] {elapsed:.0f}/{total:.0f}s", end="", flush=True)
        emit_event("reading_progress", {
            "window": check_num,
            "elapsed": round(elapsed, 1),
            "total": round(total, 1),
        })
    
    # Capture EEG
    raw_data = reader.read_seconds(window_seconds, callback=progress_callback)
    
    if verbose:
        print(f"\n   Captured {raw_data.shape[1]} samples")
    
    if raw_data.shape[1] == 0:
        return {
            "assurance_score": 0.0,
            "confidence": "error",
            "error": "No data captured",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Preprocess
    if verbose:
        print(f"   Preprocessing...", end=" ")
    filtered = preprocess_eeg(raw_data)
    if verbose:
        print("OK")
    
    # Extract features
    if verbose:
        print(f"   Extracting features...", end=" ")
    try:
        features = extract_eeg_features(filtered)
    except Exception as e:
        return {
            "assurance_score": 0.0,
            "confidence": "error",
            "error": f"Feature extraction failed: {e}",
            "timestamp": datetime.now().isoformat(),
        }
    if verbose:
        print(f"OK ({len(features)} dimensions)")
    
    # Normalize using enrolled scaler
    normalized = normalize_features(features, model["scaler_mean"], model["scaler_scale"])
    
    # Compute assurance
    if verbose:
        print(f"   Computing assurance signal...", end=" ")
    result = compute_assurance(normalized, model)
    
    # Compute live spectral for dashboard overlay
    live_spectral = _compute_live_spectral(raw_data) if _DASHBOARD_MODE else None
    
    emit_event("assurance_result", {
        "window": check_num,
        "assurance_score": result.get("assurance_score", 0),
        "similarity": result.get("similarity", 0),
        "confidence": result.get("confidence", "unknown"),
        "live_spectral": live_spectral,
    })
    if verbose:
        print("OK")
    
    return result


def format_assurance_display(result: Dict[str, Any]) -> str:
    """Format assurance result for CLI display."""
    score = result.get("assurance_score", 0)
    confidence = result.get("confidence", "unknown")
    similarity = result.get("similarity", 0)
    
    # Visual bar
    bar_width = 30
    filled = int(bar_width * score)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # Color indicators
    if score >= 0.8:
        indicator = "[VERY HIGH]"
    elif score >= 0.6:
        indicator = "[HIGH]     "
    elif score >= 0.3:
        indicator = "[MODERATE] "
    else:
        indicator = "[LOW]      "
    
    lines = [
        f"",
        f"   Assurance Signal",
        f"   ──────────────────────────────────────",
        f"   Score:      [{bar}] {score:.2f}",
        f"   Confidence: {indicator}",
        f"   Similarity: {similarity:.4f}",
        f"   Distance:   {result.get('distance', 0):.4f}",
        f"   ──────────────────────────────────────",
    ]
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="EEG Brainwave Authorization - Live identity assurance signal"
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
        "--window", type=float, default=10.0,
        help="Capture window in seconds (default: 10)"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuously, producing periodic assurance signals"
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Interval between checks in continuous mode (default: 10s)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON only (for script/tool integration)"
    )
    parser.add_argument(
        "--synthetic-seed", type=int, default=42,
        help="Random seed for synthetic identity (synthetic mode only)"
    )
    parser.add_argument(
        "--different-person", action="store_true",
        help="(Synthetic only) Use a different seed to simulate a non-enrolled person"
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
    verbose = not args.json
    
    if verbose:
        print("=" * 60)
        print("EEG BRAINWAVE AUTHORIZATION")
        print("=" * 60)
        print(f"\n   Mode: {mode}")
        print(f"   Capture window: {args.window}s")
        print(f"   Continuous: {args.continuous}")
    
    # ── Load enrolled model ──
    if verbose:
        print(f"\n   Loading enrolled identity model...")
    
    try:
        model = load_eeg_identity_model(MODELS_DIR)
    except FileNotFoundError as e:
        if args.json:
            print(json.dumps({"error": str(e), "assurance_score": 0.0}))
        else:
            print(f"\n   ERROR: {e}")
        sys.exit(1)
    
    if verbose:
        stats = model["statistics"]
        print(f"   Model loaded:")
        print(f"   - Feature dimensions: {stats.get('feature_dim', 'unknown')}")
        print(f"   - Training samples: {stats.get('num_samples', 'unknown')}")
        print(f"   - Mean similarity: {stats.get('mean_similarity', 0):.4f}")
        print(f"   - Threshold (1 sigma): {stats.get('similarity_threshold_1std', 0):.4f}")
    
    # Dashboard event — send model shape data for the static overlay
    spectral_data = None
    spectral_path = MODELS_DIR / "spectral_profile.json"
    if spectral_path.exists():
        try:
            with open(spectral_path, 'r') as f:
                sp = json.load(f)
            spectral_data = sp.get("aggregate", {})
        except Exception:
            pass
    
    emit_event("auth_start", {
        "config": model["config"],
        "statistics": {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                       for k, v in model["statistics"].items()
                       if k != "percentiles"},
        "spectral_profile": spectral_data,
    })
    
    # ── Connect to device ──
    if verbose:
        print(f"\n   Connecting to device...")
    
    # For the --different-person flag, use a different synthetic seed
    synthetic_seed = args.synthetic_seed
    if args.different_person:
        synthetic_seed = args.synthetic_seed + 999  # Completely different identity
        if verbose:
            print(f"   (Simulating DIFFERENT person with seed {synthetic_seed})")
    
    reader = EmotivReader(
        mode=mode,
        serial_number=args.serial,
        is_research=(args.device_type == "research"),
        synthetic_seed=synthetic_seed,
    )
    
    if not reader.connect():
        if args.json:
            print(json.dumps({"error": "Failed to connect", "assurance_score": 0.0}))
        else:
            print("\n   Failed to connect to EMOTIV device.")
        sys.exit(1)
    
    if verbose:
        print(f"   Connected!")
    
    try:
        if args.continuous:
            # ── Continuous authorization ──
            if verbose:
                print(f"\n   Starting continuous authorization (Ctrl+C to stop)")
                print(f"   Check interval: {args.interval}s, Window: {args.window}s")
            
            check_num = 0
            results_history = []
            
            while True:
                check_num += 1
                
                if verbose:
                    print(f"\n{'─' * 60}")
                    print(f"   Check #{check_num}")
                
                result = authorize_single(reader, model, args.window, verbose=verbose, check_num=check_num)
                result["check_number"] = check_num
                results_history.append(result)
                
                # Compute rolling average (last 5 checks)
                recent_scores = [r["assurance_score"] for r in results_history[-5:] 
                                if "error" not in r]
                if recent_scores:
                    result["rolling_average"] = round(float(np.mean(recent_scores)), 4)
                
                if args.json:
                    print(json.dumps(result))
                else:
                    print(format_assurance_display(result))
                    if "rolling_average" in result:
                        print(f"   Rolling avg (last {min(5, len(recent_scores))}): {result['rolling_average']:.4f}")
                
                # Wait for next interval
                if verbose:
                    print(f"\n   Next check in {args.interval}s... (Ctrl+C to stop)")
                time.sleep(args.interval)
        
        else:
            # ── Single authorization check ──
            if verbose:
                print(f"\n   Please sit still and relax for {args.window} seconds.")
                print(f"   Starting in 3 seconds...")
                time.sleep(1)
                print(f"   2...")
                time.sleep(1)
                print(f"   1...")
                time.sleep(1)
            
            result = authorize_single(reader, model, args.window, verbose=verbose, check_num=1)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(format_assurance_display(result))
                
                # Summary
                print(f"\n{'=' * 60}")
                print("AUTHORIZATION RESULT")
                print("=" * 60)
                score = result.get("assurance_score", 0)
                if score >= 0.6:
                    print(f"\n   PASS - Brainwave patterns match enrolled identity")
                    print(f"   Assurance score: {score:.2f} ({result.get('confidence', 'unknown')})")
                elif score >= 0.3:
                    print(f"\n   PARTIAL - Some similarity to enrolled identity")
                    print(f"   Assurance score: {score:.2f} ({result.get('confidence', 'unknown')})")
                else:
                    print(f"\n   FAIL - Brainwave patterns do not match enrollment")
                    print(f"   Assurance score: {score:.2f} ({result.get('confidence', 'unknown')})")
    
    except KeyboardInterrupt:
        if verbose:
            print(f"\n\n   Authorization stopped by user.")
    
    finally:
        reader.disconnect()
        emit_event("auth_stopped", {"message": "Authorization session ended"})
        if verbose:
            print(f"   Device disconnected.")


if __name__ == "__main__":
    main()

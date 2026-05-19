"""Calibrate synthetic EEG demo settings against an enrolled identity model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from emotiv_reader import EmotivReader, preprocess_eeg
from enroll_brainwaves import extract_eeg_features


def _authorize_api():
    from authorize_brainwaves import (
        compute_assurance,
        load_eeg_identity_model,
        normalize_features,
    )

    return compute_assurance, load_eeg_identity_model, normalize_features


def _feature_vector_from_reader(
    reader: EmotivReader,
    window_seconds: float,
) -> np.ndarray | None:
    raw = reader.read_seconds(window_seconds, task_label="baseline")
    if raw.shape[1] == 0:
        return None
    filtered = preprocess_eeg(raw)
    features = extract_eeg_features(filtered)
    return features


def calibrate_demo_synthetic_seed(
    models_dir: Path,
    *,
    seed_min: int = 0,
    seed_max: int = 500,
    window_seconds: float = 10.0,
) -> dict[str, Any]:
    """Find a synthetic identity_seed that best matches the enrolled centroid."""
    _, load_eeg_identity_model, normalize_features = _authorize_api()
    model = load_eeg_identity_model(models_dir)
    centroid = model["centroid"]
    scaler_mean = model["scaler_mean"]
    scaler_scale = model["scaler_scale"]

    best_seed = 42
    best_similarity = -1.0

    for seed in range(seed_min, seed_max + 1):
        reader = EmotivReader(mode="synthetic", synthetic_seed=seed)
        if not reader.connect():
            continue
        try:
            features = _feature_vector_from_reader(reader, window_seconds)
        finally:
            reader.disconnect()

        if features is None:
            continue

        normalized = normalize_features(features, scaler_mean, scaler_scale)
        sim = float(
            np.dot(normalized, centroid)
            / (np.linalg.norm(normalized) * np.linalg.norm(centroid) + 1e-8)
        )
        if sim > best_similarity:
            best_similarity = sim
            best_seed = seed

    return {
        "demo_synthetic_seed": best_seed,
        "calibration_similarity": round(best_similarity, 4),
        "seed_search_range": [seed_min, seed_max],
        "window_seconds": window_seconds,
    }


def assurance_from_centroid_demo(
    model: dict[str, Any],
    *,
    noise_scale: float = 0.015,
) -> dict[str, Any]:
    """Replay enrolled signature (centroid + tiny noise) for headset-free demos."""
    compute_assurance, _, _ = _authorize_api()
    centroid = model["centroid"].astype(np.float64)
    if noise_scale > 0:
        noisy = centroid + np.random.randn(*centroid.shape) * noise_scale
        norm = np.linalg.norm(noisy)
        features = noisy / norm if norm > 0 else centroid
    else:
        features = centroid

    result = compute_assurance(features, model)
    result["mode"] = "synthetic_demo_replay"
    return result


def update_config_demo_fields(models_dir: Path, fields: dict[str, Any]) -> Path:
    config_path = models_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config.update(fields)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path

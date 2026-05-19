#!/usr/bin/env python3
"""Calibrate headset-free EEG demo settings for an enrolled model."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from demo_calibration import calibrate_demo_synthetic_seed, update_config_demo_fields  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate synthetic EEG demo for enrollment")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "eeg_identity",
        help="EEG model directory",
    )
    parser.add_argument("--seed-max", type=int, default=500, help="Upper bound for seed search")
    parser.add_argument(
        "--prefer-replay",
        action="store_true",
        default=True,
        help="Mark config to use centroid demo replay for synthetic authorize (default)",
    )
    args = parser.parse_args()

    if not (args.models_dir / "config.json").is_file():
        print(f"No enrollment at {args.models_dir}")
        sys.exit(1)

    print(f"Calibrating demo synthetic seed in {args.models_dir} ...")
    cal = calibrate_demo_synthetic_seed(args.models_dir, seed_max=args.seed_max)
    fields = {
        **cal,
        "demo_replay_preferred": args.prefer_replay,
    }
    path = update_config_demo_fields(args.models_dir, fields)
    print(json.dumps({"saved_to": str(path), **fields}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parse all uploaded memory exports (ChatGPT + Claude).
====================================================

Runs parse_memories.py when memory/memories.json exists,
and parse_anthropic_memories.py when memory/anthropic_memories.json exists.
Used by the dashboard "Parse Memories" pipeline step.

Usage:
  python parse_all_memories.py
  python parse_all_memories.py --clean
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

USER_ID = os.environ.get("USER_ID")

CHATGPT_SCRIPT = SCRIPT_DIR / "parse_memories.py"
ANTHROPIC_SCRIPT = SCRIPT_DIR / "parse_anthropic_memories.py"

OUTPUT_FILES = ("user.context.jsonl", "claude.context.jsonl")


def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    if user_id:
        return base_dir / user_id
    return base_dir


MEMORY_DIR = get_user_dir(PROJECT_ROOT / "memory", USER_ID)


def run_parser(label: str, script: Path, extra_args: List[str]) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    print("$", " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def clean_memory_outputs() -> None:
    for name in OUTPUT_FILES:
        path = MEMORY_DIR / name
        if path.exists():
            path.unlink()
            print(f"Removed {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ChatGPT and/or Claude memory uploads when present"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove user.context.jsonl and claude.context.jsonl before parsing",
    )
    args = parser.parse_args()

    chatgpt_upload = MEMORY_DIR / "memories.json"
    anthropic_upload = MEMORY_DIR / "anthropic_memories.json"

    print(f"Memory directory: {MEMORY_DIR}")
    print(f"  ChatGPT upload: {'found' if chatgpt_upload.exists() else 'not found'} ({chatgpt_upload})")
    print(
        f"  Claude upload:  {'found' if anthropic_upload.exists() else 'not found'} ({anthropic_upload})"
    )

    if not chatgpt_upload.exists() and not anthropic_upload.exists():
        print("\nError: No memory uploads found.")
        print("Upload via Data Explorer:")
        print("  - ChatGPT memories.json -> memories.json")
        print("  - Claude memories export -> anthropic_memories.json")
        sys.exit(1)

    if args.clean:
        clean_memory_outputs()

    exit_code = 0
    ran = 0

    if chatgpt_upload.exists():
        code = run_parser("ChatGPT (memories.json)", CHATGPT_SCRIPT, [])
        if code != 0:
            exit_code = code
        ran += 1
    else:
        print("\nSkipping ChatGPT parser (memories.json not uploaded).")

    if anthropic_upload.exists():
        code = run_parser("Claude (anthropic_memories.json)", ANTHROPIC_SCRIPT, [])
        if code != 0 and exit_code == 0:
            exit_code = code
        ran += 1
    else:
        print("\nSkipping Claude parser (anthropic_memories.json not uploaded).")

    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print(f"Done — ran {ran} parser(s).")
    else:
        print(f"Finished with errors (exit {exit_code}) from one or more parsers.")
    print(f"{'=' * 60}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

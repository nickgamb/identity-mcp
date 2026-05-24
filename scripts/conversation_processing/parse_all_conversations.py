#!/usr/bin/env python3
"""
Parse all uploaded conversation exports (ChatGPT + Claude).
============================================================

Runs parse_conversations.py when conversations/conversations.json exists,
and parse_anthropic_conversations.py when conversations/anthropic_conversations.json
exists. Used by the dashboard "Parse Conversations" pipeline step.

Usage:
  python parse_all_conversations.py
  python parse_all_conversations.py --clean
  python parse_all_conversations.py --force   # Reprocess all (CLI only; dashboard does not pass this)

Default: skip conversations that already have conversation_<id>.jsonl.
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


def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    if user_id:
        return base_dir / user_id
    return base_dir


CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations", USER_ID)

CHATGPT_SCRIPT = SCRIPT_DIR / "parse_conversations.py"
ANTHROPIC_SCRIPT = SCRIPT_DIR / "parse_anthropic_conversations.py"


def run_parser(label: str, script: Path, extra_args: List[str]) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    print("$", " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ChatGPT and/or Claude conversation uploads when present"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing conversation_*.jsonl/md before parsing (once, then both parsers)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Reprocess all conversations (both parsers); not used by the dashboard UI",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal parser output")
    args = parser.parse_args()

    chatgpt_upload = CONVERSATIONS_DIR / "conversations.json"
    anthropic_upload = CONVERSATIONS_DIR / "anthropic_conversations.json"

    print(f"Conversations directory: {CONVERSATIONS_DIR}")
    print(f"  ChatGPT upload: {'found' if chatgpt_upload.exists() else 'not found'} ({chatgpt_upload})")
    print(
        f"  Claude upload:  {'found' if anthropic_upload.exists() else 'not found'} ({anthropic_upload})"
    )

    if not chatgpt_upload.exists() and not anthropic_upload.exists():
        print("\nError: No conversation uploads found.")
        print("Upload via Data Explorer:")
        print("  - ChatGPT conversations.json -> conversations.json")
        print("  - Claude conversations.json -> anthropic_conversations.json")
        sys.exit(1)

    # --clean only on the first parser that runs (both write to the same output dir)
    clean_pending = args.clean
    exit_code = 0
    ran = 0

    if chatgpt_upload.exists():
        chatgpt_args: List[str] = []
        if clean_pending:
            chatgpt_args.append("--clean")
            clean_pending = False
        if args.force:
            chatgpt_args.append("--force")
        if args.quiet:
            chatgpt_args.append("--quiet")
        code = run_parser("ChatGPT (conversations.json)", CHATGPT_SCRIPT, chatgpt_args)
        if code != 0:
            exit_code = code
        ran += 1
    else:
        print("\nSkipping ChatGPT parser (conversations.json not uploaded).")

    if anthropic_upload.exists():
        anthropic_args: List[str] = []
        if clean_pending:
            anthropic_args.append("--clean")
        if args.force:
            anthropic_args.append("--force")
        if args.quiet:
            anthropic_args.append("--quiet")
        code = run_parser("Claude (anthropic_conversations.json)", ANTHROPIC_SCRIPT, anthropic_args)
        if code != 0 and exit_code == 0:
            exit_code = code
        ran += 1
    else:
        print("\nSkipping Claude parser (anthropic_conversations.json not uploaded).")

    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print(f"Done — ran {ran} parser(s). Next: Analyze Patterns.")
    else:
        print(f"Finished with errors (exit {exit_code}) from one or more parsers.")
    print(f"{'=' * 60}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

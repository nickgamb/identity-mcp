#!/usr/bin/env python3
"""
Claude Memory Parser — anthropic_memories.json → claude.context.jsonl
=====================================================================

Converts Claude's memory export (list of conversations_memory blobs) into
tagged JSONL records, parallel to parse_memories.py for ChatGPT.

Input (anthropic_memories.json):
  [
    {"conversations_memory": "..."},
    ...
  ]

Output (claude.context.jsonl):
  {"id": "...", "type": "claude_memory", "content": "...", "tags": [...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from memory_parser_common import (
    USER_ID,
    chunk_text,
    extract_tags,
    get_user_dir,
    load_keywords,
)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def load_claude_memory_items(data: Any) -> List[Dict[str, Any]]:
    """Normalize Claude upload JSON into a list of memory objects."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("memories", "items", "data"):
            nested = data.get(key)
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse anthropic_memories.json to tagged JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=str, default=None, help="Path to anthropic_memories.json")
    parser.add_argument("--output", type=str, default=None, help="Path for output JSONL")
    args = parser.parse_args()

    memory_dir = get_user_dir(PROJECT_ROOT / "memory", USER_ID)
    input_path = Path(args.input) if args.input else memory_dir / "anthropic_memories.json"
    output_path = Path(args.output) if args.output else memory_dir / "claude.context.jsonl"
    patterns_path = memory_dir / "patterns.jsonl"

    if not input_path.exists():
        print(f"Error: anthropic_memories.json not found at {input_path}")
        sys.exit(1)

    keywords = load_keywords(patterns_path)
    if keywords:
        print(f"Loaded {len(keywords)} keywords from patterns.jsonl")
    else:
        print("No patterns.jsonl found - memories will have basic tags")
        print("  Tip: Run 'python analyze_patterns.py' first for auto-tagging")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    items = load_claude_memory_items(data)
    if not items:
        print("Error: Invalid anthropic_memories.json format")
        print('Expected: [{"conversations_memory": "..."}, ...]')
        sys.exit(1)

    output_lines: List[str] = []
    tag_counts: Dict[str, int] = {}
    record_count = 0

    for idx, item in enumerate(items):
        blob = (item.get("conversations_memory") or item.get("content") or "").strip()
        if not blob:
            continue

        chunks = chunk_text(blob)
        for chunk_idx, content in enumerate(chunks):
            tags = extract_tags(content, keywords)
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            suffix = f"-{chunk_idx}" if len(chunks) > 1 else ""
            record: Dict[str, Any] = {
                "id": f"claude-memory-{idx}{suffix}",
                "type": "claude_memory",
                "content": content,
                "source": "claude_memories",
                "source_index": idx,
                "tags": tags,
            }
            if len(chunks) > 1:
                record["chunk"] = chunk_idx
            output_lines.append(json.dumps(record, ensure_ascii=False))
            record_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + ("\n" if output_lines else ""))

    print(f"\n✅ Parsed {len(items)} Claude memory export(s) → {record_count} record(s) in {output_path}")

    if tag_counts:
        print("\nTop tags applied:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()

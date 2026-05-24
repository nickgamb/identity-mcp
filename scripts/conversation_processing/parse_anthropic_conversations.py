#!/usr/bin/env python3
"""
Anthropic/Claude Conversation Parser
=====================================

Converts Claude's conversations.json export into the same JSONL/Markdown format
used by the ChatGPT parser (parse_conversations.py). This makes Claude
conversations work seamlessly with the rest of the identity-mcp pipeline:
pattern analysis, identity analysis, Letta ingest, and the dashboard.

Usage:
  python parse_anthropic_conversations.py
  python parse_anthropic_conversations.py --input path/to/conversations.json
  python parse_anthropic_conversations.py --clean
  python parse_anthropic_conversations.py --force   # Reprocess all, even if already parsed

Input:
  conversations/anthropic_conversations.json (default)

Output (identical to parse_conversations.py):
  conversations/conversation_<id>.jsonl
  conversations/conversation_<id>.md

  JSONL: {"timestamp": "...", "role": "user|assistant", "content": "...", "line_number": 1}

Dependencies:
  pip install tqdm  (optional, for progress bars)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

USER_ID = os.environ.get("USER_ID")


def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    if user_id:
        return base_dir / user_id
    return base_dir


CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations", USER_ID)


def claude_output_id(convo: Dict) -> str:
    """ID used in conversation_<id>.jsonl filenames (first 8 chars of uuid)."""
    convo_id = convo.get("uuid", "")
    return convo_id[:8] if convo_id else ""


def clean_output_directory(output_dir: Path) -> int:
    """Remove existing conversation_*.jsonl/md from output directory."""
    removed = 0
    for pattern in ("conversation_*.jsonl", "conversation_*.md"):
        for file_path in output_dir.glob(pattern):
            try:
                file_path.unlink()
                removed += 1
            except OSError as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    return removed


def get_existing_conversation_ids(output_dir: Path) -> Set[str]:
    """IDs already present as conversation_<id>.jsonl files."""
    existing: Set[str] = set()
    for jsonl_file in output_dir.glob("conversation_*.jsonl"):
        name = jsonl_file.stem
        if name.startswith("conversation_"):
            existing.add(name[len("conversation_") :])
    return existing


def extract_text_from_content(content_items: List[Dict]) -> str:
    """Extract readable text from Claude's structured content array."""
    parts = []
    for item in content_items:
        item_type = item.get("type", "")
        if item_type == "text":
            text = item.get("text", "").strip()
            if text:
                parts.append(text)
        elif item_type == "thinking":
            thinking = item.get("thinking", "").strip()
            if thinking:
                parts.append(f"<thinking>\n{thinking}\n</thinking>")
    return "\n\n".join(parts)


def process_claude_conversation(convo: Dict, output_dir: Path) -> Optional[Dict]:
    """Process a single Claude conversation into JSONL and Markdown."""
    convo_id = convo.get("uuid", "")
    if not convo_id:
        return None

    short_id = convo_id[:8]
    title = convo.get("name", "Untitled Conversation") or "Untitled"
    chat_messages = convo.get("chat_messages", [])
    if not chat_messages:
        return None

    messages = []
    for msg in chat_messages:
        sender = msg.get("sender", "")
        role = "user" if sender == "human" else "assistant" if sender == "assistant" else None
        if not role:
            continue

        content_items = msg.get("content", [])
        if isinstance(content_items, list):
            text = extract_text_from_content(content_items)
        else:
            text = str(content_items)

        if not text or not text.strip():
            fallback = msg.get("text", "")
            if fallback and fallback.strip():
                text = fallback.strip()
            else:
                continue

        text = text.encode("utf-8", errors="replace").decode("utf-8")
        text = text.replace("\x00", "")

        timestamp = msg.get("created_at", "")
        if not timestamp:
            for item in msg.get("content", []):
                s = item.get("start_timestamp")
                if s:
                    timestamp = s
                    break

        messages.append({
            "timestamp": timestamp or "Unknown",
            "role": role,
            "content": text,
            "line_number": 0,
        })

    if not messages:
        return None

    for i, msg in enumerate(messages):
        msg["line_number"] = i + 1

    filename_base = f"conversation_{short_id}"
    jsonl_path = output_dir / f"{filename_base}.jsonl"
    md_path = output_dir / f"{filename_base}.md"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Conversation ID:** {convo_id}\n")
        f.write(f"**Source:** Claude/Anthropic export\n")
        f.write(f"**Messages:** {len(messages)}\n\n")
        f.write("---\n\n")
        for msg in messages:
            role_display = msg["role"].title()
            f.write(f"**{role_display}** [{msg['timestamp']}]:\n\n")
            f.write(f"{msg['content']}\n\n")
            f.write("---\n\n")

    return {
        "id": convo_id,
        "short_id": short_id,
        "title": title,
        "message_count": len(messages),
    }


def main():
    ap = argparse.ArgumentParser(description="Parse Claude/Anthropic conversation exports")
    ap.add_argument("--input", "-i", type=Path, help="Input JSON file")
    ap.add_argument("--output", "-o", type=Path, help="Output directory (default: conversations/)")
    ap.add_argument("--clean", action="store_true", help="Remove existing conversation files before processing")
    ap.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Reprocess all conversations, even if already parsed",
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = ap.parse_args()

    output_dir = Path(args.output) if args.output else CONVERSATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or (CONVERSATIONS_DIR / "anthropic_conversations.json")
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        print("Upload your Claude conversations.json via the dashboard or place it at:")
        print(f"  {CONVERSATIONS_DIR / 'anthropic_conversations.json'}")
        sys.exit(1)

    if not args.quiet:
        print(f"Loading from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Expected a JSON array of conversations.")
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(data)} conversations")

    if args.clean:
        removed = clean_output_directory(output_dir)
        if not args.quiet:
            print(f"Removed {removed} existing conversation files")

    # Skip already parsed (same behavior as parse_conversations.py)
    conversations = data
    if not args.force and not args.clean:
        existing_ids = get_existing_conversation_ids(output_dir)
        if existing_ids:
            original_count = len(conversations)
            conversations = [
                c for c in conversations if claude_output_id(c) not in existing_ids
            ]
            skipped_existing = original_count - len(conversations)
            if skipped_existing > 0 and not args.quiet:
                print(
                    f"Skipping {skipped_existing} already processed conversations "
                    "(use --force to reprocess)"
                )

    if not conversations:
        print("All Claude conversations already processed. Use --force to reprocess.")
        return

    if not args.quiet:
        print(f"Processing {len(conversations)} conversations...")

    iterator = (
        tqdm(conversations, desc="Parsing Claude", unit="conv")
        if HAS_TQDM and not args.quiet
        else conversations
    )

    results = []
    skipped_empty = 0
    for convo in iterator:
        result = process_claude_conversation(convo, output_dir)
        if result:
            results.append(result)
        else:
            skipped_empty += 1

    total_msgs = sum(r["message_count"] for r in results)
    if not args.quiet:
        print(f"\n✅ Processed {len(results)} conversations ({skipped_empty} empty/skipped in export)")
        print(f"   Total messages: {total_msgs}")
        print(f"   Output directory: {output_dir}")
        print("\nDone. Run the analysis pipeline next:")
        print("  python scripts/conversation_processing/analyze_patterns.py")


if __name__ == "__main__":
    main()

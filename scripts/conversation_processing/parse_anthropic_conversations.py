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
from typing import List, Dict, Optional
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
    ap.add_argument("--clean", action="store_true", help="Remove existing output first")
    args = ap.parse_args()

    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = args.input or (CONVERSATIONS_DIR / "anthropic_conversations.json")
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        print("Upload your Claude conversations.json via the dashboard or place it at:")
        print(f"  {CONVERSATIONS_DIR / 'anthropic_conversations.json'}")
        sys.exit(1)

    print(f"Loading from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Expected a JSON array of conversations.")
        sys.exit(1)

    print(f"Found {len(data)} conversations")

    if args.clean:
        existing = list(CONVERSATIONS_DIR.glob("conversation_*.jsonl")) + \
                   list(CONVERSATIONS_DIR.glob("conversation_*.md"))
        for p in existing:
            p.unlink()
        print(f"Cleaned {len(existing)} existing files")

    iterator = tqdm(data, desc="Parsing", unit="conv") if HAS_TQDM else data

    results = []
    skipped = 0
    for convo in iterator:
        result = process_claude_conversation(convo, CONVERSATIONS_DIR)
        if result:
            results.append(result)
        else:
            skipped += 1

    print(f"\nProcessed {len(results)} conversations ({skipped} skipped)")
    total_msgs = sum(r["message_count"] for r in results)
    print(f"Total messages: {total_msgs}")

    print("Done. Run the analysis pipeline next:")
    print("  python scripts/conversation_processing/analyze_patterns.py")


if __name__ == "__main__":
    main()

"""Shared helpers for memory parsers and downstream analysis scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PARSED_MEMORY_FILES = (
    ("user.context.jsonl", "chatgpt_memories"),
    ("claude.context.jsonl", "claude_memories"),
)

USER_ID = os.environ.get("USER_ID")


def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    if user_id:
        return base_dir / user_id
    return base_dir


def load_keywords(patterns_path: Path) -> List[str]:
    keywords: List[str] = []
    if not patterns_path.exists():
        return keywords
    try:
        for line in patterns_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("type") == "pattern.keywords":
                    keywords = record.get("keywords", [])
                    break
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        print(f"Warning: Error loading patterns: {exc}")
    return keywords


def extract_tags(content: str, keywords: List[str]) -> List[str]:
    tags: List[str] = []
    lower = content.lower()
    for keyword in keywords:
        if keyword.lower() in lower:
            tags.append(keyword)
    return tags[:10]


def load_parsed_memories(memory_dir: Path) -> List[Dict[str, Any]]:
    """Load ChatGPT and Claude memory JSONL produced by parse_memories / parse_anthropic_memories."""
    records: List[Dict[str, Any]] = []
    for filename, source in PARSED_MEMORY_FILES:
        path = memory_dir / filename
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                content = rec.get("content") or rec.get("text") or ""
                if not content:
                    continue
                records.append(
                    {
                        "id": rec.get("id", f"{source}-{len(records)}"),
                        "type": rec.get("type", source),
                        "content": content,
                        "source": source,
                        "tags": rec.get("tags", []),
                    }
                )
        except Exception as exc:
            print(f"Warning: could not load {path}: {exc}")
    return records


def memories_as_corpus_entries(memory_dir: Path) -> List[Dict[str, str]]:
    """Format parsed memories like files/ corpus entries for analyze_patterns."""
    return [
        {
            "filepath": f"memory/{rec['source']}/{rec['id']}",
            "content": rec["content"],
        }
        for rec in load_parsed_memories(memory_dir)
    ]


def memories_as_synthetic_conversation(memory_dir: Path) -> Dict[str, Any]:
    """Bundle parsed memories as one pseudo-conversation of user messages."""
    messages = [
        {"role": "user", "content": rec["content"], "timestamp": ""}
        for rec in load_parsed_memories(memory_dir)
        if rec.get("content")
    ]
    return {
        "id": "parsed_memories",
        "file": "memory/*.context.jsonl",
        "messages": messages,
        "first_timestamp": "",
        "last_timestamp": "",
    }


def chunk_text(text: str, max_chars: int = 12_000) -> List[str]:
    """Split large memory blobs into paragraph-aware chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    chunks: List[str] = []
    current: List[str] = []
    size = 0

    for para in paragraphs:
        para_len = len(para) + (2 if current else 0)
        if size + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            size = 0
        if len(para) > max_chars and not current:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        current.append(para)
        size += para_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks

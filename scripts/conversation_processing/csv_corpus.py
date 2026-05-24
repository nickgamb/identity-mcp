#!/usr/bin/env python3
"""
Generic CSV/TSV handling for identity-mcp corpus tools.

Any tabular upload is parsed row-by-row (schema-agnostic): each row becomes
labeled key: value lines. Used by Letta ingest and analyze_patterns instead of
dumping the raw file as one blob.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

TABULAR_EXTENSIONS = {".csv", ".tsv"}

DATE_COLUMN_RE = re.compile(r"date|time|created|updated|posted|timestamp", re.I)
MIN_ROW_CHARS = 10
MIN_PASSAGE_CHARS = 20


def is_tabular_file(path: Path) -> bool:
    return path.suffix.lower() in TABULAR_EXTENSIONS


def read_tabular_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV or TSV into list of row dicts (string values, stripped)."""
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            return [], []
        fieldnames = [str(h).strip() for h in reader.fieldnames if h]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            row = {str(k).strip(): (v or "").strip() for k, v in raw.items() if k}
            if any(row.values()):
                rows.append(row)
        return fieldnames, rows


def format_row(row: Dict[str, str], fieldnames: List[str]) -> str:
    """Human-readable single row for embedding / analysis."""
    lines: List[str] = []
    seen = set()
    for key in fieldnames:
        if key in seen:
            continue
        seen.add(key)
        val = row.get(key, "")
        if val:
            lines.append(f"{key}: {val}")
    for key, val in row.items():
        if key not in seen and val:
            lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _row_timestamp(row: Dict[str, str], fieldnames: List[str]) -> Optional[str]:
    for key in fieldnames:
        if DATE_COLUMN_RE.search(key):
            val = row.get(key, "")
            if val:
                return val
    return None


def iter_row_bodies(path: Path) -> Iterator[Tuple[int, str, Optional[str]]]:
    """
    Yield (row_number, body_text, optional_timestamp) for each non-empty row.
    Falls back to whole-file text if the file is not a valid table.
    """
    try:
        fieldnames, rows = read_tabular_rows(path)
    except csv.Error:
        fieldnames, rows = [], []

    if not rows:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) >= MIN_ROW_CHARS:
            yield 1, text, None
        return

    for i, row in enumerate(rows, start=1):
        body = format_row(row, fieldnames)
        if len(body) >= MIN_ROW_CHARS:
            yield i, body, _row_timestamp(row, fieldnames)


def corpus_text_for_analysis(path: Path) -> str:
    """Flatten all rows into one document for vocabulary / pattern analysis."""
    blocks = [body for _, body, _ in iter_row_bodies(path)]
    return "\n\n".join(blocks)


def search_matching_rows(path: Path, query: str, limit: int = 20) -> List[str]:
    """Return formatted row blocks whose text contains query (case-insensitive)."""
    q = query.lower().strip()
    if not q:
        return []
    out: List[str] = []
    for row_num, body, _ in iter_row_bodies(path):
        if q in body.lower():
            out.append(f"--- row {row_num} ---\n{body}")
            if len(out) >= limit:
                break
    return out


def iter_tabular_passages(
    path: Path,
    source_name: str,
    target_chars: int = 1500,
) -> Iterator[Tuple[str, Optional[str]]]:
    """
    Group rows into passages ~target_chars for archival embedding.
    Yields (passage_text_with_header, optional_timestamp).
    """
    header_prefix = f"[tabular | {source_name}]"
    chunk_rows: List[str] = []
    chunk_ts: Optional[str] = None
    chunk_len = 0

    def flush() -> Optional[Tuple[str, Optional[str]]]:
        nonlocal chunk_rows, chunk_ts, chunk_len
        if not chunk_rows:
            return None
        body = "\n\n".join(chunk_rows)
        passage = f"{header_prefix}\n{body}"
        ts, chunk_rows, chunk_ts, chunk_len = chunk_ts, [], None, 0
        return passage, ts

    for row_num, body, ts in iter_row_bodies(path):
        row_block = f"--- row {row_num} ---\n{body}"
        row_len = len(row_block) + 2

        if row_len > target_chars and not chunk_rows:
            yield f"{header_prefix}\n{row_block}", ts
            continue

        if chunk_len + row_len > target_chars and chunk_rows:
            out = flush()
            if out:
                yield out

        if chunk_ts is None and ts:
            chunk_ts = ts
        chunk_rows.append(row_block)
        chunk_len += row_len

    out = flush()
    if out:
        yield out


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: csv_corpus.py corpus <path> | csv_corpus.py search <path> <query> [limit]",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = sys.argv[1]
    file_path = Path(sys.argv[2])

    if cmd == "corpus":
        print(corpus_text_for_analysis(file_path), end="")
    elif cmd == "search":
        query = sys.argv[3] if len(sys.argv) > 3 else ""
        row_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        print(json.dumps(search_matching_rows(file_path, query, row_limit)))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)

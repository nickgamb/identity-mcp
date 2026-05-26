"""
Persist reasoning-loop guard hits for logs and dashboard Activity (via JSONL on memory volume).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from reasoning_guard import LoopDetectionDetail

log = logging.getLogger("letta-bridge.guard")

GUARD_LOG_PATH = os.getenv(
    "LETTA_GUARD_LOG_PATH",
    "/app/memory/bridge-guard-events.jsonl",
)
GUARD_LOG_MAX_BYTES = int(os.getenv("LETTA_GUARD_LOG_MAX_BYTES", "5242880"))  # 5 MiB rotate


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.is_file() and path.stat().st_size > GUARD_LOG_MAX_BYTES:
            backup = path.with_suffix(".jsonl.1")
            if backup.exists():
                backup.unlink()
            path.rename(backup)
    except OSError as exc:
        log.warning("guard log rotate skipped: %s", exc)


def record_guard_event(
    detail: LoopDetectionDetail,
    *,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    source: str = "stream",
) -> None:
    """Append one JSON line and emit a structured warning log."""
    ts = datetime.now(timezone.utc).isoformat()
    row: Dict[str, Any] = {
        "ts": ts,
        "kind": "reasoning_loop",
        "reason": detail.reason.value,
        "source": source,
        "agent_id": agent_id,
        "run_id": run_id,
        "total_chars": detail.total_chars,
        "sample": detail.sample,
    }
    if detail.pattern:
        row["pattern"] = detail.pattern
    if detail.marker_count is not None:
        row["marker_count"] = detail.marker_count

    log.warning(
        "Reasoning loop guard: reason=%s chars=%s run_id=%s — %s",
        detail.reason.value,
        detail.total_chars,
        run_id or "-",
        detail.sample[:200].replace("\n", " "),
        extra={"guard_event": row},
    )

    path = Path(GUARD_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("Could not write guard log %s: %s", path, exc)

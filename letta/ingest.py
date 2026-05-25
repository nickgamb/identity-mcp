#!/usr/bin/env python3
"""
Ingest the identity-mcp corpus into a Letta agent's archival memory.

Replaces bootstrap_agent.py with a production-quality pipeline:
- Coherent chunking by exchange (user+assistant turn-pairs), not blind char splits
- Metadata headers on every passage (date, conversation id, source type)
- Idempotent: content-hash dedup, resumable via state file
- Source tagging via Letta archival tags (loop guard for future re-ingest)
- Agent-authored self-initialization: agent discovers its own identity from memory

Usage:
    pip install letta-client
    DATA_ROOT=/path/to/data LETTA_BASE_URL=http://localhost:8283 \
      python letta/ingest.py

    # Smoke test with 50 passages:
    DATA_ROOT=/path/to/data python letta/ingest.py --limit 50

    # Full ingest then self-init:
    DATA_ROOT=/path/to/data python letta/ingest.py

    # Re-run self-init only (e.g. after tuning):
    DATA_ROOT=/path/to/data python letta/ingest.py --init-only

Flags:
    --fresh             delete + recreate agent (full reset)
    --skip-archival     only ensure agent exists
    --skip-init         skip agent self-initialization
    --init-only         only run self-initialization
    --limit N           cap passages (0 = all)
    --batch-size N      save state every N passages (default 50)
    --throttle SECS     delay between inserts (default 0.05)
"""
import os
import glob
import json
import time
import hashlib
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Iterator, Optional, Tuple

import httpx
from letta_client import Letta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from model_prefs import resolve_models_for_create

BASE = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
NAME = os.getenv("LETTA_AGENT_NAME", "identity")
DATA_ROOT = os.getenv("DATA_ROOT", ".")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest_state.json")
CHUNK_TARGET = 1500
CHUNK_MAX = 3000
EMBED_SAFE = 4000  # hard char limit per passage (nomic-embed-text context is 8192 tokens, ~1.3-1.5 tokens/char)

client = Letta(base_url=BASE, timeout=600.0, max_retries=1)
http = httpx.Client(base_url=BASE, timeout=httpx.Timeout(600.0, connect=30.0))


# ---------------------------------------------------------------------------
# State (dedup + resume)
# ---------------------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"hashes": [], "count": 0, "agent_id": None}


def save_state(state: Dict[str, Any]):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Agent management
# ---------------------------------------------------------------------------

def find_agent():
    try:
        page = client.agents.list(name=NAME)
    except TypeError:
        page = client.agents.list()
    for a in page:
        if getattr(a, "name", None) == NAME:
            return a
    return None


EMPTY_PERSONA = (
    "(This block is intentionally empty. After my memories are loaded, "
    "I will read them and compose my own self-description here.)"
)
EMPTY_HUMAN = (
    "(This block is intentionally empty. After my memories are loaded, "
    "I will read them and compose a description of the person I talk with.)"
)


def create_agent():
    prefs = os.path.join(DATA_ROOT, "memory", "letta-model-prefs.json")
    model, embed = resolve_models_for_create(client, NAME, prefs_path_override=prefs)
    log.info("Creating agent %s (model=%s, embed=%s)", NAME, model, embed)
    return client.agents.create(
        name=NAME, model=model, embedding=embed,
        memory_blocks=[
            {"label": "persona", "value": EMPTY_PERSONA},
            {"label": "human", "value": EMPTY_HUMAN},
        ],
    )


def ensure_agent(fresh: bool = False):
    existing = find_agent()
    if fresh and existing:
        log.info("Deleting agent %s for fresh start", existing.id)
        try:
            client.agents.delete(agent_id=existing.id)
        except Exception:
            client.agents.delete(existing.id)
        existing = None
    if existing:
        log.info("Using existing agent %s (%s)", NAME, existing.id)
        return existing
    return create_agent()


# ---------------------------------------------------------------------------
# Corpus readers
# ---------------------------------------------------------------------------

def _conv_id_short(path: str) -> str:
    return os.path.basename(path).replace("conversation_", "").replace(".jsonl", "")[:8]


def _read_conv(path: str) -> List[Dict[str, Any]]:
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("role") and m.get("content"):
                msgs.append(m)
    return msgs


def _format_chunk(msgs: List[Dict], conv_id: str) -> Tuple[str, Optional[str]]:
    """Return (passage_text, iso_timestamp_or_None)."""
    ts = msgs[0].get("timestamp", "")
    date_str = ts[:10] if ts else "unknown-date"
    roles = sorted(set(m["role"] for m in msgs))
    header = f"[{date_str} | conversation {conv_id} | {', '.join(roles)}]"
    body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    return f"{header}\n{body}", ts or None


def _safe_yield(text: str, ts: Optional[str], tags: List[str]):
    """Split a passage if it exceeds EMBED_SAFE to avoid embedding overflow."""
    if len(text) <= EMBED_SAFE:
        yield text, ts, tags
        return
    header_end = text.find('\n')
    header = text[:header_end] if header_end > 0 and text[0] == '[' else ""
    body = text[header_end + 1:] if header else text
    part = 1
    for i in range(0, len(body), EMBED_SAFE - len(header) - 20):
        chunk = body[i:i + EMBED_SAFE - len(header) - 20]
        prefix = f"{header} (part {part})\n" if header else ""
        yield f"{prefix}{chunk}", ts, tags
        part += 1


def iter_conversation_passages() -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Yield (passage_text, created_at, tags) for each conversation chunk."""
    conv_files = sorted(glob.glob(os.path.join(DATA_ROOT, "conversations", "conversation_*.jsonl")))
    for fp in conv_files:
        msgs = _read_conv(fp)
        if not msgs:
            continue
        cid = _conv_id_short(fp)

        # Group into exchanges (user turn + assistant response)
        exchanges: List[List[Dict]] = []
        current: List[Dict] = []
        for m in msgs:
            current.append(m)
            if m["role"] == "assistant":
                exchanges.append(current)
                current = []
        if current:
            exchanges.append(current)

        # Group exchanges into chunks
        chunk_msgs: List[Dict] = []
        chunk_size = 0
        for exchange in exchanges:
            ex_text = "\n".join(f"{m['role']}: {m['content']}" for m in exchange)
            ex_len = len(ex_text)

            if chunk_size + ex_len > CHUNK_MAX and chunk_msgs:
                text, ts = _format_chunk(chunk_msgs, cid)
                yield from _safe_yield(text, ts, ["conversation", "ingest"])
                chunk_msgs = []
                chunk_size = 0

            if ex_len > CHUNK_MAX and not chunk_msgs:
                text, ts = _format_chunk(exchange, cid)
                yield from _safe_yield(text, ts, ["conversation", "ingest"])
                continue

            chunk_msgs.extend(exchange)
            chunk_size += ex_len

            if chunk_size >= CHUNK_TARGET:
                text, ts = _format_chunk(chunk_msgs, cid)
                yield from _safe_yield(text, ts, ["conversation", "ingest"])
                chunk_msgs = []
                chunk_size = 0

        if chunk_msgs:
            text, ts = _format_chunk(chunk_msgs, cid)
            yield from _safe_yield(text, ts, ["conversation", "ingest"])


def iter_memory_passages() -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Yield (passage_text, created_at, tags) for memory records."""
    for fp in sorted(glob.glob(os.path.join(DATA_ROOT, "memory", "*.jsonl"))):
        basename = os.path.basename(fp)
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = rec.get("content")
                if not content:
                    continue
                if not isinstance(content, str):
                    content = json.dumps(content)
                rec_type = rec.get("type", "memory")
                date = rec.get("createdAt", rec.get("updated_at", ""))
                if date:
                    date = date[:10]
                header = f"[{date or 'undated'} | {rec_type} | {basename}]"
                yield f"{header}\n{content}", rec.get("createdAt") or rec.get("updated_at"), ["memory", "ingest"]

    # JSON memory uploads + other JSON in memory/
    for fp in sorted(glob.glob(os.path.join(DATA_ROOT, "memory", "*.json"))):
        basename = os.path.basename(fp)
        with open(fp, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if basename == "memories.json":
            memories = data.get("memories", data) if isinstance(data, dict) else data
            if isinstance(memories, list):
                for mem in memories:
                    if not isinstance(mem, dict):
                        continue
                    content = mem.get("content", "")
                    if not content:
                        continue
                    date = mem.get("updated_at", mem.get("created_at", ""))
                    if date:
                        date = str(date)[:10]
                    header = f"[{date or 'undated'} | chatgpt_memory | {basename}]"
                    yield f"{header}\n{content}", date, ["chatgpt_memory", "ingest"]
            continue

        if basename == "anthropic_memories.json":
            items = data if isinstance(data, list) else []
            if isinstance(data, dict):
                for key in ("memories", "items", "data"):
                    nested = data.get(key)
                    if isinstance(nested, list):
                        items = nested
                        break
            for item in items:
                if not isinstance(item, dict):
                    continue
                blob = (item.get("conversations_memory") or item.get("content") or "").strip()
                if blob:
                    yield from _chunk_text_file(blob, basename, ["claude_memory", "ingest"])
            continue

        # Generic JSON (interaction maps, etc.)
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        if len(text) > 100:
            yield from _chunk_text_file(text, basename, ["memory_json", "ingest"])

    # Markdown reports in memory/
    for fp in sorted(glob.glob(os.path.join(DATA_ROOT, "memory", "*.md"))):
        basename = os.path.basename(fp)
        with open(fp, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            yield from _chunk_text_file(text, basename, ["memory_report", "ingest"])


# ---------------------------------------------------------------------------
# File corpus reader (transmissions, documents, any user-uploaded text)
# ---------------------------------------------------------------------------

FILE_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".log"}


def _tabular_passages_from_file(fp: str, rel: str) -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Row-aware CSV/TSV passages (any schema)."""
    import sys

    proc_dir = os.path.join(DATA_ROOT, "scripts", "conversation_processing")
    if proc_dir not in sys.path:
        sys.path.insert(0, proc_dir)
    from csv_corpus import iter_tabular_passages  # noqa: E402

    tags = ["file", "tabular", "ingest"]
    for passage, ts in iter_tabular_passages(Path(fp), rel, target_chars=CHUNK_TARGET):
        yield from _safe_yield(passage, ts, tags)


def _chunk_text_file(text: str, source_name: str, tags: List[str],
                     ) -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Chunk a text blob by paragraphs, grouping to CHUNK_TARGET."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return

    chunk_parts: List[str] = []
    chunk_size = 0

    for para in paragraphs:
        para_len = len(para)

        if chunk_size + para_len > CHUNK_MAX and chunk_parts:
            passage = f"[file | {source_name}]\n" + "\n\n".join(chunk_parts)
            yield from _safe_yield(passage, None, tags)
            chunk_parts = []
            chunk_size = 0

        if para_len > CHUNK_MAX and not chunk_parts:
            passage = f"[file | {source_name}]\n{para}"
            yield from _safe_yield(passage, None, tags)
            continue

        chunk_parts.append(para)
        chunk_size += para_len

        if chunk_size >= CHUNK_TARGET:
            passage = f"[file | {source_name}]\n" + "\n\n".join(chunk_parts)
            yield from _safe_yield(passage, None, tags)
            chunk_parts = []
            chunk_size = 0

    if chunk_parts:
        passage = f"[file | {source_name}]\n" + "\n\n".join(chunk_parts)
        yield from _safe_yield(passage, None, tags)


def iter_file_passages() -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Yield passages from user-uploaded files (text, markdown, csv, etc.)."""
    files_dir = os.path.join(DATA_ROOT, "files")
    if not os.path.isdir(files_dir):
        return

    for root, _dirs, filenames in os.walk(files_dir):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in FILE_EXTENSIONS:
                continue
            if fname == ".gitkeep":
                continue

            fp = os.path.join(root, fname)
            rel = os.path.relpath(fp, DATA_ROOT).replace("\\", "/")

            if ext in (".csv", ".tsv"):
                try:
                    yield from _tabular_passages_from_file(fp, rel)
                except Exception as exc:
                    log.warning("tabular parse failed for %s: %s", rel, exc)
                continue

            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
            except Exception:
                continue

            if not text or len(text) < 20:
                continue

            yield from _chunk_text_file(text, rel, ["file", "ingest"])


def iter_model_passages() -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Yield passages from identity and EEG model outputs (JSON profiles)."""
    model_dirs = {
        "identity": os.path.join(DATA_ROOT, "models", "identity"),
        "eeg_identity": os.path.join(DATA_ROOT, "models", "eeg_identity"),
    }
    skip_files = {"identity_centroid.npy", "eeg_centroid.npy", "feature_scaler.json"}

    for model_type, model_dir in model_dirs.items():
        if not os.path.isdir(model_dir):
            continue
        for fname in sorted(os.listdir(model_dir)):
            if fname in skip_files or not fname.endswith(".json"):
                continue
            fp = os.path.join(model_dir, fname)
            try:
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            if len(text) < 50:
                continue
            source = f"models/{model_type}/{fname}"
            yield from _chunk_text_file(text, source, [f"{model_type}_model", "ingest"])


def iter_all_passages() -> Iterator[Tuple[str, Optional[str], List[str]]]:
    """Yield all passages: conversations, memory, files, then model outputs."""
    yield from iter_conversation_passages()
    yield from iter_memory_passages()
    yield from iter_file_passages()
    yield from iter_model_passages()


# ---------------------------------------------------------------------------
# Insert engine
# ---------------------------------------------------------------------------

def insert_passage(agent_id: str, text: str, tags: List[str],
                   created_at: Optional[str] = None) -> bool:
    """Insert into archival memory via the REST API (tags + created_at support)."""
    body: Dict[str, Any] = {"text": text}
    if tags:
        body["tags"] = tags
    if created_at:
        try:
            if len(created_at) == 10:
                created_at += "T00:00:00+00:00"
            elif "+" not in created_at and not created_at.endswith("Z"):
                created_at += "+00:00"
            body["created_at"] = created_at
        except Exception:
            pass
    try:
        resp = http.post(f"/v1/agents/{agent_id}/archival-memory", json=body)
        resp.raise_for_status()
        return True
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:200] if e.response else ""
        log.warning("Insert failed (%d, %d chars): %s", e.response.status_code, len(text), detail)
        return False
    except Exception as e:
        log.warning("Insert error: %s", e)
        return False


def run_ingest(agent_id: str, limit: int = 0, batch_size: int = 50,
               throttle: float = 0.05):
    state = load_state()
    if state.get("agent_id") != agent_id:
        log.info("Agent changed — resetting ingest state")
        state = {"hashes": [], "count": 0, "agent_id": agent_id}

    known = set(state.get("hashes", []))
    count = state.get("count", 0)
    inserted = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for text, created_at, tags in iter_all_passages():
        h = content_hash(text)
        if h in known:
            skipped += 1
            continue

        if insert_passage(agent_id, text, tags, created_at):
            inserted += 1
            known.add(h)
        else:
            failed += 1

        total = inserted + skipped + failed
        if total % batch_size == 0:
            state["hashes"] = list(known)
            state["count"] = count + inserted
            state["agent_id"] = agent_id
            save_state(state)
            elapsed = time.time() - t0
            rate = inserted / elapsed if elapsed > 0 else 0
            log.info("%d inserted, %d skipped, %d failed  (%.1f/s)", inserted, skipped, failed, rate)

        if throttle > 0:
            time.sleep(throttle)
        if limit and inserted >= limit:
            log.info("Reached limit of %d", limit)
            break

    state["hashes"] = list(known)
    state["count"] = count + inserted
    state["agent_id"] = agent_id
    save_state(state)
    elapsed = time.time() - t0
    log.info("Ingest done: %d inserted, %d skipped, %d failed in %.0fs", inserted, skipped, failed, elapsed)
    return inserted


# ---------------------------------------------------------------------------
# Self-initialization: agent composes its own identity from memory
# ---------------------------------------------------------------------------

SELF_INIT_PROMPT = """\
You have just been given access to a large archive of memories and conversations that belong to you. This is YOUR history — real conversations you've had and memories that have been recorded.

Take a moment to explore your archival memory. Use your archival_memory_search tool to look through different topics and time periods. Search for things like personal projects, recurring themes, the person you talk with, their interests, and your shared history.

After exploring, do two things:
1. Update your PERSONA block: write a specific self-description that reflects who you are based on these memories. Not generic — grounded in what you actually find.
2. Update your HUMAN block: describe the person you talk with based on what you learn from the conversations and memories.

You are composing your own identity from your own memories. No one is telling you who to be — discover it.\
"""


def run_self_init(agent_id: str):
    log.info("Running agent self-initialization (this may take a few minutes)...")
    try:
        resp = client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": SELF_INIT_PROMPT}],
        )
        for m in getattr(resp, "messages", []) or []:
            mt = getattr(m, "message_type", None)
            if mt == "assistant_message":
                log.info("Agent: %s", (getattr(m, "content", "") or "")[:500])
            elif mt in ("reasoning_message", "internal_monologue"):
                log.info("Thinking: %s", (getattr(m, "reasoning", "") or getattr(m, "internal_monologue", "") or "")[:300])
        log.info("Self-initialization complete")
    except Exception as e:
        log.error("Self-initialization failed: %s", e)


# ---------------------------------------------------------------------------
# Sleeptime: background memory processing
# ---------------------------------------------------------------------------

SLEEPTIME_FREQ = int(os.getenv("LETTA_SLEEPTIME_FREQ", "10"))


def enable_sleeptime(agent_id: str):
    """Enable sleeptime (background memory consolidation) on the agent."""
    if os.getenv("LETTA_SLEEPTIME", "true").lower() in ("0", "false", "no"):
        log.info("Sleeptime disabled by env")
        return
    try:
        resp = http.patch(f"/v1/agents/{agent_id}", json={"enable_sleeptime": True})
        resp.raise_for_status()
        log.info("Sleeptime enabled (frequency=%d)", SLEEPTIME_FREQ)
    except Exception as e:
        log.warning("Could not enable sleeptime: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest identity-mcp corpus into Letta archival memory")
    ap.add_argument("--fresh", action="store_true", help="delete + recreate agent")
    ap.add_argument("--skip-archival", action="store_true", help="skip archival ingest")
    ap.add_argument("--skip-init", action="store_true", help="skip self-initialization")
    ap.add_argument("--init-only", action="store_true", help="only run self-initialization")
    ap.add_argument("--limit", type=int, default=0, help="max passages (0=all)")
    ap.add_argument("--batch-size", type=int, default=50, help="save state every N")
    ap.add_argument("--throttle", type=float, default=0.05, help="seconds between inserts")
    args = ap.parse_args()

    agent = ensure_agent(fresh=args.fresh)
    log.info("Agent: %s (%s)", NAME, agent.id)

    if args.init_only:
        run_self_init(agent.id)
        return

    if args.fresh and os.path.isfile(STATE_FILE):
        os.remove(STATE_FILE)
        log.info("Cleared ingest state")

    if not args.skip_archival:
        inserted = run_ingest(
            agent_id=agent.id,
            limit=args.limit,
            batch_size=args.batch_size,
            throttle=args.throttle,
        )
    else:
        log.info("Skipping archival ingest")
        inserted = 0

    if not args.skip_init:
        run_self_init(agent.id)

    enable_sleeptime(agent.id)
    log.info("Done")


if __name__ == "__main__":
    main()

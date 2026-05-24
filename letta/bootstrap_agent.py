#!/usr/bin/env python3
"""
Bootstrap / enrich the Letta "identity" agent from the identity-mcp corpus.

  1. ensure the agent exists (Ollama LLM + embedding handles)
  2. seed the persona core-memory block from memory/identity.jsonl
  3. backfill archival memory from conversations/*.jsonl + memory/*.jsonl
     (this is what makes the agent recall your real history ambiently)

Run on the HOST (needs the data dirs) against a running Letta server:
    pip install letta-client
    DATA_ROOT=/path/to/data LETTA_BASE_URL=http://localhost:8283 \
      LETTA_MODEL=ollama/qwen3:32b python letta/bootstrap_agent.py

Flags:
    --skip-archival     only ensure agent + seed persona (fast)
    --limit N           cap archival passages (0 = all; useful for a smoke test)

Idempotent for the agent + persona. Archival re-inserts on re-run, so use
--skip-archival (or --limit) when you just want to refresh the persona.

NOTE: letta-client method names have shifted across versions; the calls below
include fallbacks. If one errors, check `pip show letta-client` and the docs.
"""
import os
import glob
import json
import argparse
from letta_client import Letta

BASE = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
NAME = os.getenv("LETTA_AGENT_NAME", "identity")
MODEL = os.getenv("LETTA_MODEL", "ollama/qwen3:32b")
EMBED = os.getenv("LETTA_EMBEDDING", "ollama/nomic-embed-text")
DATA_ROOT = os.getenv("DATA_ROOT", ".")

client = Letta(base_url=BASE)


def ensure_agent():
    # letta-client >=0.16 returns SyncArrayPage (iterable, not a plain list) — normalize.
    try:
        page = client.agents.list(name=NAME)
    except TypeError:
        page = client.agents.list()
    matched = [a for a in page if getattr(a, "name", None) == NAME]
    if matched:
        print(f"agent exists: {matched[0].id}")
        return matched[0]
    print(f"creating agent {NAME} (model={MODEL}, embed={EMBED})")
    return client.agents.create(
        name=NAME, model=MODEL, embedding=EMBED,
        memory_blocks=[
            {"label": "persona",
             "value": "I am a persistent identity that remembers and notices itself over time."},
            {"label": "human", "value": "(unknown so far)"},
        ],
    )


def load_identity_text():
    path = os.path.join(DATA_ROOT, "memory", "identity.jsonl")
    if not os.path.isfile(path):
        return None
    facts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("content")
            if c:
                facts.append(c if isinstance(c, str) else json.dumps(c))
    if not facts:
        return None
    return "Who I am (from my identity record):\n" + "\n".join(f"- {x}" for x in facts[:50])


def update_persona(agent, text):
    for attempt in (
        lambda: client.agents.blocks.modify(agent_id=agent.id, block_label="persona", value=text),
        lambda: client.agents.core_memory.modify_block(agent_id=agent.id, block_label="persona", value=text),
        lambda: client.agents.core_memory.update(agent_id=agent.id, label="persona", value=text),
    ):
        try:
            attempt()
            return True
        except Exception:  # noqa: BLE001
            continue
    print("  ! could not update persona block (check letta-client version / method name)")
    return False


def iter_passages():
    # Conversations -> chunked passages
    for fp in sorted(glob.glob(os.path.join(DATA_ROOT, "conversations", "conversation_*.jsonl"))):
        msgs = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r, c = m.get("role"), m.get("content")
                if r and c:
                    msgs.append(f"{r}: {c}")
        if msgs:
            convo = "\n".join(msgs)
            for i in range(0, len(convo), 2000):
                yield convo[i:i + 2000]
    # Memory records -> one passage each
    for fp in sorted(glob.glob(os.path.join(DATA_ROOT, "memory", "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = rec.get("content")
                if c:
                    yield c if isinstance(c, str) else json.dumps(c)


def insert_passage(agent, text):
    for attempt in (
        lambda: client.agents.passages.create(agent_id=agent.id, text=text),
        lambda: client.agents.archival_memory.insert(agent_id=agent.id, content=text),
    ):
        try:
            attempt()
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-archival", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max archival passages (0=all)")
    args = ap.parse_args()

    agent = ensure_agent()
    print(f"agent: {agent.id} ({NAME})")

    idtext = load_identity_text()
    if idtext:
        if update_persona(agent, idtext):
            print("persona seeded from memory/identity.jsonl")
    else:
        print("no memory/identity.jsonl found — leaving default persona")

    if args.skip_archival:
        print("skipping archival backfill (--skip-archival)")
        return

    n, fail = 0, 0
    for p in iter_passages():
        if insert_passage(agent, p):
            n += 1
        else:
            fail += 1
        if (n + fail) % 50 == 0:
            print(f"  inserted {n} passages ({fail} failed)...")
        if args.limit and n >= args.limit:
            break
    print(f"archival backfill complete: {n} inserted, {fail} failed")


if __name__ == "__main__":
    main()

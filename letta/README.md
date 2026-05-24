# Letta memory layer

Adds **Letta** (stateful agent with self-editing memory) to the stack so memory feels
*ambient* — the agent maintains a persona it reflects on/edits and recalls your history
inline, instead of memory being an MCP tool the model has to choose to call.

```
LibreChat ──/v1/chat/completions──▶ letta-bridge ──▶ Letta agent ──▶ Ollama (inference + embeddings)
                                                          │
                                                          ├──▶ Postgres + pgvector (agent memory)
                                                          └──▶ identity-mcp tools (optional, follow-up)
```

**Design A:** one model does both chat + memory (no separate memory model → no VRAM
contention). Pick a reliable native-tool-calling driver (`qwen3:32b`, `llama3.3:70b`,
`gpt-oss:20b`, or your `gpt-oss-20b-finetuned`) — **not** a Q2 ARK model.

## Components
- `letta-bridge/` — Async FastAPI OpenAI-compat shim with true streaming. Forwards only the
  latest user turn to a stateful Letta agent (Letta keeps its own recall memory). Streams
  token-by-token via Letta's SSE endpoint; reasoning/internal monologue is surfaced as
  `<think>` blocks that LibreChat renders as collapsible thinking sections.
  `app.py`, `Dockerfile`, `requirements.txt`.
- `bootstrap_agent.py` — creates/enriches the `identity` agent: seeds persona from
  `memory/identity.jsonl`, backfills archival memory from `conversations/*.jsonl` + `memory/*.jsonl`.
- `docker-compose.letta.yml` — portable fragment: `letta`, `letta-postgres`, `letta-bridge`
  (profile `letta`). Kept separate from the main `docker-compose.yml` so it runs merged on
  the server OR standalone on localhost.

## Prereqs (in the Ollama that Letta points at)
```
ollama pull nomic-embed-text     # embeddings (required by Letta)
# driver model already present, e.g. qwen3:32b
```

## Deploy — all on server (merge with the main stack)
```
cd ~/ai
docker compose -f docker-compose.yml -f letta/docker-compose.letta.yml \
  --profile ollama --profile identity --profile letta up -d --build
```

## Deploy — localhost, pointing at remote Ollama
```
OLLAMA_BASE_URL=http://<server-ip>:11434/v1 \
LETTA_MODEL=ollama/qwen3:32b \
docker compose -f letta/docker-compose.letta.yml --profile letta up -d --build
```
(LibreChat can run in the same local compose; it reaches `letta-bridge:8284` over the shared network.)

## Bootstrap the agent (seed persona + backfill history)
```
pip install letta-client
DATA_ROOT=/path/to/data LETTA_BASE_URL=http://localhost:8283 LETTA_MODEL=ollama/qwen3:32b \
  python letta/bootstrap_agent.py --limit 200      # smoke test with 200 passages first
# then full:
DATA_ROOT=/path/to/data python letta/bootstrap_agent.py
```
(The bridge auto-creates a minimal agent on first request too, but bootstrap is what gives it
your persona + real history.)

## Ingest corpus into archival memory
```
LETTA_BASE_URL=http://localhost:8283 LETTA_AGENT_NAME=identity DATA_ROOT=/path/to/data \
  python letta/ingest.py --limit 200      # smoke test
# full run (may take hours for large corpora):
LETTA_BASE_URL=http://localhost:8283 LETTA_AGENT_NAME=identity DATA_ROOT=/path/to/data \
  python letta/ingest.py
```
Ingests conversations, memory files, file corpus, and identity/EEG model outputs into
Letta's pgvector-backed archival memory. Deduplicates on re-run. Use `--fresh` to wipe
and re-ingest.

## Register identity-mcp tools on the agent
```
LETTA_BASE_URL=http://localhost:8283 MCP_SERVER_URL=http://mcp-server:4000 \
  python letta/register_tools.py
# remove tools:
python letta/register_tools.py --clean
```
Mounts **10** identity-mcp tools on the Letta agent (plus Letta's built-in memory/conversation tools):
`identity_search_corpus`, `identity_search_all`, `identity_memory_search`,
`identity_file_search`, `identity_conversation_get`, `identity_get_profile`,
`identity_analysis_summary`, `identity_interaction_summary`, `identity_get_momentum`,
`identity_model_status`.

Re-run after changing `register_tools.py` (idempotent: updates source, attaches new tools).

Or use **Memory → Maintenance** in the dashboard (same run/output UI as Pipeline).

## Maintenance: what is manual vs automatic?

| Task | Who does it | When |
|------|-------------|------|
| **Archival ingest** (`ingest.py`) | **You** (batch job) | New bulk data: conversations export, memory upload, files corpus, re-embed into pgvector. Dedupes on re-run. **Not** automatic when you add files in the dashboard. |
| **Bootstrap** (`bootstrap_agent.py`) | **You** | First-time agent + persona seed from `memory/identity.jsonl` + optional archival backfill. Use `--skip-archival` to refresh persona only. |
| **Register tools** (`register_tools.py`) | **You** | After changing tool wrappers or first deploy. |
| **Pipeline scripts** (dashboard) | **You** | Parse conversations, `analyze_identity.py`, `train_identity_model.py` — updates **disk** JSONL/models, not Letta archival. |
| **Letta sleeptime** | **Agent** (automatic) | Every N messages: consolidates thread into core/archival via `memory_insert` / `memory_replace` and built-in search. |
| **Self-init** (`ingest.py --init-only`) | **One-shot prompt** | Agent explores archival and rewrites persona/human — optional after ingest. |
| **Live chat recall** | **Agent** (automatic) | Embeddings + core blocks; may call identity-mcp tools for files/conversations/disk memory. |

**You do not need to re-bootstrap** for normal use. Re-run **ingest** when the on-disk corpus grows significantly and you want that material in **pgvector archival**. The agent can search **files** and **conversations** on disk via `identity_search_all` / `identity_file_search` without re-ingest, but archival semantic search only covers ingested passages.

**Typical refresh cadence**
1. Upload new data in dashboard (or drop files in `conversations/`, `memory/`, `files/`).
2. Run pipeline scripts if you want updated identity analysis / verification model.
3. `python letta/ingest.py` (incremental; skips content hashes already inserted).
4. Optional: `register_tools.py` after tool changes.
5. Let sleeptime run — no scheduled job required.

## Use it
In LibreChat, pick the **"Identity (Letta memory)"** model spec (endpoint `Letta`).
Success criterion: ask about something only in your past conversations — it recalls
**without** an explicit tool call.

## Env vars
| var | default | purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434/v1` | where Letta runs inference + embeddings |
| `LETTA_MODEL` | `ollama/qwen3:32b` | agent driver (chat + memory) |
| `LETTA_EMBEDDING` | `ollama/nomic-embed-text` | archival embeddings |
| `LETTA_AGENT_NAME` | `identity` | the agent the bridge talks to |
| `LETTA_BASE_URL` | `http://letta:8283` | bridge → Letta server |
| `LETTA_STREAM_TIMEOUT` | `600` | seconds; long for cold-loaded big models on Pascal |

## Known limitations / follow-ups
- **One continuous identity.** A single agent with persistent memory is the whole point. Every
  inbound turn appends to its timeline. Edit/branch/regenerate in LibreChat are simply
  *remembered* — divergences become part of the conversation history, truer to memory than
  "unseeing" them. Multi-user (user→agent map) is a later config knob, not now.


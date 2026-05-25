#!/usr/bin/env python3
"""
letta-bridge: an OpenAI-compatible shim in front of a stateful Letta agent.

LibreChat (or any OpenAI client) calls /v1/chat/completions here. We forward ONLY
the latest user turn to a Letta agent -- Letta is stateful and keeps its own recall
memory, so replaying the whole history each turn would double-count. The agent's
self-editing memory (persona/human core blocks + archival) is what makes recall feel
ambient ("it just knows") instead of a tool the model has to call.

Design A: the agent's single model does both chat + memory. Pick a reliable
native-tool-calling driver (qwen3:32b / llama3.3:70b / gpt-oss:20b) -- NOT a Q2.

One continuous identity: a single agent with persistent memory IS the whole point.
Every inbound turn appends to its timeline. Edit/branch/regenerate in LibreChat are
simply remembered -- divergences become part of the conversation history, truer to
memory than "unseeing" them. Multi-user (user->agent map) is a later config knob.

Streaming: calls Letta's SSE streaming endpoint (stream_steps + stream_tokens) and
translates to OpenAI-compatible SSE. Reasoning / internal monologue from the agent
is surfaced as <think> blocks that LibreChat renders as collapsible thinking sections.
Memory tool calls (archival_memory_search, core_memory_append, etc.) appear as brief
annotations inside the thinking block for observability.

Env:
  LETTA_BASE_URL         default http://letta:8283
  LETTA_AGENT_NAME       default "identity"
  LETTA_AGENT_ID         optional explicit id (overrides name lookup)
  LETTA_MODEL            default ollama/qwen3:32b
  LETTA_EMBEDDING        default ollama/nomic-embed-text:latest
  MODEL_ID               default "letta-identity"
  PORT                   default 8284
  LETTA_STREAM_TIMEOUT   default 600 (seconds; long for cold-loaded big models)
"""
import os
import re
import time
import json
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from letta_client import Letta

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("letta-bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://letta:8283")
AGENT_NAME = os.getenv("LETTA_AGENT_NAME", "identity")
AGENT_ID_ENV = os.getenv("LETTA_AGENT_ID")
LETTA_MODEL = os.getenv("LETTA_MODEL", "ollama/qwen3:32b")
LETTA_EMBEDDING = os.getenv("LETTA_EMBEDDING", "ollama/nomic-embed-text:latest")
MODEL_ID = os.getenv("MODEL_ID", "letta-identity")
STREAM_TIMEOUT = float(os.getenv("LETTA_STREAM_TIMEOUT", "600"))

# Sync SDK client for agent management (create, list, health).
client = Letta(base_url=LETTA_BASE_URL, timeout=STREAM_TIMEOUT, max_retries=0)
_agent_id: Optional[str] = AGENT_ID_ENV
_http: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http
    _http = httpx.AsyncClient(timeout=httpx.Timeout(STREAM_TIMEOUT, connect=30.0))
    yield
    await _http.aclose()


app = FastAPI(title="letta-bridge", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Agent management (sync SDK -- called via asyncio.to_thread from async paths)
# ---------------------------------------------------------------------------

def ensure_agent() -> str:
    """Resolve the agent by name (or LETTA_AGENT_ID); create a minimal one if missing.
    Ingest (ingest.py) enriches persona + archival later."""
    global _agent_id
    if _agent_id:
        return _agent_id
    try:
        page = client.agents.list(name=AGENT_NAME)
    except TypeError:
        page = client.agents.list()
    matched = [a for a in page if getattr(a, "name", None) == AGENT_NAME]
    if matched:
        _agent_id = matched[0].id
        log.info("Using existing agent %s (%s)", AGENT_NAME, _agent_id)
        return _agent_id
    log.info("Creating agent %s (model=%s, embed=%s)", AGENT_NAME, LETTA_MODEL, LETTA_EMBEDDING)
    agent = client.agents.create(
        name=AGENT_NAME,
        model=LETTA_MODEL,
        embedding=LETTA_EMBEDDING,
        memory_blocks=[
            {"label": "persona",
             "value": "I am a persistent identity. I remember our past conversations and "
                      "notice my own continuity over time."},
            {"label": "human",
             "value": "(Learning about the person I'm talking with.)"},
        ],
    )
    _agent_id = agent.id
    return _agent_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _extract_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "".join(
            p if isinstance(p, str)
            else (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") or "")
            for p in obj
        )
    return ""


_THINK_RE = re.compile(r'</?think(?:ing)?>')


def _strip_think_tags(text: str) -> str:
    return _THINK_RE.sub('', text)


# ---------------------------------------------------------------------------
# OpenAI SSE chunk helper
# ---------------------------------------------------------------------------

def _oai_chunk(cid: str, created: int, delta: Dict[str, Any],
               finish: Optional[str] = None) -> str:
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk",
        "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


# ---------------------------------------------------------------------------
# Streaming path: Letta SSE --> OpenAI SSE with <think> blocks
# ---------------------------------------------------------------------------

async def _stream_letta(agent_id: str, user_text: str) -> AsyncGenerator[str, None]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    yield _oai_chunk(cid, created, {"role": "assistant"})

    in_think = False
    has_assistant = False

    url = f"{LETTA_BASE_URL}/v1/agents/{agent_id}/messages/stream"
    body = {
        "messages": [{"role": "user", "content": user_text}],
        "stream_tokens": True,
    }

    try:
        async with _http.stream("POST", url, json=body) as resp:
            ct = resp.headers.get("content-type", "")

            if resp.status_code != 200:
                err = (await resp.aread()).decode(errors="replace")
                log.error("Letta stream error %d: %s", resp.status_code, err[:500])
                yield _oai_chunk(cid, created, {"content": f"[Letta error {resp.status_code}]"})
                yield _oai_chunk(cid, created, {}, finish="stop")
                yield "data: [DONE]\n\n"
                return

            # If Letta returned plain JSON instead of SSE, extract and emit as one chunk.
            if "text/event-stream" not in ct:
                raw = (await resp.aread()).decode(errors="replace")
                try:
                    data = json.loads(raw)
                    reasoning_parts, text_parts = [], []
                    for m in data.get("messages", []):
                        mt = m.get("message_type", "")
                        if mt in ("reasoning_message", "internal_monologue"):
                            r = m.get("reasoning", "") or m.get("internal_monologue", "")
                            if r:
                                reasoning_parts.append(_strip_think_tags(r))
                        elif mt == "assistant_message":
                            text_parts.append(_extract_text(
                                m.get("assistant_message", "") or m.get("content", "")
                            ))
                    reasoning = "\n".join(reasoning_parts).strip()
                    text = "".join(text_parts).strip() or "(no response)"
                    if reasoning:
                        text = f"<think>\n{reasoning}\n</think>\n\n{text}"
                except (json.JSONDecodeError, AttributeError):
                    text = raw[:500] if raw else "(empty response)"
                yield _oai_chunk(cid, created, {"content": text})
                yield _oai_chunk(cid, created, {}, finish="stop")
                yield "data: [DONE]\n\n"
                return

            # --- Parse Letta SSE events ---
            # Letta can go silent for minutes while Ollama processes a
            # large prompt.  We pipe lines through an asyncio.Queue so
            # the main loop can emit SSE keepalive comments every 15s
            # without cancelling the httpx read (asyncio.wait_for on
            # an async-iterator __anext__ corrupts the stream).
            KEEPALIVE_INTERVAL = 15  # seconds
            line_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            async def _sse_reader():
                try:
                    async for raw_line in resp.aiter_lines():
                        await line_queue.put(raw_line)
                except Exception as exc:
                    log.warning("SSE reader error: %s", exc)
                finally:
                    await line_queue.put(None)  # sentinel

            reader_task = asyncio.create_task(_sse_reader())

            while True:
                try:
                    line = await asyncio.wait_for(
                        line_queue.get(), timeout=KEEPALIVE_INTERVAL
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if line is None:
                    break  # reader finished

                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if payload.startswith("[DONE"):
                    continue

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                msg_type = event.get("message_type", "")

                # --- Reasoning / internal monologue ---
                if msg_type in ("reasoning_message", "internal_monologue"):
                    text = _strip_think_tags(
                        event.get("reasoning", "")
                        or event.get("internal_monologue", "")
                    )
                    if not text:
                        continue
                    if not in_think:
                        in_think = True
                        yield _oai_chunk(cid, created, {"content": "<think>\n"})
                    yield _oai_chunk(cid, created, {"content": text})

                # --- Tool calls (show memory operations in thinking) ---
                elif msg_type in ("tool_call_message", "function_call"):
                    fc = event.get("function_call") or event.get("tool_call") or {}
                    name = fc.get("name", "") if isinstance(fc, dict) else str(fc)
                    if name and name != "send_message":
                        if not in_think:
                            in_think = True
                            yield _oai_chunk(cid, created, {"content": "<think>\n"})
                        yield _oai_chunk(cid, created, {"content": f"\n[{name}]\n"})

                # --- Assistant message (the actual response) ---
                elif msg_type == "assistant_message":
                    text = _extract_text(
                        event.get("assistant_message", "")
                        or event.get("content", "")
                    )
                    if not text:
                        continue
                    if in_think:
                        in_think = False
                        yield _oai_chunk(cid, created, {"content": "\n</think>\n\n"})
                    has_assistant = True
                    yield _oai_chunk(cid, created, {"content": text})

            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    except httpx.ReadTimeout:
        log.error("Letta stream timed out after %ss", STREAM_TIMEOUT)
        if in_think:
            yield _oai_chunk(cid, created, {"content": "\n</think>\n\n"})
        yield _oai_chunk(cid, created, {"content": "[response timed out]"})
    except httpx.HTTPError as exc:
        log.exception("Letta stream HTTP error")
        if in_think:
            yield _oai_chunk(cid, created, {"content": "\n</think>\n\n"})
        yield _oai_chunk(cid, created, {"content": f"[connection error: {exc}]"})
    else:
        if in_think:
            yield _oai_chunk(cid, created, {"content": "\n</think>\n\n"})
        if not has_assistant:
            yield _oai_chunk(cid, created, {"content": "(no response)"})

    yield _oai_chunk(cid, created, {}, finish="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Non-streaming path (sync SDK, kept as fallback)
# ---------------------------------------------------------------------------

def _assistant_text(resp) -> str:
    out: List[str] = []
    for m in getattr(resp, "messages", []) or []:
        if getattr(m, "message_type", None) != "assistant_message":
            continue
        c = getattr(m, "content", "")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for p in c:
                t = getattr(p, "text", None)
                if t is None and isinstance(p, dict):
                    t = p.get("text")
                if t:
                    out.append(t)
    return "\n".join(s for s in out if s).strip()


def _reasoning_text(resp) -> str:
    out: List[str] = []
    for m in getattr(resp, "messages", []) or []:
        mt = getattr(m, "message_type", None)
        if mt in ("reasoning_message", "internal_monologue"):
            t = getattr(m, "reasoning", None) or getattr(m, "internal_monologue", "")
            if t:
                out.append(_strip_think_tags(t))
    return "\n".join(out).strip()


def _completion_json(text: str, reasoning: str = "") -> Dict[str, Any]:
    content = f"<think>\n{reasoning}\n</think>\n\n{text}" if reasoning else text
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@app.get("/health")
async def health():
    try:
        aid = await asyncio.to_thread(ensure_agent)
        return {"status": "ok", "agent": aid, "letta": LETTA_BASE_URL}
    except Exception as e:
        raise HTTPException(503, f"letta not ready: {e}")


@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": 0,
                       "owned_by": "letta-bridge"}]}


@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    aid = await asyncio.to_thread(ensure_agent)
    user_text = _last_user_message(req.messages)
    if not user_text:
        raise HTTPException(400, "no user message found in request")

    if req.stream:
        return StreamingResponse(
            _stream_letta(aid, user_text),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        resp = await asyncio.to_thread(
            client.agents.messages.create,
            agent_id=aid,
            messages=[{"role": "user", "content": user_text}],
        )
    except Exception as e:
        log.exception("letta message failed")
        raise HTTPException(502, f"letta error: {e}")

    text = _assistant_text(resp) or "(no assistant message returned)"
    reasoning = _reasoning_text(resp)
    return _completion_json(text, reasoning)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8284")))

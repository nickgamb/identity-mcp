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

Reasoning loop guard: if internal monologue repeats (e.g. "(Writing)."/"(End)." loops),
the bridge closes that reasoning segment and opens a new one for post-tool thinking (see
reasoning_guard.py). Tool calls and returns each get their own collapsible thinking block.

Env:
  LETTA_BASE_URL              default http://letta:8283
  LETTA_AGENT_NAME            default "identity"
  LETTA_AGENT_ID              optional explicit id (overrides name lookup)
  LETTA_MODEL_PREFS_FILE      default /app/memory/letta-model-prefs.json (from dashboard)
  LETTA_MODEL / LETTA_EMBEDDING  optional override for new-agent creation only
  MODEL_ID                    default "letta-identity"
  PORT                        default 8284
  LETTA_STREAM_TIMEOUT        default 600; 0 = no HTTP/cancel timeout (unlimited stream)
  LETTA_TOOL_RETURN_MAX_CHARS default 4000 (truncate tool results in thinking UI)
  LETTA_GUARD_LOG_PATH        default /app/memory/bridge-guard-events.jsonl
  LETTA_GUARD_SAMPLE_CHARS    default 500 (raw tail saved when guard fires)
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

from model_prefs import load_prefs_file, models_from_existing_agent, resolve_models_for_create
from guard_log import record_guard_event
from reasoning_guard import (
    LoopReason,
    ReasoningLoopGuard,
    THINKING_CLOSE_NOTE,
    trim_reasoning_for_display,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("letta-bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://letta:8283")
AGENT_NAME = os.getenv("LETTA_AGENT_NAME", "identity")
AGENT_ID_ENV = os.getenv("LETTA_AGENT_ID")
MODEL_ID = os.getenv("MODEL_ID", "letta-identity")
_raw_stream_timeout = os.getenv("LETTA_STREAM_TIMEOUT", "600")
STREAM_TIMEOUT: Optional[float] = (
    None if float(_raw_stream_timeout) <= 0 else float(_raw_stream_timeout)
)
TOOL_RETURN_MAX_CHARS = int(os.getenv("LETTA_TOOL_RETURN_MAX_CHARS", "4000"))

# Sync SDK client for agent management (create, list, health).
_client_timeout = STREAM_TIMEOUT if STREAM_TIMEOUT is not None else None
client = Letta(base_url=LETTA_BASE_URL, timeout=_client_timeout, max_retries=0)
_agent_id: Optional[str] = AGENT_ID_ENV
_http: Optional[httpx.AsyncClient] = None


def _httpx_timeout() -> httpx.Timeout:
    if STREAM_TIMEOUT is None:
        return httpx.Timeout(None, connect=30.0)
    return httpx.Timeout(STREAM_TIMEOUT, connect=30.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http
    _http = httpx.AsyncClient(timeout=_httpx_timeout())
    try:
        ensure_agent()
    except Exception as e:
        log.warning("Agent not ready at startup (will retry on first request): %s", e)
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
        model, embed = models_from_existing_agent(matched[0])
        log.info(
            "Using existing agent %s (%s) model=%s embed=%s",
            AGENT_NAME,
            _agent_id,
            model,
            embed,
        )
        return _agent_id
    model, embed = resolve_models_for_create(client, AGENT_NAME)
    log.info("Creating agent %s (model=%s, embed=%s)", AGENT_NAME, model, embed)
    agent = client.agents.create(
        name=AGENT_NAME,
        model=model,
        embedding=embed,
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


def _truncate_display(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n… ({len(text) - limit} more chars truncated)"


def _format_tool_args(fc: Dict[str, Any]) -> str:
    args = fc.get("arguments") or fc.get("args")
    if args is None:
        return ""
    if isinstance(args, str):
        return _truncate_display(args.strip(), 1200)
    try:
        return _truncate_display(json.dumps(args, indent=2, ensure_ascii=False), 1200)
    except (TypeError, ValueError):
        return _truncate_display(str(args), 1200)


def _format_tool_return(event: Dict[str, Any]) -> tuple[str, str]:
    """Return (status_label, body) for a tool_return_message event."""
    status = event.get("status") or "success"
    if event.get("is_err"):
        status = "error"

    parts: List[str] = []
    tool_returns = event.get("tool_returns")
    if isinstance(tool_returns, list) and tool_returns:
        for tr in tool_returns:
            if not isinstance(tr, dict):
                continue
            st = tr.get("status") or status
            parts.append(_extract_tool_return_value(tr.get("tool_return"), st))
    else:
        parts.append(
            _extract_tool_return_value(event.get("tool_return"), status)
        )

    for stream_name in ("stdout", "stderr"):
        lines = event.get(stream_name)
        if lines:
            label = stream_name.upper()
            body = "\n".join(lines) if isinstance(lines, list) else str(lines)
            parts.append(f"--- {label} ---\n{body}")

    body = "\n\n".join(p for p in parts if p).strip() or "(empty)"
    body = _truncate_display(body, TOOL_RETURN_MAX_CHARS)
    label = "error" if status == "error" else "success"
    return label, body


def _extract_tool_return_value(raw: Any, status: str) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return _extract_text(raw)
    if isinstance(raw, dict):
        return _extract_text(raw.get("text", "")) or json.dumps(raw, ensure_ascii=False)
    return str(raw)


class _ThinkingSegments:
    """One LibreChat collapsible block per reasoning burst or tool round-trip."""

    __slots__ = ("_open", "_kind", "_guard")

    def __init__(self) -> None:
        self._open = False
        self._kind: Optional[str] = None  # "reasoning" | "tool"
        self._guard = ReasoningLoopGuard()

    def _open_tag(self) -> List[str]:
        if self._open:
            return []
        self._open = True
        return ["<think>\n"]

    def _close_tag(self, trailer: str = "") -> List[str]:
        if not self._open:
            return []
        self._open = False
        self._kind = None
        self._guard = ReasoningLoopGuard()
        return [f"{trailer}</think>\n\n"]

    def close_all(self, trailer: str = "") -> List[str]:
        return self._close_tag(trailer)

    def _close_for_new_kind(self, kind: str) -> List[str]:
        if self._open and self._kind != kind:
            return self._close_tag()
        return []

    def feed_reasoning(self, text: str) -> tuple[List[str], bool]:
        """Returns (content chunks, True if this chunk closed the segment due to loop)."""
        out = self._close_for_new_kind("reasoning")
        if not self._open:
            out.extend(self._open_tag())
            self._kind = "reasoning"

        loop_reason = self._guard.feed(text)
        out.append(text)
        closed_loop = False
        if loop_reason is not None:
            note = THINKING_CLOSE_NOTE.get(
                loop_reason,
                "\n\n_(continuing with your reply.)_\n",
            )
            out.extend(self._close_tag(note))
            closed_loop = True
        return out, closed_loop

    def feed_tool_call(self, name: str, fc: Dict[str, Any]) -> List[str]:
        if not name or name == "send_message":
            return []
        out = self._close_tag() if self._open else []
        out.extend(self._open_tag())
        self._kind = "tool"
        out.append(f"**Tool call:** `{name}`\n")
        args_s = _format_tool_args(fc)
        if args_s:
            out.append(f"```json\n{args_s}\n```\n")
        return out

    def feed_tool_return(self, event: Dict[str, Any]) -> List[str]:
        status_label, body = _format_tool_return(event)
        out: List[str] = []
        if not self._open or self._kind != "tool":
            out.extend(self._open_tag())
            self._kind = "tool"
        out.append(f"**Tool result** ({status_label}):\n```\n{body}\n```\n")
        out.extend(self._close_tag())
        return out


def _thinking_blocks_from_messages(messages: List[Any]) -> str:
    """Non-streaming: build segmented thinking blocks in timeline order."""
    blocks: List[str] = []

    def flush_block(parts: List[str]) -> None:
        if not parts:
            return
        blocks.append("<think>\n" + "".join(parts) + "\n</think>")

    reasoning_buf: List[str] = []
    tool_buf: List[str] = []

    def flush_reasoning() -> None:
        nonlocal reasoning_buf
        if not reasoning_buf:
            return
        text = "".join(reasoning_buf)
        reasoning_buf = []
        safe, trimmed, detail = trim_reasoning_for_display(text)
        if trimmed:
            safe += "\n\n_(planning trimmed — reply below)_\n"
            if detail:
                record_guard_event(detail, source="batch")
        flush_block([safe])

    def flush_tool() -> None:
        nonlocal tool_buf
        if not tool_buf:
            return
        flush_block(tool_buf)
        tool_buf = []

    for m in messages:
        if isinstance(m, dict):
            mt = m.get("message_type", "")
            reasoning = m.get("reasoning", "") or m.get("internal_monologue", "")
            assistant = m.get("assistant_message", "") or m.get("content", "")
            fc = m.get("function_call") or m.get("tool_call") or {}
        else:
            mt = getattr(m, "message_type", None)
            reasoning = getattr(m, "reasoning", None) or getattr(m, "internal_monologue", "")
            assistant = getattr(m, "content", "")
            fc = getattr(m, "function_call", None) or getattr(m, "tool_call", None) or {}

        if mt in ("reasoning_message", "internal_monologue"):
            flush_tool()
            t = _strip_think_tags(reasoning or "")
            if t:
                reasoning_buf.append(t)
        elif mt in ("tool_call_message", "function_call"):
            flush_reasoning()
            name = fc.get("name", "") if isinstance(fc, dict) else str(fc)
            if name and name != "send_message":
                tool_buf.append(f"**Tool call:** `{name}`\n")
                args_s = _format_tool_args(fc) if isinstance(fc, dict) else ""
                if args_s:
                    tool_buf.append(f"```json\n{args_s}\n```\n")
        elif mt == "tool_return_message":
            flush_reasoning()
            label, body = _format_tool_return(m if isinstance(m, dict) else {})
            if not isinstance(m, dict):
                label, body = _format_tool_return(
                    {
                        "status": getattr(m, "status", None),
                        "tool_return": getattr(m, "tool_return", None),
                        "tool_returns": getattr(m, "tool_returns", None),
                        "stdout": getattr(m, "stdout", None),
                        "stderr": getattr(m, "stderr", None),
                        "is_err": getattr(m, "is_err", None),
                    }
                )
            tool_buf.append(f"**Tool result** ({label}):\n```\n{body}\n```\n")
            flush_tool()
        elif mt == "assistant_message":
            flush_reasoning()
            flush_tool()

    flush_reasoning()
    flush_tool()
    return "\n\n".join(blocks)


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


def _event_run_id(event: Dict[str, Any]) -> Optional[str]:
    for key in ("run_id", "job_id"):
        val = event.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


async def _cancel_agent_runs(agent_id: str, run_id: Optional[str] = None) -> None:
    """Stop a stuck Letta run (frees GPU). Best-effort; requires Redis on Letta."""
    body: Dict[str, Any] = {}
    if run_id:
        body["run_ids"] = [run_id]
    try:
        resp = await _http.post(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/messages/cancel",
            json=body,
        )
        if resp.status_code >= 400:
            log.warning("Letta cancel returned %s: %s", resp.status_code, resp.text[:200])
        else:
            log.info("Letta run cancel requested", extra={"run_id": run_id})
    except Exception as exc:
        log.warning("Letta cancel failed: %s", exc)


# ---------------------------------------------------------------------------
# Streaming path: Letta SSE --> OpenAI SSE with <think> blocks
# ---------------------------------------------------------------------------

async def _stream_letta(agent_id: str, user_text: str) -> AsyncGenerator[str, None]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    yield _oai_chunk(cid, created, {"role": "assistant"})

    segments = _ThinkingSegments()
    has_assistant = False
    trimmed_waiting_reply = False
    trimmed_at: Optional[float] = None
    active_run_id: Optional[str] = None
    cancel_sent = False

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
                    reasoning, trimmed, detail = trim_reasoning_for_display(reasoning)
                    if trimmed and detail:
                        record_guard_event(detail, source="json_fallback")
                    text = "".join(text_parts).strip() or "(no response)"
                    if reasoning:
                        suffix = (
                            "\n\n_(planning trimmed — reply below)_\n"
                            if trimmed
                            else "\n"
                        )
                        text = (
                            f"<think>\n{reasoning}{suffix}"
                            f"</think>\n\n{text}"
                        )
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
                    if (
                        STREAM_TIMEOUT is not None
                        and trimmed_waiting_reply
                        and not has_assistant
                        and trimmed_at is not None
                        and not cancel_sent
                        and (time.time() - trimmed_at) >= STREAM_TIMEOUT
                    ):
                        cancel_sent = True
                        log.warning(
                            "No assistant after reasoning trim — requesting Letta cancel"
                        )
                        await _cancel_agent_runs(agent_id, active_run_id)
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

                rid = _event_run_id(event)
                if rid:
                    active_run_id = rid

                msg_type = event.get("message_type", "")

                # --- Reasoning / internal monologue (own collapsible per burst) ---
                if msg_type in ("reasoning_message", "internal_monologue"):
                    text = _strip_think_tags(
                        event.get("reasoning", "")
                        or event.get("internal_monologue", "")
                    )
                    if not text:
                        continue

                    parts, closed_loop = segments.feed_reasoning(text)
                    for part in parts:
                        yield _oai_chunk(cid, created, {"content": part})
                    if closed_loop:
                        detail = segments._guard.last_detection
                        if detail:
                            record_guard_event(
                                detail,
                                agent_id=agent_id,
                                run_id=active_run_id,
                                source="stream",
                            )
                        trimmed_waiting_reply = True
                        trimmed_at = time.time()

                # --- Tool call + return (separate collapsible blocks) ---
                elif msg_type in ("tool_call_message", "function_call"):
                    fc = event.get("function_call") or event.get("tool_call") or {}
                    name = fc.get("name", "") if isinstance(fc, dict) else str(fc)
                    for part in segments.feed_tool_call(name, fc if isinstance(fc, dict) else {}):
                        yield _oai_chunk(cid, created, {"content": part})

                elif msg_type == "tool_return_message":
                    for part in segments.feed_tool_return(event):
                        yield _oai_chunk(cid, created, {"content": part})

                # --- Assistant message (the actual response) ---
                elif msg_type == "assistant_message":
                    text = _extract_text(
                        event.get("assistant_message", "")
                        or event.get("content", "")
                    )
                    if not text:
                        continue
                    for part in segments.close_all():
                        yield _oai_chunk(cid, created, {"content": part})
                    trimmed_waiting_reply = False
                    has_assistant = True
                    yield _oai_chunk(cid, created, {"content": text})

            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    except httpx.ReadTimeout:
        log.error("Letta stream timed out after %ss", STREAM_TIMEOUT)
        for part in segments.close_all():
            yield _oai_chunk(cid, created, {"content": part})
        yield _oai_chunk(cid, created, {"content": "[response timed out]"})
    except httpx.HTTPError as exc:
        log.exception("Letta stream HTTP error")
        for part in segments.close_all():
            yield _oai_chunk(cid, created, {"content": part})
        yield _oai_chunk(cid, created, {"content": f"[connection error: {exc}]"})
    else:
        for part in segments.close_all():
            yield _oai_chunk(cid, created, {"content": part})
        if not has_assistant:
            if trimmed_waiting_reply:
                yield _oai_chunk(
                    cid,
                    created,
                    {
                        "content": (
                            "\n\n*(The model did not finish a visible reply after "
                            "internal planning was trimmed. Say **continue** or "
                            "send your message again.)*"
                        )
                    },
                )
            else:
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


def _completion_json(text: str, thinking: str = "") -> Dict[str, Any]:
    content = f"{thinking}\n\n{text}" if thinking else text
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
    thinking = _thinking_blocks_from_messages(getattr(resp, "messages", []) or [])
    return _completion_json(text, thinking)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8284")))

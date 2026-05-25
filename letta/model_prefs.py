"""Resolve Letta model/embedding handles from prefs file or existing agent — no baked-in defaults."""

from __future__ import annotations

import json
import os
from typing import Any, Tuple

PREFS_ENV = "LETTA_MODEL_PREFS_FILE"
DEFAULT_PREFS_PATH = "/app/memory/letta-model-prefs.json"


def prefs_path() -> str:
    return os.getenv(PREFS_ENV, DEFAULT_PREFS_PATH)


def load_prefs_file(path: str | None = None) -> Tuple[str, str]:
    p = path or prefs_path()
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"No model prefs at {p}. Choose models in Memory Explorer and click "
            '"Update models" (writes letta-model-prefs.json).'
        )
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    model = (data.get("model") or "").strip()
    embedding = (data.get("embedding") or "").strip()
    if not model or not embedding:
        raise ValueError(f"Invalid prefs at {p}: requires non-empty model and embedding")
    return model, embedding


def _handle_from_agent(agent: Any) -> Tuple[str, str]:
    """Extract ollama handles from a Letta agent object or dict."""
    if isinstance(agent, dict):
        llm = agent.get("llm_config") or {}
        emb = agent.get("embedding_config") or {}
        model = llm.get("handle") or agent.get("model") or ""
        embedding = emb.get("handle") or agent.get("embedding") or ""
    else:
        llm = getattr(agent, "llm_config", None) or {}
        emb = getattr(agent, "embedding_config", None) or {}
        if hasattr(llm, "handle"):
            model = llm.handle or getattr(agent, "model", "")
        elif isinstance(llm, dict):
            model = llm.get("handle") or getattr(agent, "model", "")
        else:
            model = getattr(agent, "model", "")
        if hasattr(emb, "handle"):
            embedding = emb.handle or getattr(agent, "embedding", "")
        elif isinstance(emb, dict):
            embedding = emb.get("handle") or getattr(agent, "embedding", "")
        else:
            embedding = getattr(agent, "embedding", "")
    return str(model).strip(), str(embedding).strip()


def resolve_models_for_create(
    client: Any,
    agent_name: str,
    *,
    prefs_path_override: str | None = None,
) -> Tuple[str, str]:
    """
    Models for agent creation: explicit env override, else prefs file.
    If the agent already exists, returns its current handles (caller should not create).
    """
    env_model = os.getenv("LETTA_MODEL", "").strip()
    env_embed = os.getenv("LETTA_EMBEDDING", "").strip()
    if env_model and env_embed:
        return env_model, env_embed

    p = prefs_path_override or prefs_path()
    if os.path.isfile(p):
        return load_prefs_file(p)

    raise FileNotFoundError(
        f"No {p} and LETTA_MODEL/LETTA_EMBEDDING not set. "
        'Use Memory Explorer → Update models first.'
    )


def models_from_existing_agent(agent: Any) -> Tuple[str, str]:
    model, embedding = _handle_from_agent(agent)
    if not model:
        raise ValueError("Existing agent has no model handle")
    if not embedding:
        raise ValueError("Existing agent has no embedding handle")
    return model, embedding

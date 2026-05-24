#!/usr/bin/env python3
"""
Register identity-mcp tools on the Letta agent.

Gives the agent (and sleeptime) access to the identity corpus
via the identity-mcp HTTP API running in the mcp-server container.

Usage:
    LETTA_BASE_URL=http://localhost:8283 \
    MCP_SERVER_URL=http://mcp-server:4000 \
      python letta/register_tools.py

    # Remove previously registered tools:
    python letta/register_tools.py --clean
"""
import os
import json
import argparse
import logging

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("register_tools")

BASE = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:4000")
AGENT_NAME = os.getenv("LETTA_AGENT_NAME", "identity")

http = httpx.Client(base_url=BASE, timeout=30.0, follow_redirects=True)

TAG = "identity_mcp"


def _tool_source(
    func_name: str,
    docstring: str,
    endpoint: str,
    params: str = "",
    body_expr: str = "{}",
    method: str = "POST",
) -> str:
    """Generate Letta Python tool source that calls identity-mcp HTTP API."""
    if method.upper() == "GET":
        return "\n".join(
            [
                f"def {func_name}({params}) -> str:",
                f'    """{docstring}"""',
                "    import urllib.request",
                "    import json as _json",
                "",
                f'    url = "{MCP_URL}{endpoint}"',
                "    req = urllib.request.Request(url, method='GET')",
                "    try:",
                "        with urllib.request.urlopen(req, timeout=60) as resp:",
                "            data = _json.loads(resp.read().decode('utf-8'))",
                "            return _json.dumps(data, indent=2)[:8000]",
                "    except Exception as e:",
                "        return f'Error calling identity-mcp: {e}'",
            ]
        ) + "\n"

    return "\n".join(
        [
            f"def {func_name}({params}) -> str:",
            f'    """{docstring}"""',
            "    import urllib.request",
            "    import json as _json",
            "",
            f'    url = "{MCP_URL}{endpoint}"',
            f"    body = _json.dumps({body_expr}).encode('utf-8')",
            "    req = urllib.request.Request(url, data=body, method='POST',",
            "                                headers={'Content-Type': 'application/json'})",
            "    try:",
            "        with urllib.request.urlopen(req, timeout=60) as resp:",
            "            data = _json.loads(resp.read().decode('utf-8'))",
            "            return _json.dumps(data, indent=2)[:8000]",
            "    except Exception as e:",
            "        return f'Error calling identity-mcp: {e}'",
        ]
    ) + "\n"


TOOL_SOURCES = [
    # ── Search & corpus access ─────────────────────────────────────────────
    _tool_source(
        "identity_search_corpus",
        "Semantic search across Letta archival memory (pgvector). Use for meaning-based recall of ingested history.\n\n    Args:\n        query: Natural language search query\n        limit: Max results (default 20)\n\n    Returns:\n        str: JSON results",
        "/mcp/search.semantic",
        params="query: str, limit: int = 20",
        body_expr='{"query": query, "limit": limit or 20}',
    ),
    _tool_source(
        "identity_search_all",
        "Search the full identity corpus: memory JSONL, uploaded files, parsed conversations, and Letta archival memory. Prefer this when you need complete coverage or files/conversations not found in archival alone.\n\n    Args:\n        query: Natural language or keyword query\n        limit: Max results per source (default 15)\n\n    Returns:\n        str: JSON with memories, files, conversations, and letta sections",
        "/mcp/search.all",
        params="query: str, limit: int = 15",
        body_expr='{"query": query, "limit": limit or 15, "sources": ["memories", "files", "conversations", "letta"]}',
    ),
    _tool_source(
        "identity_memory_search",
        "Full-text keyword search across memory/*.jsonl files on disk.\n\n    Args:\n        query: Search query\n        limit: Max results (default 20)\n\n    Returns:\n        str: JSON results",
        "/mcp/memory.search",
        params="query: str, limit: int = 20",
        body_expr='{"query": query, "limit": limit or 20}',
    ),
    _tool_source(
        "identity_file_search",
        "Full-text search across documents in the files/ RAG corpus (reports, notes, uploads). Use when archival search misses file-only content.\n\n    Args:\n        query: Search query\n        folder: Optional subfolder under files/\n        limit: Not used by API; kept for compatibility\n\n    Returns:\n        str: JSON file search results",
        "/mcp/file.search",
        params='query: str, folder: str = ""',
        body_expr='({"query": query, "folder": folder} if folder else {"query": query})',
    ),
    _tool_source(
        "identity_conversation_get",
        "Retrieve a full parsed conversation by ID (from conversation search or data explorer). Use for deep context when updating human memory or summarizing a thread.\n\n    Args:\n        conversation_id: Conversation ID string\n\n    Returns:\n        str: JSON conversation with messages",
        "/mcp/conversation.get",
        params="conversation_id: str",
        body_expr='{"conversationId": conversation_id}',
    ),
    # ── Identity bundles & analysis ────────────────────────────────────────
    _tool_source(
        "identity_get_profile",
        "Retrieve the full identity bundle including breath, vows, prime directives, and all memory JSONL files.\n\n    Returns:\n        str: JSON identity bundle",
        "/mcp/identity.get_full",
        body_expr="{}",
    ),
    _tool_source(
        "identity_analysis_summary",
        "Overview of identity pattern analysis: relational patterns, clusters, and summary stats from analyze_identity.py.\n\n    Returns:\n        str: JSON analysis summary",
        "/mcp/identity.analysis_summary",
        method="GET",
    ),
    _tool_source(
        "identity_interaction_summary",
        "Summary of human interaction patterns: event counts, topic and tone distribution.\n\n    Returns:\n        str: JSON interaction summary",
        "/mcp/interaction.summary",
        method="GET",
    ),
    _tool_source(
        "identity_get_momentum",
        "Patterns rising or falling over time in the user's communication (identity evolution). Use when refreshing persona/human blocks or reflecting on change.\n\n    Returns:\n        str: JSON with rising, falling, and stable patterns",
        "/mcp/identity.momentum",
        method="GET",
    ),
    _tool_source(
        "identity_model_status",
        "Trained identity verification model status: stylistic profile, vocabulary fingerprint, and temporal_analysis (similarity over time). Requires train_identity_model.py to have been run.\n\n    Returns:\n        str: JSON model status",
        "/mcp/identity.model_status",
        method="GET",
    ),
]


def _find_agent():
    resp = http.get(f"/v1/agents/?name={AGENT_NAME}")
    resp.raise_for_status()
    for a in resp.json():
        if a.get("name") == AGENT_NAME:
            return a
    return None


def _get_agent_tools(agent_id: str):
    resp = http.get(f"/v1/agents/{agent_id}/tools/")
    resp.raise_for_status()
    return resp.json()


def clean_tools():
    resp = http.get("/v1/tools/")
    resp.raise_for_status()
    removed = 0
    for t in resp.json():
        if TAG in (t.get("tags") or []):
            http.delete(f"/v1/tools/{t['id']}/")
            log.info("Removed tool: %s (%s)", t["name"], t["id"])
            removed += 1
    log.info("Cleaned %d identity-mcp tools", removed)


def register_tools():
    agent = _find_agent()
    if not agent:
        log.error("Agent '%s' not found", AGENT_NAME)
        return

    agent_id = agent["id"]
    log.info("Agent: %s (%s)", AGENT_NAME, agent_id)

    existing_tools = http.get("/v1/tools/").json()
    existing_names = {t["name"]: t for t in existing_tools}

    agent_tools = _get_agent_tools(agent_id)
    agent_tool_ids = {t["id"] for t in agent_tools}

    for source_code in TOOL_SOURCES:
        func_line = source_code.split("\n")[0]
        name = func_line.split("(")[0].replace("def ", "")

        if name in existing_names:
            existing = existing_names[name]
            resp = http.patch(f"/v1/tools/{existing['id']}/", json={
                "source_code": source_code,
                "tags": [TAG],
            })
            resp.raise_for_status()
            tool_id = existing["id"]
            log.info("Updated tool: %s (%s)", name, tool_id)
        else:
            resp = http.post("/v1/tools/", json={
                "source_code": source_code,
                "source_type": "python",
                "tags": [TAG],
            })
            if resp.status_code != 200:
                log.error("Failed to create tool %s: %s", name, resp.text[:300])
                continue
            tool_id = resp.json()["id"]
            log.info("Created tool: %s (%s)", name, tool_id)

        if tool_id not in agent_tool_ids:
            resp = http.patch(f"/v1/agents/{agent_id}/", json={
                "tool_ids": list(agent_tool_ids | {tool_id}),
            })
            resp.raise_for_status()
            agent_tool_ids.add(tool_id)
            log.info("Attached tool %s to agent", name)

    log.info("Done: %d identity-mcp tools on agent %s", len(TOOL_SOURCES), AGENT_NAME)


def main():
    ap = argparse.ArgumentParser(description="Register identity-mcp tools on Letta agent")
    ap.add_argument("--clean", action="store_true", help="Remove previously registered tools")
    args = ap.parse_args()

    if args.clean:
        clean_tools()
    else:
        register_tools()


if __name__ == "__main__":
    main()

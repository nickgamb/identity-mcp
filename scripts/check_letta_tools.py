#!/usr/bin/env python3
"""Quick check: Letta agent tools and recent tool messages."""
import os
import json
import httpx

BASE = os.getenv("LETTA_BASE_URL", "http://letta:8283")
NAME = os.getenv("LETTA_AGENT_NAME", "identity")

http = httpx.Client(base_url=BASE, timeout=30.0)
agents = http.get("/v1/agents/", params={"name": NAME}).json()
if isinstance(agents, list):
    agent = next((a for a in agents if a.get("name") == NAME), agents[0] if agents else None)
else:
    agent = agents
if not agent:
    print("No agent found")
    raise SystemExit(1)
aid = agent["id"]
print(f"agent: {aid} ({agent.get('name')})")
tools = http.get(f"/v1/agents/{aid}/tools").json()
print(f"tools: {len(tools)}")
for t in tools[:2]:
    src = t.get("source_code") or ""
    print(f"  {t.get('name')}: 127.0.0.1={('127.0.0.1' in src)} mcp-server={('mcp-server' in src)}")

msgs = http.get(f"/v1/agents/{aid}/messages", params={"limit": 30, "order": "desc"}).json()
if isinstance(msgs, dict):
    msgs = msgs.get("messages", [])
try:
    import urllib.request

    req = urllib.request.Request(
        "http://mcp-server:4000/mcp/search.semantic",
        data=json.dumps({"query": "test", "limit": 1}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"mcp-server from letta: OK ({resp.status})")
except Exception as e:
    print(f"mcp-server from letta: FAIL ({e})")

for m in msgs[:25]:
    mt = m.get("message_type", m.get("role", "?"))
    c = m.get("content") or m.get("tool_return") or m.get("reasoning") or ""
    if isinstance(c, list):
        c = json.dumps(c)[:200]
    s = str(c).replace("\n", " ")[:140]
    if "unavail" in s.lower() or "error" in s.lower() or "identity" in s.lower() or "tool" in mt:
        print(f"  [{mt}] {s}")

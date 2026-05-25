#!/bin/bash
set -e
echo "=== semantic ==="
curl -s -X POST http://127.0.0.1:4000/mcp/search.semantic \
  -H 'Content-Type: application/json' \
  -d '{"query":"Nick","limit":5}' | head -c 800
echo
echo "=== search.all ==="
curl -s -X POST http://127.0.0.1:4000/mcp/search.all \
  -H 'Content-Type: application/json' \
  -d '{"query":"Nick","limit":5,"sources":["memories","files","conversations","letta"]}' | head -c 1200
echo
echo "=== letta archival direct ==="
AGENT=$(curl -s 'http://127.0.0.1:8283/v1/agents/?name=identity' | python3 -c "import sys,json; a=json.load(sys.stdin); print(next((x['id'] for x in a if x.get('name')=='identity'),''))")
echo "agent=$AGENT"
if [ -n "$AGENT" ]; then
  curl -s "http://127.0.0.1:8283/v1/agents/${AGENT}/archival-memory/search?query=Nick&top_k=5" | head -c 600
  echo
fi
echo "=== from letta container to mcp-server ==="
docker exec letta curl -sf -X POST http://mcp-server:4000/mcp/search.semantic \
  -H 'Content-Type: application/json' \
  -d '{"query":"Nick","limit":3}' | head -c 500 || echo "letta->mcp failed"
echo

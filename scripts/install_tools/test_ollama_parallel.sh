#!/usr/bin/env bash
# Fire N concurrent Ollama /api/generate requests and print /api/ps + nvidia-smi snapshots.
# Usage: ./scripts/test_ollama_parallel.sh [N] [model]
# Example: OLLAMA_URL=http://127.0.0.1:11434 ./scripts/test_ollama_parallel.sh 2 qwen3:32b

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL="${2:-${MODEL:-qwen3:32b}}"
N="${1:-2}"
NUM_PREDICT="${NUM_PREDICT:-80}"
TMPDIR="${TMPDIR:-/tmp}"
STAMP=$(date +%s)

echo "=== Ollama parallel test ==="
echo "URL=$OLLAMA_URL  model=$MODEL  concurrent=$N  num_predict=$NUM_PREDICT"
echo "Server parallel env (if docker):"
docker exec ollama-server printenv OLLAMA_NUM_PARALLEL 2>/dev/null || true

snap() {
  local label=$1
  echo "--- $label ---"
  curl -sS "$OLLAMA_URL/api/ps" 2>/dev/null | python3 -m json.tool 2>/dev/null || curl -sS "$OLLAMA_URL/api/ps" || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  fi
}

snap "before"

run_one() {
  local id=$1
  local out="$TMPDIR/ollama-par-${STAMP}-${id}.json"
  local start
  start=$(date +%s)
  echo "[req $id] start $(date -Iseconds)"
  if curl -sS "$OLLAMA_URL/api/generate" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Load-test worker ${id}. Reply in exactly 4 short sentences about waiting in a queue.\",\"stream\":false,\"options\":{\"num_predict\":${NUM_PREDICT}}}" \
    -o "$out" \
    -w "[req $id] finished http=%{http_code} curl_time=%{time_total}s\n"; then
    echo "[req $id] wall=$(( $(date +%s) - start ))s"
  else
    echo "[req $id] FAILED wall=$(( $(date +%s) - start ))s"
  fi
}

(
  for t in 3 6 9 12 15 18 21 24 27 30 45 60 90 120; do
    sleep 3
    snap "t+${t}s (while running)"
  done
) &
MON_PID=$!

for i in $(seq 1 "$N"); do
  run_one "$i" &
done
wait

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

snap "after all complete"
echo "=== done ==="

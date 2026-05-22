#!/usr/bin/env bash
# Toggle LLM inference between Ollama and HuggingFace (mutually exclusive on GPU).
# Core stack (mcp-server, librechat, dashboard, identity-service) stays up.
#
# Usage:
#   ./switch-model-provider.sh          # toggle ollama <-> hf
#   ./switch-model-provider.sh status   # show which provider is running
#   ./switch-model-provider.sh ollama   # switch to Ollama
#   ./switch-model-provider.sh hf        # switch to HuggingFace
#
# Run from project root (directory with docker-compose.yml), e.g. ~/ai on the server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${COMPOSE_DIR:-$SCRIPT_DIR}"

if [[ ! -f docker-compose.yml ]]; then
  echo "Error: docker-compose.yml not found in $(pwd)" >&2
  echo "Copy docker-compose-template.yml to docker-compose.yml or set COMPOSE_DIR." >&2
  exit 1
fi

COMPOSE=(docker compose)
IDENTITY_PROFILE=(--profile identity)

is_running() {
  local name="$1"
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$name"
}

detect_provider() {
  local ollama_up=false
  local hf_up=false
  is_running ollama-server && ollama_up=true
  is_running hf-service && hf_up=true

  if $ollama_up && $hf_up; then
    echo "both"
  elif $ollama_up; then
    echo "ollama"
  elif $hf_up; then
    echo "hf"
  else
    echo "none"
  fi
}

print_status() {
  local p
  p="$(detect_provider)"
  echo "=== Model provider status ==="
  echo "Directory: $(pwd)"
  case "$p" in
    ollama)
      echo "Active:  Ollama (ollama-server)"
      echo "LibreChat endpoint: LocalOllama → http://ollama:11434/v1"
      ;;
    hf)
      echo "Active:  HuggingFace (hf-service)"
      echo "LibreChat endpoint: HFService → http://hf-service:8000/v1"
      if is_running ollama-server; then
        echo "Warning: ollama-server is also running (GPU contention likely)"
      fi
      ;;
    both)
      echo "Active:  BOTH ollama-server and hf-service (not recommended)"
      echo "Run: $0 ollama   or   $0 hf   to leave only one up"
      ;;
    none)
      echo "Active:  neither (no LLM container running)"
      echo "Run: $0 ollama   or   $0 hf"
      ;;
  esac
  echo ""
  docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep -E 'ollama-server|hf-service|mcp-server|librechat-api' || true
}

wait_healthy() {
  local name="$1"
  local url="$2"
  local i
  echo "Waiting for $name..."
  for i in $(seq 1 60); do
    if docker exec "$name" wget -qO- "$url" &>/dev/null 2>&1; then
      echo "$name is ready."
      return 0
    fi
    sleep 2
  done
  echo "Warning: $name did not respond at $url within ~2 minutes (may still be starting)." >&2
  return 1
}

start_ollama() {
  echo "=== Switching to Ollama ==="
  if is_running hf-service; then
    echo "Stopping hf-service..."
    "${COMPOSE[@]}" --profile hf stop hf-service
  fi
  echo "Starting Ollama + core stack..."
  "${COMPOSE[@]}" "${IDENTITY_PROFILE[@]}" --profile ollama up -d ollama
  # Ensure core services exist without tearing down
  "${COMPOSE[@]}" "${IDENTITY_PROFILE[@]}" --profile ollama up -d \
    mcp-server identity-service librechat-api dashboard mongodb redis meilisearch
  wait_healthy ollama-server "http://127.0.0.1:11434/" || true
  echo ""
  echo "Use LibreChat presets: Qwen3 32B (MCP), Llama 3.1 8B (Tools), GPT-OSS 20B, etc."
  echo "HF presets will fail until you run: $0 hf"
}

start_hf() {
  echo "=== Switching to HuggingFace ==="
  if is_running ollama-server; then
    echo "Stopping ollama-server (frees GPU for HF)..."
    "${COMPOSE[@]}" --profile ollama stop ollama
  fi
  echo "Starting hf-service + core stack..."
  "${COMPOSE[@]}" "${IDENTITY_PROFILE[@]}" --profile hf up -d hf-service
  "${COMPOSE[@]}" "${IDENTITY_PROFILE[@]}" --profile hf up -d \
    mcp-server identity-service librechat-api dashboard mongodb redis meilisearch
  wait_healthy hf-service "http://127.0.0.1:8000/health" || true
  echo ""
  echo "Use LibreChat presets: GLM-4.5-Air (HF + Tools), GPT-OSS fine-tuned (HF)"
  echo "First HF request may take 30–60s while the model loads into GPU."
}

toggle() {
  case "$(detect_provider)" in
    ollama) start_hf ;;
    hf) start_ollama ;;
    both)
      echo "Both providers running. Stopping both, then starting Ollama..."
      "${COMPOSE[@]}" --profile hf stop hf-service 2>/dev/null || true
      "${COMPOSE[@]}" --profile ollama stop ollama 2>/dev/null || true
      start_ollama
      ;;
    none) start_ollama ;;
  esac
}

cmd="${1:-toggle}"
case "$cmd" in
  status|-s)
    print_status
    ;;
  ollama|ollama-server)
    start_ollama
    print_status
    ;;
  hf|huggingface|hf-service)
    start_hf
    print_status
    ;;
  toggle|switch|"")
    toggle
    print_status
    ;;
  -h|--help|help)
    sed -n '2,12p' "$0"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: $0 [status|ollama|hf|toggle]" >&2
    exit 1
    ;;
esac

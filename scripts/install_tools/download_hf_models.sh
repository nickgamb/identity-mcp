#!/usr/bin/env bash
# Download HuggingFace models on the HOST into ~/models/hf_models/<short_name>/
# (flat layout — matches hf-service hf_api.py)
#
# One-time setup on the server:
#   python3 -m venv ~/ai/.venv-hf
#   ~/ai/.venv-hf/bin/pip install -U "huggingface_hub>=0.23"
#
# Run:
#   export HF_TOKEN=hf_...
#   ./scripts/install_tools/download_hf_models.sh
#
# Or install hub only and download yourself (see bottom of script / README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${HF_VENV:-$AI_ROOT/.venv-hf}"
HOST_HF_CACHE="${HF_CACHE_DIR:-$HOME/models/hf_models}"

# HF models are ONLY for cases Ollama can't cover well on this hardware (2x P40):
#   - CPU/RAM offload for a big MoE that has no good Ollama GGUF
#   - Tool calling Ollama mishandles (e.g. GLM's XML tool-call envelope)
#   - The base model required for LoRA fine-tuning
#
# Do NOT add a model here if it already has a solid Ollama GGUF match — those go
# in download_ollama_models.sh (library pull) or download_ollama_gguf_imports.sh
# (Unsloth/HF GGUF + ollama create; e.g. Qwen3.6 UD-IQ4_NL / UD-Q5_K_M).
#
# P40 has NO FP8 and NO BF16 tensor cores. FP8 repos are excluded: they upcast to
# FP16/FP32 and blow past 48GB VRAM + 112GB RAM. Frontier MoEs (Qwen3.5-397B,
# DeepSeek-V3/V4, GLM-4.6) should be pulled as GGUF in Ollama with RAM offload.
MODEL_SPECS=(
  "zai-org/GLM-4.5-Air|glm-4.5-air"  # GLM XML tools + offload; no faithful Ollama match
  "openai/gpt-oss-20b|gpt-oss-20b"   # LoRA fine-tune base (not an inference dup)
)
# Removed — poor fit for P40 (do not re-add without switching format):
#   Qwen/Qwen3.5-397B-A17B-FP8           -> FP8 unsupported on Pascal; use a GGUF Q3/Q4 in Ollama instead
#   deepseek-ai/DeepSeek-V3              -> FP8-native + ~1.3TB at BF16; use a low-quant GGUF in Ollama if wanted
#   mistralai/Mixtral-8x22B-Instruct-v0.1 -> dated (early 2024); big-MoE needs better served by GGUF-in-Ollama

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
  fi
  if ! "$VENV_DIR/bin/python" -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub ..."
    "$VENV_DIR/bin/pip" install -U "huggingface_hub>=0.23"
  fi
}

model_complete() {
  local short_name="$1"
  local repo_id="$2"
  local dir="$HOST_HF_CACHE/$short_name"

  local hub_name="models--$(echo "$repo_id" | sed 's/\//--/g')"
  if [ -d "$HOST_HF_CACHE/$hub_name/snapshots" ]; then
    local snap
    snap="$(find "$HOST_HF_CACHE/$hub_name/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
    if [ -n "$snap" ] && find "$snap" \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null | grep -q .; then
      [ "$short_name" = "gpt-oss-20b" ] && return 0
    fi
  fi
  [ -d "$HOST_HF_CACHE/gpt-oss-20b-finetuned" ] && [ "$short_name" = "gpt-oss-20b" ] && return 0
  [ -d "$HOST_HF_CACHE/glm-4.5-air" ] && [ "$short_name" = "glm-4.5-air" ] && \
    find "$HOST_HF_CACHE/glm-4.5-air" \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null | grep -q . && return 0

  [ -f "$dir/.download_complete" ] && return 0
  [ -d "$dir" ] && find "$dir" \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null | grep -q .
}

download_one() {
  local repo_id="$1"
  local short_name="$2"
  HF_CACHE_DIR="$HOST_HF_CACHE" REPO_ID="$repo_id" SHORT_NAME="$short_name" \
    HF_HUB_DISABLE_XET=1 \
    "$VENV_DIR/bin/python" -u <<'PY'
import os
from huggingface_hub import hf_hub_download, list_repo_files

cache = os.environ["HF_CACHE_DIR"]
repo_id = os.environ["REPO_ID"]
short = os.environ["SHORT_NAME"]
token = os.environ.get("HF_TOKEN")
local_dir = os.path.join(cache, short)

os.makedirs(local_dir, exist_ok=True)
files = list_repo_files(repo_id, token=token)
total = len(files)
print(f"  {total} files", flush=True)

for i, filename in enumerate(files, 1):
    dest = os.path.join(local_dir, filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  [{i}/{total}] skip {filename}", flush=True)
        continue
    print(f"  [{i}/{total}] {filename}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )

with open(os.path.join(local_dir, ".download_complete"), "w", encoding="utf-8") as f:
    f.write(repo_id + "\n")
print(f"  ✓ {local_dir}", flush=True)
PY
}

# --- only run downloads when executed (not when sourced for manual helpers) ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "=== HuggingFace Model Download (host venv) ==="
  echo "Cache: $HOST_HF_CACHE"
  echo "Venv:  $VENV_DIR"
  echo ""

  ensure_venv
  mkdir -p "$HOST_HF_CACHE"

  if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set — set it for faster, reliable downloads"
    echo ""
  fi

  for spec in "${MODEL_SPECS[@]}"; do
    repo_id="${spec%%|*}"
    short_name="${spec##*|}"
    echo "=== $repo_id → $short_name ==="
    if model_complete "$short_name" "$repo_id"; then
      echo "  ✓ already complete"
      continue
    fi
    echo "  → Downloading..."
    if download_one "$repo_id" "$short_name"; then
      :
    else
      echo "  ✗ Failed: $repo_id"
    fi
  done

  echo ""
  echo "=== Done ==="
  du -sh "$HOST_HF_CACHE"/* 2>/dev/null | head -25 || true
fi

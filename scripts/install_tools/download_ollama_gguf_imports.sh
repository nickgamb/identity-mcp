#!/usr/bin/env bash
# Download a specific GGUF quant from HuggingFace and import it into Ollama.
#
# For frontier / "ARK" models that are NOT in the Ollama library but DO have a
# community GGUF (unsloth, bartowski, etc.). On 2x P40 (Pascal: no FP8, no BF16
# tensor cores) GGUF + llama.cpp RAM-offload beats the HF service for big models,
# so anything with a GGUF belongs here, not on hf-service.
#
#   - Models that ARE in the Ollama library  -> download_ollama_models.sh (plain `ollama pull`)
#   - Models with a GGUF but NOT in library   -> THIS script (download + `ollama create`)
#   - HF safetensors (LoRA base / tool fidelity) -> download_hf_models.sh
#
# Composes two existing patterns in this repo:
#   - host venv + hf_hub_download   (download_hf_models.sh)
#   - Modelfile + `ollama create`   (merge_and_convert_lora.sh)
#
# One-time setup on the server:
#   python3 -m venv ~/ai/.venv-hf && ~/ai/.venv-hf/bin/pip install -U "huggingface_hub>=0.23"
#
# Run on the server (host, not in container):
#   export HF_TOKEN=hf_...            # optional, for gated/faster downloads
#   export ARK_MAX_GB=2000            # total disk budget across all imports (default 2000)
#   ./scripts/install_tools/download_ollama_gguf_imports.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${HF_VENV:-$AI_ROOT/.venv-hf}"
# Bind-mounted into ollama-server at /imports:ro (see docker-compose.yml)
IMPORTS_DIR="${OLLAMA_IMPORTS_DIR:-$HOME/models/ollama_models/imports}"
ARK_MAX_GB="${ARK_MAX_GB:-2000}"
OLLAMA="docker exec ollama-server ollama"

# repo_id | quant pattern (substring matched against .gguf filenames) | ollama model name
# Sizes are the chosen quant's footprint; all fit the ~160GB RAM+VRAM run budget except
# the Tier-C archives (Kimi/etc.) which are archival (run slow via heavy offload).
MODEL_SPECS=(
  # --- Tier B: frontier MoE that RUNS at usable speed (RAM offload) ---
  "unsloth/MiniMax-M2.5-GGUF|UD-Q3_K_XL|minimax-m2.5"        # 230B/10B-active, ~101GB
  "unsloth/Qwen3.5-397B-A17B-GGUF|UD-IQ2_M|qwen3.5-397b"     # 397B/17B-active, ~120GB (repo has no UD-Q2_K_XL)
  "unsloth/GLM-4.6-GGUF|UD-Q2_K_XL|glm-4.6"                  # 357B MoE, ~135GB, strong tools

  # --- Tier C: ARK archive (huge; archival + occasional use). Auto-skips if no GGUF yet. ---
  "unsloth/GLM-4.7-GGUF|UD-Q2_K_XL|glm-4.7"                  # 358B, newer GLM
  "unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF|UD-Q3_K_XL|qwen3-vl-235b"  # multimodal MoE
  "unsloth/Kimi-K2.6-GGUF|UD-Q2_K_XL|kimi-k2.6"             # 1T flagship (slow; archive)
  # "unsloth/DeepSeek-V4-Flash-GGUF|UD-Q2_K_XL|deepseek-v4-flash"  # enable when GGUF ships
)

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

ollama_has() {
  $OLLAMA list 2>/dev/null | awk '{print $1}' | grep -qx "$1:latest" && return 0
  $OLLAMA list 2>/dev/null | awk '{print $1}' | grep -qx "$1" && return 0
  return 1
}

# Download matched GGUF shards within budget. Prints "FROM=<host_path>" on success,
# "SKIP_TOOBIG=<gb>" if over the remaining budget, or "SKIP_NOFILES" if none match.
download_quant() {
  local repo_id="$1" quant="$2" name="$3" remaining_gb="$4"
  HF_REPO="$repo_id" HF_QUANT="$quant" DEST="$IMPORTS_DIR/$name" REMAINING_GB="$remaining_gb" \
    HF_HUB_DISABLE_XET=1 \
    "$VENV_DIR/bin/python" -u <<'PY'
import os, sys, time
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")  # default 10s is too short for this link
from huggingface_hub import HfApi, hf_hub_download

repo = os.environ["HF_REPO"]
quant = os.environ["HF_QUANT"]
dest = os.environ["DEST"]
remaining = float(os.environ["REMAINING_GB"])
token = os.environ.get("HF_TOKEN")

api = HfApi()
try:
    info = api.model_info(repo, files_metadata=True, token=token)
except Exception as e:
    print(f"SKIP_NOFILES  ({e})", flush=True)
    sys.exit(0)

# Match .gguf files whose path contains the quant tag (case-insensitive).
matched = [s for s in info.siblings
           if s.rfilename.lower().endswith(".gguf") and quant.lower() in s.rfilename.lower()]
if not matched:
    print("SKIP_NOFILES", flush=True)
    sys.exit(0)

total_bytes = sum((s.size or 0) for s in matched)
total_gb = total_bytes / (1024**3)
if total_gb > remaining:
    print(f"SKIP_TOOBIG={total_gb:.0f}", flush=True)
    sys.exit(0)

os.makedirs(dest, exist_ok=True)
files = sorted(s.rfilename for s in matched)
print(f"  {len(files)} shard(s), {total_gb:.0f}GB", flush=True)
for i, fn in enumerate(files, 1):
    print(f"  [{i}/{len(files)}] {fn}", flush=True)
    # Retry-with-resume: hf_hub_download resumes from the partial .incomplete file each call,
    # so we can survive timeouts / RemoteProtocolError mid-shard on a flaky link.
    for attempt in range(1, 13):
        try:
            hf_hub_download(repo_id=repo, filename=fn, local_dir=dest, token=token)
            break
        except Exception as e:
            print(f"    attempt {attempt}/12 failed ({type(e).__name__}); resuming in 15s", flush=True)
            time.sleep(15)
    else:
        print("FAIL_AFTER_RETRIES", flush=True)
        sys.exit(1)

# First shard is what the Modelfile FROM points at (Ollama auto-loads sibling shards).
print(f"FROM={os.path.join(dest, files[0])}", flush=True)
print(f"SIZE_GB={total_gb:.0f}", flush=True)
PY
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "=== Ollama GGUF Imports (host venv) ==="
  echo "Imports dir: $IMPORTS_DIR"
  echo "Budget:      ${ARK_MAX_GB}GB"
  echo ""
  ensure_venv
  mkdir -p "$IMPORTS_DIR"
  [ -z "${HF_TOKEN:-}" ] && echo "WARNING: HF_TOKEN not set — set it for faster/gated downloads" && echo ""

  remaining="$ARK_MAX_GB"
  for spec in "${MODEL_SPECS[@]}"; do
    repo_id="${spec%%|*}"; rest="${spec#*|}"
    quant="${rest%%|*}"; name="${rest##*|}"
    echo "=== $repo_id ($quant) -> $name ==="

    if ollama_has "$name"; then
      echo "  ✓ already in Ollama — skipping"
      continue
    fi

    out="$(download_quant "$repo_id" "$quant" "$name" "$remaining")"
    echo "$out" | grep -vE '^(FROM=|SIZE_GB=)' || true

    if echo "$out" | grep -q '^SKIP_NOFILES'; then
      echo "  ✗ no matching GGUF in repo (not released yet?) — skipping"; echo ""; continue
    fi
    if echo "$out" | grep -q '^SKIP_TOOBIG='; then
      sz="$(echo "$out" | sed -n 's/^SKIP_TOOBIG=//p')"
      echo "  ✗ ${sz}GB exceeds remaining budget ${remaining}GB — skipping"; echo ""; continue
    fi

    host_from="$(echo "$out" | sed -n 's/^FROM=//p')"
    size_gb="$(echo "$out" | sed -n 's/^SIZE_GB=//p')"
    if [ -z "$host_from" ]; then
      echo "  ✗ download failed — skipping"; echo ""; continue
    fi

    # Container sees IMPORTS_DIR as /imports (read-only mount)
    container_from="/imports/${host_from#$IMPORTS_DIR/}"
    modelfile="$IMPORTS_DIR/$name/Modelfile"
    # Minimal Modelfile: Ollama (>=0.4) reads the chat template + params from GGUF metadata.
    printf 'FROM %s\n' "$container_from" > "$modelfile"

    echo "  → ollama create $name"
    if $OLLAMA create "$name" -f "/imports/$name/Modelfile"; then
      echo "  ✓ imported $name"
      remaining=$(( remaining - ${size_gb%.*} ))
      echo "  remaining budget: ${remaining}GB"
    else
      echo "  ✗ ollama create failed for $name (chat template may need a manual TEMPLATE)"
    fi
    echo ""
  done

  echo "=== Done ==="
  $OLLAMA list
fi

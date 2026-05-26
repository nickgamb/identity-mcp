#!/usr/bin/env bash
# Download a specific GGUF quant from HuggingFace and import it into Ollama.
#
# For frontier / "ARK" models that are NOT in the Ollama library but DO have a
# community GGUF (unsloth, bartowski, etc.). On 2x P40 (Pascal: no FP8, no BF16
# tensor cores) GGUF + llama.cpp RAM-offload beats the HF service for big models,
# so anything with a GGUF belongs here, not on hf-service.
#
#   - Models that ARE in the Ollama library  -> download_ollama_models.sh (plain `ollama pull`)
#   - Models with a GGUF but NOT in library   -> THIS script (HF hub download + `ollama create`)
#   - HF safetensors (LoRA base / tool fidelity) -> download_hf_models.sh  (NOT GGUF)
#
# Qwen3.6 optional quants (UD-IQ4_NL, UD-Q5_K_M) live here — not in download_hf_models.sh.
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
#   export OLLAMA_GGUF_ONLY=qwen3.6-35b-iq4nl,qwen3.6-35b-q5km  # optional subset for A/B
#   ./scripts/install_tools/download_ollama_gguf_imports.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${HF_VENV:-$AI_ROOT/.venv-hf}"
# Bind-mounted into ollama-server at /imports:ro (see docker-compose.yml)
IMPORTS_DIR="${OLLAMA_IMPORTS_DIR:-$HOME/models/ollama_models/imports}"
ARK_MAX_GB="${ARK_MAX_GB:-2000}"
OLLAMA="docker exec ollama-server ollama"

# repo_id | quant pattern (substring matched against .gguf filenames) | ollama model name | renderer
# The optional 4th field sets RENDERER + PARSER in the Modelfile so Ollama reports the
# correct capabilities (tools, thinking, vision). Without it the GGUF import is
# completion-only and invisible to tool-calling frameworks like Letta.
# Available renderer names: https://github.com/ollama/ollama/tree/main/model/renderers
#
# Sizes are the chosen quant's footprint; all fit the ~160GB RAM+VRAM run budget except
# the Tier-C archives (Kimi/etc.) which are archival (run slow via heavy offload).
MODEL_SPECS=(
  # --- Tier A: Qwen3.6-35B-A3B identity experiments (fits 1x P40 at Q4/Q5; optional vs ollama pull qwen3.6:35b) ---
  "unsloth/Qwen3.6-35B-A3B-GGUF|A3B-UD-IQ4_NL|qwen3.6-35b-iq4nl|qwen3.5"   # ~18GB imatrix Q4-class
  "unsloth/Qwen3.6-35B-A3B-GGUF|A3B-UD-Q5_K_M|qwen3.6-35b-q5km|qwen3.5"   # ~26GB; Letta ≥Q5 guidance

  # --- Tier B: frontier MoE that RUNS at usable speed (RAM offload) ---
  "unsloth/MiniMax-M2.5-GGUF|UD-Q3_K_XL|minimax-m2.5"        # 230B/10B-active, ~101GB
  "unsloth/Qwen3.5-397B-A17B-GGUF|UD-IQ2_M|qwen3.5-397b|qwen3.5"     # 397B/17B-active, ~120GB (repo has no UD-Q2_K_XL)
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

# GGUF already on disk under IMPORTS_DIR/$name (complete prior download).
import_on_disk() {
  local name="$1" quant="$2"
  local dir="$IMPORTS_DIR/$name"
  [ -f "$dir/.download_complete" ] || return 1
  find "$dir" -maxdepth 1 -type f -iname '*.gguf' 2>/dev/null | grep -qi "$quant" || return 1
  return 0
}

find_gguf_from() {
  local name="$1" quant="$2"
  find "$IMPORTS_DIR/$name" -maxdepth 1 -type f -iname '*.gguf' 2>/dev/null \
    | grep -i "$quant" | sort | head -1
}

create_ollama_import() {
  local name="$1" host_from="$2" size_gb="${3:-0}" renderer="${4:-}"
  local container_from="/imports/${host_from#$IMPORTS_DIR/}"
  local modelfile="$IMPORTS_DIR/$name/Modelfile"
  printf 'FROM %s\n' "$container_from" > "$modelfile"
  if [ -n "$renderer" ]; then
    printf 'RENDERER %s\nPARSER %s\n' "$renderer" "$renderer" >> "$modelfile"
  fi
  echo "  → ollama create $name"
  if $OLLAMA create "$name" -f "/imports/$name/Modelfile"; then
    echo "  ✓ imported $name"
    return 0
  fi
  echo "  ✗ ollama create failed for $name (chat template may need a manual TEMPLATE)"
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
# Avoid UD-IQ4_NL also picking up UD-IQ4_NL_XL (substring collision).
matched = [s for s in info.siblings
           if s.rfilename.lower().endswith(".gguf") and quant.lower() in s.rfilename.lower()]
if "ud-iq4_nl" in quant.lower() and "xl" not in quant.lower():
    matched = [s for s in matched if "iq4_nl_xl" not in s.rfilename.lower()]
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
with open(os.path.join(dest, ".download_complete"), "w", encoding="utf-8") as f:
    f.write(repo + "\n")
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
    quant="${rest%%|*}"; rest2="${rest#*|}"
    name="${rest2%%|*}"
    renderer=""; [[ "$rest2" == *"|"* ]] && renderer="${rest2#*|}"

    if [ -n "${OLLAMA_GGUF_ONLY:-}" ]; then
      wanted=0
      IFS=',' read -ra only_names <<< "$OLLAMA_GGUF_ONLY"
      for on in "${only_names[@]}"; do
        on="${on// /}"
        [ "$on" = "$name" ] && wanted=1 && break
      done
      if [ "$wanted" -eq 0 ]; then
        continue
      fi
    fi

    echo "=== $repo_id ($quant) -> $name ==="

    if ollama_has "$name"; then
      echo "  ✓ already in Ollama — skipping"
      echo ""
      continue
    fi

    if import_on_disk "$name" "$quant"; then
      host_from="$(find_gguf_from "$name" "$quant")"
      if [ -n "$host_from" ]; then
        echo "  ✓ already on disk — importing only"
        create_ollama_import "$name" "$host_from" "0" "$renderer"
        echo ""
        continue
      fi
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

    if create_ollama_import "$name" "$host_from" "$size_gb" "$renderer"; then
      remaining=$(( remaining - ${size_gb%.*} ))
      echo "  remaining budget: ${remaining}GB"
    fi
    echo ""
  done

  echo "=== Done ==="
  $OLLAMA list
fi

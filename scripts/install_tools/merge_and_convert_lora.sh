#!/bin/bash
# Merge LoRA adapter into base model and convert to GGUF for Ollama
#
# Usage: ./merge_and_convert_lora.sh <adapter_path> <output_name>
# Example: ./merge_and_convert_lora.sh /home/nick/ai/adapters/lora-gpt-oss-20b-1766515801769 gpt-oss-20b-finetuned
#
# Prerequisites:
# - Python with transformers, peft, torch installed
# - llama.cpp repo cloned (for convert script)
#
# After running, add to docker-compose.yml ollama volumes:
#   - /home/nick/models/ollama_models/imports:/imports:ro

set -e

# Activate the Python venv with torch/transformers/peft
source /home/nick/ai/venv/bin/activate

ADAPTER_PATH="${1:-/home/nick/ai/adapters/lora-gpt-oss-20b-1766515801769}"
OUTPUT_NAME="${2:-gpt-oss-20b-finetuned}"
BASE_MODEL_PATH="/home/nick/models/hf_models/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
# Keep merged HF models in hf_models folder (organized with other HF models)
MERGED_OUTPUT_DIR="/home/nick/models/hf_models/${OUTPUT_NAME}"
# GGUF staging area - Ollama will import this into its blob storage
# Using ollama_models/imports/ as staging (add volume mount to docker-compose)
GGUF_OUTPUT_DIR="/home/nick/models/ollama_models/imports"
LLAMA_CPP_PATH="/home/nick/tools/llama.cpp"
# Container path for GGUF staging (needs volume mount in docker-compose.yml):
#   - /home/nick/models/ollama_models/imports:/imports:ro
CONTAINER_GGUF_PATH="/imports"

echo "=== LoRA Merge and GGUF Conversion ==="
echo "Adapter path: ${ADAPTER_PATH}"
echo "Output name: ${OUTPUT_NAME}"
echo "Base model: ${BASE_MODEL_PATH}"
echo ""

# Export environment variables for Python script
export ADAPTER_PATH BASE_MODEL_PATH MERGED_OUTPUT_DIR

# Step 1: Merge LoRA adapter into base model
echo "Step 1: Merging LoRA adapter into base model..."
python3 << 'PYTHON_SCRIPT'
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = os.environ.get('ADAPTER_PATH')
base_model_path = os.environ.get('BASE_MODEL_PATH')
merged_output_dir = os.environ.get('MERGED_OUTPUT_DIR')

print(f"Loading base model from {base_model_path}...")
# Load base model on CPU to avoid GPU memory issues
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print(f"Loading LoRA adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter weights into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to {merged_output_dir}...")
os.makedirs(merged_output_dir, exist_ok=True)
merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_output_dir)

print("Merge complete!")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "Error: Failed to merge LoRA adapter"
    exit 1
fi

echo ""
echo "Step 2: Converting to GGUF format..."

# Check if llama.cpp exists
if [ ! -d "${LLAMA_CPP_PATH}" ]; then
    echo "llama.cpp not found at ${LLAMA_CPP_PATH}"
    echo "Cloning llama.cpp..."
    mkdir -p "$(dirname ${LLAMA_CPP_PATH})"
    git clone https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_PATH}"
fi

# Install llama.cpp Python dependencies if needed
pip install -q gguf sentencepiece

# Create GGUF output directory
mkdir -p "${GGUF_OUTPUT_DIR}"

# Convert to GGUF (bf16 first, then can quantize)
GGUF_FILE="${GGUF_OUTPUT_DIR}/${OUTPUT_NAME}.gguf"
echo "Converting to GGUF: ${GGUF_FILE}"

python3 "${LLAMA_CPP_PATH}/convert_hf_to_gguf.py" \
    "${MERGED_OUTPUT_DIR}" \
    --outfile "${GGUF_FILE}" \
    --outtype bf16

if [ $? -ne 0 ]; then
    echo "Error: GGUF conversion failed"
    exit 1
fi

echo ""
echo "Step 3: Registering with Ollama..."

# Create Modelfile for Ollama (use container path for GGUF file)
MODELFILE_PATH="${GGUF_OUTPUT_DIR}/${OUTPUT_NAME}.Modelfile"
CONTAINER_GGUF_FILE="${CONTAINER_GGUF_PATH}/${OUTPUT_NAME}.gguf"
cat > "${MODELFILE_PATH}" << EOF
# Fine-tuned GPT-OSS 20B model
FROM ${CONTAINER_GGUF_FILE}

PARAMETER temperature 0.7
PARAMETER top_p 0.9

TEMPLATE """{{ .System }}

{{ .Prompt }}"""
EOF

echo "Created Modelfile at ${MODELFILE_PATH}"
echo "Modelfile references: ${CONTAINER_GGUF_FILE}"

# Register with Ollama (via docker exec since Ollama runs in container)
echo ""
echo "Registering model '${OUTPUT_NAME}' with Ollama..."
echo "NOTE: Make sure docker-compose.yml has this volume mount for ollama:"
echo "  - /home/nick/models/ollama_models/imports:/imports:ro"
echo ""
docker exec ollama-server ollama create "${OUTPUT_NAME}" -f "${CONTAINER_GGUF_PATH}/${OUTPUT_NAME}.Modelfile"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Success! ==="
    echo "Model '${OUTPUT_NAME}' is now available in Ollama"
    echo ""
    echo "Test with: docker exec ollama-server ollama run ${OUTPUT_NAME} 'Hello!'"
    echo ""
    echo "Add to LibreChat by including '${OUTPUT_NAME}' in your librechat.yaml Ollama models list"
else
    echo "Warning: Ollama registration failed. You may need to register manually."
    echo "Command: ollama create ${OUTPUT_NAME} -f ${MODELFILE_PATH}"
fi

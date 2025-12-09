#!/bin/bash
# Script to download HuggingFace models configured for the HF service
# Note: Models should be downloaded to /home/<USERNAME>/models/hf_models/

echo "=== HuggingFace Model Download Script ==="
echo "This script downloads models for the HF service"
echo ""

# Models needed for HF service
# These match what's configured in docker-compose.yml and librechat.yaml
MODELS=(
    "openai/gpt-oss-20b"
    "zai-org/GLM-4.5-Air"
)

# Default cache directory (adjust if needed)
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/models/hf_models}"

echo "HF Cache Directory: $HF_CACHE_DIR"
echo ""

# Check if Python and huggingface_hub are available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install --user huggingface_hub
fi

echo "Downloading models..."
echo ""

for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    
    # Check if model already exists in cache
    # Check both direct directory name and HuggingFace cache structure
    model_dir_name=$(echo "$model" | tr '/' '-')
    hf_cache_name="models--$(echo $model | sed 's/\//--/g')"
    
    if [ -d "$HF_CACHE_DIR/$model_dir_name" ] || \
       [ -d "$HF_CACHE_DIR/$hf_cache_name" ] || \
       [ -d "$HF_CACHE_DIR/glm-4.5-air" ] || \
       [ -d "$HF_CACHE_DIR/gpt-oss-20b" ]; then
        echo "  ✓ $model already exists in cache"
    else
        echo "  → Downloading $model (this may take a while)..."
        python3 << EOF
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_CACHE_DIR', os.path.expanduser('~/models/hf_models'))
model_id = "$model"

try:
    print(f"Downloading {model_id} to {cache_dir}...")
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
        resume_download=True
    )
    print(f"  ✓ Successfully downloaded {model_id}")
except Exception as e:
    print(f"  ✗ Failed to download {model_id}: {e}")
    exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully downloaded $model"
        else
            echo "  ✗ Failed to download $model"
        fi
    fi
done

echo ""
echo "=== Download complete ==="
echo "Models should be in: $HF_CACHE_DIR"
echo ""
echo "Note: Models are mounted in docker-compose.yml as:"
echo "  - $HF_CACHE_DIR/glm-4.5-air:/root/.cache/huggingface/models/glm-4.5-air:ro"
echo "  - $HF_CACHE_DIR:/root/.cache/huggingface/hub:ro"


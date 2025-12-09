#!/bin/bash
# Script to download all Ollama models configured in librechat.yaml

echo "=== Downloading Ollama Models ==="
echo "This will download all models configured in librechat.yaml"
echo ""

# Models from librechat.yaml
MODELS=(
    "gpt-oss:20b"
    "qwen3:30b-a3b"
    "qwen3:32b"
    "qwen2.5:32b"
    "qwen2.5:14b"
    "qwen2.5:7b"
    "glm-4.5-air"
    "codellama:7b"
    "deepseek-coder:6.7b"
    "deepseek-r1"
    "gemma2:9b"
    "gemma2:27b"
    "llama3.1:8b"
    "llama3.1:70b"
    "llama3.2:3b"
    "llama3.2:11b"
    "mistral:latest"
    "mistral-nemo"
    "mixtral:8x7b"
    "phi3.5:3.8b"
)

# Check if running in Docker or locally
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    OLLAMA_CMD="ollama"
else
    # If running on host, use docker exec
    OLLAMA_CMD="docker exec ollama-server ollama"
fi

echo "Waiting for Ollama to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if $OLLAMA_CMD list > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts: Waiting for Ollama..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: Ollama did not become ready"
    exit 1
fi

echo ""
echo "Current models:"
$OLLAMA_CMD list

echo ""
echo "Downloading models..."
for model in "${MODELS[@]}"; do
    echo ""
    echo "=== $model ==="
    
    # Check if model already exists
    if $OLLAMA_CMD list | grep -q "^$model"; then
        echo "  ✓ $model already exists"
    else
        # Special handling for GLM-4.5-Air (needs to be imported from HuggingFace)
        if [ "$model" = "glm-4.5-air" ]; then
            echo "  → GLM-4.5-Air requires import from HuggingFace (unsloth GGUF)"
            echo "  → Checking if model exists in HF cache..."
            
            # Check HF cache for GLM model
            GLM_HF_PATH="/home/nick/models/hf_models/glm-4.5-air"
            GLM_HUB_PATH="/home/nick/models/hf_models/models--zai-org--GLM-4.5-Air"
            
            if [ -d "$GLM_HF_PATH" ] || [ -d "$GLM_HUB_PATH" ]; then
                echo "  → Found GLM model in HF cache"
                echo "  → Note: To import to Ollama, you need to:"
                echo "     1. Convert to GGUF using unsloth or llama.cpp"
                echo "     2. Use 'ollama create glm-4.5-air -f Modelfile' to import"
                echo "  → For now, use the HF service endpoint for GLM-4.5-Air"
                echo "  → Attempting direct pull (may not be available)..."
                if $OLLAMA_CMD pull "$model" 2>/dev/null; then
                    echo "  ✓ Successfully pulled $model from Ollama registry"
                else
                    echo "  ⚠ GLM-4.5-Air not in Ollama registry - use HF service endpoint"
                fi
            else
                echo "  → GLM model not found in HF cache"
                echo "  → Download from HuggingFace first using download_hf_models.sh"
                echo "  → Then convert to GGUF and import to Ollama"
            fi
        else
            echo "  → Downloading $model (this may take a while)..."
            if $OLLAMA_CMD pull "$model"; then
                echo "  ✓ Successfully downloaded $model"
            else
                echo "  ✗ Failed to download $model"
            fi
        fi
    fi
done

echo ""
echo "=== Download complete ==="
echo "Available models:"
$OLLAMA_CMD list


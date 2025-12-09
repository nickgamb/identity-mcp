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
    "glm-4.5-air"
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
        echo "  → Downloading $model (this may take a while)..."
        if $OLLAMA_CMD pull "$model"; then
            echo "  ✓ Successfully downloaded $model"
        else
            echo "  ✗ Failed to download $model"
        fi
    fi
done

echo ""
echo "=== Download complete ==="
echo "Available models:"
$OLLAMA_CMD list


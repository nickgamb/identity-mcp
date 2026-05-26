#!/usr/bin/env bash
# Download all Ollama models from librechat.yaml.
# Weights persist on the host when docker-compose bind-mounts:
#   ~/models/ollama_models/blobs
#   ~/models/ollama_models/manifests

echo "=== Downloading Ollama Models ==="
echo "This will download all models configured in librechat.yaml"
echo ""

# Models from librechat.yaml
MODELS=(
    # Coding (2026 — replaces ancient codellama:7b / deepseek-coder:6.7b)
    "devstral:24b"          # agentic coding: multi-file edits, codebase tools (Mistral + All Hands)
    "qwen2.5-coder:32b"     # local coding king (~GPT-4o on Aider); fits 1x P40
    "qwen3-coder-next"      # newest MoE coder: 80B/3B-active, 256K ctx; ~52GB, light RAM offload

    # Newer additions (see librechat-config/MODELS.md)
    "qwen3.6:35b"
    "gemma4:31b-it-q8_0"    # Gemma 4 31B dense Q8 (34GB); fits 2x P40
    "gemma4:26b-a4b-it-q8_0" # Gemma 4 26B-A4B MoE Q8 (28GB); fits 2x P40
    "deepseek-r1:32b"
    "qwq:32b"
    "llama3.3:70b"
    "gpt-oss:20b"
    "qwen3:30b-a3b"
    "qwen3:32b"
    "qwen2.5:32b"
    "qwen2.5:14b"
    "qwen2.5:7b"
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


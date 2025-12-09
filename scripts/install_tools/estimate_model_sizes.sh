#!/bin/bash
# Estimate disk space needed for all models

echo "=== Ollama Model Size Estimates ==="
echo ""

# Model sizes in GB (approximate, actual may vary)
declare -A sizes=(
    ["gpt-oss:20b"]=13
    ["qwen3:30b-a3b"]=18
    ["qwen3:32b"]=19
    ["qwen2.5:32b"]=19
    ["qwen2.5:14b"]=8
    ["qwen2.5:7b"]=4.5
    ["glm-4.5-air"]=8
    ["codellama:7b"]=3.8
    ["deepseek-coder:6.7b"]=3.7
    ["deepseek-llm:7b"]=4.2
    ["deepseek-r1"]=4.2
    ["gemma:7b"]=4.5
    ["gemma2:9b"]=5.5
    ["gemma2:27b"]=15
    ["llama3:8b"]=4.7
    ["llama3:latest"]=4.7
    ["llama3.1:8b"]=4.7
    ["llama3.1:70b"]=40
    ["llama3.2:3b"]=2
    ["llama3.2:11b"]=6.5
    ["mistral:latest"]=4.1
    ["mistral-nemo"]=7
    ["mixtral:8x7b"]=26
    ["phi3:3.8b"]=2.3
    ["phi3.5:3.8b"]=2.3
    ["starling-lm:7b"]=4.2
    ["neural-chat:7b"]=4.2
)

total=0
echo "Model sizes:"
echo "----------------------------------------"
for model in "${!sizes[@]}"; do
    size=${sizes[$model]}
    total=$(echo "$total + $size" | bc)
    printf "%-25s %6.1f GB\n" "$model" "$size"
done

echo "----------------------------------------"
printf "%-25s %6.1f GB\n" "TOTAL" "$total"
echo ""
echo "Note: Actual sizes may vary. Ollama uses quantization,"
echo "so models are typically smaller than full precision."
echo ""
echo "Recommendation: Start with essential models and add others as needed."


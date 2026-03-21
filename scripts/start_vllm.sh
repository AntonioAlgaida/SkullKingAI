#!/bin/bash

# Model: Llama-3-8B-Instruct (Quantized for Speed/VRAM)
# MODEL="casperhansen/llama-3-8b-instruct-awq"
# MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# MODEL="cyankiwi/Qwen3.5-9B-AWQ-4bit"
MODEL="openai/gpt-oss-20b"
# Get the absolute path to the models directory
PROJECT_ROOT=$(pwd)
MODEL_DIR="$PROJECT_ROOT/models"

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Storage: $MODEL_DIR"
echo "Port: 8000"

# Create the directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Launch vLLM with the custom download directory
uv run python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --download-dir "$MODEL_DIR" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port 8000
#!/bin/bash
# scripts/start_vllm.sh — Launch vLLM server for SkullZero
#
# Previous alternatives (still fit on RTX 3090):
#   casperhansen/llama-3-8b-instruct-awq         (older, ~5 GB)
#   hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
#   Qwen/Qwen3-8B                                (FP16, ~16 GB — no quantisation loss)
#   bartowski/DeepSeek-R1-Distill-Qwen-14B-exl2  (INT4, ~9 GB — pure reasoning)
#   mistralai/Mistral-Small-3.1-24B-Instruct-2503 (AWQ, ~13 GB — stronger, slower)

MODEL="Qwen/Qwen3-14B-AWQ"

# Always resolve paths relative to the project root, not CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models"

echo "Starting vLLM server..."
echo "Model:   $MODEL"
echo "Storage: $MODEL_DIR"
echo "Port:    8000"

mkdir -p "$MODEL_DIR"

uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --download-dir "$MODEL_DIR" \
    --dtype float16 \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --port 8000

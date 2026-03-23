#!/bin/bash
# scripts/start_vllm.sh — Launch vLLM server for SkullZero
#
# The model is read from conf/config.yaml (llm.model_name) so there is a single
# source of truth. To change the model, edit config.yaml only.
#
# Previous alternatives (still fit on RTX 3090):
#   casperhansen/llama-3-8b-instruct-awq         (older, ~5 GB)
#   hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
#   Qwen/Qwen3-8B                                (FP16, ~16 GB — no quantisation loss)
#   Qwen/Qwen3-14B-AWQ                           (INT4, ~9 GB — previous default)
#   QuixiAI/Qwen3-30B-A3B-AWQ                   (MoE INT4, ~17 GB — 3B active params)
#   bartowski/DeepSeek-R1-Distill-Qwen-14B-exl2  (INT4, ~9 GB — pure reasoning)
#   mistralai/Mistral-Small-3.1-24B-Instruct-2503 (AWQ, ~13 GB — stronger, slower)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_ROOT/conf/config.yaml"

# Read model name from config.yaml — no external tools required, pure Python
MODEL=$(python3 -c "
import sys
with open('$CONFIG') as f:
    for line in f:
        line = line.strip()
        if line.startswith('model_name:'):
            print(line.split(':', 1)[1].strip().strip('\"'))
            sys.exit(0)
")

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
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --port 8000

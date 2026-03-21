#!/usr/bin/env bash
# run_loop.sh — Full SkullZero training loop
#
# Usage:
#   ./run_loop.sh [ITERATIONS]
#
# Each iteration:
#   1. run_parallel.py  — Play N concurrent games, save traces, update ELO
#   2. run_sleep_cycle.py — Reflect on new traces, write rules to ChromaDB
#   3. run_pruning.py   — Audit ChromaDB, delete hallucinations/fluff
#
# Prerequisites: vLLM server must be running on localhost:8000
#   bash scripts/start_vllm.sh

set -euo pipefail

ITERATIONS=${1:-3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  SkullZero Training Loop — ${ITERATIONS} iteration(s)"
echo "============================================"

# Verify vLLM is reachable before starting
if ! curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
    echo "[ERROR] vLLM server not reachable at http://localhost:8000"
    echo "        Run: bash scripts/start_vllm.sh"
    exit 1
fi

for i in $(seq 1 "$ITERATIONS"); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ITERATION ${i} / ${ITERATIONS}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo ""
    echo ">>> [1/3] Wake Cycle — Playing games..."
    cd "$SCRIPT_DIR"
    uv run python run_parallel.py

    echo ""
    echo ">>> [2/3] Sleep Cycle — Reflecting on traces..."
    cd "$SCRIPT_DIR"
    uv run python run_sleep_cycle.py

    echo ""
    echo ">>> [3/3] Pruning Cycle — Cleaning memory..."
    cd "$SCRIPT_DIR"
    uv run python run_pruning.py

    echo ""
    echo "  Iteration ${i} complete."
done

echo ""
echo "============================================"
echo "  All ${ITERATIONS} iteration(s) done."
echo "  Memory: data/chroma_db/"
echo "  ELO:    data/elo_ratings.json"
echo "  Traces: outputs/"
echo "============================================"

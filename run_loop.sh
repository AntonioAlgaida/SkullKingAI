#!/usr/bin/env bash
# run_loop.sh — Full SkullZero training loop
#
# Usage:
#   ./run_loop.sh [ITERATIONS] [MINI_BATCHES]
#
# Each outer iteration:
#   MINI_BATCHES × (run_parallel.py → run_sleep_cycle.py)   ← learn frequently
#   run_pruning.py                                            ← clean once per outer iter
#
# Example: ./run_loop.sh 5 3
#   → 5 outer iterations, each with 3 mini-batches of games+sleep before pruning.
#   → Agents receive updated rules 3× more often than a flat loop.
#
# Prerequisites: vLLM server must be running on localhost:8000
#   bash scripts/start_vllm.sh

set -euo pipefail

ITERATIONS=${1:-3}
MINI_BATCHES=${2:-3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  SkullZero Training Loop"
echo "  Iterations : ${ITERATIONS}"
echo "  Mini-batches per iteration : ${MINI_BATCHES}"
echo "  (games+sleep × ${MINI_BATCHES}, then prune once)"
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

    for b in $(seq 1 "$MINI_BATCHES"); do
        echo ""
        echo "  ── Mini-batch ${b} / ${MINI_BATCHES} ──"

        echo "  >>> [Wake]  Playing games..."
        cd "$SCRIPT_DIR"
        uv run python run_parallel.py

        echo "  >>> [Sleep] Reflecting on new traces..."
        cd "$SCRIPT_DIR"
        uv run python run_sleep_cycle.py
    done

    echo ""
    echo "  >>> [Prune] Cleaning memory (once per iteration)..."
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

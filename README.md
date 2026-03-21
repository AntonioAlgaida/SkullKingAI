# SkullKingAI: Neuro-Symbolic LLM Agents for Skull King

SkullKingAI is a research framework for studying **Asymmetric Co-Evolution** in Imperfect Information Games (IIG). It applies a **Neuro-Symbolic architecture** to the trick-taking game *Skull King*.

## Architecture Highlights
- **Symbolic Layer:** Python state machine enforcing all game rules and legal action masking.
- **Neural Layer:** LLM-based reasoning (optimized for vLLM/Qwen/Llama) using Chain-of-Thought.
- **Reflective RAG:** A self-improvement loop where agents analyze game logs (Sleep Cycle) to discover and store strategies in a Vector DB.
- **Memory Consolidation:** An automated Pruning Cycle that uses an LLM-as-a-Judge to maintain high-quality strategic heuristics.

## Setup
Developed using `uv` and `vLLM` on WSL2.

```bash
uv sync
./scripts/start_vllm.sh
uv run python run_parallel.py
```
```
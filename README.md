# SkullKingAI: Neuro-Symbolic LLM Agents for Imperfect Information Games

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/inference-vLLM-purple.svg)](https://github.com/vllm-project/vllm)
[![ChromaDB](https://img.shields.io/badge/vectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SkullKingAI** is a research framework exploring **Asymmetric Co-Evolution** and **Verbal Reinforcement Learning** in complex, multi-agent Imperfect Information Games (IIGs). 

Traditional Reinforcement Learning (RL) like PPO struggles with trick-taking games that feature dynamic alliances, strict suit-following constraints, and highly punitive exact-bidding mechanics. This project solves these bottlenecks using a **Neuro-Symbolic Architecture**, combining a deterministic Python state machine with Large Language Models (LLMs) acting as strategic reasoning agents.

## 🧠 Key Research Features

- **Neuro-Symbolic Engine**: Python strictly enforces game physics and legal action masking (Symbolic), while the LLM manages Theory of Mind and strategic planning (Neural).
- **Reflective RAG (Wake/Sleep Cycle)**: Agents play games relying on their current "Grimoire" of strategies. Offline, they analyze game traces to identify the root causes of failed bids and generate new natural language rules (Hindsight Experience Replay).
- **Automated Memory Consolidation**: To prevent context degradation, a `MemoryPruner` uses semantic vector distance (L2) and an *LLM-as-a-Judge* to audit the database, deleting hallucinations, contradictions, and duplicates.
- **High-Throughput Asynchronous Simulation**: Fully integrated with **vLLM** and `asyncio`, enabling continuous batching of concurrent games to maximize GPU utilization (tested on RTX 3090 24GB).

## 🏴‍☠️ The Environment: Skull King
Skull King is a highly volatile trick-taking game played over 10 rounds. 
- **The Challenge**: Players must bid the *exact* number of tricks they will win. Overbidding or underbidding results in heavy penalties.
- **The "Zero Meta"**: Bidding `0` offers massive rewards but devastating penalties if even a single trick is won. This creates a fascinating "Tragedy of the Commons" dynamic where agents actively try to force each other to win.
- **Non-Transitive Mechanics**: The game features a Rock-Paper-Scissors hierarchy (Mermaid > Skull King > Pirate > Mermaid) that overrides standard trump logic.

## 🏗️ Architecture & Pipeline

1. **Phase 1: Game Execution (The Wake Cycle)**
   - Agents translate the numeric game state into a semantic "Threat HUD" (e.g., assessing if an opponent is *STARVING* or *FULL* based on their bid).
   - They query **ChromaDB** for specific rules relevant to their current hand and phase.
   - LLMs use **Chain-of-Thought (CoT)** to select an action.
2. **Phase 2: Hindsight Reflection (The Sleep Cycle)**
   - After a batch of games, the Reflector isolates tricks where agents failed their bids.
   - The LLM analyzes the play-by-play log and the agent's starting hand to deduce strategic errors.
   - A new rule is written and embedded into ChromaDB.
3. **Phase 3: Garbage Collection (The Pruning Cycle)**
   - The DB is audited to remove low-quality or contradictory rules, maintaining a high signal-to-noise ratio in the RAG pipeline.

## 🚀 Getting Started

### Prerequisites
- Linux / WSL2
- NVIDIA GPU (24GB VRAM recommended for 27B+ parameter models)
-[uv](https://github.com/astral-sh/uv) (Python package manager)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SkullKingAI.git
cd SkullKingAI

# 2. Pin Python version and sync dependencies
uv python pin 3.11
uv sync
```

### Running the Inference Server
This project is optimized for running quantized reasoning models (e.g., Qwen3.5-27B AWQ or Llama-3-8B AWQ) using vLLM.

```bash
# Start the vLLM server (Defaults to port 8000)
./scripts/start_vllm.sh
```

### Running the Pipeline
Open a new terminal. You can manage the entire pipeline using the provided scripts:

```bash
# 1. Run 10 asynchronous concurrent games
uv run python run_parallel.py

# 2. Trigger the Sleep Cycle to analyze logs and generate rules
uv run python run_sleep_cycle.py

# 3. Clean the memory database of hallucinations
uv run python run_pruning.py

# 4. Export the AI's learned strategies to a Markdown file
uv run python scripts/visualize_memory.py
```

## 📂 Repository Structure

```text
SkullKingAI/
├── conf/                  # Hydra configuration files (Agents, LLM settings, etc.)
├── data/
│   ├── chroma_db/         # Local Vector Database (Ignored in Git)
│   └── artifacts/         # Exported strategy markdowns
├── scripts/               # Bash scripts for inference servers
├── src/
│   ├── agents/            # LLM Client, Base Agents, and Heuristic Baselines
│   ├── engine/            # Deterministic Python State Machine and Physics
│   ├── memory/            # RAG Engine, Reflector (Sleep), and Pruner
│   ├── prompts/           # Modular text files for rules and personas
│   └── utils/             # Semantic Translators and Prompt Loaders
├── main.py                # Single-game sequential execution
├── run_parallel.py        # Asynchronous multi-game execution
├── run_sleep_cycle.py     # Hindsight reflection trigger
└── run_pruning.py         # Memory consolidation trigger
```

## 💡 Example of Emergent Strategy
Without hardcoding, the agentic pipeline independently discovered the concept of "Sloughing" (discarding high-value cards legally):

> **[RULE]:** *When bidding 0, always prioritize sloughing off high cards or Specials when you are void in the lead suit to avoid forcing an opponent to win tricks.*
> 
> **[RULE]:** *If you bid 0 and hold a high Special card (Pirate, King, etc.), prioritize playing it in a subsequent trick where you are likely to lose or force another player to win, rather than holding onto it for an uncertain outcome.*


**Author:** Antonio Guillen-Perez
**License:** MIT
```
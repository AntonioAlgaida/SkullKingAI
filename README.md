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
- **Engine-Verified Counterfactual Simulation**: At the critical trick where a player's bid failure became inevitable, the engine replays the trick with every legal alternative card (ceteris-paribus). The LLM receives ground-truth evidence ("if you had played X, you *would have* lost the trick") rather than having to speculate.
- **Symbolic Action Labelling (SAL)**: A deterministic extension of legal-action masking. Before each decision, the physics engine annotates every legal card with its guaranteed or probable trick outcome — `→ WINS ✓`, `→ LEADS`, `→ LOSES ✗`, or `→ DESTROYS TRICK`. When the agent plays last, outcomes are fully certain. This eliminates a class of positional reasoning errors (e.g. playing a second Pirate believing it will win when the first Pirate already leads). SAL is an engine *observation*, not a value function.
- **Current Trick Winner Annotation**: Every PLAYING prompt includes a real-time annotation of who is currently winning the trick and exactly what card type would beat them (e.g. *"⚠ CURRENT WINNER: P2 with [Pirate]. Only [Skull King] beats it. A second Pirate CANNOT win."*). Computed by calling `physics.resolve_trick()` on the partial trick.
- **Bid Status Context**: The PLAYING prompt always shows the agent's current bid progress — *"I need N more win(s)"*, *"⚠ BID MET — must LOSE all remaining tricks"*, or *"⚠ OVERBID — bid already failed"* — so the LLM knows exactly what is at stake without having to recompute it from raw numbers.
- **Rule Fitness Tracking**: Every ChromaDB rule carries a fitness score updated after each round. Rules relevant to a winning situation receive `+0.8`; rules active during a failure receive `−0.2`, clamped to `[−2.0, 5.0]`. Fitness adjusts retrieval ranking (`effective_distance = semantic_distance − fitness × 0.05`), so proven rules rise to the surface over time.
- **Critical-Trick Credit Assignment**: PLAYING fitness credit is anchored to the state at the exact trick where failure became inevitable — the cards in play, the player's remaining hand, and what they actually played — rather than a general round-level description. BIDDING rules retain round-level credit (bid is placed at round start). Success credit uses round-level context (no single failure point). This is the credit assignment analogue of Hindsight Experience Replay applied to a symbolic rule base.
- **Automated Memory Consolidation**: A `MemoryPruner` uses an *LLM-as-a-Judge* — auditing PLAYING and BIDDING rules separately against their respective strategy bundles — to delete hallucinations, contradictions, and fluff. Full rule text is logged before deletion for audit trail.
- **Semantic Deduplication**: Before storing any new rule, ChromaDB is queried for the closest existing rule. Rules with semantic distance < 0.50 (L2) are rejected as near-paraphrases of already-known insights, preventing the Grimoire from filling with redundant variations of the same lesson.
- **vLLM Prefix Caching**: The static game-rules bundle (~3000 tokens) is sent as the system role so vLLM can cache and reuse its KV representation across all concurrent game calls. Only the dynamic game state (~300–500 tokens) changes per request.
- **High-Throughput Asynchronous Simulation**: Fully integrated with **vLLM** and `asyncio`, enabling continuous batching of concurrent games to maximise GPU utilisation (tested on RTX 3090 24 GB). Default model: `QuixiAI/Qwen3-30B-A3B-AWQ` (MoE, 30B parameters, 3B active — fast inference with large-model knowledge).
- **Centralised Model Configuration**: `scripts/start_vllm.sh` reads `llm.model_name` directly from `conf/config.yaml`. Changing the model requires editing one file only.
- **Async Sleep Cycle**: All reflection LLM calls within a trace are fired concurrently via `asyncio.gather()`, with prefix-cached system prompts — eliminating the serial 3-min-per-call bottleneck of a naive implementation.
- **Stratified Starting Rounds**: Concurrent games are assigned different starting rounds (e.g., 3 games → rounds 1, 4, 7) so every mini-batch covers early, mid, and late-game situations. The engine supports starting from any round via `env.reset(starting_round=k)`. Each game plays only its assigned round(s) (`rounds_per_game: 1`) before the sleep cycle fires, maximising reflection frequency.
- **Randomised Starting Player Offset**: The leading seat is randomised each game so no persona is systematically disadvantaged by always acting first.
- **Mini-Batch Training Loop**: `run_loop.sh` runs `MINI_BATCHES × (games + sleep)` before each pruning pass, so agents receive updated rules several times per outer iteration rather than once. Supports resuming from a specific iteration: `./run_loop.sh [ITERATIONS] [MINI_BATCHES] [START_ITER]`.
- **Fitness-Tiered RAG Injection**: Retrieved rules are presented to the LLM in three tiers — `★ PROVEN` (fitness ≥ 2.0), `◆ EXPERIMENTAL` (fitness 0–2), and `✗ WEAK` (fitness < 0) — with the semantic retrieval query shown as a header so the LLM understands *why* each rule applies. Phase-specific instructions force the agent to explicitly link a proven rule to its action rather than treating memory as optional background.
- **Evaluation Protocol**: `run_eval.py` runs a fixed set of games with deterministic seeds and a read-only Grimoire (no rule updates), appending per-iteration metrics to `data/eval_log.jsonl`. Configurable round limit (`eval_rounds_per_game`, default 3) for fast checkpointing. Each eval game writes a full debug log and play-by-play to `logs/eval_iter{N}_game{M}.log`.
- **Training Curve Plots**: `scripts/plot_training_curves.py` reads `eval_log.jsonl` and generates PNG plots for mean score, bid error, Grimoire size, mean fitness, zero-success rate, and ELO trajectory — with ± std deviation bands.
- **ELO Tracking**: `EloTracker` maintains pairwise ELO ratings across all games (training and evaluation). ELO history is snapshotted at each evaluation checkpoint and logged to `eval_log.jsonl`.
- **Per-Game Logging**: Each concurrent game writes to its own `logs/game_{id}.log` and a line-buffered `logs/game_{id}_play.txt` play-by-play file, `tail -f` compatible for live inspection.

## 🏴‍☠️ The Environment: Skull King

Skull King is a highly volatile trick-taking game played over 10 rounds.
- **The Challenge**: Players must bid the *exact* number of tricks they will win. Overbidding or underbidding results in heavy penalties.
- **The "Zero Meta"**: Bidding `0` offers massive rewards but devastating penalties if even a single trick is won. This creates a fascinating "Tragedy of the Commons" dynamic where agents actively try to force each other to win unwanted tricks.
- **Non-Transitive Mechanics**: The game features a Rock-Paper-Scissors hierarchy (Mermaid > Skull King > Pirate > Mermaid) that overrides standard trump logic.

## 🏗️ Architecture & Pipeline

### Wake Cycle (Game Execution)
1. The **SemanticTranslator** converts the numeric game state into a rich text "Threat HUD" — assessing whether each opponent is *STARVING*, *FULL*, *OVERBOARD*, etc.
2. **StrategyMemory** queries ChromaDB for rules relevant to the current phase and situation, re-ranking candidates by fitness-adjusted distance.
3. The LLM uses **Chain-of-Thought (CoT)** reasoning to select an action from the legal action set.

### Sleep Cycle (Hindsight Reflection)
1. The **Reflector** loads completed game traces and groups events by player and round.
2. For each failed round, it identifies the **critical trick** (the exact decision point where failure became inevitable) and builds a credit query anchored to that state — the cards in play, the player's remaining hand, and the card actually played.
3. **CounterfactualSimulator** reconstructs the player's hand at the critical trick, enumerates follow-suit-legal alternatives, and calls `physics.resolve_trick()` for each — providing engine-verified evidence to the LLM.
4. New rules are embedded into ChromaDB with `fitness=0.0`. PLAYING rules relevant to the critical trick state have their fitness updated (`FITNESS_WIN=+0.8` / `FITNESS_LOSS=−0.2`, clamped to `[−2.0, 5.0]`). BIDDING rules use a round-level query (bid is placed at round start).
5. All LLM calls for a trace are batched into a single `asyncio.gather()`. Static game rules are in the system role for prefix caching; only the play-by-play and task are in the user role.
6. `<think>` blocks are stripped from LLM responses before `[RULE]:` extraction, ensuring rule parsing succeeds even when chain-of-thought fills most of the token budget.

### Pruning Cycle (Garbage Collection)
1. The **MemoryPruner** audits PLAYING and BIDDING rules in separate batches, each against its matching strategy bundle.
2. An *LLM-as-a-Judge* prompt classifies each rule as valid or one of: HALLUCINATION, CONTRADICTION, USELESS FLUFF, or ARTIFACT.
3. Flagged rules are permanently deleted, maintaining a high signal-to-noise ratio in the RAG pipeline.

### Full Loop
```bash
# MINI_BATCHES × (games + sleep) per iteration, then prune + eval once
./run_loop.sh [ITERATIONS] [MINI_BATCHES] [START_ITER]

# Example: 20 outer iterations, 3 mini-batches each → 9 rounds + 3 sleep cycles per prune
./run_loop.sh 20 3

# Resume from iteration 15 (continues eval_log.jsonl without duplicate entries)
./run_loop.sh 6 3 15
```

## 🚀 Getting Started

### Prerequisites
- Linux / WSL2
- NVIDIA GPU (24 GB VRAM recommended for 14B+ parameter models)
- [uv](https://github.com/astral-sh/uv) (Python package manager)

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
The project is optimised for AWQ-quantized models via vLLM. The active model is set in `conf/config.yaml` (`llm.model_name`) and read automatically by `scripts/start_vllm.sh` — change it in one place only.

Default: `QuixiAI/Qwen3-30B-A3B-AWQ` (MoE, 30B total / 3B active parameters, ~17 GB VRAM).

```bash
# Start the vLLM server with prefix caching enabled (port 8000)
bash scripts/start_vllm.sh
```

Key flags set by the script: `--max-model-len 8192`, `--enable-prefix-caching`, `--max-num-seqs 16`.

The prefix cache hit rate should reach ~91% once the server is warm — verify with the vLLM stats log line: `Prefix cache hit rate: XX%`.

Alternative models that fit in 24 GB on a single RTX 3090:

| Model | VRAM | Notes |
|---|---|---|
| `QuixiAI/Qwen3-30B-A3B-AWQ` | ~17 GB | Default — MoE, fast inference |
| `Qwen/Qwen3-14B-AWQ` | ~9 GB | Previous default — smaller, slower |
| `Qwen/Qwen3-8B` | ~16 GB | FP16, no quantisation loss |

### Running the Pipeline
```bash
# Run concurrent games (configurable in conf/config.yaml)
uv run python run_parallel.py

# Trigger the Sleep Cycle (counterfactual analysis + rule generation + fitness update)
uv run python run_sleep_cycle.py

# Clean the memory database of hallucinations and contradictions
uv run python run_pruning.py

# Evaluate agents with fixed seeds and read-only Grimoire
uv run python run_eval.py experiment.eval_iteration=1

# Export learned strategies to Markdown
uv run python scripts/visualize_memory.py

# Plot training curves from eval_log.jsonl
uv run python scripts/plot_training_curves.py

# Run the full loop: 20 iterations × 3 mini-batches, starting from iteration 1
./run_loop.sh 20 3

# Resume from iteration 15
./run_loop.sh 6 3 15
```

### Monitoring Live Games
Each concurrent game writes a line-buffered play-by-play file:
```bash
tail -f logs/game_3_play.txt
```

Each evaluation game writes a full debug log and play-by-play:
```bash
tail -f logs/eval_iter0001_game0_play.txt
```

## 📂 Repository Structure

```text
SkullKingAI/
├── conf/                  # Hydra configuration (agents, LLM settings, etc.)
├── data/
│   ├── chroma_db/         # Local Vector Database (not tracked in Git)
│   └── artifacts/         # Exported strategy markdowns
├── logs/                  # Per-game logs and play-by-play files
├── scripts/               # Bash scripts (vLLM server, memory visualisation, training curve plots)
├── src/
│   ├── agents/            # LLM client, agent base class, heuristic baselines
│   ├── engine/            # Deterministic game state machine and physics
│   ├── memory/
│   │   ├── rag_engine.py         # ChromaDB wrapper + fitness re-ranking
│   │   ├── reflector.py          # Sleep cycle: trace analysis + rule generation
│   │   ├── counterfactual.py     # Engine-verified alternative card simulation
│   │   ├── pruner.py             # LLM-as-Judge memory garbage collector
│   │   └── elo_tracker.py        # Per-agent ELO tracking across games
│   ├── prompts/           # Modular text files (game rules, persona prompts)
│   └── utils/             # Semantic translators, prompt loaders, play-by-play
├── tests/                 # Pytest suite (physics, env, counterfactual, RAG, reflector)
├── run_parallel.py        # Async multi-game execution
├── run_sleep_cycle.py     # Sleep cycle trigger
├── run_pruning.py         # Pruning cycle trigger
├── run_eval.py            # Evaluation protocol (fixed seeds, read-only Grimoire, eval_log.jsonl)
└── run_loop.sh            # Full pipeline loop orchestrator
```

## 💡 Example of Emergent Strategy

Without hardcoding, the agentic pipeline independently discovered the concept of "Sloughing" (discarding high-value cards legally):

> **[Strategy]:** *When holding any trump of value 5 or lower (Black 1–5) without a higher trump or a Pirate/Skull King to back it up, do **not** count those low trumps toward your bid; instead subtract one from the estimated number of tricks you expect to win.*

> **[Strategy]:** *If you hold a **Pirate** in a hand that also contains low-value suit cards and the round's bid is modest (≤ 3), avoid leading a suit card that you can beat with a higher suit or trump. Instead, lead the Pirate immediately. This forces any opponent who holds the corresponding higher-ranked suit or a Trump to win the trick (or lose it if they also have a special), thereby preventing them from using their high cards later and giving you a better chance to hit your exact bid.*

## 🔄 Clean Reset

To restart training from scratch, removing all learned rules, ELO ratings, evaluation logs, and game traces:

```bash
cd /home/anton/SkullZero

# Optional: archive the previous Grimoire export before wiping
cp data/artifacts/Rules_Found.md data/artifacts/Rules_Found_run1.md

# Grimoire (vector DB)
rm -rf data/chroma_db/

# Persistent state files
rm -f data/elo_ratings.json
rm -f data/eval_log.jsonl
rm -f data/processed_traces.json

# All game traces and Hydra output directories
rm -rf outputs/

echo "Reset complete. Ready to run: ./run_loop.sh 20 3"
```

**What is kept:** `data/artifacts/` (research notes, plots), all source code, configuration, and scripts.

Then start a fresh run:

```bash
./run_loop.sh 20 3
```

---

**Author:** Antonio Guillen-Perez
**License:** MIT

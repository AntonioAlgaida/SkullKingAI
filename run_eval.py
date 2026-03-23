# run_eval.py
#
# Evaluation protocol: run K games with a READ-ONLY Grimoire (retrieve only,
# no rule generation, no fitness updates, no pruning) using fixed seeds so
# every iteration is evaluated on identical card deals.
#
# Results are appended to data/eval_log.jsonl — one JSON line per checkpoint.
# Call from run_loop.sh after each prune step:
#   uv run python run_eval.py experiment.eval_iteration=N

import asyncio
import datetime
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.engine.state import SkullKingEnv
from src.agents.heuristic import HeuristicAgent
from src.agents.llm_agent import LLMAgent
from src.agents.llm_client import LLMClient
from src.memory.elo_tracker import EloTracker
from src.memory.rag_engine import StrategyMemory
from src.utils.translators import SemanticTranslator
from src.utils.play_by_play import PlayByPlay

logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# Path is relative to the project root, not the Hydra output dir
EVAL_LOG = Path(__file__).resolve().parent / "data" / "eval_log.jsonl"


def _make_eval_logger(game_id: int, iteration: int) -> logging.Logger:
    """Creates a per-game logger that writes to logs/eval_iter{N}_game{M}.log."""
    os.makedirs("logs", exist_ok=True)
    name = f"eval.game.{game_id}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Remove any handlers added by a previous eval run in the same process
    logger.handlers.clear()
    fh = logging.FileHandler(f"logs/eval_iter{iteration:04d}_game{game_id}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Single evaluation game
# ─────────────────────────────────────────────────────────────────────────────

async def _run_eval_game(
    game_id: int,
    seed: int,
    iteration: int,
    ending_round: int,
    cfg: DictConfig,
    client: LLMClient,
    memory: StrategyMemory,
) -> Optional[Dict]:
    """
    Runs one eval game from round 1 through ending_round with a fixed numpy seed.
    The Grimoire is used for retrieval only — no memorize_rule, no update_fitness.
    Returns a dict of per-round metrics, or None if the game aborted.
    """
    # Fix the card shuffle and starting player for this game_id so all
    # iterations see identical conditions (deterministic but varied across games).
    import random as _random
    np.random.seed(seed)
    _random.seed(seed)

    env        = SkullKingEnv(num_players=cfg.game.num_players)
    start_player_offset = _random.randint(0, cfg.game.num_players - 1)
    state      = env.reset(starting_round=1, start_player_offset=start_player_offset)
    translator = SemanticTranslator(env.physics)

    agents: list = []
    player_personas: Dict[int, str] = {}
    game_logger = _make_eval_logger(game_id, iteration)

    for i in range(cfg.game.num_players):
        agent_type = cfg.agents.get(f"p{i}", "heuristic")
        if agent_type == "heuristic":
            agents.append(HeuristicAgent(env.physics))
        else:
            agent = LLMAgent(
                client=client, translator=translator,
                persona=agent_type, memory=memory,
                logger=game_logger,
            )
            agent.max_retries = cfg.llm.max_retries
            agents.append(agent)
        player_personas[i] = agent_type

    # Per-round tracking for bid accuracy
    round_records: List[Dict] = []   # [{round, bids, won}]

    # Play-by-play for human-readable per-game log
    pbp_path = f"logs/eval_iter{iteration:04d}_game{game_id}_play.txt"
    pbp = PlayByPlay(game_id, pbp_path, translator)
    _pbp_round   = -1
    _round_hands = {}
    _round_bids  = []
    _prev_totals = [0] * cfg.game.num_players

    done   = False
    failed = False
    while not done:
        pid = state["current_player_id"]
        try:
            if hasattr(agents[pid], "a_act"):
                action_id = await agents[pid].a_act(state)
            else:
                action_id = agents[pid].act(state)
        except Exception as e:
            game_logger.error(f"[Eval game {game_id}] P{pid} error: {e}")
            failed = True
            break

        # Play-by-play recording (mirrors run_parallel.py)
        if state["phase"] == "BIDDING":
            if state["round_num"] != _pbp_round:
                _pbp_round   = state["round_num"]
                _round_hands = {}
                _round_bids  = []
            _round_hands[pid] = list(state["my_hand"])
            _round_bids.append((pid, player_personas[pid], action_id))
        elif state["phase"] == "PLAYING":
            if _round_bids:
                pbp.round_start(state["round_num"], _round_hands, player_personas)
                for _p, _persona, _bid in _round_bids:
                    pbp.bid(_p, _persona, _bid)
                _round_bids = []
                pbp.trick_start(pid)
            pbp.play(pid, player_personas[pid], action_id)

        new_state, _, done, info = env.step(action_id)

        if "trick_winner" in info:
            if info.get("trick_destroyed"):
                next_leader = new_state["current_player_id"]
                pbp.trick_destroyed(next_leader)
                if not done and new_state["phase"] == "PLAYING":
                    pbp.trick_start(next_leader)
            elif info["trick_winner"] != -1:
                pbp.trick_end(info["trick_winner"], info.get("trick_bonus", 0))
                if not done and new_state["phase"] == "PLAYING":
                    pbp.trick_start(new_state["current_player_id"])

        if "round_rewards" in info:
            new_totals   = list(new_state["scores"])
            round_deltas = [new_totals[i] - _prev_totals[i] for i in range(len(new_totals))]
            pbp.round_end(state["round_num"], list(info["bids"]), list(info["won"]),
                          round_deltas, new_totals, player_personas)
            _prev_totals = new_totals
            round_records.append({
                "round":  state["round_num"],
                "bids":   list(info["bids"]),
                "won":    list(info["won"]),
                "scores": list(new_state["scores"]),
            })
            game_logger.info(
                f"[Eval game {game_id}] Round {state['round_num']} done. "
                f"Scores: {new_totals}  Bids: {list(info['bids'])}  Won: {list(info['won'])}"
            )
            if state["round_num"] >= ending_round:
                done = True

        state = new_state

    if failed:
        return None

    final_scores = state["scores"]
    pbp.game_end(final_scores, player_personas)
    game_logger.info(f"[Eval game {game_id}] FINISHED. Final scores: {final_scores}")
    log.info(f"[Eval] Game {game_id} done. Final scores: {final_scores}")

    return {
        "game_id":         game_id,
        "seed":            seed,
        "final_scores":    final_scores,
        "player_personas": player_personas,
        "round_records":   round_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    game_results: List[Dict],
    player_personas: Dict[int, str],
    memory: StrategyMemory,
    elo: EloTracker,
) -> Dict:
    """Aggregates raw game results into the metrics dict saved to eval_log.jsonl."""

    # Collect per-persona final scores and per-round bid errors
    scores_by_persona:     Dict[str, List[float]] = {}
    bid_errors_by_persona: Dict[str, List[float]] = {}
    zero_success_by_round: Dict[str, List[int]]   = {}   # forced_zero only

    for result in game_results:
        if result is None:
            continue
        personas = result["player_personas"]

        # Update ELO from this eval game
        player_results = [(personas[pid], result["final_scores"][pid]) for pid in sorted(personas.keys())]
        elo.update_from_game(player_results)

        for pid, persona in personas.items():
            scores_by_persona.setdefault(persona, []).append(result["final_scores"][pid])

        for rec in result["round_records"]:
            for pid, persona in personas.items():
                bid = rec["bids"][pid]
                won = rec["won"][pid]
                if bid < 0:   # bid not yet placed (shouldn't happen at round_end)
                    continue
                error = abs(bid - won)
                bid_errors_by_persona.setdefault(persona, []).append(error)
                if persona == "forced_zero":
                    zero_success_by_round.setdefault(persona, []).append(1 if won == 0 else 0)

    def _stats(values: List[float]) -> Dict:
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(statistics.mean(values), 3),
            "std":  round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "n":    len(values),
        }

    metrics: Dict = {}
    for persona in set(player_personas.values()):
        entry = {
            "score":     _stats(scores_by_persona.get(persona, [])),
            "bid_error": _stats(bid_errors_by_persona.get(persona, [])),
        }
        if persona == "forced_zero":
            zs = zero_success_by_round.get(persona, [])
            entry["zero_success_rate"] = round(statistics.mean(zs), 3) if zs else None
        metrics[persona] = entry

    # Grimoire stats (read-only — just count and mean fitness)
    grimoire: Dict = {}
    for persona_key, collection in [
        ("rational",    memory.rational_collection),
        ("forced_zero", memory.zero_collection),
    ]:
        count = collection.count()
        mean_fitness = None
        if count > 0:
            data = collection.get(include=["metadatas"])
            fitnesses = [float(m.get("fitness", 0.0)) for m in data["metadatas"]]
            mean_fitness = round(statistics.mean(fitnesses), 4) if fitnesses else None
        grimoire[persona_key] = {"size": count, "mean_fitness": mean_fitness}

    # ELO snapshot after eval games
    elo_snapshot: Dict = {
        persona: round(elo.get_elo(persona), 1)
        for persona in set(player_personas.values())
    }

    return {"by_persona": metrics, "grimoire": grimoire, "elo": elo_snapshot}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    iteration        = int(getattr(cfg.experiment, "eval_iteration",      0))
    eval_games       = int(getattr(cfg.experiment, "eval_games",          10))
    seed_offset      = int(getattr(cfg.experiment, "eval_seed_offset",  9000))
    rounds_per_game  = getattr(cfg.experiment, "eval_rounds_per_game",   None)
    ending_round     = int(rounds_per_game) if rounds_per_game else 10

    log.info(f"=== EVALUATION — iteration {iteration}, {eval_games} games, rounds 1–{ending_round} ===")
    log.info(f"Seeds: {seed_offset} … {seed_offset + eval_games - 1}  (fixed, read-only Grimoire)")

    client = LLMClient(
        base_url=cfg.llm.base_url,
        model_name=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
        enable_thinking=cfg.llm.enable_thinking,
    )
    # Read-only: pass to agents for retrieval; no sleep/prune calls in this script
    memory = StrategyMemory(persistence_path="data/chroma_db")
    elo    = EloTracker(persistence_path="data/elo_ratings.json")

    # Resolve player_personas from config (same mapping as run_parallel.py)
    player_personas = {i: cfg.agents.get(f"p{i}", "heuristic") for i in range(cfg.game.num_players)}

    async def run_all():
        tasks = [
            _run_eval_game(
                game_id=i,
                seed=seed_offset + i,
                iteration=iteration,
                ending_round=ending_round,
                cfg=cfg,
                client=client,
                memory=memory,
            )
            for i in range(eval_games)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    raw_results = asyncio.run(run_all())

    # Filter out exceptions (treat as None / aborted)
    game_results = [
        r if isinstance(r, dict) else None
        for r in raw_results
    ]
    completed = sum(1 for r in game_results if r is not None)
    log.info(f"Completed {completed}/{eval_games} eval games.")

    metrics = _compute_metrics(game_results, player_personas, memory, elo)

    # ── Save eval traces to eval_traces/ (ignored by run_sleep_cycle.py) ──
    os.makedirs("eval_traces", exist_ok=True)
    for result in game_results:
        if result is None:
            continue
        path = f"eval_traces/iter{iteration:04d}_game{result['game_id']}_trace.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    # ── Append one line to data/eval_log.jsonl ──
    record = {
        "iteration":   iteration,
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_games":  eval_games,
        "completed":   completed,
        **metrics,
    }

    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    # ── Human-readable summary ──
    log.info("─" * 55)
    log.info(f"  Iteration {iteration} evaluation summary")
    log.info("─" * 55)
    for persona, stats in metrics["by_persona"].items():
        score = stats["score"]
        berr  = stats["bid_error"]
        log.info(
            f"  {persona:15s}  score={score['mean']:+.1f} ± {score['std']:.1f}"
            f"  bid_err={berr['mean']:.3f}"
            + (f"  zero_ok={stats['zero_success_rate']:.2%}"
               if "zero_success_rate" in stats and stats["zero_success_rate"] is not None
               else "")
        )
    for persona, g in metrics["grimoire"].items():
        log.info(f"  Grimoire [{persona:12s}]  rules={g['size']}  mean_fitness={g['mean_fitness']}")
    log.info("  ELO: " + "  ".join(f"{p}={v}" for p, v in sorted(metrics["elo"].items())))
    log.info("─" * 55)
    log.info(f"  Results appended to: {EVAL_LOG}")


if __name__ == "__main__":
    main()

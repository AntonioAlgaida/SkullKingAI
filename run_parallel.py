# run_parallel.py

import asyncio
import logging
import random
import hydra
import json
import os
from omegaconf import DictConfig, OmegaConf
from src.engine.state import SkullKingEnv
from src.agents.heuristic import HeuristicAgent
from src.agents.llm_agent import LLMAgent
from src.agents.llm_client import LLMClient
from src.utils.translators import SemanticTranslator
from src.memory.rag_engine import StrategyMemory
from src.memory.elo_tracker import EloTracker
from src.utils.play_by_play import PlayByPlay

# Mute the HTTPX spam that floods the async logs
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

def _starting_round(game_id: int, num_games: int, cfg: DictConfig) -> int:
    """
    Returns the starting round for a given game based on the configured strategy.

    fixed      — all games start at round 1 (full game).
    stratified — games are evenly spread across [min_starting_round, 10].
                 e.g. 3 games, min=1 → rounds 1, 4, 7.
    random     — uniform sample from [min_starting_round, 10].
    """
    strategy  = getattr(cfg.experiment, "starting_round_strategy", "fixed")
    min_round = int(getattr(cfg.experiment, "min_starting_round", 1))
    min_round = max(1, min(min_round, 10))

    if strategy == "stratified":
        span = 10 - min_round          # playable range above min
        step = span / max(num_games, 1)
        return min_round + round(game_id * step)
    elif strategy == "random":
        return random.randint(min_round, 10)
    else:  # "fixed" or anything unrecognised
        return 1


def _make_game_logger(game_id: int) -> logging.Logger:
    """Creates a logger that writes exclusively to logs/game_{id}.log."""
    os.makedirs("logs", exist_ok=True)
    game_logger = logging.getLogger(f"game.{game_id}")
    game_logger.setLevel(logging.DEBUG)
    game_logger.propagate = False   # keep game logs out of the shared run_parallel.log
    fh = logging.FileHandler(f"logs/game_{game_id}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    game_logger.addHandler(fh)
    return game_logger

async def run_single_game(game_id: int, cfg: DictConfig, client: LLMClient, memory: StrategyMemory, elo: EloTracker, starting_round: int = 1):
    """Runs a game starting from starting_round through round 10 asynchronously."""
    game_log = _make_game_logger(game_id)
    game_log.info(f"[Game {game_id}] Initialization started... (starting round: {starting_round})")

    env = SkullKingEnv(num_players=cfg.game.num_players)
    state = env.reset(starting_round=starting_round)
    translator = SemanticTranslator(env.physics)
    
    agents =[]
    player_personas = {}
    
    for i in range(cfg.game.num_players):
        agent_type = cfg.agents.get(f"p{i}", "heuristic")
        if agent_type == "heuristic":
            agents.append(HeuristicAgent(env.physics))
        else:
            agent = LLMAgent(client=client, translator=translator, persona=agent_type, memory=memory, logger=game_log)
            agent.max_retries = cfg.llm.max_retries
            agents.append(agent)
        player_personas[i] = agent_type

    game_trace = {
        "game_id": game_id,
        "starting_round": starting_round,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "player_map": player_personas,
        "events": []
    }

    # Play-by-play: human-readable step-by-step file, flushed after every event
    pbp = PlayByPlay(game_id, f"logs/game_{game_id}_play.txt", translator)
    _pbp_round     = -1          # last round we've opened a section for
    _round_hands   = {}          # {pid: [card_ids]} collected during bidding
    _round_bids    = []          # [(pid, persona, bid)] collected during bidding
    _prev_totals   = [0] * cfg.game.num_players

    done = False
    failed = False
    while not done:
        pid = state["current_player_id"]
        current_agent = agents[pid]

        try:
            # Use async act if it's an LLM, otherwise sync act
            if hasattr(current_agent, "a_act"):
                action_id = await current_agent.a_act(state)
            else:
                action_id = current_agent.act(state)
        except Exception as e:
            game_log.error(f"[Game {game_id}] CRITICAL ERROR on P{pid}: {e}")
            failed = True
            break

        # --- Play-by-play: record before stepping the env ---
        if state["phase"] == "BIDDING":
            if state["round_num"] != _pbp_round:
                _pbp_round   = state["round_num"]
                _round_hands = {}
                _round_bids  = []
            _round_hands[pid] = list(state["my_hand"])
            _round_bids.append((pid, player_personas[pid], action_id))

        elif state["phase"] == "PLAYING":
            # Flush the round header + all bids on the very first play of each round
            if _round_bids:
                pbp.round_start(state["round_num"], _round_hands, player_personas)
                for _p, _persona, _bid in _round_bids:
                    pbp.bid(_p, _persona, _bid)
                _round_bids = []
                pbp.trick_start(pid)
            pbp.play(pid, player_personas[pid], action_id)

        # Log to trace
        move_info = {
            "round": state["round_num"],
            "phase": state["phase"],
            "player": pid,
            "action_id": action_id,
            "card_text": translator.translate_card(action_id) if state["phase"] == "PLAYING" else str(action_id),
            "my_hand": list(state["my_hand"]),
        }
        if state["phase"] == "BIDDING": move_info["bid_amount"] = action_id
        game_trace["events"].append(move_info)

        new_state, _, done, info = env.step(action_id)

        if "trick_winner" in info:
            if info.get("trick_destroyed"):
                # Kraken (or White Whale all-specials): trick is destroyed, no winner.
                game_trace["events"].append({
                    "event_type": "trick_end",
                    "winner": -1,
                    "bonus": 0,
                    "destroyed": True,
                })
                next_leader = new_state["current_player_id"]
                pbp.trick_destroyed(next_leader)
                if not done and new_state["phase"] == "PLAYING":
                    pbp.trick_start(next_leader)
            elif info["trick_winner"] != -1:
                game_trace["events"].append({
                    "event_type": "trick_end",
                    "winner": info["trick_winner"],
                    "bonus": info["trick_bonus"]
                })
                pbp.trick_end(info["trick_winner"], info.get("trick_bonus", 0))
                # Start next trick header if the round continues
                if not done and new_state["phase"] == "PLAYING":
                    pbp.trick_start(new_state["current_player_id"])

        if "round_rewards" in info:
            new_totals   = list(new_state["scores"])
            round_deltas = [new_totals[i] - _prev_totals[i] for i in range(len(new_totals))]
            pbp.round_end(state["round_num"], list(info["bids"]), list(info["won"]),
                          round_deltas, new_totals, player_personas)
            _prev_totals = new_totals
            game_trace["events"].append({
                "event_type": "round_end",
                "round": state["round_num"],
                "scores": new_totals,
                "bids": list(info["bids"]),
                "won": list(info["won"])
            })
            game_log.info(f"[Game {game_id}] Round {state['round_num']} Complete. Current Scores: {new_totals}")

        state = new_state

    final_scores = state['scores']
    if failed:
        game_log.warning(f"*** [Game {game_id}] ABORTED due to error. Skipping ELO update. ***")
        log.warning(f"[Game {game_id}] ABORTED — see logs/game_{game_id}.log")
        return None

    pbp.game_end(final_scores, player_personas)
    game_log.info(f"*** [Game {game_id}] FINISHED! Final Scores: {final_scores} ***")
    log.info(f"[Game {game_id}] FINISHED. Scores: {final_scores}")

    # Update ELO: pair each player's persona with their final score
    player_results = [(player_personas[i], final_scores[i]) for i in range(len(final_scores))]
    elo.update_from_game(player_results)

    # Save individual trace file into a subfolder
    os.makedirs("traces", exist_ok=True)
    trace_path = f"traces/game_{game_id}_trace.json"
    with open(trace_path, "w") as f:
        json.dump(game_trace, f, indent=2)

    return final_scores

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info("=== LAUNCHING PARALLEL MULTI-AGENT SIMULATION ===")
    
    # Wake-cycle client: fast, no chain-of-thought (enable_thinking=False)
    client = LLMClient(
        base_url=cfg.llm.base_url,
        model_name=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
        enable_thinking=cfg.llm.enable_thinking,
    )
    memory = StrategyMemory(persistence_path="data/chroma_db")
    elo    = EloTracker(persistence_path="data/elo_ratings.json")

    concurrent_games = cfg.experiment.concurrent_games

    async def run_all():
        tasks = [
            run_single_game(
                i, cfg, client, memory, elo,
                starting_round=_starting_round(i, concurrent_games, cfg),
            )
            for i in range(concurrent_games)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    asyncio.run(run_all())

    log.info("=== ALL PARALLEL GAMES COMPLETE ===")
    log.info(f"Traces saved to: {os.path.join(os.getcwd(), 'traces/')}")
    log.info("\n" + elo.get_leaderboard())

if __name__ == "__main__":
    main()
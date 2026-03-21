# run_parallel.py

import asyncio
import logging
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

# Mute the HTTPX spam that floods the async logs
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

async def run_single_game(game_id: int, cfg: DictConfig, client: LLMClient, memory: StrategyMemory):
    """Runs a complete 10-round game asynchronously."""
    log.info(f"[Game {game_id}] Initialization started...")
    
    env = SkullKingEnv(num_players=cfg.game.num_players)
    state = env.reset()
    translator = SemanticTranslator(env.physics)
    
    agents =[]
    player_personas = {}
    
    for i in range(cfg.game.num_players):
        agent_type = cfg.agents.get(f"p{i}", "heuristic")
        if agent_type == "heuristic":
            agents.append(HeuristicAgent(env.physics))
        else:
            agent = LLMAgent(client=client, translator=translator, persona=agent_type, memory=memory)
            agent.max_retries = cfg.llm.max_retries
            agents.append(agent)
        player_personas[i] = agent_type

    game_trace = {
        "game_id": game_id,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "player_map": player_personas,
        "events": []
    }

    done = False
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
            log.error(f"[Game {game_id}] CRITICAL ERROR on P{pid}: {e}")
            break

        # Log to trace
        move_info = {
            "round": state["round_num"],
            "phase": state["phase"],
            "player": pid,
            "action_id": action_id, 
            "card_text": translator.translate_card(action_id) if state["phase"] == "PLAYING" else str(action_id)
        }
        if state["phase"] == "BIDDING": move_info["bid_amount"] = action_id
        game_trace["events"].append(move_info)

        new_state, _, done, info = env.step(action_id)
        
        if "trick_winner" in info and info["trick_winner"] != -1:
            game_trace["events"].append({
                "event_type": "trick_end", 
                "winner": info["trick_winner"], 
                "bonus": info["trick_bonus"]
            })

        if "round_rewards" in info:
            game_trace["events"].append({
                "event_type": "round_end", 
                "round": state["round_num"],
                "scores": list(new_state['scores']), 
                "bids": list(info['bids']), 
                "won": list(info['won'])
            })
            log.info(f"[Game {game_id}] Round {state['round_num']} Complete. Current Scores: {new_state['scores']}")

        state = new_state

    log.info(f"*** [Game {game_id}] FINISHED! Final Scores: {state['scores']} ***")
    
    # Save individual trace file into a subfolder
    os.makedirs("traces", exist_ok=True)
    trace_path = f"traces/game_{game_id}_trace.json"
    with open(trace_path, "w") as f:
        json.dump(game_trace, f, indent=2)
        
    return state['scores']

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info("=== LAUNCHING PARALLEL MULTI-AGENT SIMULATION ===")
    
    client = LLMClient(base_url=cfg.llm.base_url, model_name=cfg.llm.model_name)
    memory = StrategyMemory(persistence_path="data/chroma_db")
    
    # Number of concurrent games to run
    CONCURRENT_GAMES = 10 
    
    async def run_all():
        tasks =[run_single_game(i, cfg, client, memory) for i in range(CONCURRENT_GAMES)]
        results = await asyncio.gather(*tasks)
        return results
        
    all_scores = asyncio.run(run_all())
    
    log.info("=== ALL PARALLEL GAMES COMPLETE ===")
    log.info(f"Traces saved to: {os.path.join(os.getcwd(), 'traces/')}")

if __name__ == "__main__":
    main()
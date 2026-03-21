# main.py

import sys
import os
import json
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from src.engine.state import SkullKingEnv
from src.agents.heuristic import HeuristicAgent
from src.agents.llm_agent import LLMAgent
from src.agents.llm_client import LLMClient
from src.utils.translators import SemanticTranslator
from src.memory.rag_engine import StrategyMemory # The Grimoire

# Set up the logger
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info(f"=== STARTING CO-EVOLUTION SIMULATION: {cfg.experiment.name} ===")
    
    # 1. Setup Environment
    env = SkullKingEnv(num_players=cfg.game.num_players)
    state = env.reset()
    translator = SemanticTranslator(env.physics)

    # 2. Setup Agents
    # Player 0: The LLM Agent
    client = LLMClient(
        base_url=cfg.llm.base_url, 
        model_name=cfg.llm.model_name
    )
    
    try:
        # Quick connectivity check
        log.info("Connecting to Local LLM...")
        # (Optional: You could do a quick dry-run prompt here)
    except Exception as e:
        log.error(f"LLM Connection failed: {e}")
        sys.exit(1)

    # Initialize RAG Memory (The Grimoire)
    # Using absolute path logic handled inside StrategyMemory class
    memory = StrategyMemory(persistence_path="data/chroma_db") 

    # 3. Dynamic Agent Instantiation
    agents = []
    player_personas = {} # For logging/trace

    # Iterate through p0, p1, p2, p3 in config
    for i in range(cfg.game.num_players):
        agent_type = cfg.agents.get(f"p{i}", "heuristic")
        
        if agent_type == "heuristic":
            log.info(f"Player {i}: Heuristic Bot")
            agents.append(HeuristicAgent(env.physics))
            player_personas[i] = "heuristic"
        else:
            log.info(f"Player {i}: LLM Agent ({agent_type})")
            # Create LLM Agent with Memory
            agent = LLMAgent(
                client=client, 
                translator=translator, 
                persona=agent_type,
                memory=memory  # Give them access to the Grimoire
            )
            agent.max_retries = cfg.llm.max_retries
            agents.append(agent)
            player_personas[i] = agent_type

    game_trace = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "player_map": player_personas,
        "events": []
    }

    
    log.info("--- GAME START ---")

    game_over = False
    
    while not game_over:
        current_pid = state["current_player_id"]
        current_agent = agents[current_pid]
        
        # 3. Get Action from Agent
        # The agent internally handles logic, translation, and LLM querying
        # We wrap in try/except to catch any unexpected disconnects/crashes
        try:
            action_id = current_agent.act(state)
        except Exception as e:
            log.error(f"CRITICAL ERROR on Player {current_pid}: {e}")
            break
        
        # 4. Logging & Tracing
        move_info = {
            "round": state["round_num"],
            "phase": state["phase"],
            "player": current_pid,
            "action_id": action_id,
            "card_text": translator.translate_card(action_id) if state["phase"] == "PLAYING" else str(action_id),
            "my_hand": state["my_hand"],
        }
        
        # 4. Logging the Move
        if state["phase"] == "BIDDING":
            log.info(f"[Round {state['round_num']}] Player {current_pid} Bids: {action_id}")
        else:
            # If playing phase, translate the card ID to text for readable log
            log.info(f"  > P{current_pid} plays: {move_info['card_text']}")

        game_trace["events"].append(move_info)
        
        # 5. Execute Step
        new_state, reward, done, info = env.step(action_id)
        
       # 6. Post-Move Logging
        if "trick_winner" in info and info["trick_winner"] != -1:
            w_id = info["trick_winner"]
            pts = info["trick_bonus"]
            log.info(f"  --- Trick Winner: Player {w_id} (Bonus: {pts}) ---")
            game_trace["events"].append({"event_type": "trick_end", "winner": w_id, "bonus": pts})

        if "round_rewards" in info:
            log.info(f"\n=== ROUND {state['round_num']} END ===")
            log.info(f"Bids: {info['bids']}")
            log.info(f"Won:  {info['won']}")
            log.info(f"Scores update: {info['round_rewards']}")
            log.info(f"Current Totals: {new_state['scores']}\n")
            
            game_trace["events"].append({
                "event_type": "round_end",
                "round": state["round_num"],
                "scores": list(new_state['scores']),
                "bids": list(info['bids']),
                "won": list(info['won'])
            })
            if not done:
                log.info(f"--- STARTING ROUND {new_state['round_num']} ---")

        # Update loop state
        state = new_state
        game_over = done

    log.info("=== GAME OVER ===")
    log.info(f"Final Scores: {state['scores']}")
    
    # 7. Save Detailed JSON Trace (For Hindsight Analysis)
    if cfg.experiment.save_json_logs:
        # Hydra changes CWD to the output folder, so just save filename
        with open("game_trace.json", "w") as f:
            json.dump(game_trace, f, indent=2)
        log.info(f"Game trace saved to: {os.getcwd()}/game_trace.json")

if __name__ == "__main__":
    main()
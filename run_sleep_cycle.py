# run_sleep_cycle.py

import logging
import os
import glob
from pathlib import Path
import hydra
from omegaconf.dictconfig import DictConfig
from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.memory.reflector import SleepCycleReflector

logging.basicConfig(level=logging.INFO)

# Set up the logger
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Point this to the latest Hydra output folder
    # (You can automate finding the newest folder, but for now we hardcode the file)
    # E.g., trace_path = "outputs/2026-03-01/16-00-00/game_trace.json"
    
    # 1. Get the absolute path to the project root (where this script lives)
    project_root = Path(__file__).resolve().parent
    
    # Construct the absolute search pattern: /home/anton/SkullZero/outputs/*/*/game_trace.json
    search_pattern = str(project_root / "outputs" / "*" / "*" / "game_trace.json")
    
    # 2. Find all traces
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        logging.error(f"No game traces found! Searched in: {search_pattern}")
        return
        
    latest_trace = max(list_of_files, key=os.path.getctime)
    print(f"Running Sleep Cycle on: {latest_trace}")

    # 2. Initialize the AI components
    client = LLMClient(
    base_url=cfg.llm.base_url, 
    model_name=cfg.llm.model_name
    
    )
    memory = StrategyMemory(persistence_path="data/chroma_db")
    reflector = SleepCycleReflector(client, memory)

    # 3. Define the LLM Players map
    # Currently, only Player 0 is an LLM. 
    # In Phase 4, you will change this to: {0: "forced_zero", 1: "rational"}
    llm_players_map = {
        0: "forced_zero",
        1: "rational",
        2: "rational",
        3: "rational"
    }

    # 4. Run the reflection!
    reflector.process_trace(latest_trace, llm_players_map)

if __name__ == "__main__":
    main()
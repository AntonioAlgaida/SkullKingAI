# run_pruning.py

import logging

import hydra
from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.memory.pruner import MemoryPruner
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== STARTING MEMORY PRUNING CYCLE ===")
    
    # Use your high-speed vLLM connection
    client = LLMClient(base_url=cfg.llm.base_url, model_name=cfg.llm.model_name)
    memory = StrategyMemory(persistence_path="data/chroma_db") 
    pruner = MemoryPruner(client, memory)

    # Prune both personas
    pruner.prune_persona("forced_zero")
    print("-" * 40)
    pruner.prune_persona("rational")
    
    print("\n=== PRUNING COMPLETE ===")
    print("Run `uv run python scripts/visualize_memory.py` to see the cleaned database.")

if __name__ == "__main__":
    main()
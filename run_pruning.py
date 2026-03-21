# run_pruning.py

import logging

import hydra
from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.memory.pruner import MemoryPruner
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info("=== STARTING MEMORY PRUNING CYCLE ===")
    
    # Pruning client: strict, analytical, chain-of-thought for better judgment
    client = LLMClient(
        base_url=cfg.llm.base_url,
        model_name=cfg.llm.model_name,
        temperature=cfg.llm.pruner_temperature,
        top_p=cfg.llm.pruner_top_p,
        max_tokens=cfg.llm.pruner_max_tokens,
        enable_thinking=cfg.llm.pruner_enable_thinking,
    )
    memory = StrategyMemory(persistence_path="data/chroma_db") 
    pruner = MemoryPruner(client, memory)

    # Prune both personas
    pruner.prune_persona("forced_zero")
    log.info("-" * 40)
    pruner.prune_persona("rational")

    log.info("=== PRUNING COMPLETE ===")
    log.info("Run `uv run python scripts/visualize_memory.py` to see the cleaned database.")

if __name__ == "__main__":
    main()
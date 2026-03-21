# run_sleep_cycle.py

import logging
import os
import json
import glob
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.memory.reflector import SleepCycleReflector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Tracks which trace files have already been processed so re-runs are safe.
PROCESSED_LOG = Path(__file__).resolve().parent / "data" / "processed_traces.json"


def _load_processed() -> set:
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, "r") as f:
            return set(json.load(f))
    return set()


def _save_processed(processed: set):
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_LOG, "w") as f:
        json.dump(sorted(processed), f, indent=2)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    project_root = Path(__file__).resolve().parent

    # Find ALL game traces across every output folder
    search_pattern = str(project_root / "outputs" / "*" / "*" / "game_trace.json")
    parallel_pattern = str(project_root / "outputs" / "*" / "*" / "traces" / "game_*_trace.json")

    all_traces = glob.glob(search_pattern) + glob.glob(parallel_pattern)

    if not all_traces:
        log.error(f"No game traces found. Searched:\n  {search_pattern}\n  {parallel_pattern}")
        return

    already_processed = _load_processed()
    new_traces = [t for t in all_traces if os.path.abspath(t) not in already_processed]

    if not new_traces:
        log.info(f"All {len(all_traces)} trace(s) already processed. Nothing to do.")
        return

    log.info(f"Found {len(all_traces)} total trace(s). Processing {len(new_traces)} new one(s).")

    # Sleep-cycle client: deeper reasoning with chain-of-thought enabled
    client = LLMClient(
        base_url=cfg.llm.base_url,
        model_name=cfg.llm.model_name,
        temperature=cfg.llm.reflection_temperature,
        top_p=cfg.llm.reflection_top_p,
        max_tokens=cfg.llm.reflection_max_tokens,
        enable_thinking=cfg.llm.reflection_enable_thinking,
    )
    memory   = StrategyMemory(persistence_path="data/chroma_db")
    reflector = SleepCycleReflector(client, memory)

    newly_done = set()
    for trace_path in sorted(new_traces):
        log.info(f"--- Processing: {trace_path}")
        try:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
            # player_map in trace is {str_key: persona}, reflector expects {int_key: persona}
            llm_players_map = {int(k): v for k, v in trace_data.get("player_map", {}).items()}
            reflector.process_trace(trace_path, llm_players_map)
            newly_done.add(os.path.abspath(trace_path))
        except Exception as e:
            log.error(f"Failed on {trace_path}: {e}")

    _save_processed(already_processed | newly_done)
    log.info(f"Sleep cycle complete. Processed {len(newly_done)} new trace(s).")


if __name__ == "__main__":
    main()

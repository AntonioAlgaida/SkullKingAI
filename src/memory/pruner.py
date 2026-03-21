# src/memory/pruner.py

import logging
import re
from typing import List, Dict, Any
from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class MemoryPruner:
    """
    Acts as a 'Garbage Collector' for the RAG Memory.
    Uses LLM-as-a-Judge to identify and delete hallucinations, 
    contradictions, or useless fluff.
    """
    def __init__(self, client: LLMClient, memory: StrategyMemory):
        self.client = client
        self.memory = memory

    def prune_persona(self, persona: str):
        """Runs the audit process on a specific persona's memory."""
        logger.info(f"Starting Pruning Cycle for Persona: {persona.upper()}")
        
        collection = self.memory._get_collection(persona)
        if collection.count() == 0:
            logger.info("Memory empty. Nothing to prune.")
            return

        data = collection.get()
        ids = data.get("ids", [])
        documents = data.get("documents",[])
        
        # Batch processing (LLMs struggle to judge 30 rules at once)
        batch_size = 10
        ids_to_delete =[]

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            
            logger.info(f"Auditing batch {i//batch_size + 1} ({len(batch_ids)} rules)...")
            
            trash_ids = self._audit_batch(persona, batch_ids, batch_docs)
            ids_to_delete.extend(trash_ids)

        if ids_to_delete:
            logger.warning(f"Deleting {len(ids_to_delete)} bad rules from DB: {ids_to_delete}")
            collection.delete(ids=ids_to_delete)
        else:
            logger.info("Memory is clean. No rules deleted.")

    def _audit_batch(self, persona: str, ids: List[str], docs: List[str]) -> List[str]:
        """Asks the LLM to judge a batch of rules."""
        
        # 1. Format the rules for the prompt
        rules_text = ""
        for rule_id, text in zip(ids, docs):
            rules_text += f"ID: [{rule_id}]\nRULE: {text}\n\n"

        # 2. LOAD GAME CONTEXT (Fixed: Using bundles now)
        # We use the playing bundle because these rules are for the Playing Phase
        bundle = PromptLoader.get_playing_bundle(persona)
        
        # 3. Construct Prompt
        prompt = f"""[SYSTEM]
You are the Supreme Auditor for a Skull King AI. Your job is to delete bad memories from the AI's database.

[GAME RULES & STRATEGY]
{bundle}

[AUDIT CRITERIA]
You must DELETE a rule if it violates ANY of the following:
1. HALLUCINATION: Suggests something illegal (e.g., "don't follow suit").
2. CONTRADICTION: Suggests something mathematically stupid (e.g., advising a Zero-bidder to "take control" or "win just one trick").
3. USELESS FLUFF: Vague advice that doesn't actually tell the AI what card to play (e.g., "play strategically", "be careful").
4. ARTIFACTS: Contains weird text artifacts like "[ACTION]: <SKULL KING>".

[RULES TO AUDIT]
{rules_text}

[TASK]
Analyze each rule. Write brief CoT reasoning. 
Then, list the IDs of the rules that MUST BE DELETED.
If a rule is good, DO NOT list it.
Output your final answer strictly as a comma-separated list of IDs inside brackets.
Example: [DELETE]: rule_1, rule_5
If no rules should be deleted, output: [DELETE]: NONE
"""
        try:
            # Low temperature because we want strict, analytical judgment
            raw_content, _ = self.client.get_move_with_content(prompt, temperature=0.1)
            
            # Regex to find [DELETE]: rule_1, rule_2
            match = re.search(r"\[DELETE\]:\s*(.*)", raw_content, re.IGNORECASE)
            if match:
                result_str = match.group(1).strip().lower()
                if "none" in result_str:
                    return[]
                
                # Extract words that look like rule_X
                bad_ids = re.findall(r"(rule_\d+)", result_str)
                return bad_ids
            else:
                logger.error(f"Failed to parse audit response:\n{raw_content}")
                return[]
                
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return[]
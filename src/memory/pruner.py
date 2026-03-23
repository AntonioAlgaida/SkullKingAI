# src/memory/pruner.py

import logging
import re
from typing import List

from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class MemoryPruner:
    """
    Acts as a 'Garbage Collector' for the RAG Memory.
    Uses LLM-as-a-Judge to identify and delete hallucinations,
    contradictions, or useless fluff.

    Audits PLAYING and BIDDING rules separately so game context
    matches the phase the rule was generated for.
    """

    AUDIT_CRITERIA = """\
DELETE a rule if it violates ANY of the following:
1. HALLUCINATION : Suggests something illegal (e.g. "don't follow suit when you have the lead suit").
2. CONTRADICTION : Mathematically impossible advice (e.g. advising a Zero-bidder to "win just one trick").
3. USELESS FLUFF : Vague advice that gives no card-specific guidance (e.g. "play strategically", "be careful").
4. ARTIFACT      : Contains raw output noise like "[ACTION]: <SKULL KING>" or incomplete sentences.

If a rule is good, do NOT list it. When in doubt, keep it."""

    def __init__(self, client: LLMClient, memory: StrategyMemory):
        self.client = client
        self.memory = memory

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def prune_persona(self, persona: str):
        """Runs the full audit cycle on a specific persona's memory."""
        logger.info(f"[Pruner] Starting cycle for: {persona.upper()}")

        collection  = self.memory._get_collection(persona)
        total_before = collection.count()
        if total_before == 0:
            logger.info("[Pruner] Memory empty. Nothing to prune.")
            return

        ids_to_delete = []

        # Audit PLAYING and BIDDING rules separately — different bundles as context
        for phase in ("PLAYING", "BIDDING"):
            bundle = (
                PromptLoader.get_playing_bundle(persona)
                if phase == "PLAYING"
                else PromptLoader.get_bidding_bundle(persona)
            )

            try:
                data = collection.get(where={"phase": phase})
            except Exception:
                # ChromaDB version without where-on-get support — fall back to all
                data = collection.get()

            ids  = data.get("ids", [])
            docs = data.get("documents", [])

            if not ids:
                logger.info(f"[Pruner] No {phase} rules for {persona}. Skipping.")
                continue

            logger.info(f"[Pruner] Auditing {len(ids)} {phase} rules for {persona}...")

            batch_size = 10
            for i in range(0, len(ids), batch_size):
                batch_ids  = ids[i : i + batch_size]
                batch_docs = docs[i : i + batch_size]
                logger.info(f"  Batch {i // batch_size + 1} / {(len(ids) - 1) // batch_size + 1} ({len(batch_ids)} rules)")
                trash = self._audit_batch(persona, batch_ids, batch_docs, bundle, phase)
                ids_to_delete.extend(trash)

        # Apply deletions — log full rule text before deleting for audit trail
        if ids_to_delete:
            deleted_data = collection.get(ids=ids_to_delete, include=["documents"])
            deleted_texts = deleted_data.get("documents", [])
            for rule_id, rule_text in zip(ids_to_delete, deleted_texts):
                logger.warning(f"[Pruner] DELETING [{rule_id}]: {rule_text}")
            logger.warning(
                f"[Pruner] {persona.upper()}: deleting {len(ids_to_delete)}/{total_before} rules → "
                f"{total_before - len(ids_to_delete)} remain."
            )
            collection.delete(ids=ids_to_delete)
        else:
            logger.info(f"[Pruner] {persona.upper()}: memory clean. 0/{total_before} rules deleted.")

    # ------------------------------------------------------------------ #
    # Batch auditor                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_strategy_section(bundle: str) -> str:
        """
        Returns only the '--- STRATEGY GUIDE ---' section of the bundle.
        Drops the full card hierarchy, scoring tables, and examples (~2000 tokens)
        that are not needed for judging whether a rule is tactically valid.
        Falls back to the full bundle if the marker is absent.
        """
        marker = "--- STRATEGY GUIDE ---"
        idx = bundle.find(marker)
        if idx != -1:
            return bundle[idx:]
        return bundle

    def _audit_batch(
        self,
        persona: str,
        ids: List[str],
        docs: List[str],
        bundle: str,
        phase: str,
    ) -> List[str]:
        """Asks the LLM to judge one batch of rules. Returns IDs to delete."""

        rules_text = ""
        for rule_id, text in zip(ids, docs):
            rules_text += f"ID: [{rule_id}]\nRULE: {text}\n\n"

        # Keep the system prompt short: full bundle causes token overflow when
        # thinking is disabled and the completion budget is tight.
        # The strategy section alone is sufficient for judging rule quality.
        strategy_section = self._extract_strategy_section(bundle)
        system_prompt = (
            f"You are the memory auditor for a Skull King AI. "
            f"Delete bad {phase} rules from the database.\n\n"
            f"[STRATEGY CONTEXT]\n{strategy_section}\n\n"
            f"[AUDIT CRITERIA]\n{self.AUDIT_CRITERIA}"
        )

        # Dynamic audit task + rule batch → user role
        user_prompt = (
            f"[{phase} RULES TO AUDIT]\n{rules_text}\n"
            f"[TASK]\n"
            f"Analyse each rule. Write brief CoT reasoning.\n"
            f"Then list the IDs of rules that MUST BE DELETED.\n"
            f"Output strictly as: [DELETE]: rule_1, rule_5\n"
            f"If no rules should be deleted, output: [DELETE]: NONE"
        )

        try:
            raw_content = self.client.generate(user_prompt, system_prompt=system_prompt)

            match = re.search(r"\[DELETE\]\s*:\s*(.*)", raw_content, re.IGNORECASE)
            if not match:
                logger.error(f"[Pruner] Failed to parse audit response:\n{raw_content[:300]}")
                return []

            result_str = match.group(1).strip().lower()
            if "none" in result_str:
                return []

            bad_ids = re.findall(r"rule_\d+", result_str)
            if bad_ids:
                logger.info(f"[Pruner] Batch marked for deletion: {bad_ids}")
            return bad_ids

        except Exception as e:
            logger.error(f"[Pruner] Audit batch failed: {e}")
            return []

# src/agents/llm_agent.py

import random
import logging
from typing import Dict, Any, Optional

from src.agents.llm_client import LLMClient
from src.utils.translators import SemanticTranslator
from src.memory.rag_engine import StrategyMemory
from src.memory.action_cache import SemanticActionCache

_default_logger = logging.getLogger(__name__)

class LLMAgent:
    """
    The Autonomous Agent.
    1. Receives Game State (Dict)
    2. Translates to Semantic Text (via Translator)
    3. Retrieves Strategic Rules (via RAG - Placeholder for Phase 3)
    4. Queries LLM (via Client)
    5. Validates & Executes Action
    """
    def __init__(self,
                 client: LLMClient,
                 translator: SemanticTranslator,
                 persona: str = "Rational",
                 memory: Optional[StrategyMemory] = None,
                 action_cache: Optional[SemanticActionCache] = None,
                 logger: Optional[logging.Logger] = None):

        self.client = client
        self.translator = translator
        self.persona = persona
        self.memory = memory
        self.action_cache = action_cache
        self.max_retries = 2
        self.logger = logger or _default_logger

    def act(self, state_dict: Dict[str, Any]) -> int:
        """
        Main decision loop.
        Returns: Integer Action ID (0-10 for bidding, 0-74 for playing).
        """
        # 1. Get Legal Moves for Validation
        phase = state_dict["phase"]
        legal_actions = state_dict["legal_actions"]
        
        # --- OPTIMIZATION 1: FORCED ZERO BIDDING ---
        normalized_persona = self.persona.lower().replace("-", "_")
        if phase == "BIDDING" and (self.persona.lower() == "forced-zero" or normalized_persona == "forced_zero"):
            self.logger.info(f"[{self.persona}] Bidding Phase: Automatically bidding 0 as per persona constraint.")
            return 0  # Action ID 0 is Bid 0

        # --- OPTIMIZATION 2: SINGLE LEGAL MOVE ---
        if len(legal_actions) == 1:
            self.logger.info(f"[{self.persona}] Only one legal move: {legal_actions[0]}. Skipping LLM.")
            return legal_actions[0]

        # 2. Build system prompt (static bundle → cached by vLLM) and user context (dynamic state)
        system_prompt = self.translator.get_system_prompt(phase, self.persona)
        prompt        = self.translator.build_user_context(state_dict)

        # 3. Semantic Action Cache lookup (before touching the LLM)
        cache_query = self.memory.generate_query_context(state_dict) if self.memory else None
        if self.action_cache and cache_query:
            cached = self.action_cache.lookup(cache_query, legal_actions, phase)
            if cached is not None:
                return cached

        # 4. Inject Memory (if available)
        if self.memory:
            strategic_advice = self.memory.retrieve_rules(state_dict, self.persona)
            prompt += f"\n\n{strategic_advice}\n"
            if phase == "BIDDING":
                prompt += (
                    "\n[DECISION]\nUsing the game state above and your Strategic Memory, "
                    "state your bid reasoning in one sentence, then output your ACTION."
                )
            else:
                prompt += (
                    "\n[DECISION]\nUsing the game state above and your Strategic Memory, "
                    "state which proven rule (if any) applies to THIS trick, then output your ACTION."
                )
            self.logger.debug(f"[{self.persona}] Injected Memory into Prompt:\n{strategic_advice}")

        self.logger.debug(f"--- SYSTEM PROMPT ---\n{system_prompt}\n--- USER CONTEXT ---\n{prompt}\n--------------------------")

        # 5. Query LLM with Retry Loop
        for attempt in range(self.max_retries + 1):
            raw_content, action_id = self.client.get_move_with_content(prompt, system_prompt=system_prompt)

            player_id = state_dict.get("current_player_id", "?")
            self.logger.info(f"[P{player_id}/{self.persona}] LLM Response (Attempt {attempt+1}):\n{raw_content}")

            if action_id in legal_actions:
                # Store successful decision in cache for future reuse
                if self.action_cache and cache_query:
                    self.action_cache.store(cache_query, action_id, phase, self.persona)
                return action_id

            self.logger.warning(f"[{self.persona}] Illegal move {action_id}. Not in {legal_actions}")
            if attempt < self.max_retries:
                prompt += f"\n\n[SYSTEM ERROR]: ID {action_id} is ILLEGAL. Pick from: {legal_actions}. Think again."

        # Fallback
        fallback_action = random.choice(legal_actions)
        self.logger.error(f"[{self.persona}] LLM failed. Playing random fallback: {fallback_action}")
        return fallback_action

    async def a_act(self, state_dict: Dict[str, Any]) -> int:
        """Asynchronous decision loop."""
        phase = state_dict["phase"]
        legal_actions = state_dict["legal_actions"]

        normalized_persona = self.persona.lower().replace("-", "_")
        if phase == "BIDDING" and normalized_persona == "forced_zero":
            self.logger.info(f"[{self.persona}] Bidding Phase: Automatically bidding 0 as per persona constraint.")
            return 0  

        if len(legal_actions) == 1:
            self.logger.info(f"[{self.persona}] Only one legal move: {legal_actions[0]}. Skipping LLM.")
            return legal_actions[0]

        system_prompt = self.translator.get_system_prompt(phase, self.persona)
        prompt        = self.translator.build_user_context(state_dict)

        # Semantic Action Cache lookup
        cache_query = self.memory.generate_query_context(state_dict) if self.memory else None
        if self.action_cache and cache_query:
            cached = self.action_cache.lookup(cache_query, legal_actions, phase)
            if cached is not None:
                return cached

        if self.memory:
            strategic_advice = self.memory.retrieve_rules(state_dict, self.persona)
            prompt += f"\n\n{strategic_advice}\n"
            if phase == "BIDDING":
                prompt += (
                    "\n[DECISION]\nUsing the game state above and your Strategic Memory, "
                    "state your bid reasoning in one sentence, then output your ACTION."
                )
            else:
                prompt += (
                    "\n[DECISION]\nUsing the game state above and your Strategic Memory, "
                    "state which proven rule (if any) applies to THIS trick, then output your ACTION."
                )
            self.logger.debug(f"[{self.persona}] Injected Memory into Prompt:\n{strategic_advice}")

        self.logger.debug(f"--- SYSTEM PROMPT ---\n{system_prompt}\n--- USER CONTEXT ---\n{prompt}\n--------------------------")

        for attempt in range(self.max_retries + 1):
            raw_content, action_id = await self.client.a_get_move_with_content(prompt, system_prompt=system_prompt)

            player_id = state_dict.get("current_player_id", "?")
            self.logger.info(f"[P{player_id}/{self.persona}] LLM Response (Attempt {attempt+1}):\n{raw_content}")

            if action_id in legal_actions:
                if self.action_cache and cache_query:
                    self.action_cache.store(cache_query, action_id, phase, self.persona)
                return action_id

            self.logger.warning(f"[{self.persona}] Illegal move {action_id}. Not in {legal_actions}")
            if attempt < self.max_retries:
                prompt += f"\n\n[SYSTEM ERROR]: ID {action_id} is ILLEGAL. Pick from: {legal_actions}. Think again."

        fallback_action = random.choice(legal_actions)
        self.logger.error(f"[{self.persona}] LLM failed. Playing random fallback: {fallback_action}")
        return fallback_action
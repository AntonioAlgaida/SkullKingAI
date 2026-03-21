# src/agents/llm_agent.py

import random
import logging
from typing import Dict, Any, Optional

from src.agents.llm_client import LLMClient
from src.utils.translators import SemanticTranslator
from src.memory.rag_engine import StrategyMemory

logger = logging.getLogger(__name__)

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
                 memory: Optional[StrategyMemory] = None):
        
        self.client = client
        self.translator = translator
        self.persona = persona
        self.memory = memory
        self.max_retries = 2

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
            logger.info(f"[{self.persona}] Bidding Phase: Automatically bidding 0 as per persona constraint.")
            return 0  # Action ID 0 is Bid 0

        # --- OPTIMIZATION 2: SINGLE LEGAL MOVE ---
        if len(legal_actions) == 1:
            logger.info(f"[{self.persona}] Only one legal move: {legal_actions[0]}. Skipping LLM.")
            return legal_actions[0]

        # 2. Build the Base Prompt
        prompt = self.translator.build_llm_prompt_context(state_dict, self.persona)
        
        # 3. Inject Memory (if available)
        if self.memory:
            # Retrieve top 3 relevant rules
            strategic_advice = self.memory.retrieve_rules(state_dict, self.persona)
            
            # We append a new section to the prompt
            prompt += f"\n[RECALLED STRATEGIES]\n{strategic_advice}\n"
            prompt += "\n[FINAL INSTRUCTION]\nCombine the System Rules, Threat Level, and Recalled Strategies to pick the best move."
            
            logger.debug(f"[{self.persona}] Injected Memory into Prompt:\n{strategic_advice}")

        # Log the prompt at DEBUG level (viewable in main.log if configured)
        logger.debug(f"--- PROMPT SENT TO LLM ---\n{prompt}\n--------------------------")

        # 2. Query LLM with Retry Loop
        for attempt in range(self.max_retries + 1):
            raw_content, action_id = self.client.get_move_with_content(prompt)
            
            # Log the Full Response (Reasoning + Action)
            logger.info(f"[{self.persona}] LLM Response (Attempt {attempt+1}):\n{raw_content}")

            if action_id in legal_actions:
                return action_id
            
            # Error Handling
            logger.warning(f"[{self.persona}] Illegal move {action_id}. Not in {legal_actions}")
            if attempt < self.max_retries:
                prompt += f"\n\n[SYSTEM ERROR]: ID {action_id} is ILLEGAL. Pick from: {legal_actions}. Think again."

        # Fallback
        fallback_action = random.choice(legal_actions)
        logger.error(f"[{self.persona}] LLM failed. Playing random fallback: {fallback_action}")
        return fallback_action

    async def a_act(self, state_dict: Dict[str, Any]) -> int:
        """Asynchronous decision loop."""
        phase = state_dict["phase"]
        legal_actions = state_dict["legal_actions"]

        normalized_persona = self.persona.lower().replace("-", "_")
        if phase == "BIDDING" and normalized_persona == "forced_zero":
            logger.info(f"[{self.persona}] Bidding Phase: Automatically bidding 0 as per persona constraint.")
            return 0  

        if len(legal_actions) == 1:
            logger.info(f"[{self.persona}] Only one legal move: {legal_actions[0]}. Skipping LLM.")
            return legal_actions[0]

        prompt = self.translator.build_llm_prompt_context(state_dict, self.persona)

        if self.memory:
            strategic_advice = self.memory.retrieve_rules(state_dict, self.persona)
            prompt += f"\n[RECALLED STRATEGIES]\n{strategic_advice}\n"
            prompt += "\n[FINAL INSTRUCTION]\nCombine the System Rules, Threat Level, and Recalled Strategies to pick the best move."
            logger.debug(f"[{self.persona}] Injected Memory into Prompt:\n{strategic_advice}")

        logger.debug(f"--- PROMPT SENT TO LLM ---\n{prompt}\n--------------------------")

        for attempt in range(self.max_retries + 1):
            raw_content, action_id = await self.client.a_get_move_with_content(prompt)
            
            logger.info(f"[{self.persona}] LLM Response (Attempt {attempt+1}):\n{raw_content}")
            
            if action_id in legal_actions:
                return action_id
            
            logger.warning(f"[{self.persona}] Illegal move {action_id}. Not in {legal_actions}")
            if attempt < self.max_retries:
                prompt += f"\n\n[SYSTEM ERROR]: ID {action_id} is ILLEGAL. Pick from: {legal_actions}. Think again."

        fallback_action = random.choice(legal_actions)
        logger.error(f"[{self.persona}] LLM failed. Playing random fallback: {fallback_action}")
        return fallback_action
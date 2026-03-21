# src/memory/reflector.py

import json
import logging
import re
from typing import Dict, List, Any
from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory
from src.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class SleepCycleReflector:
    """
    Analyzes game traces offline. Identifies failed bids for LLM agents,
    prompts the LLM to write a new strategic rule, and saves it to ChromaDB.
    """
    def __init__(self, client: LLMClient, memory: StrategyMemory):
        self.client = client
        self.memory = memory

    def process_trace(self, trace_path: str, llm_players: Dict[int, str]):
        """
        trace_path: Path to the game_trace.json
        llm_players: Dict mapping Player ID to their Persona e.g., {0: "forced_zero", 1: "rational"}
        """
        logger.info(f"Initiating Sleep Cycle on trace: {trace_path}")
        
        try:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trace: {e}")
            return

        events = trace_data.get("events", trace_data) if isinstance(trace_data, dict) else trace_data
        
        # Group events by round
        rounds_data = self._group_by_round(events)

        for round_num, r_data in rounds_data.items():
            end_event = r_data.get("end_event")
            if not end_event:
                continue

            bids = end_event["bids"]
            won = end_event["won"]

            # Check if any LLM player failed their bid
            for pid, persona in llm_players.items():
                if bids[pid] != won[pid]:
                    logger.info(f"[{persona.upper()}] P{pid} failed in Round {round_num} (Bid: {bids[pid]}, Won: {won[pid]}). Generating reflection...")
                    
                    # Extract the starting hand we tracked
                    starting_hand = r_data.get("starting_hands", {}).get(pid,[])
                    
                     # 1. ALWAYS generate a rule for Card Playing
                    self._generate_playing_rule(pid, persona, round_num, bids[pid], won[pid], r_data["tricks"], starting_hand)
                    
                    # 2. NEW: Generate a rule for Bidding (ONLY for Rational agents)
                    # We skip this for Forced-Zero because their bid is hardcoded to 0.
                    if persona.lower() != "forced_zero":
                        self._generate_bidding_rule(pid, persona, round_num, bids[pid], won[pid], starting_hand)

    def _group_by_round(self, events: List[Dict]) -> Dict[int, Dict]:
        """Parses the flat event log into structured rounds, tricks, and starting hands."""
        rounds = {}
        current_trick =[]
        
        # We need the SemanticTranslator to translate the hand IDs
        from src.utils.translators import SemanticTranslator
        from src.engine.physics import GamePhysics
        translator = SemanticTranslator(GamePhysics())
        
        for event in events:
            r = event.get("round")
            if r is None:
                continue
                
            if r not in rounds:
                rounds[r] = {"tricks":[], "end_event": None, "starting_hands": {}}

            # Capture the starting hand for each player on their first move of the round
            if "my_hand" in event and event.get("phase") == "PLAYING":
                pid = event["player"]
                if pid not in rounds[r]["starting_hands"]:
                    # Translate IDs to text
                    hand_text = [translator.translate_card(cid) for cid in event["my_hand"]]
                    rounds[r]["starting_hands"][pid] = hand_text

            # Handle Trick Plays
            if "action_id" in event and event.get("phase") == "PLAYING":
                card_str = event.get("card_text", str(event["action_id"]))
                current_trick.append(f"P{event['player']} played {card_str}")
                
            # Handle Trick Ends
            elif event.get("event_type") == "trick_end":
                if current_trick:
                    rounds[r]["tricks"].append({
                        "plays": current_trick.copy(),
                        "winner": event.get("winner")
                    })
                    current_trick =[]
                    
            # Handle Round Ends
            elif event.get("event_type") == "round_end":
                rounds[r]["end_event"] = event

        return rounds

    def _generate_playing_rule(self, pid: int, persona: str, round_num: int, bid: int, won: int, tricks: List[Dict], starting_hand: List[str]):
        """Builds the hindsight prompt, queries the LLM, and saves the rule."""
        
        history_text = ""
        for i, trick in enumerate(tricks):
            plays = ", ".join(trick["plays"])
            history_text += f"Trick {i+1}: {plays} --> WINNER: P{trick['winner']}\n"

        goal_text = "win EXACTLY 0 tricks." if persona == "forced_zero" else f"win EXACTLY {bid} tricks."

        # LOAD GAME CONTEXT
        game_intro = PromptLoader.load("rules", "game_intro")
        card_hierarchy = PromptLoader.load("rules", "card_hierarchy")
        game_rules = f"{game_intro}\n\n{card_hierarchy}"
        trick_mechanics = PromptLoader.load("rules", "trick_mechanics")
        persona_guide = PromptLoader.get_persona(persona)
        
        # Format Starting Hand
        hand_str = ", ".join(starting_hand) if starting_hand else "Unknown"
        
        # Dynamic Constraints based on Persona
        constraints = ""
        if persona == "forced_zero":
            constraints = """
1. ZERO TOLERANCE: For a Zero Bidder, winning even 1 trick is a TOTAL FAILURE (-10 points per card). Never imply that winning "only one" trick is acceptable.
2. SUICIDE PLAYS: Never advise leading a high Black card (Trump) or a Pirate/King. This forces a win.
3. SLOUGHING LOGIC: If advising to discard a high card, you must specify the condition: "Only play this when you are VOID in the lead suit" or "Only when someone else has already played a higher card."
"""
        else:
            constraints = """
1. PRECISION: The goal is the exact bid. Winning extra tricks is a penalty.
2. TRUMP MANAGEMENT: Advise on when to save trumps for the end vs. using them early.
"""

        prompt = f"""[SYSTEM]
You are a master Skull King tactician analyzing game logs. Most strategy guides are generic and useless. Your task is to extract "Secret Tech" or "Exploitative Heuristics" that actually win games.

[RULES CONTEXT]
{game_rules}
{trick_mechanics}

[PERSONA & GOAL]
{persona_guide}
Player {pid} Goal: {goal_text}
Result: Player {pid} bid {bid}, but actually won {won} tricks. FAILED.

[ROUND {round_num} STARTING HAND]
Player {pid} started with: {hand_str}

[ROUND {round_num} PLAY-BY-PLAY]
{history_text}

[REFLECTION CONSTRAINTS]
{constraints}
4. NO GENERIC ADVICE: Do not output rules like "Play low" or "Save cards." That is useless.
5. EXPLOIT THE META: Focus on how Player {pid} could have leveraged the specific game state (e.g., "P2 is FULL, therefore lead X," or "The Mermaid is still in the deck, therefore P0 should hold X").
6. CARD SPECIFIC: Rules MUST mention specific cards or interactions (e.g., White Whale, Kraken, Mermaid/King interaction).
7. PSYCHOLOGICAL: Focus on predicting what the opponents are likely to do based on their bid and tricks won.

[TASK]
Analyze the play-by-play. Find the exact trick where the bid failure became inevitable. 
What specific card interaction or opponent motivation did Player {pid} misread?
Output ONE specific rule starting with [RULE]:
Example: [RULE]: When P2 is 'FULL' and leads a low suit, they are trying to lose; if you hold a Pirate, play it now to force P2 to win the trick and fail their bid.
"""

        # Query and Save Logic (Same as before)
        try:
            raw_content, _ = self.client.get_move_with_content(prompt, temperature=0.4) 
            
            match = re.search(r"\[RULE\]\s*:\s*(.*)", raw_content, re.IGNORECASE)
            if match:
                new_rule = match.group(1).strip()
                logger.info(f"Learned New Rule for {persona}: {new_rule}")
                
                metadata = {
                    "round_num": round_num,
                    "bid": bid,
                    "won": won,
                    "phase": "PLAYING"
                }
                self.memory.memorize_rule(new_rule, persona, metadata)
            else:
                logger.warning(f"Failed to extract [RULE] from LLM response:\n{raw_content}")
                
        except Exception as e:
            logger.error(f"Reflection failed: {e}")

    def _generate_bidding_rule(self, pid: int, persona: str, round_num: int, bid: int, won: int, starting_hand: List[str]):
        """NEW: Generates rules for the BIDDING phase (Hand Evaluation)."""
        
        # Load the Bidding Bundle (Intro + Hierarchy + Scoring + Persona)
        bundle = PromptLoader.get_bidding_bundle(persona)
        hand_str = ", ".join(starting_hand) if starting_hand else "Unknown"

        prompt = f"""[SYSTEM]
You are an AI Strategy Architect for Skull King. Analyze the failed round and write a strict, 1-sentence rule about BIDDING.

[RULES CONTEXT]
{bundle}

[FAILURE REPORT]
Player {pid} Starting Hand: {hand_str}
Player {pid} Target Bid: {bid}
Actual Tricks Won: {won}
Result: FAILED BID. The player misjudged the strength of their hand.

[REFLECTION CONSTRAINTS]
1. Focus entirely on HAND EVALUATION. Why did this hand win {won} tricks instead of the predicted {bid}?
2. Identify specific card combinations that are misleading (e.g., "Low trumps are not guaranteed wins", or "Pirates are forced wins").
3. Do not give card-playing advice. Give BIDDING advice (e.g., "Add 1 to your bid if...", "Do not count X as a win").

[TASK]
Analyze the starting hand. Write your Chain-of-Thought reasoning.
Then, output a single strategic rule for the BIDDING PHASE starting with [RULE]:
Example: [RULE]: When holding multiple low-value Trumps (Black 1-5) and no Pirates, do not count them as guaranteed wins, so lower your estimated bid by 1.
"""
        try:
            raw_content, _ = self.client.get_move_with_content(prompt, temperature=0.4) 
            match = re.search(r"\[RULE\]\s*:\s*(.*)", raw_content, re.IGNORECASE)
            
            if match:
                new_rule = match.group(1).strip()
                logger.info(f"Learned New BIDDING Rule for {persona}: {new_rule}")
                
                metadata = {
                    "round_num": round_num,
                    "bid": bid,
                    "won": won,
                    "phase": "BIDDING" # <--- Critical! Saves it to the Bidding Context
                }
                self.memory.memorize_rule(new_rule, persona, metadata)
            else:
                logger.warning(f"Failed to extract [RULE] from BIDDING reflection.")
        except Exception as e:
            logger.error(f"Bidding Reflection failed: {e}")
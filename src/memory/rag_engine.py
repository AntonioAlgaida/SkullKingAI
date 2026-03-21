# src/memory/rag_engine.py

import os
import chromadb
import logging
from typing import List, Dict, Any
from chromadb.config import Settings
from src.engine.physics import GamePhysics, CardType, Suit

logger = logging.getLogger(__name__)

class StrategyMemory:
    """
    The 'Grimoire'. Stores learned strategic rules in a Vector Database.
    Separates memory by Persona (Zero vs Rational).
    """
    def __init__(self, persistence_path="data/chroma_db"):
        self.physics = GamePhysics() 
        
        # Force the path to be absolute, relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        abs_path = os.path.join(base_dir, persistence_path)
        
        os.makedirs(abs_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=abs_path)
        
        self.zero_collection = self.client.get_or_create_collection(name="forced_zero_strategies")
        self.rational_collection = self.client.get_or_create_collection(name="rational_strategies")

    def _get_collection(self, persona: str):
        # Normalize the string (converts hyphens to underscores, makes lowercase)
        normalized_persona = persona.lower().replace("-", "_").strip()
        
        if normalized_persona == "forced_zero":
            return self.zero_collection
        return self.rational_collection

    def retrieve_rules(self, state_dict: Dict[str, Any], persona: str, n_results=10) -> str:
        """
        WAKE CYCLE: Finds rules relevant to the current game state.
        """
        collection = self._get_collection(persona)
        current_phase = state_dict["phase"]
        
        # --- DEBUG: Print what the database thinks is inside ---
        # Look at the first 3 items in the collection
        # sample = collection.peek(3)
        # logger.info(f"DEBUG: Checking {persona} collection. Sample metadata: {sample['metadatas']}")
        
        # Check if collection is empty
        if collection.count() == 0:
            return f"Memory empty for {persona.upper()}. Playing mostly on instinct, based on the rules of the game and your goals."
        
        # 1. Create a "Semantic Query" from the state
        # We describe the situation in text so the vector search matches similar situations.
        # e.g., "Round 8. Holding Skull King. Opponents are full."
        query_text = self._generate_query_context(state_dict)
        
        # 2. Metadata Filtering (Hard constraints)
        # Only look for rules applicable to this Phase (Bidding vs Playing)
        current_phase = state_dict["phase"]
        
        try:
            # We ask for distance metrics now
            results = collection.query(
                query_texts=[query_text],
                n_results=min(n_results, collection.count()), # Don't ask for more than exists
                where={"phase": current_phase}
            )
            
            # results['documents'] is a list of lists.
            if not results['documents'] or not results['documents'][0]:
                return f"No specific past strategies found for {current_phase} phase."
            
            docs = results['documents'][0]
            distances = results['distances'][0]
            
            # --- DYNAMIC FILTERING ---
            # Standard threshold for "Relevant": < 0.5 (for L2 distance)
            # If the best match is 0.2 and the 10th match is 0.45, keep all.
            # If the best is 0.2 and the next is 0.8, only keep the first.
            
            relevant_rules = []
            threshold = 0.55  # Tune this: Lower = Stricter, Higher = More Rules
            
            for doc, dist in zip(docs, distances):
                if dist < threshold:
                    relevant_rules.append(doc)
            
            # Fallback: If nothing meets the threshold, take the top 1 anyway
            if not relevant_rules and docs:
                relevant_rules.append(docs[0])
                
            # Limit the prompt size to max 5-7 rules to prevent context pollution
            final_rules = relevant_rules[:7]

            rules_text = "\n".join([f"- {rule}" for rule in final_rules])
            return rules_text
            
        except Exception as e:
            # If collection is empty, standard behavior
            logger.error(f"Error retrieving rules: {e}")
            return f"Memory empty for {persona.upper()}. Playing mostly on instinct."

    def memorize_rule(self, rule_text: str, persona: str, metadata: Dict[str, Any]):
        """
        SLEEP CYCLE: Saves a newly discovered rule into the DB.
        Includes Semantic Deduplication to prevent repetitive rules.
        """
        collection = self._get_collection(persona)
        
        # --- DEDUPLICATION CHECK ---
        if self._is_duplicate(collection, rule_text):
            logger.info(f"[{persona}] Rule rejected (Duplicate/Too Similar).")
            return

        # Generate ID based on count
        rule_id = f"rule_{collection.count() + 1}"
        
        logger.info(f"[{persona}] Memorizing new strategy: {rule_id}")
        
        collection.add(
            documents=[rule_text],
            metadatas=[metadata],
            ids=[rule_id]
        )
        
    def _is_duplicate(self, collection, new_rule: str, threshold: float = 0.31) -> bool:
        """
        Checks if the new rule is semantically identical to an existing one.
        Threshold 0.31 (L2 distance) is a good baseline for 'very similar sentences'.
        """
        if collection.count() == 0:
            return False
            
        results = collection.query(
            query_texts=[new_rule],
            n_results=1
        )
        
        if not results['distances'] or not results['distances'][0]:
            return False
            
        # Chroma returns distance (smaller = closer)
        closest_distance = results['distances'][0][0]
        closest_text = results['documents'][0][0]
        
        if closest_distance < threshold:
            logger.info(f"Duplicate detected (Dist: {closest_distance:.4f}).\n   New: {new_rule}\n   Old: {closest_text}")
            return True
        else:
            logger.info(f"No duplicate (Dist: {closest_distance:.4f}).\n   New: {new_rule}\n   Closest: {closest_text}")
            
        return False
    
    def _generate_query_context(self, state: Dict[str, Any]) -> str:
        """
        Generates a Rich Semantic Query.
        Includes Phase, Round, AND specific high-value cards held.
        """
        if state["phase"] == "BIDDING":
            return f"Bidding Phase. Round {state['round_num']}. Hand evaluation."
        
        # --- ENHANCED CONTEXT GENERATION ---
        round_num = state["round_num"]
        round_stage = "Late Game" if round_num > 7 else ("Mid Game" if round_num > 4 else "Early Game")
        
        # Identify Key Cards in Hand to make query specific
        hand_keywords = []
        for cid in state["my_hand"]:
            if cid == 74: 
                hand_keywords.append("Tigress Escape")
                continue
            
            card = self.physics.deck[cid]
            if card.card_type == CardType.SKULL_KING:
                hand_keywords.append("Holding Skull King")
            elif card.card_type == CardType.PIRATE:
                hand_keywords.append("Holding Pirate")
            elif card.card_type == CardType.MERMAID:
                hand_keywords.append("Holding Mermaid")
            elif card.card_type == CardType.NUMBER and card.suit == Suit.BLACK and card.value >= 10:
                hand_keywords.append("Holding High Black Trump")
            elif card.card_type == CardType.NUMBER and card.value == 14:
                hand_keywords.append(f"Holding {card.suit.name} 14")

        hand_context = ", ".join(set(hand_keywords)) # Remove dupes
        
        # Final Query String
        # Example: "Playing Phase. Late Game. Holding Skull King, Holding Mermaid."
        return f"Playing Phase. {round_stage}. {hand_context}"
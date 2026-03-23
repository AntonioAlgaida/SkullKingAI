# src/memory/rag_engine.py

import os
import chromadb
import logging
from typing import Dict, Any, List, Tuple
from src.engine.physics import GamePhysics, CardType, Suit

# How much a rule's fitness score shifts its effective retrieval distance.
# Positive fitness → effectively closer (easier to retrieve).
# Set conservatively so fitness nudges ranking without dominating semantics.
FITNESS_WEIGHT  = 0.05
FITNESS_MAX     =  5.0
FITNESS_MIN     = -2.0   # tighter floor — prevents rules from dying permanently
FITNESS_WIN     =  0.8   # raised: reward success more aggressively
FITNESS_LOSS    = -0.2   # lowered: softer penalty — failures dominate in early training

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
            # Fetch 2× more results than needed so fitness re-ranking can surface
            # high-fitness rules that are slightly less semantically close.
            # Retry with 1 if ChromaDB rejects because n_results > filtered count.
            fetch_n = min(n_results * 2, collection.count())
            results = None
            for attempt_n in [fetch_n, 1]:
                try:
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=attempt_n,
                        where={"phase": current_phase},
                        include=["documents", "distances", "metadatas"],
                    )
                    break
                except Exception:
                    if attempt_n == 1:
                        raise
            if results is None:
                raise RuntimeError("All query attempts failed")

            if not results['documents'] or not results['documents'][0]:
                return f"No specific past strategies found for {current_phase} phase."

            docs      = results['documents'][0]
            distances = results['distances'][0]
            metas     = results['metadatas'][0]

            # --- FITNESS-ADJUSTED RE-RANKING ---
            # effective_score = distance - fitness * weight
            # (lower is better — high fitness shrinks effective distance)
            scored: List[Tuple[float, str]] = []
            threshold = 0.55
            for doc, dist, meta in zip(docs, distances, metas):
                fitness  = float(meta.get("fitness", 0.0))
                eff_dist = dist - fitness * FITNESS_WEIGHT
                if eff_dist < threshold:
                    scored.append((eff_dist, doc))

            scored.sort(key=lambda x: x[0])

            # Fallback: if nothing passes threshold, take raw top-1
            if not scored and docs:
                scored = [(distances[0], docs[0])]

            # Re-attach metadata for fitness-tier formatting
            # Build (eff_dist, doc, fitness) triples using the original zip
            meta_map = {doc: float(meta.get("fitness", 0.0))
                        for doc, meta in zip(docs, metas)}
            ranked = [(eff_dist, doc, meta_map.get(doc, 0.0))
                      for eff_dist, doc in scored[:7]]

            return self._format_rules(ranked, query_text)

        except Exception as e:
            logger.error(f"Error retrieving rules: {e}")
            return f"Memory empty for {persona.upper()}. Playing mostly on instinct."

    # ── Fitness-tier formatting ───────────────────────────────────────────── #

    @staticmethod
    def _format_rules(ranked: list, query_text: str) -> str:
        """
        Format retrieved rules into tiers based on fitness so the LLM
        knows which rules are proven, experimental, or questionable.

        Tiers:
          PROVEN       fitness >= 2.0  — strong positive signal
          EXPERIMENTAL fitness 0..2.0  — new or mildly positive
          WEAK         fitness < 0     — previously led to failures
        """
        proven, experimental, weak = [], [], []
        for _eff_dist, doc, fitness in ranked:
            if fitness >= 2.0:
                proven.append((doc, fitness))
            elif fitness >= 0.0:
                experimental.append((doc, fitness))
            else:
                weak.append((doc, fitness))

        lines = [
            f"[STRATEGIC MEMORY — retrieved for: \"{query_text}\"]",
            "Rules from similar past situations, sorted by proven effectiveness:",
        ]

        if proven:
            lines.append("\n[PROVEN STRATEGIES] — follow these confidently:")
            for rule, fit in proven:
                lines.append(f"  ★ (fitness {fit:+.1f}) {rule}")

        if experimental:
            lines.append("\n[EXPERIMENTAL STRATEGIES] — useful hints, apply with judgment:")
            for rule, fit in experimental:
                lines.append(f"  ◆ (fitness {fit:+.1f}) {rule}")

        if weak:
            lines.append("\n[WEAK STRATEGIES] — these previously led to failures; avoid unless situation is very different:")
            for rule, fit in weak:
                lines.append(f"  ✗ (fitness {fit:+.1f}) {rule}")

        return "\n".join(lines)

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

        # Initialise fitness at 0 so the rule starts neutral and must earn its rank
        metadata.setdefault("fitness", 0.0)

        collection.add(
            documents=[rule_text],
            metadatas=[metadata],
            ids=[rule_id]
        )
        
    def _is_duplicate(self, collection, new_rule: str, threshold: float = 0.50) -> bool:
        """
        Checks if the new rule is semantically too similar to an existing one.
        Threshold 0.50 (L2 distance) — catches near-paraphrases of the same insight,
        not just exact copies. 0.31 was too tight; paraphrased duplicates were slipping
        through and filling the Grimoire with redundant variations of the same lesson.
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
    
    def query_rule_ids(
        self, query_text: str, persona: str, phase: str, n_results: int = 5
    ) -> List[str]:
        """
        Returns the IDs of rules semantically relevant to query_text.
        Used by the reflector for offline fitness credit assignment.
        """
        collection = self._get_collection(persona)
        if collection.count() == 0:
            return []
        # ChromaDB raises when n_results exceeds the number of documents that
        # match the where-filter. Retry with n_results=1 as a safe fallback.
        # Note: some ChromaDB versions don't support "ids" in query include —
        # use "documents" (always valid) and read IDs from results["ids"].
        for attempt_n in [min(n_results, collection.count()), 1]:
            try:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=attempt_n,
                    where={"phase": phase},
                    include=["documents"],
                )
                if results["ids"] and results["ids"][0]:
                    return results["ids"][0]
                break
            except Exception as e:
                logger.debug(f"query_rule_ids error (attempt_n={attempt_n}): {e}")
                if attempt_n == 1:
                    return []
        return []

    def update_fitness(self, rule_ids: List[str], delta: float):
        """
        Adjusts the fitness score of each rule by delta (clamped to [FITNESS_MIN, FITNESS_MAX]).
        Positive delta → rule contributed to a win.
        Negative delta → rule was active during a failure.
        """
        if not rule_ids:
            return

        # We need to determine which persona collection holds each rule.
        # Rule IDs are unique across collections, so we try both.
        for collection in (self.zero_collection, self.rational_collection):
            try:
                data = collection.get(ids=rule_ids)
            except Exception:
                continue

            if not data["ids"]:
                continue

            for rid, meta in zip(data["ids"], data["metadatas"]):
                current = float(meta.get("fitness", 0.0))
                new_val = max(FITNESS_MIN, min(FITNESS_MAX, current + delta))
                updated_meta = {**meta, "fitness": new_val}
                collection.update(ids=[rid], metadatas=[updated_meta])
                logger.debug(f"[fitness] {rid}: {current:.2f} → {new_val:.2f} (Δ{delta:+.2f})")

    def generate_query_context(self, state: Dict[str, Any]) -> str:
        """Public alias — used by the action cache and any external caller."""
        return self._generate_query_context(state)

    def _generate_query_context(self, state: Dict[str, Any]) -> str:
        """
        Generates a rich semantic query string for RAG retrieval.
        For BIDDING: includes hand toxicity profile and round stakes.
        For PLAYING: includes key cards held, round stage, and hunger context.
        """
        round_num = state["round_num"]
        round_stage = "Late Game" if round_num > 7 else ("Mid Game" if round_num > 4 else "Early Game")

        if state["phase"] == "BIDDING":
            # Classify hand toxicity inline (mirrors SemanticTranslator.categorize_hand_toxicity)
            toxic = volatile = safe = escapes = 0
            for cid in state["my_hand"]:
                if cid == 74:
                    escapes += 1
                    continue
                c = self.physics.deck[cid]
                if c.card_type in (CardType.ESCAPE, CardType.LOOT):
                    escapes += 1
                elif c.card_type in (CardType.SKULL_KING, CardType.PIRATE, CardType.MERMAID, CardType.TIGRESS):
                    toxic += 1
                elif c.card_type == CardType.NUMBER:
                    if c.suit == Suit.BLACK:
                        if c.value >= 10:
                            toxic += 1
                        else:
                            volatile += 1
                    else:
                        if c.value >= 10:
                            volatile += 1
                        else:
                            safe += 1

            # Summarise opponent states (bids may be -1 if not yet revealed)
            opp_states = []
            my_id = state["current_player_id"]
            for pid, (bid, won) in enumerate(zip(state["bids"], state["tricks_won"])):
                if pid == my_id or bid == -1:
                    continue
                delta = bid - won
                tricks_left = round_num - sum(state["tricks_won"])
                if delta == 0:
                    opp_states.append("FULL")
                elif delta < 0:
                    opp_states.append("OVERBOARD")
                elif delta == tricks_left:
                    opp_states.append("STARVING")
                else:
                    opp_states.append("HUNGRY")

            opp_summary = ", ".join(opp_states) if opp_states else "unknown"
            return (
                f"Bidding Phase. {round_stage}. Round {round_num}. "
                f"Hand: {toxic} Toxic, {volatile} Volatile, {safe} Safe, {escapes} Escapes. "
                f"Opponents: {opp_summary}."
            )

        # PLAYING phase — key cards and situation
        hand_keywords = []
        for cid in state["my_hand"]:
            if cid == 74:
                hand_keywords.append("Tigress as Escape")
                continue
            card = self.physics.deck[cid]
            if card.card_type == CardType.SKULL_KING:
                hand_keywords.append("Holding Skull King")
            elif card.card_type == CardType.PIRATE:
                hand_keywords.append("Holding Pirate")
            elif card.card_type == CardType.MERMAID:
                hand_keywords.append("Holding Mermaid")
            elif card.card_type == CardType.KRAKEN:
                hand_keywords.append("Holding Kraken")
            elif card.card_type == CardType.WHITE_WHALE:
                hand_keywords.append("Holding White Whale")
            elif card.card_type == CardType.NUMBER and card.suit == Suit.BLACK and card.value >= 10:
                hand_keywords.append("Holding High Black Trump")
            elif card.card_type == CardType.NUMBER and card.value == 14:
                hand_keywords.append(f"Holding {card.suit.name} 14")

        hand_context = ", ".join(set(hand_keywords))

        # Tricks needed gives context for whether agent is hunting or ducking
        my_id = state["current_player_id"]
        my_bid = state["bids"][my_id]
        my_won = state["tricks_won"][my_id]
        tricks_needed = my_bid - my_won if my_bid >= 0 else 0
        if tricks_needed == 0:
            mode = "ducking (bid met)"
        elif tricks_needed < 0:
            mode = "over bid (trying to lose)"
        else:
            mode = f"hunting {tricks_needed} more trick(s)"

        return f"Playing Phase. {round_stage}. Round {round_num}. {hand_context}. {mode}."
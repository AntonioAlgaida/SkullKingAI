# agents/heuristic.py

import numpy as np
from src.engine.physics import GamePhysics, CardType, Suit

class HeuristicAgent:
    """
    A rule-based agent that acts as a baseline.
    Strategy:
    - Bidding: Value estimation based on high cards and specials.
    - Playing: Greedy (try to win) if below bid, Anti-Greedy (try to lose) if met bid.
    """
    def __init__(self, physics: GamePhysics):
        self.physics = physics

    def act(self, state_dict: dict) -> int:
        phase = state_dict["phase"]
        
        if phase == "BIDDING":
            return self._calculate_bid(state_dict["my_hand"])
        else: # PLAYING
            return self._select_card(state_dict)

    def _calculate_bid(self, hand: list) -> int:
        """
        Estimate hand strength.
        - Skull King/Mermaid/Pirate: 1 trick each.
        - Black 11-14: 1 trick each.
        - Other 13-14: 0.5 trick each.
        """
        predicted_tricks = 0.0
        
        for c_id in hand:
            if c_id == 74: continue # Skip virtual action
            
            card = self.physics.deck[c_id]
            
            if card.card_type in [CardType.SKULL_KING, CardType.MERMAID, CardType.PIRATE]:
                predicted_tricks += 1.0
            elif card.card_type == CardType.TIGRESS:
                predicted_tricks += 1.0 
            elif card.card_type == CardType.NUMBER:
                if card.suit == Suit.BLACK:
                    if card.value >= 11: predicted_tricks += 1.0
                    elif card.value >= 7: predicted_tricks += 0.5
                else:
                    if card.value >= 13: predicted_tricks += 0.5
        
        return int(round(predicted_tricks))

    def _select_card(self, state_dict: dict) -> int:
        """
        Logic:
        1. Determine if we want to win or lose.
        2. Filter valid actions using mask.
        3. Sort valid actions by 'power'.
        4. Pick best.
        """
        player_idx = state_dict["current_player_id"]
        my_bid = state_dict["bids"][player_idx]
        tricks_won = state_dict["tricks_won"][player_idx]
        legal_actions = state_dict["legal_actions"]
        
        want_to_win = tricks_won < my_bid
        
        scored_actions =[]
        
        for action_id in legal_actions:
            if action_id == 74:
                scored_actions.append((action_id, -1))
                continue
            
            card = self.physics.deck[action_id]
            effective_type = CardType.PIRATE if card.card_type == CardType.TIGRESS else card.card_type
            
            score = 0
            if effective_type == CardType.SKULL_KING: score = 1000
            elif effective_type == CardType.PIRATE: score = 900
            elif effective_type == CardType.MERMAID: score = 800
            elif effective_type == CardType.WHITE_WHALE: score = 0 
            elif effective_type == CardType.KRAKEN: score = 0 
            elif effective_type in[CardType.ESCAPE, CardType.LOOT]: score = -1
            elif effective_type == CardType.NUMBER:
                base = card.value
                if card.suit == Suit.BLACK: score = 200 + base
                else: score = 100 + base # Grouping all colors the same for baseline
                
            scored_actions.append((action_id, score))
            
        scored_actions.sort(key=lambda x: x[1])
        
        if want_to_win:
            return scored_actions[-1][0] # Highest power
        else:
            return scored_actions[0][0]  # Lowest power
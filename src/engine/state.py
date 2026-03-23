# src/engine/state.py

import numpy as np
from collections import deque
from typing import List, Dict, Any, Tuple

from .physics import GamePhysics, CardType, Suit, TrickResult

class SkullKingEnv():
    """
    A pure Python state machine for Skull King (No Gymnasium dependencies).
    Designed to feed clean dictionaries to LLM Semantic Translators.
    """
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.physics = GamePhysics()
        
        # --- ACTION SPACE ---
        # Action Mapping Logic in GamePhysics:
            # Action Mapping Logic
            # 0-13: Green 1-14
            # 14-27: Purple 1-14
            # 28-41: Yellow 1-14
            # 42-55: Black 1-14
            # 56-60: Pirates (5)
            # 61: Tigress as Pirate (Action)
            # 62: Skull King (1)
            # 63-64: Mermaids (2)
            # 65-69: Escapes (5)
            # 70-71: Loot (2)
            # 72: Kraken (1)
            # 73: White Whale (1)
            # 74: Tigress as Escape (Virtual Action)
        # Game State
        self.hands: List[set] = [set() for _ in range(num_players)]
        self.bids: List[int] = [-1] * num_players
        self.tricks_won: List[int] =[0] * num_players
        self.scores: List[int] = [0] * num_players
        self.current_round: int = 1
        self.start_player_index: int = 0
        self.current_player_index: int = 0
        self.phase: int = 0  # 0: Bidding, 1: Playing
        
        # Tracking for AI memory
        self.cards_played_in_round: set = set()
        self.round_bonuses: List[int] = [0] * num_players
        self.active_alliances: List[tuple] = []
        self.current_trick_actions: List[tuple] = []


    def reset(self, starting_round: int = 1, start_player_offset: int = 0) -> Dict[str, Any]:
        """
        Reset the environment for a new game.

        starting_round: first round to simulate (1–10).
          - 1  → full 10-round game (default).
          - k  → game starts at round k and plays through to round 10.
                 Scores begin at 0 regardless of k; only the hand size and
                 round-number dynamics change.
        start_player_offset: shifts which seat leads round 1 (and all subsequent
          rounds by the same amount). Randomising this across games prevents P0
          from always bidding/playing first and biasing the experience distribution.
        """
        self.current_round = max(1, min(starting_round, 10))
        self.scores = [0] * self.num_players
        # Rotate the first leader: base rotation + random seat offset
        self.start_player_index = (self.current_round - 1 + start_player_offset) % self.num_players

        self._deal_round()
        return self.get_state_dict()
    
    def step(self, action_id: int) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """
        Executes an action. Returns (New State, Is_Game_Over, Info_Dict).
        """
        player_id = self.current_player_index
        info = {}
        
        # --- PHASE 0: BIDDING ---
        if self.phase == 0:
            # Action 0-10 is the bid
            self.bids[player_id] = min(action_id, 10)
            
            # Move to next player for bidding
            # In simultaneous bidding simulations, we usually collect all at once.
            # But for standard Gym, we step sequentially. 
            # We assume the wrapper handles the "Simultaneous" aspect by hiding other bids.
            self.current_player_index = (self.current_player_index + 1) % self.num_players
            
            # If we circled back to start_player, bidding is done
            if self.current_player_index == self.start_player_index:
                self.phase = 1
                self.current_trick_actions = []
            
            return self.get_state_dict(), 0, False, info

        # --- PHASE 1: PLAYING ---
        if self.phase == 1:
            # Validate Move
            if not self._is_legal_move(player_id, action_id):
                raise ValueError(f"Player {player_id} attempted illegal move {action_id}")

            # Execute Play
            self._remove_card_from_hand(player_id, action_id)
            self.current_trick_actions.append((player_id, action_id))
            
           # Update Graveyard Memory
            if action_id == 74: # Tigress as Escape
                tigress_id = next(k for k,v in self.physics.deck.items() if v.card_type == CardType.TIGRESS)
                self.cards_played_in_round.add(tigress_id)
            else:
                self.cards_played_in_round.add(action_id)

            # Trick Complete?
            if len(self.current_trick_actions) == self.num_players:
                return self._resolve_trick_end()
            
            self.current_player_index = (self.current_player_index + 1) % self.num_players
            return self.get_state_dict(), 0, False, info

    def _resolve_trick_end(self):
        # Cache the player who played the final card of the trick
        last_player_id = self.current_player_index 
        
        result: TrickResult = self.physics.resolve_trick(self.current_trick_actions)
        
        # Update Tricks Won
        if result.winner_id is not None:
            self.tricks_won[result.winner_id] += 1
            self.round_bonuses[result.winner_id] += result.bonus_points
            
        next_leader = result.next_lead_id
        
        # Track Alliances
        if result.alliance_formed:
            loot_player = -1
            for pid, aid in self.current_trick_actions:
                if self.physics.actions[aid]['as_type'] == CardType.LOOT:
                    loot_player = pid
                    break
            if loot_player != -1:
                self.active_alliances.append((loot_player, result.winner_id))
        
        safe_winner_id = result.winner_id if result.winner_id is not None else -1
        
        trick_info = {
            "trick_winner": safe_winner_id,
            "trick_bonus": result.bonus_points,
            "trick_destroyed": result.destroyed,
            "trick_alliance": result.alliance_formed
        }

        # Check Round End
        if len(self.hands[0]) == 0:
            round_rewards = self._calculate_round_rewards()
            final_bids = list(self.bids)
            final_tricks = list(self.tricks_won)
            
            for i in range(self.num_players):
                self.scores[i] += round_rewards[i]
            
            # Check Game End
            if self.current_round == 10:
                return self.get_state_dict(), round_rewards[last_player_id], True, {
                    "final_scores": list(self.scores),
                    "round_rewards": round_rewards,
                    "bids": final_bids,
                    "won": final_tricks,
                    **trick_info
                }
            
            # Setup Next Round
            self.current_round += 1
            self.start_player_index = (self.start_player_index + 1) % self.num_players
            self._deal_round()
            
            return self.get_state_dict(), round_rewards[last_player_id], False, {
                "round_rewards": round_rewards,
                "bids": final_bids,
                "won": final_tricks,
                **trick_info
            }
        
        # Setup Next Trick
        self.current_player_index = next_leader
        self.current_trick_actions = []
        return self.get_state_dict(), 0, False, trick_info

    def _calculate_round_rewards(self) -> List[int]:
        # 1. Determine who made their bid
        bid_success = [False] * self.num_players
        for i in range(self.num_players):
            if self.bids[i] == self.tricks_won[i]:
                bid_success[i] = True

        # 2. Calculate Base Scores (Individual Performance)
        rewards = [0] * self.num_players
        for i in range(self.num_players):
            bid = self.bids[i]
            won = self.tricks_won[i]
            score = 0
            
            if bid == 0:
                # Zero Bid Logic
                score = 10 * self.current_round if won == 0 else -10 * self.current_round
            else:
                # Standard Bid Logic
                if won == bid:
                    score = 20 * bid
                    score += self.round_bonuses[i] # Capture bonuses (Pirates, 14s, etc.)
                else:
                    diff = abs(won - bid)
                    score = -10 * diff
            
            rewards[i] = score

        # 3. Apply Cooperative Bonuses (Loot Alliances)
        # Done outside the main loop to explicitly handle the multi-agent interaction
        for looter, winner in self.active_alliances:
            # Rule: "If BOTH get their bid correct, they are EACH awarded 20 bonus points"
            if bid_success[looter] and bid_success[winner]:
                rewards[looter] += 20
                rewards[winner] += 20
                # Optional: Logging for debugging
                # print(f"\tAlliance Bonus: +20 to Player {looter} and {winner}")

        return rewards

    def _deal_round(self):
        """Resets state for a new round."""
        self.bids = [-1] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick_actions = []
        self.current_player_index = self.start_player_index
        self.phase = 0 
        self.cards_played_in_round = set()
        self.round_bonuses = [0] * self.num_players
        self.active_alliances = []
        
        deck_keys = list(self.physics.deck.keys())
        np.random.shuffle(deck_keys)
        
        ptr = 0
        num_cards_to_deal = self.current_round
        if self.num_players == 8 and self.current_round >= 9:
            num_cards_to_deal = 8
            
        for i in range(self.num_players):
            self.hands[i] = set()
            for _ in range(num_cards_to_deal):
                self.hands[i].add(deck_keys[ptr])
                ptr += 1

    def _remove_card_from_hand(self, player_id, action_id):
        """Removes the card associated with the action."""
        if action_id == 74:
            # Remove the Tigress card ID (61)
            tigress_id = next(k for k,v in self.physics.deck.items() if v.card_type == CardType.TIGRESS)
            self.hands[player_id].remove(tigress_id)
        else:
            self.hands[player_id].remove(action_id)


    def get_legal_actions(self, player_id: int) -> List[int]:
        """Returns a list of valid integer Action IDs for the player."""
        if self.phase == 0:
            return list(range(11)) # Bids 0-10
            
        legal_actions = []
        hand = self.hands[player_id]
        if not hand:
            return[]

        tigress_id = next(k for k,v in self.physics.deck.items() if v.card_type == CardType.TIGRESS)
        valid_cards_in_hand =[self.physics.deck[c_id] for c_id in hand]

        # Determine Lead Suit
        lead_suit = None
        if len(self.current_trick_actions) > 0:
            for pid, aid in self.current_trick_actions:
                config = self.physics.actions[aid]
                c_type, c_suit = config['as_type'], config['card'].suit
                
                if self.current_trick_actions[0] == (pid, aid) and c_type in[CardType.PIRATE, CardType.SKULL_KING, CardType.MERMAID]:
                    break # No lead suit
                
                if c_type not in [CardType.ESCAPE, CardType.LOOT] and c_suit != Suit.SPECIAL:
                    lead_suit = c_suit
                    break
        
        has_suit = any(c.suit == lead_suit for c in valid_cards_in_hand)

        for c_id in hand:
            card = self.physics.deck[c_id]
            is_special = card.suit == Suit.SPECIAL
            
            if is_special:
                legal_actions.append(c_id)
                if c_id == tigress_id:
                    legal_actions.append(74) # Add Tigress Escape action
                continue
            
            if lead_suit is None or not has_suit or card.suit == lead_suit:
                legal_actions.append(c_id)

        return legal_actions


    def _is_legal_move(self, player_id: int, action_id: int) -> bool:
        return action_id in self.get_legal_actions(player_id)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Creates a clean dictionary of the current state.
        This replaces `_get_obs()` and is fed directly to the LLM Prompt Translator.
        """
        return {
            "round_num": self.current_round,
            "phase": "BIDDING" if self.phase == 0 else "PLAYING",
            "current_player_id": self.current_player_index,
            "my_hand": list(self.hands[self.current_player_index]),
            "legal_actions": self.get_legal_actions(self.current_player_index),
            "bids": self.bids,           # Note: Prompt Translator should hide unrevealed bids if phase==0
            "tricks_won": self.tricks_won,
            "scores": self.scores,
            "current_trick": self.current_trick_actions, # List of (player_id, action_id)
            "graveyard": list(self.cards_played_in_round),
        }
# tests/test_env.py

import unittest
import numpy as np
from src.engine.state import SkullKingEnv
from src.engine.physics import CardType, Suit

class TestSkullKingEnv(unittest.TestCase):
    def setUp(self):
        self.env = SkullKingEnv(num_players=4)
        self.env.reset()
        
    def get_card_id(self, c_type, suit=None, val=None):
        """Helper to find ID from deck"""
        for cid, card in self.env.physics.deck.items():
            if card.card_type == c_type:
                if c_type == CardType.NUMBER:
                    if card.suit == suit and card.value == val: return cid
                else:
                    return cid
        return -1

    def test_round_deal_counts(self):
        """Test if correct number of cards are dealt per round."""
        # Round 1
        self.env.reset()
        self.assertEqual(len(self.env.hands[0]), 1)
        
        # Round 5
        self.env.current_round = 5
        self.env._deal_round()
        self.assertEqual(len(self.env.hands[0]), 5)
        
        # 8-player cap (Round 10 -> 8 cards)
        env8 = SkullKingEnv(num_players=8)
        env8.current_round = 10
        env8._deal_round()
        self.assertEqual(len(env8.hands[0]), 8, "8-player game should cap at 8 cards in round 10")

    def test_action_mask_follow_suit(self):
        """CRITICAL: If I have lead suit, I MUST play it (or special)."""
        self.env.phase = 1
        
        # P0 plays Green 14
        g14 = self.get_card_id(CardType.NUMBER, Suit.GREEN, 14)
        self.env.current_trick_actions = [(0, g14)]
        
        # Setup P1 Hand: Has Green 5, Yellow 5, and Pirate
        g5 = self.get_card_id(CardType.NUMBER, Suit.GREEN, 5)
        y5 = self.get_card_id(CardType.NUMBER, Suit.YELLOW, 5)
        pir = self.get_card_id(CardType.PIRATE)
        
        self.env.hands[1] = {g5, y5, pir}
        
        legal_actions = self.env.get_legal_actions(player_id=1)

        # Assertions
        self.assertIn(g5, legal_actions, "Should be allowed to follow suit")
        self.assertIn(pir, legal_actions, "Should always be allowed to play Special")
        self.assertNotIn(y5, legal_actions, "Should NOT be allowed to play off-suit (Yellow)")

    def test_action_mask_void_suit(self):
        """If I am void in lead suit, I can play anything."""
        self.env.phase = 1
        
        # P0 plays Green 14
        g14 = self.get_card_id(CardType.NUMBER, Suit.GREEN, 14)
        self.env.current_trick_actions = [(0, g14)]
        
        # Setup P1 Hand: No Green. Has Yellow 5 and Pirate.
        y5 = self.get_card_id(CardType.NUMBER, Suit.YELLOW, 5)
        pir = self.get_card_id(CardType.PIRATE)
        
        self.env.hands[1] = {y5, pir}
        
        legal_actions = self.env.get_legal_actions(player_id=1)
        
        # Assertions
        self.assertIn(y5, legal_actions, "Can play off-suit if void")
        self.assertIn(pir, legal_actions, "Can play special")

    def test_tigress_duality_mask(self):
        """Tigress should enable both Action ID (Pirate) and Action 74 (Escape)."""
        self.env.phase = 1
        tigress_id = self.get_card_id(CardType.TIGRESS)
        
        self.env.hands[0] = {tigress_id}
        legal_actions = self.env.get_legal_actions(player_id=0)
        
        self.assertIn(tigress_id, legal_actions, "Can play Tigress as Pirate")
        self.assertIn(74, legal_actions, "Can play Tigress as Escape")

    def test_scoring_standard(self):
        """Test Standard Bid Scoring (+20 / -10)."""
        # Scenario: Bid 2, Won 2.
        self.env.bids = [2, 0, 0, 0]
        self.env.tricks_won = [2, 0, 0, 0]
        
        rewards = self.env._calculate_round_rewards()
        self.assertEqual(rewards[0], 40) # 2 * 20
        
        # Scenario: Bid 2, Won 3.
        self.env.bids = [2, 0, 0, 0]
        self.env.tricks_won = [3, 0, 0, 0]
        
        rewards = self.env._calculate_round_rewards()
        self.assertEqual(rewards[0], -10) # |3-2| * -10
        
    def test_scoring_loot_alliance(self):
        """Test Loot Alliance Scoring (+20 to both)."""
        self.env.bids = [1, 1, 0, 0]
        self.env.tricks_won = [1, 1, 0, 0]
        # P0 played Loot, P1 won trick
        self.env.active_alliances = [(0, 1)]
        
        rewards = self.env._calculate_round_rewards()
        # P0: 20 (base) + 20 (alliance) = 40
        # P1: 20 (base) + 20 (alliance) = 40
        self.assertEqual(rewards[0], 40)
        self.assertEqual(rewards[1], 40)

    def test_scoring_zero_bid(self):
        """Test Zero Bid Scoring (+10*Round / -10*Round)."""
        self.env.current_round = 5
        
        # Scenario: Bid 0, Won 0 (Success)
        self.env.bids = [0, 1, 1, 1]
        self.env.tricks_won = [0, 1, 1, 1]
        rewards = self.env._calculate_round_rewards()
        self.assertEqual(rewards[0], 50) # 10 * 5
        
        # Scenario: Bid 0, Won 1 (Fail)
        self.env.tricks_won = [1, 1, 1, 1]
        rewards = self.env._calculate_round_rewards()
        self.assertEqual(rewards[0], -50) # -10 * 5

    def test_phase_transition(self):
        """Test transition from Bidding to Playing."""
        self.env.reset()
        self.assertEqual(self.env.phase, 0) # Start in Bidding
        
        # 4 Players make bids
        self.env.step(1) # P0 bids 1
        self.env.step(0) # P1 bids 0
        self.env.step(2) # P2 bids 2
        self.env.step(1) # P3 bids 1
        
        self.assertEqual(self.env.phase, 1, "Should switch to Playing Phase")
        self.assertEqual(self.env.current_player_index, 0, "P0 should start leading")
    
    def test_get_state_dict_formatting(self):
        """Ensure the state_dict outputs the clean semantic format expected by the LLM."""
        state = self.env.reset()
        
        # Check keys
        expected_keys =["round_num", "phase", "current_player_id", "my_hand", 
                         "legal_actions", "bids", "tricks_won", "scores", 
                         "current_trick", "graveyard"]
        for key in expected_keys:
            self.assertIn(key, state)
            
        self.assertEqual(state["phase"], "BIDDING")
        self.assertEqual(len(state["my_hand"]), 1)
        
        # Bids should not crash the dictionary representation
        self.assertEqual(len(state["bids"]), 4)

if __name__ == '__main__':
    unittest.main()
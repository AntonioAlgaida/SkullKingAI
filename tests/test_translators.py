# tests/test_translators.py

import unittest
from src.utils.translators import SemanticTranslator
from src.engine.physics import GamePhysics, CardType, Suit

class TestSemanticTranslator(unittest.TestCase):
    def setUp(self):
        self.physics = GamePhysics()
        self.translator = SemanticTranslator(self.physics)

    def get_id(self, c_type, suit=None, val=None):
        """Helper to find IDs"""
        for cid, card in self.physics.deck.items():
            if card.card_type == c_type:
                if c_type == CardType.NUMBER:
                    if card.suit == suit and card.value == val: return cid
                else: return cid
        return -1

    def test_card_translation(self):
        # Standard Number
        g14 = self.get_id(CardType.NUMBER, Suit.GREEN, 14)
        self.assertEqual(self.translator.translate_card(g14), "[Green 14]")
        
        # Skull King
        sk = self.get_id(CardType.SKULL_KING)
        self.assertEqual(self.translator.translate_card(sk), "[Skull King]")
        
        # Tigress Duality
        tig_id = self.get_id(CardType.TIGRESS)
        self.assertEqual(self.translator.translate_card(tig_id), "[Tigress (Played as Pirate)]")
        self.assertEqual(self.translator.translate_card(74), "[Tigress (Played as Escape)]")
        
        # Kraken
        kraken_id = self.get_id(CardType.KRAKEN)
        self.assertEqual(self.translator.translate_card(kraken_id), "[Kraken]")
        
        # Mermaid
        mer_id = self.get_id(CardType.MERMAID)
        self.assertEqual(self.translator.translate_card(mer_id), "[Mermaid]")
        
        # Pirate
        pir_id = self.get_id(CardType.PIRATE)
        self.assertEqual(self.translator.translate_card(pir_id), "[Pirate]")

    # --- 2. HUNGER MATRIX (PSYCHOLOGY) TESTS ---
    def test_hunger_matrix_states(self):
        """Test the psychological state calculations."""
        state = {
            "round_num": 5,
            "current_player_id": 0,
            "bids": [0, 2, 1, 3], 
            # P0 (Me): Bid 0
            # P1: Bid 2
            # P2: Bid 1
            # P3: Bid 3
            "tricks_won": [0, 2, 2, 0] 
            # Total won: 4. Tricks left in round: 5 - 4 = 1.
        }
        
        matrix = self.translator.get_hunger_matrix(state)
        
        # Player 1: Bid 2, Won 2 -> Delta 0 -> FULL
        self.assertIn("Player 1: FULL", matrix)
        
        # Player 2: Bid 1, Won 2 -> Delta -1 -> OVERBOARD
        self.assertIn("Player 2: OVERBOARD", matrix)
        
        # Player 3: Bid 3, Won 0. Needs 3 wins. Only 1 trick left. -> DOOMED
        self.assertIn("Player 3: DOOMED", matrix)

    def test_starving_logic(self):
        """Test 'STARVING' (Must win everything) vs 'HUNGRY' (Wants to win)."""
        # Scenario: Round 10. 0 tricks played. 10 tricks left.
        state = {
            "round_num": 10,
            "current_player_id": 0,
            "bids": [0, 5, 10, 0],
            "tricks_won": [0, 0, 0, 0] 
        }
        matrix = self.translator.get_hunger_matrix(state)

        # Player 1: Needs 5 wins. 10 left. -> HUNGRY (Selective)
        self.assertIn("Player 1: HUNGRY", matrix)
        
        # Player 2: Needs 10 wins. 10 left. -> STARVING (Desperate)
        self.assertIn("Player 2: STARVING", matrix)

    def test_hidden_bids(self):
        """Ensure Phase 0 (Bidding) doesn't crash calculations with -1 bids."""
        state = {
            "round_num": 1,
            "current_player_id": 0,
            "bids": [-1, -1, -1, -1],
            "tricks_won": [0, 0, 0, 0]
        }
        matrix = self.translator.get_hunger_matrix(state)
        self.assertIn("Player 1: BID HIDDEN", matrix)
        
        # --- 3. TOXICITY BUCKETING TESTS ---
    def test_toxicity_boundaries(self):
        """
        Verify the exact cutoffs for Toxic vs Volatile.
        Toxic: Black 10-14, Specials.
        Volatile: Black 1-9, Color 10-14.
        Safe: Color 1-9.
        """
        # Boundary Cases
        b9 = self.get_id(CardType.NUMBER, Suit.BLACK, 9)   # Volatile
        b10 = self.get_id(CardType.NUMBER, Suit.BLACK, 10) # Toxic
        
        g9 = self.get_id(CardType.NUMBER, Suit.GREEN, 9)   # Safe
        g10 = self.get_id(CardType.NUMBER, Suit.GREEN, 10) # Volatile
        
        sk = self.get_id(CardType.SKULL_KING)              # Toxic
        esc = self.get_id(CardType.ESCAPE)                 # Escape
        loot = self.get_id(CardType.LOOT)                  # Escape
        
        hand = [b9, b10, g9, g10, sk, esc, loot]
        
        profile = self.translator.categorize_hand_toxicity(hand)
        
        # Expected Counts:
        # Toxic: 2 (Black 10 + Skull King)
        # Volatile: 2 (Black 9 + Green 10)
        # Safe: 1 (Green 9)
        # Escapes: 2 (Escape + Loot)
        
        self.assertIn("2 Toxic", profile)
        self.assertIn("2 Volatile", profile)
        self.assertIn("1 Safe", profile)
        self.assertIn("2 Escapes", profile)
        
    def test_toxicity_bucketing(self):
        sk = self.get_id(CardType.SKULL_KING)
        b14 = self.get_id(CardType.NUMBER, Suit.BLACK, 14)
        g2 = self.get_id(CardType.NUMBER, Suit.GREEN, 2)
        esc = self.get_id(CardType.ESCAPE)
        
        hand = [sk, b14, g2, esc]
        # sk (Toxic), b14 (Toxic), g2 (Safe), esc (Escape)
        profile = self.translator.categorize_hand_toxicity(hand)
        
        self.assertIn("2 Toxic", profile)
        self.assertIn("1 Safe", profile)
        self.assertIn("1 Escapes", profile)

    def test_graveyard_summary(self):
        sk = self.get_id(CardType.SKULL_KING)
        pirate = self.get_id(CardType.PIRATE)
        b1 = self.get_id(CardType.NUMBER, Suit.BLACK, 1)
        
        summary = self.translator.summarize_graveyard([sk, pirate, b1])
        
        self.assertIn("Skull King: PLAYED", summary)
        self.assertIn("Pirates Played: 1/5", summary)
        self.assertIn("Jolly Rogers (Black Trumps) Played: 1/14", summary)

    def test_prompt_context_structure(self):
        """Ensure the final prompt contains the correct semantic headers."""
        state = {
            "round_num": 1,
            "phase": "PLAYING",
            "current_player_id": 0,
            "my_hand": [74],
            "legal_actions": [74],
            "bids": [0, 1, 0, 0],
            "tricks_won": [0, 0, 0, 0],
            "current_trick": [],
            "graveyard": []
        }
        prompt = self.translator.build_llm_prompt_context(state, "Forced-Zero")
        
        self.assertIn("[STATE TRANSLATION]", prompt)
        self.assertIn("Persona: FORCED-ZERO", prompt)
        self.assertIn("[OPPONENT THREAT LEVEL]", prompt)
        self.assertIn("[GRAVEYARD HUD]", prompt)

    def test_tigress_action_74_toxicity(self):
        """Action 74 (Tigress as Escape) should count as an Escape in hand profiling."""
        # Note: Usually hand contains card IDs, but if the agent is considering moves, 
        # we want to ensure 74 is handled gracefully if it appears.
        hand = [74]
        profile = self.translator.categorize_hand_toxicity(hand)
        self.assertIn("1 Escapes", profile)

    # --- 4. GRAVEYARD HUD TESTS ---
    def test_graveyard_summary_counts(self):
        sk = self.get_id(CardType.SKULL_KING)
        pirate = self.get_id(CardType.PIRATE)
        b1 = self.get_id(CardType.NUMBER, Suit.BLACK, 1)
        kraken = self.get_id(CardType.KRAKEN)
        
        summary = self.translator.summarize_graveyard([sk, pirate, b1, kraken])
        
        self.assertIn("Skull King: PLAYED", summary)
        self.assertIn("Kraken: PLAYED", summary)
        self.assertIn("White Whale: UNPLAYED", summary)
        self.assertIn("Pirates Played: 1/5", summary)
        self.assertIn("Jolly Rogers (Black Trumps) Played: 1/14", summary)

    def test_graveyard_virtual_action_safety(self):
        """Ensure Action 74 in graveyard doesn't crash the summarizer."""
        # Action 74 is not in self.physics.deck keys.
        summary = self.translator.summarize_graveyard([74])
        # Should just run without error
        self.assertIsInstance(summary, str)

    # --- 5. PROMPT STRUCTURE TESTS ---
    def test_playing_phase_prompt(self):
        state = {
            "round_num": 1,
            "phase": "PLAYING",
            "current_player_id": 0,
            "my_hand": [],
            "legal_actions": [],
            "bids": [0, 0, 0, 0],
            "tricks_won": [0, 0, 0, 0],
            "current_trick": [],
            "graveyard": []
        }
        prompt = self.translator.build_llm_prompt_context(state, "Forced-Zero")
        
        self.assertIn("Phase: PLAYING", prompt)
        self.assertIn("[OPPONENT THREAT LEVEL]", prompt)
        self.assertIn("You are leading the trick", prompt)

    def test_bidding_phase_prompt(self):
        """Bidding prompt is simpler and focused on hand analysis."""
        state = {
            "round_num": 1,
            "phase": "BIDDING",
            "current_player_id": 0,
            "my_hand": [],
            "legal_actions": [], # irrelevant for text gen but required by type
            "bids": [-1]*4,
            "tricks_won": [0]*4,
            "current_trick": [],
            "graveyard": []
        }
        prompt = self.translator.build_llm_prompt_context(state, "Rational")
        
        self.assertIn("Phase: BIDDING", prompt)
        self.assertIn("[YOUR HAND]", prompt)
        self.assertNotIn("[GRAVEYARD HUD]", prompt) # No graveyard in bidding
        self.assertIn("Output your CoT reasoning", prompt)


if __name__ == '__main__':
    unittest.main()
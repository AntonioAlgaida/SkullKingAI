# tests/test_physics.py

import unittest
from src.engine.physics import GamePhysics, Suit, CardType, TrickResult

class TestSkullKingPhysics(unittest.TestCase):
    def setUp(self):
        self.physics = GamePhysics()
        
    def get_action_id(self, card_type, suit=None, value=None, as_escape=False):
        """Helper to find the Action ID for a specific card description."""
        # Special case: Tigress played as Escape
        if card_type == CardType.TIGRESS and as_escape:
            return 74

        for aid, config in self.physics.actions.items():
            # Skip the virtual Tigress-Escape action (74) unless requested
            if aid == 74: continue
            
            c = config['card']
            if c.card_type == card_type:
                # If it's a number card, match suit and value
                if card_type == CardType.NUMBER:
                    if c.suit == suit and c.value == value:
                        return aid
                # If it's a special card, just return the first one found
                # (Since all Pirates are identical in logic, etc.)
                else:
                    return aid
        raise ValueError(f"Card not found: {card_type} {suit} {value}")

    # --- 1. SUIT HIERARCHY TESTS ---
    
    def test_standard_suit_hierarchy(self):
        # Green 14 leads, Green 5 follows. 14 should win.
        p1 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        p2 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 5)
        
        result = self.physics.resolve_trick([(0, p1), (1, p2)])
        self.assertEqual(result.winner_id, 0)
        self.assertEqual(result.bonus_points, 10) # 10 pts for Green 14

    def test_off_suit_loses(self):
        # Green 14 leads, Yellow 13 follows. Green wins (Yellow is off-suit).
        p1 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        p2 = self.get_action_id(CardType.NUMBER, Suit.YELLOW, 13)
        
        result = self.physics.resolve_trick([(0, p1), (1, p2)])
        self.assertEqual(result.winner_id, 0)

    def test_trump_wins(self):
        # Green 14 leads, Black 2 (Trump) follows. Black wins.
        p1 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        p2 = self.get_action_id(CardType.NUMBER, Suit.BLACK, 2)
        
        result = self.physics.resolve_trick([(0, p1), (1, p2)])
        self.assertEqual(result.winner_id, 1)

    # --- 2. SPECIAL CARD HIERARCHY (RPS) ---

    def test_mermaid_beats_skull_king(self):
        # P1: Skull King, P2: Mermaid. Mermaid wins + 40 bonus.
        sk = self.get_action_id(CardType.SKULL_KING)
        mer = self.get_action_id(CardType.MERMAID)
        
        result = self.physics.resolve_trick([(0, sk), (1, mer)])
        self.assertEqual(result.winner_id, 1)
        self.assertEqual(result.bonus_points, 40)

    def test_skull_king_beats_pirate(self):
        # P1: Pirate, P2: Skull King. King wins + 30 bonus.
        pir = self.get_action_id(CardType.PIRATE)
        sk = self.get_action_id(CardType.SKULL_KING)
        
        result = self.physics.resolve_trick([(0, pir), (1, sk)])
        self.assertEqual(result.winner_id, 1)
        self.assertEqual(result.bonus_points, 30)

    def test_pirate_beats_mermaid(self):
        # P1: Mermaid, P2: Pirate. Pirate wins + 20 bonus.
        mer = self.get_action_id(CardType.MERMAID)
        pir = self.get_action_id(CardType.PIRATE)
        
        result = self.physics.resolve_trick([(0, mer), (1, pir)])
        self.assertEqual(result.winner_id, 1)
        self.assertEqual(result.bonus_points, 20)

    def test_full_rps_interaction(self):
        # Rule: If Pirate, King, AND Mermaid are in the trick, Mermaid wins.
        pir = self.get_action_id(CardType.PIRATE)
        sk = self.get_action_id(CardType.SKULL_KING)
        mer = self.get_action_id(CardType.MERMAID)
        
        # Order: Pirate -> King -> Mermaid
        result = self.physics.resolve_trick([(0, pir), (1, sk), (2, mer)])
        self.assertEqual(result.winner_id, 2) # Mermaid wins
        self.assertEqual(result.bonus_points, 40) # Bonus for killing King

    # --- 3. TIGRESS LOGIC ---

    def test_tigress_as_pirate(self):
        # Tigress (Pirate) vs Black 14. Tigress wins.
        tig = self.get_action_id(CardType.TIGRESS, as_escape=False)
        b14 = self.get_action_id(CardType.NUMBER, Suit.BLACK, 14)
        
        result = self.physics.resolve_trick([(0, tig), (1, b14)])
        self.assertEqual(result.winner_id, 0)

    def test_tigress_as_escape(self):
        # Tigress (Escape) vs Green 2. Green 2 wins.
        tig = self.get_action_id(CardType.TIGRESS, as_escape=True)
        g2 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 2)
        
        result = self.physics.resolve_trick([(0, tig), (1, g2)])
        self.assertEqual(result.winner_id, 1)

    # --- 4. DISRUPTORS (KRAKEN & WHALE) ---

    def test_kraken_destroys_trick(self):
        # P1: Green 14, P2: Kraken. Destroyed.
        g14 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        kraken = self.get_action_id(CardType.KRAKEN)
        
        result = self.physics.resolve_trick([(0, g14), (1, kraken)])
        self.assertIsNone(result.winner_id)
        self.assertTrue(result.destroyed)
        self.assertEqual(result.next_lead_id, 0) # P1 would have won

    def test_white_whale_resets_suits(self):
        # P1: Black 2 (Trump), P2: Green 14, P3: White Whale.
        # Whale turns Black 2 into just "2". Green 14 becomes "14". 14 wins.
        b2 = self.get_action_id(CardType.NUMBER, Suit.BLACK, 2)
        g14 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        whale = self.get_action_id(CardType.WHITE_WHALE)
        
        result = self.physics.resolve_trick([(0, b2), (1, g14), (2, whale)])
        self.assertEqual(result.winner_id, 1) # Green 14 (now just 14) > 2
        self.assertFalse(result.destroyed)

    def test_kraken_vs_whale_precedence(self):
        # Rule: Second one played determines action.
        kraken = self.get_action_id(CardType.KRAKEN)
        whale = self.get_action_id(CardType.WHITE_WHALE)
        g14 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        
        # Order: Kraken -> Whale -> G14
        # Whale is second disruptor -> Whale effect applies.
        # G14 wins (highest number).
        result = self.physics.resolve_trick([(0, kraken), (1, whale), (2, g14)])
        self.assertEqual(result.winner_id, 2)
        self.assertFalse(result.destroyed)

        # Order: Whale -> Kraken -> G14
        # Kraken is second -> Kraken effect applies.
        result_k = self.physics.resolve_trick([(0, whale), (1, kraken), (2, g14)])
        self.assertTrue(result_k.destroyed)

    def test_white_whale_all_specials(self):
        # P1: Pirate, P2: King, P3: Whale.
        # Whale zeros specials. No numbers played. Trick destroyed.
        pir = self.get_action_id(CardType.PIRATE)
        sk = self.get_action_id(CardType.SKULL_KING)
        whale = self.get_action_id(CardType.WHITE_WHALE)
        
        result = self.physics.resolve_trick([(0, pir), (1, sk), (2, whale)])
        self.assertTrue(result.destroyed)
        self.assertEqual(result.next_lead_id, 0) # P1 led

    # --- 5. LOOT & ESCAPES ---

    def test_loot_alliance(self):
        # P1: Loot, P2: Green 14. P2 wins. Alliance formed.
        loot = self.get_action_id(CardType.LOOT)
        g14 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        
        result = self.physics.resolve_trick([(0, loot), (1, g14)])
        self.assertEqual(result.winner_id, 1)
        self.assertTrue(result.alliance_formed)

    def test_loot_card_alliance_formation(self):
        """
        Rule: 'An alliance is formed between the player who played it 
        and the player who won the trick.'
        """
        loot = self.get_action_id(CardType.LOOT)
        king = self.get_action_id(CardType.SKULL_KING)
        
        # P0 plays Loot, P1 plays King.
        res = self.physics.resolve_trick([(0, loot), (1, king)])
        
        self.assertEqual(res.winner_id, 1)
        self.assertTrue(res.alliance_formed, "Alliance should form when Loot is captured by another")

    def test_all_escapes(self):
        # P1: Escape, P2: Escape. P1 wins.
        esc1 = self.get_action_id(CardType.ESCAPE)
        esc2 = self.get_action_id(CardType.ESCAPE) # Will act as a different escape card
        
        result = self.physics.resolve_trick([(0, esc1), (1, esc2)])
        self.assertEqual(result.winner_id, 0)
        
    def test_bonus_stacking_king_captures_pirate_and_14(self):
        """
        Scenario: P0 plays Skull King. P1 plays Pirate. P2 plays Yellow 14.
        Result: King wins.
        Bonus: 30 (for Pirate) + 10 (for Yellow 14) = 40 total.
        """
        king = self.get_action_id(CardType.SKULL_KING)
        pirate = self.get_action_id(CardType.PIRATE)
        y14 = self.get_action_id(CardType.NUMBER, Suit.YELLOW, 14)
        
        res = self.physics.resolve_trick([(0, king), (1, pirate), (2, y14)])
        
        self.assertEqual(res.winner_id, 0)
        self.assertEqual(res.bonus_points, 40, "Bonuses should stack (30 for Pirate + 10 for 14)")
        
    
    def test_white_whale_tie_breaker(self):
        """
        Scenario: P0 plays Green 14. P1 plays Purple 14. P2 plays White Whale.
        Result: Both are value 14. P0 played first. P0 wins.
        """
        g14 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        p14 = self.get_action_id(CardType.NUMBER, Suit.PURPLE, 14)
        whale = self.get_action_id(CardType.WHITE_WHALE)
        
        res = self.physics.resolve_trick([(0, g14), (1, p14), (2, whale)])
        
        self.assertEqual(res.winner_id, 0, "Tie under White Whale goes to first played")
        self.assertFalse(res.destroyed)
        
    
    def test_loot_wins_no_alliance(self):
        """
        Scenario: P0 plays Loot. P1 plays Escape. P2 plays Escape.
        Result: P0 wins (First played escape logic).
        Alliance: False (You can't ally with yourself).
        """
        loot = self.get_action_id(CardType.LOOT)
        esc = self.get_action_id(CardType.ESCAPE)
        
        res = self.physics.resolve_trick([(0, loot), (1, esc), (2, esc)])
        
        self.assertEqual(res.winner_id, 0)
        self.assertFalse(res.alliance_formed, "No alliance if Loot player wins the trick")
        
    
    def test_kraken_leads(self):
        """
        Scenario: P0 plays Kraken. P1 plays Green 5. P2 plays Green 14.
        Result: Trick destroyed.
        Next Lead: P2 (Green 14 would have beaten Green 5).
        """
        kraken = self.get_action_id(CardType.KRAKEN)
        g5 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 5)
        g14 = self.get_action_id(CardType.NUMBER, Suit.GREEN, 14)
        
        res = self.physics.resolve_trick([(0, kraken), (1, g5), (2, g14)])
        
        self.assertTrue(res.destroyed)
        self.assertEqual(res.next_lead_id, 2, "P2 should lead next because G14 > G5")
        
    
    def test_battle_of_escapes(self):
        """
        Scenario: P0 plays Tigress (as Escape). P1 plays Loot. P2 plays Escape.
        Result: P0 wins (First played escape).
        Alliance: TRUE. 
        Why? P1 played Loot, P0 won. They are different players. 
        The rule exception ('No alliance formed') only applies if the Loot player 
        themselves wins the trick.
        """
        tig_esc = self.get_action_id(CardType.TIGRESS, as_escape=True)
        loot = self.get_action_id(CardType.LOOT)
        esc = self.get_action_id(CardType.ESCAPE)
        
        res = self.physics.resolve_trick([(0, tig_esc), (1, loot), (2, esc)])
        
        self.assertEqual(res.winner_id, 0)
        # CORRECTED ASSERTION:
        self.assertTrue(res.alliance_formed, "Alliance should form: Loot player (P1) != Winner (P0)")

    def test_loot_leads_and_wins_no_alliance(self):
        """
        Rule: If leading with Loot and everyone else plays Escape, 
        Loot player wins, but NO Alliance is formed.
        """
        loot = self.get_action_id(CardType.LOOT)
        esc = self.get_action_id(CardType.ESCAPE)
        tig_esc = self.get_action_id(CardType.TIGRESS, as_escape=True)
        
        # P0 leads with Loot. P1 and P2 play escapes.
        res = self.physics.resolve_trick([(0, loot), (1, esc), (2, tig_esc)])
        
        self.assertEqual(res.winner_id, 0) # P0 wins (First played)
        self.assertFalse(res.alliance_formed, "No alliance if Loot player wins")
if __name__ == '__main__':
    unittest.main()
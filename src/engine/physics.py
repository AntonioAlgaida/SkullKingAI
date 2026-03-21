# src/engine/physics.py

from enum import IntEnum, auto
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

class Suit(IntEnum):
    GREEN = 0   # Parrot
    PURPLE = 1  # Map
    YELLOW = 2  # Chest
    BLACK = 3   # Jolly Roger (Trump)
    SPECIAL = 4 # Pirates, etc.

class CardType(IntEnum):
    NUMBER = auto()
    PIRATE = auto()
    TIGRESS = auto()    # Action logic handles the split
    SKULL_KING = auto()
    MERMAID = auto()
    ESCAPE = auto()
    LOOT = auto()
    KRAKEN = auto()
    WHITE_WHALE = auto()
    

class Card:
    __slots__ = ['card_id', 'suit', 'value', 'card_type'] # Memory efficiency for millions of sims

    def __init__(self, card_id: int, suit: Suit, value: int, card_type: CardType):
        self.card_id = card_id     # 0 to 73
        self.suit = suit           # Suit enum
        self.value = value         # 1-14 for numbers, 0 for special
        self.card_type = card_type # CardType enum

    def __repr__(self):
        return f"Card({self.suit.name}, {self.value}, {self.card_type.name})"

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


@dataclass
class TrickResult:
    winner_id: Optional[int]    # None if destroyed
    next_lead_id: int           # Crucial for Kraken logic
    bonus_points: int           # Standard capture bonuses
    destroyed: bool             # For UI/Logging
    captured_cards: List[int]   # List of card_ids taken (for scoring)
    alliance_formed: bool       # For Loot card scoring in Env
    
class GamePhysics:
    """The static engine that defines the universe of Skull King."""
    
    def __init__(self):
        self.deck = self._initialize_deck()
        # Mapping Action ID (0-74) to (Card Object, PlayMode)
        self.actions = self._initialize_actions()

    def _initialize_deck(self) -> Dict[int, Card]:
        deck = {}
        c_id = 0
        
        # 1. Suited Cards (0-55)
        # Order: Green (0-13), Purple (14-27), Yellow (28-41), Black (42-55)
        for suit in [Suit.GREEN, Suit.PURPLE, Suit.YELLOW, Suit.BLACK]:
            for val in range(1, 15):
                deck[c_id] = Card(c_id, suit, val, CardType.NUMBER)
                c_id += 1
        
        # 2. Special Cards
        # Pirates (5 cards)
        for _ in range(5):
            deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.PIRATE); c_id += 1
        
        # Tigress (1 card)
        deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.TIGRESS); c_id += 1
        
        # Skull King (1 card)
        deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.SKULL_KING); c_id += 1
        
        # Mermaids (2 cards)
        for _ in range(2):
            deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.MERMAID); c_id += 1
            
        # Escapes (5 cards)
        for _ in range(5):
            deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.ESCAPE); c_id += 1
            
        # Loot (2 cards)
        for _ in range(2):
            deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.LOOT); c_id += 1
            
        # Kraken (1 card)
        deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.KRAKEN); c_id += 1
        
        # White Whale (1 card)
        deck[c_id] = Card(c_id, Suit.SPECIAL, 0, CardType.WHITE_WHALE); c_id += 1
        
        # Total c_id should be 74
        assert c_id == 74, f"Deck init failed. Count: {c_id}"
        
        return deck

    def _initialize_actions(self):
        """
        Maps the 75 discrete RL actions to (Card, PlayMode).
        Actions 0-73: Play the specific card ID.
        Action 74: Play the Tigress (ID 61) as an ESCAPE.
        """
        actions = {}
        
        # Find the Tigress ID
        tigress_id = next(k for k, v in self.deck.items() if v.card_type == CardType.TIGRESS)

        for i in range(74):
            card = self.deck[i]
            # Default behavior for Tigress card ID is Pirate
            as_type = CardType.PIRATE if card.card_type == CardType.TIGRESS else card.card_type
            actions[i] = {"card": card, "as_type": as_type}
        
        # Action 74: Tigress played as Escape
        actions[74] = {"card": self.deck[tigress_id], "as_type": CardType.ESCAPE}
        
        return actions

    def resolve_trick(self, played_actions: List[Tuple[int, int]]) -> TrickResult:
        trick_data = []
        captured_ids = []
        for p_id, a_id in played_actions:
            config = self.actions[a_id]
            trick_data.append({
                'player_id': p_id,
                'card': config['card'],
                'effective_type': config['as_type']
            })
            captured_ids.append(config['card'].card_id)

        # --- 1. DETECT DISRUPTORS (Order Matters) ---
        kraken_idx = -1
        whale_idx = -1
        
        for i, d in enumerate(trick_data):
            if d['effective_type'] == CardType.KRAKEN:
                kraken_idx = i
            elif d['effective_type'] == CardType.WHITE_WHALE:
                whale_idx = i
        
        # Logic: "When both kraken and whale are played, the second one played determines the action"
        active_disruptor = None
        if kraken_idx > -1 and whale_idx > -1:
            active_disruptor = CardType.KRAKEN if kraken_idx > whale_idx else CardType.WHITE_WHALE
        elif kraken_idx > -1:
            active_disruptor = CardType.KRAKEN
        elif whale_idx > -1:
            active_disruptor = CardType.WHITE_WHALE

        # --- 2. HANDLE KRAKEN (Trick Destroyed) ---
        if active_disruptor == CardType.KRAKEN:
            # We must simulate the trick WITHOUT the Kraken to find who leads next.
            # We filter out the Kraken card from the calculation logic but keep the list structure.
            # Simplification: We can reuse _calculate_winner on the list, but treating Kraken as an Escape (always loses)
            # or simply looking at the logic "who would have won".
            
            # Temporary replace Kraken with Escape for calculation of "Next Lead"
            hypothetical_data = []
            for d in trick_data:
                new_d = d.copy()
                if new_d['effective_type'] == CardType.KRAKEN:
                    new_d['effective_type'] = CardType.ESCAPE # Loses to everything
                hypothetical_data.append(new_d)
                
            hypothetical_winner, _, _ = self._calculate_standard_winner(hypothetical_data)
            
            return TrickResult(
                winner_id=None,
                next_lead_id=hypothetical_winner['player_id'],
                bonus_points=0,
                destroyed=True,
                captured_cards=[], # Cards set aside
                alliance_formed=False
            )

        # --- 3. HANDLE WHITE WHALE ---
        if active_disruptor == CardType.WHITE_WHALE:
            winner_data = self._resolve_white_whale_logic(trick_data)

            # EDGE CASE FIX: If winner is None (all specials), trick is destroyed
            if winner_data is None:
                return TrickResult(
                    winner_id=None,
                    next_lead_id=trick_data[0]['player_id'],
                    bonus_points=0,
                    destroyed=True,
                    captured_cards=[],
                    alliance_formed=False
                )

            # Under White Whale, RPS bonuses don't apply but 14-capture bonuses still do.
            # Any numbered 14 in the trick earns its bonus for the winner.
            whale_bonus = 0
            if winner_data['effective_type'] not in [CardType.ESCAPE, CardType.LOOT]:
                for d in trick_data:
                    if d['card'].value == 14:
                        whale_bonus += 20 if d['card'].suit == Suit.BLACK else 10

            return TrickResult(
                winner_id=winner_data['player_id'],
                next_lead_id=winner_data['player_id'],
                bonus_points=whale_bonus,
                destroyed=False,
                captured_cards=captured_ids,
                alliance_formed=False
            )
            
        # --- 4. STANDARD RESOLUTION ---
        winner_data, bonus, alliance = self._calculate_standard_winner(trick_data)
        
        return TrickResult(
            winner_id=winner_data['player_id'],
            next_lead_id=winner_data['player_id'],
            bonus_points=bonus,
            destroyed=False,
            captured_cards=captured_ids,
            alliance_formed=alliance
        )

    def _calculate_standard_winner(self, trick_data: List[dict]) -> Tuple[dict, int, bool]:
        """
        Determines the winner based on game rules.
        Returns (Winner Dictionary, Bonus Points, Alliance Formed)
        """
        # A. Determine Lead Suit
        lead_suit = None
        is_character_lead = trick_data[0]['effective_type'] in [CardType.PIRATE, CardType.SKULL_KING, CardType.MERMAID]
        
        if not is_character_lead:
            for d in trick_data:
                # First card that isn't Escape/Loot/Special sets the suit
                if d['effective_type'] not in [CardType.ESCAPE, CardType.LOOT] and d['card'].suit != Suit.SPECIAL:
                    lead_suit = d['card'].suit
                    break
        
        # B. Determine Winner (RPS or Suited)
        current_winner = None
        bonus = 0
        
        types = [d['effective_type'] for d in trick_data]
        
        # --- PATH 1: Rock-Paper-Scissors Logic ---
        if CardType.SKULL_KING in types and CardType.MERMAID in types:
            current_winner = next(d for d in trick_data if d['effective_type'] == CardType.MERMAID)
            bonus = 40 # Mermaid captures King

        elif CardType.SKULL_KING in types:
            current_winner = next(d for d in trick_data if d['effective_type'] == CardType.SKULL_KING)
            pirates = types.count(CardType.PIRATE)
            bonus = pirates * 30 # King captures Pirates

        elif CardType.PIRATE in types:
            current_winner = next(d for d in trick_data if d['effective_type'] == CardType.PIRATE)
            mermaids = types.count(CardType.MERMAID)
            bonus = mermaids * 20 # Pirate captures Mermaids

        elif CardType.MERMAID in types:
            current_winner = next(d for d in trick_data if d['effective_type'] == CardType.MERMAID)
            bonus = 0

        # --- PATH 2: Suited Logic (Only if RPS didn't produce a winner) ---
        if current_winner is None:
            current_winner = trick_data[0]
            
            for challenger in trick_data[1:]:
                # Escapes/Loots lose to everything non-escape
                if challenger['effective_type'] in [CardType.ESCAPE, CardType.LOOT]:
                    continue
                
                # If current winner is Escape/Loot, almost anything beats it
                if current_winner['effective_type'] in [CardType.ESCAPE, CardType.LOOT]:
                    current_winner = challenger
                    continue
                
                # Standard Suit Comparison
                w_card = current_winner['card']
                c_card = challenger['card']
                
                is_winner = False
                
                # Trump (Black) Logic
                if c_card.suit == Suit.BLACK:
                    if w_card.suit != Suit.BLACK:
                        is_winner = True
                    elif c_card.value > w_card.value:
                        is_winner = True
                
                # Lead Suit Logic
                elif lead_suit is not None and c_card.suit == lead_suit:
                    if w_card.suit != Suit.BLACK:
                        if w_card.suit != lead_suit:
                            is_winner = True
                        elif c_card.value > w_card.value:
                            is_winner = True
                
                if is_winner:
                    current_winner = challenger

        # C. Calculate "14" Bonuses 
        # (This applies regardless of whether the winner was RPS or Suited)
        # Rule: "Cards numbered 14 captured in a trick earn bonus points... regardless"
        # Exception: Escapes/Loots cannot capture points (unless maybe ALL cards were escapes, but rules imply 0 points)
        if current_winner['effective_type'] not in [CardType.ESCAPE, CardType.LOOT]:
            for d in trick_data:
                if d['card'].value == 14:
                    # 20 for Black 14, 10 for others
                    bonus += 20 if d['card'].suit == Suit.BLACK else 10

        # D. Check for Loot Alliance
        alliance_formed = False
        loot_players = [d['player_id'] for d in trick_data if d['effective_type'] == CardType.LOOT]
        
        # Alliance condition: Loot was played AND Loot player did NOT win the trick
        if loot_players and current_winner['player_id'] not in loot_players:
            alliance_formed = True
        
        return current_winner, bonus, alliance_formed

    def _resolve_white_whale_logic(self, trick_data: List[dict]) -> Optional[dict]:
        """
        Returns: Winner dict, or None if trick should be destroyed (all specials).
        """
        current_winner = trick_data[0]
        max_val = -1
        has_numbered_card = False
        
        for d in trick_data:
            val = d['card'].value
            if val > 0:
                has_numbered_card = True
            
            if val > max_val:
                max_val = val
                current_winner = d
                
        if not has_numbered_card:
            return None # Trigger destruction
            
        return current_winner
# src/utils/translators.py

from typing import List, Dict, Any
from src.engine.physics import GamePhysics, CardType, Suit
from src.utils.prompt_loader import PromptLoader

class SemanticTranslator:
    """
    Translates raw numerical game states into semantic text for the LLM.
    Acts as the 'Strategic HUD' so the AI doesn't have to do raw math.
    """
    def __init__(self, physics: GamePhysics):
        self.physics = physics

    def translate_card(self, action_id: int) -> str:
        """Converts an Action ID (0-74) into a human-readable string."""
        if action_id == 74:
            return "[Tigress (Played as Escape)]"
            
        card = self.physics.deck[action_id]
        
        if card.card_type == CardType.NUMBER:
            suit_name = card.suit.name.title() # "Green", "Black", etc.
            return f"[{suit_name} {card.value}]"
        else:
            name = card.card_type.name.replace("_", " ").title()
            if card.card_type == CardType.TIGRESS:
                name = "Tigress (Played as Pirate)"
            return f"[{name}]"

    def get_hunger_matrix(self, state_dict: Dict[str, Any]) -> str:
        """
        Calculates the psychological state of each opponent.
        This is critical for Co-Evolution and Trap Detection.
        """
        bids = state_dict["bids"]
        wins = state_dict["tricks_won"]
        round_num = state_dict["round_num"]
        my_id = state_dict["current_player_id"]
        
        # Calculate how many tricks are left in the entire round
        completed_tricks = sum(wins)
        tricks_left = round_num - completed_tricks
        
        hunger_lines =[]
        for pid in range(len(bids)):
            if pid == my_id: continue
            
            bid = bids[pid]
            won = wins[pid]
            
            # If still in Bidding Phase, bids might be -1
            if bid == -1:
                hunger_lines.append(f"- Player {pid}: BID HIDDEN")
                continue
                
            delta = bid - won
            
            if delta == 0:
                status = "FULL (Danger! They met their bid and will aggressively try to lose)"
            elif delta < 0:
                status = "OVERBOARD (They failed their bid and will desperately try to lose)"
            elif delta == tricks_left:
                status = "STARVING (They MUST win every remaining trick. You are safe from them)"
            elif 0 < delta < tricks_left:
                status = "HUNGRY (They want to win, but can be selective)"
            else:
                status = "DOOMED (Mathematically impossible to meet bid. Wildcard)"
                
            hunger_lines.append(f"- Player {pid}: {status} (Bid: {bid}, Won: {won})")
            
        return "\n".join(hunger_lines)

    def summarize_graveyard(self, graveyard_ids: List[int]) -> str:
        """Counts critical unplayed cards so the LLM knows the statistical risks."""
        played_cards = [self.physics.deck[c_id] for c_id in graveyard_ids if c_id != 74]
        
        sk_played = any(c.card_type == CardType.SKULL_KING for c in played_cards)
        kraken_played = any(c.card_type == CardType.KRAKEN for c in played_cards)
        whale_played = any(c.card_type == CardType.WHITE_WHALE for c in played_cards)
        
        mermaids_played = sum(1 for c in played_cards if c.card_type == CardType.MERMAID)
        pirates_played = sum(1 for c in played_cards if c.card_type == CardType.PIRATE)
        escapes_played = sum(1 for c in played_cards if c.card_type == CardType.ESCAPE)
        
        black_trumps_played = sum(1 for c in played_cards if c.card_type == CardType.NUMBER and c.suit == Suit.BLACK)
        
        lines =[
            f"- Skull King: {'PLAYED' if sk_played else 'UNPLAYED'}",
            f"- Kraken: {'PLAYED' if kraken_played else 'UNPLAYED'}",
            f"- White Whale: {'PLAYED' if whale_played else 'UNPLAYED'}",
            f"- Mermaids Played: {mermaids_played}/2",
            f"- Pirates Played: {pirates_played}/5",
            f"- Escapes/Loot Played: {escapes_played}/7",
            f"- Jolly Rogers (Black Trumps) Played: {black_trumps_played}/14"
        ]
        return "\n".join(lines)

    def categorize_hand_toxicity(self, hand_ids: List[int]) -> str:
        """
        Groups cards into semantic buckets for the Zero-Bid meta.
        Helps the RAG system match similar hands without needing exact numeric matches.
        """
        toxic = 0      # Must win (King, Pirate, Mermaid, Black 10-14)
        volatile = 0   # Might win (Colors 10-14, Black 1-9)
        safe = 0       # Colors 1-9
        escapes = 0    # Escapes, Loot
        
        for cid in hand_ids:
            if cid == 74:
                escapes += 1
                continue
                
            c = self.physics.deck[cid]
            if c.card_type in[CardType.ESCAPE, CardType.LOOT]:
                escapes += 1
            elif c.card_type in[CardType.SKULL_KING, CardType.PIRATE, CardType.MERMAID]:
                toxic += 1
            elif c.card_type == CardType.NUMBER:
                if c.suit == Suit.BLACK:
                    if c.value >= 10: toxic += 1
                    else: volatile += 1
                else:
                    if c.value >= 10: volatile += 1
                    else: safe += 1
                    
        return f"Toxicity Profile: {toxic} Toxic, {volatile} Volatile, {safe} Safe, {escapes} Escapes."

    def get_system_prompt(self, phase: str, persona: str) -> str:
        """Returns the static rule bundle + persona. Intended for the system role.
        This content is identical across all calls of the same phase/persona and
        benefits from vLLM prefix caching."""
        if phase == "BIDDING":
            return PromptLoader.get_bidding_bundle(persona)
        else:
            return PromptLoader.get_playing_bundle(persona)

    def build_user_context(self, state_dict: Dict[str, Any]) -> str:
        """Returns only the dynamic game state. Intended for the user role.
        Changes every call, so it is intentionally kept short."""
        phase     = state_dict["phase"]
        round_num = state_dict["round_num"]

        hand_text = ", ".join([self.translate_card(c) for c in state_dict["my_hand"]])

        if phase == "BIDDING":
            legal_actions_text = "\n".join(
                f"ID {aid}: {self.translate_card(aid)}" for aid in state_dict["legal_actions"]
            )
            return f"""[CURRENT STATE]
Phase: BIDDING (Round {round_num})
Num of players: {len(state_dict['bids'])}
My Hand: {hand_text}
{self.categorize_hand_toxicity(state_dict["my_hand"])}
I am player ID: {state_dict["current_player_id"]}
NOTE: Bidding is simultaneous — opponent bids are not visible.

[TASK]
Analyze your cards and the scoring rules. Output your CoT reasoning, then output [ACTION]: <bid_number>."""

        else:  # PLAYING
            current_trick = state_dict.get("current_trick", [])
            is_leading    = len(current_trick) == 0
            trick_text    = ", ".join([f"P{pid} played {self.translate_card(aid)}" for pid, aid in current_trick])
            if not trick_text:
                trick_text = "You are leading the trick."

            winner_annotation = self._describe_current_winner(current_trick)

            my_id       = state_dict["current_player_id"]
            num_players = len(state_dict["bids"])
            my_bid      = state_dict["bids"][my_id]
            my_won      = state_dict["tricks_won"][my_id]
            tricks_done = sum(state_dict["tricks_won"])
            tricks_left = round_num - tricks_done

            need = my_bid - my_won
            if need > 0:
                bid_status = f"I need {need} more win(s) across the {tricks_left} trick(s) left (including this one)."
            elif need == 0:
                bid_status = f"⚠ BID MET — I must LOSE this trick and all {tricks_left - 1} remaining trick(s)."
            else:
                bid_status = f"⚠ OVERBID by {-need} — bid already failed. Minimise further damage."

            legal_actions_text = self._annotate_legal_actions(
                state_dict["legal_actions"], current_trick, my_id, num_players
            )

            return f"""[CURRENT STATE]
Phase: PLAYING (Round {round_num})
Num of players: {num_players}
My Target Bid: {my_bid} | My Tricks Won: {my_won} | Tricks Left This Round: {tricks_left}
{bid_status}
LEAD STATUS: {'YOU ARE LEADING THIS TRICK' if is_leading else 'AN OPPONENT IS LEADING'}

[OPPONENT THREAT LEVEL]
{self.get_hunger_matrix(state_dict)}

[TRICK HISTORY]
Current Trick: {trick_text}{winner_annotation}
[GRAVEYARD HUD]
{self.summarize_graveyard(state_dict["graveyard"])}

My Hand: {hand_text}

[YOUR LEGAL MOVES]
{legal_actions_text}

[TASK]
Analyze the threat level and recall the card hierarchy. Write your CoT reasoning, then output [ACTION]: <ID>."""

    def _annotate_legal_actions(
        self, legal_actions: list, current_trick: list, my_id: int, num_players: int
    ) -> str:
        """
        Returns the [YOUR LEGAL MOVES] block with a deterministic outcome label
        appended to each action, computed by the physics engine.

        When leading (empty trick): no annotation — every card trivially leads.
        Mid-trick: annotates whether the card beats the current leader or not.
        Last to play: outcome is fully certain — WINS or LOSES with the winner named.
        """
        if not current_trick:
            # Leading — just list the cards, no outcome to annotate
            return "\n".join(f"ID {aid}: {self.translate_card(aid)}" for aid in legal_actions)

        is_last = len(current_trick) == num_players - 1
        lines = []
        for aid in legal_actions:
            card_text = self.translate_card(aid)
            try:
                result = self.physics.resolve_trick(current_trick + [(my_id, aid)])
            except Exception:
                lines.append(f"ID {aid}: {card_text}")
                continue

            if result.destroyed:
                label = "→ DESTROYS TRICK (Kraken — no winner, you lead next)"
            elif is_last:
                if result.winner_id == my_id:
                    label = "→ WINS ✓ (outcome certain — you take this trick)"
                else:
                    label = f"→ LOSES ✗ (P{result.winner_id} wins — outcome certain)"
            else:
                if result.winner_id == my_id:
                    label = "→ LEADS (you beat all cards played so far — others still to play)"
                else:
                    label = "→ LOSES to current leader (playing this card will not win the trick as it stands)"

            lines.append(f"ID {aid}: {card_text}  {label}")
        return "\n".join(lines)

    def _describe_current_winner(self, current_trick: list) -> str:
        """
        Calls the physics engine on the partial trick to determine who is currently
        winning and returns a one-line annotation explaining what can beat them.
        Empty string when the agent is leading (no cards yet played).
        """
        if not current_trick:
            return ""

        result = self.physics.resolve_trick(current_trick)

        if result.destroyed:
            return "\n⚠ KRAKEN IS IN PLAY — this trick will be DESTROYED. No one wins it."

        winner_id = result.winner_id
        if winner_id is None:
            return ""

        # Find the winning card action id
        winning_aid = next((aid for pid, aid in current_trick if pid == winner_id), None)
        if winning_aid is None:
            return ""

        winning_text = self.translate_card(winning_aid)
        winning_type = self.physics.actions[winning_aid]["as_type"]

        if winning_type == CardType.SKULL_KING:
            beats_hint = "Only a [Mermaid] can beat it."
        elif winning_type == CardType.PIRATE:
            beats_hint = "Only [Skull King] beats it. A second Pirate CANNOT win — first Pirate played wins."
        elif winning_type == CardType.MERMAID:
            beats_hint = "Any [Pirate] beats it."
        elif winning_type == CardType.NUMBER:
            card = self.physics.deck[winning_aid]
            if card.suit == Suit.BLACK:
                beats_hint = "Any Special ([Pirate], [Skull King], [Mermaid]) beats it, or a higher [Black] card."
            else:
                suit = card.suit.name.title()
                beats_hint = (
                    f"Any [Black] card or Special beats it. "
                    f"A higher [{suit}] card also beats it if that suit was led."
                )
        else:
            beats_hint = "Check card hierarchy."

        return f"\n⚠ CURRENT WINNER: P{winner_id} with {winning_text}. {beats_hint}"

    def build_llm_prompt_context(self, state_dict: Dict[str, Any], persona: str) -> str:
        """Legacy combined prompt (system bundle + user context in one string).
        Used by reflector/pruner. New code should call get_system_prompt + build_user_context."""
        system = self.get_system_prompt(state_dict["phase"], persona)
        user   = self.build_user_context(state_dict)
        return f"{system}\n\n{user}"
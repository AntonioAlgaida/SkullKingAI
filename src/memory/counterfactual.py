# src/memory/counterfactual.py
#
# Counterfactual simulation for the sleep cycle.
#
# At the critical trick where a player's failure became inevitable, we replace
# their card with each legal alternative and call physics.resolve_trick().
# The LLM receives engine-verified evidence ("if you had played X, you would have
# LOST the trick") rather than having to guess.
#
# Ceteris-paribus assumption: all other players' cards are held fixed.
# This is a first-order approximation; when the player leads, opponents'
# follow-suit choices would differ — we flag this in the evidence text.

from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.engine.physics import GamePhysics, CardType, Suit


@dataclass
class AlternativeOutcome:
    action_id: int
    card_name: str
    would_win: bool       # True if the player would win the trick
    bonus_pts: int        # Bonus points if player wins (0 if they lose)
    destroyed: bool       # True if trick is destroyed (Kraken / all-specials Whale)


class CounterfactualSimulator:
    """
    Engine-backed counterfactual analysis for a single trick.

    Usage:
        sim = CounterfactualSimulator(physics)
        alts = sim.simulate_alternatives(
            player_id, actual_action_id, full_trick_actions, legal_alternatives
        )
        evidence = sim.format_evidence(
            player_id, actual_action_id, won_trick, alts, translator
        )
    """

    FITNESS_WEIGHT = 0.1   # How much fitness shifts effective retrieval distance

    def __init__(self, physics: GamePhysics):
        self.physics = physics

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def simulate_alternatives(
        self,
        player_id: int,
        actual_action_id: int,
        full_trick_actions: List[Tuple[int, int]],   # all players' (pid, aid) in play order
        legal_alternatives: List[int],               # action_ids the player COULD have played
    ) -> List[AlternativeOutcome]:
        """
        For each legal alternative (excluding the card actually played),
        simulate the trick and return the outcome.
        """
        player_pos = next(
            (i for i, (pid, _) in enumerate(full_trick_actions) if pid == player_id),
            None,
        )
        if player_pos is None:
            return []

        results = []
        for alt_aid in legal_alternatives:
            if alt_aid == actual_action_id:
                continue
            alt_trick = list(full_trick_actions)
            alt_trick[player_pos] = (player_id, alt_aid)
            try:
                tr = self.physics.resolve_trick(alt_trick)
                results.append(AlternativeOutcome(
                    action_id=alt_aid,
                    card_name=self._card_name(alt_aid),
                    would_win=(tr.winner_id == player_id),
                    bonus_pts=tr.bonus_points if tr.winner_id == player_id else 0,
                    destroyed=tr.destroyed,
                ))
            except Exception:
                continue
        return results

    def get_legal_alternatives(
        self,
        player_id: int,
        actual_action_id: int,
        full_trick_actions: List[Tuple[int, int]],
        remaining_hand_ids: List[int],           # all cards player still had (including actual)
    ) -> List[int]:
        """
        Returns only the cards from remaining_hand_ids that were legal at
        the player's position in the trick (respects follow-suit rule).
        """
        player_pos = next(
            (i for i, (pid, _) in enumerate(full_trick_actions) if pid == player_id),
            None,
        )
        if player_pos is None:
            return list(remaining_hand_ids)

        prior_plays = full_trick_actions[:player_pos]
        lead_suit = self._lead_suit(prior_plays)

        if lead_suit is None:
            return list(remaining_hand_ids)   # No suit constraint

        # Does the player have any card of the lead suit?
        has_lead = any(
            aid != 74 and self.physics.deck[aid].suit == lead_suit
            for aid in remaining_hand_ids
        )
        if not has_lead:
            return list(remaining_hand_ids)   # No card of that suit → play anything

        # Must play lead suit OR a special
        legal = []
        for aid in remaining_hand_ids:
            if aid == 74:                           # Tigress as Escape (special)
                legal.append(aid)
                continue
            card = self.physics.deck[aid]
            if card.suit == Suit.SPECIAL or card.suit == lead_suit:
                legal.append(aid)
        return legal

    def reconstruct_hand_at_trick(
        self,
        player_id: int,
        starting_hand_ids: List[int],
        tricks_before: List[dict],    # tricks[i]["actions"] for i < critical_trick_index
    ) -> List[int]:
        """
        Reconstructs the player's remaining hand at the start of the critical trick
        by subtracting all cards played in earlier tricks of this round.
        """
        played = set()
        for trick in tricks_before:
            for pid, aid in trick.get("actions", []):
                if pid == player_id:
                    # Tigress action 74 → underlying card is 61
                    played.add(61 if aid == 74 else aid)

        # starting_hand_ids contains underlying card IDs (action_ids for non-tigress cards)
        remaining = []
        for aid in starting_hand_ids:
            underlying = 61 if aid == 74 else aid
            if underlying not in played:
                remaining.append(aid)
        return remaining

    def format_evidence(
        self,
        player_id: int,
        actual_action_id: int,
        actual_won_trick: bool,
        alternatives: List[AlternativeOutcome],
        translator,
    ) -> str:
        """Formats counterfactual evidence as a prompt section for the LLM."""
        actual_card = translator.translate_card(actual_action_id)
        outcome_str = "WON trick (bad for over-bid)" if actual_won_trick else "LOST trick (bad for under-bid)"

        lines = [
            "═" * 50,
            "ENGINE-VERIFIED COUNTERFACTUAL (what else P{} could have played):".format(player_id),
            f"  Actual play:  {actual_card}  →  {outcome_str}",
            "  Alternatives from the same hand position:",
        ]

        for alt in alternatives:
            if alt.destroyed:
                result = "→ trick DESTROYED (Kraken / all-specials)"
            elif alt.would_win:
                bonus = f"  (+{alt.bonus_pts} bonus pts)" if alt.bonus_pts else ""
                result = f"→ WOULD WIN the trick{bonus}"
            else:
                result = "→ would LOSE the trick  ✓ safer option"
            lines.append(f"    {alt.card_name:<30} {result}")

        if not alternatives:
            lines.append("    (No legal alternatives — forced play.)")

        lines += [
            "  NOTE: Other players' cards held fixed (ceteris-paribus).",
            "═" * 50,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _lead_suit(self, prior_plays: List[Tuple[int, int]]) -> Optional[Suit]:
        """Determine the lead suit from plays that happened before this player."""
        if not prior_plays:
            return None
        first_pid, first_aid = prior_plays[0]
        first_type = self.physics.actions[first_aid]["as_type"]
        if first_type in (CardType.PIRATE, CardType.SKULL_KING, CardType.MERMAID):
            return None   # Character lead — no suit constraint
        for _, aid in prior_plays:
            cfg = self.physics.actions[aid]
            if cfg["as_type"] not in (CardType.ESCAPE, CardType.LOOT) and cfg["card"].suit != Suit.SPECIAL:
                return cfg["card"].suit
        return None

    def _card_name(self, action_id: int) -> str:
        if action_id == 74:
            return "[Tigress as Escape]"
        card = self.physics.deck[action_id]
        if card.card_type == CardType.NUMBER:
            return f"[{card.suit.name.title()} {card.value}]"
        name = card.card_type.name.replace("_", " ").title()
        return f"[{name}]"

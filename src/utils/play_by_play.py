# src/utils/play_by_play.py
#
# Writes a human-readable, incrementally-flushed play-by-play log per game.
# Shows card names (not IDs), per-player hands, bids, plays, and round summaries.

from typing import Dict, List


class PlayByPlay:
    """Streams a readable play-by-play file that is flushed after every event."""

    def __init__(self, game_id: int, path: str, translator):
        self._f = open(path, "w", buffering=1)   # line-buffered → flush on every \n
        self._tr = translator
        self._game_id = game_id
        self._trick_num = 0
        self._w(f"{'=' * 56}")
        self._w(f"  GAME {game_id}")
        self._w(f"{'=' * 56}")

    # ------------------------------------------------------------------ internal
    def _w(self, line: str = "") -> None:
        self._f.write(line + "\n")

    def _hand_str(self, hand: List[int]) -> str:
        return "  |  ".join(self._tr.translate_card(c) for c in sorted(hand))

    # ------------------------------------------------------------------ public API
    def round_start(self, round_num: int, hands: Dict[int, List[int]], personas: Dict[int, str]) -> None:
        self._trick_num = 0
        self._w()
        self._w(f"{'─' * 56}")
        self._w(f"  ROUND {round_num}  ({round_num} cards dealt)")
        self._w(f"{'─' * 56}")
        for pid, hand in hands.items():
            persona = personas.get(pid, "?")
            self._w(f"  P{pid} ({persona:12s}): {self._hand_str(hand)}")
        self._w()
        self._w("  BIDS")
        self._w(f"  {'─' * 20}")

    def bid(self, player_id: int, persona: str, bid_amount: int) -> None:
        self._w(f"    P{player_id} ({persona}): bid {bid_amount}")

    def trick_start(self, player_id: int) -> None:
        self._trick_num += 1
        self._w()
        self._w(f"  Trick {self._trick_num}  (led by P{player_id})")
        self._w(f"  {'─' * 20}")

    def play(self, player_id: int, persona: str, action_id: int) -> None:
        card = self._tr.translate_card(action_id)
        self._w(f"    P{player_id} ({persona}): {card}")

    def trick_end(self, winner_id: int, bonus: int) -> None:
        bonus_str = f"  (+{bonus} bonus pts)" if bonus else ""
        self._w(f"  → P{winner_id} wins trick {self._trick_num}{bonus_str}")

    def trick_destroyed(self, next_leader_id: int) -> None:
        """Called when Kraken (or White Whale with all specials) destroys the trick."""
        self._w(f"  → Trick {self._trick_num} DESTROYED (Kraken). P{next_leader_id} leads next.")

    def round_end(self, round_num: int, bids: List[int], won: List[int],
                  round_scores: List[int], total_scores: List[int], personas: Dict[int, str]) -> None:
        self._w()
        self._w(f"  ROUND {round_num} RESULT")
        self._w(f"  {'─' * 40}")
        header = "  " + "".join(f"  P{i}({personas.get(i,'?')[:3]})" for i in range(len(bids)))
        self._w(header)
        self._w("  Bid:   " + "".join(f"  {b:5}" for b in bids))
        self._w("  Won:   " + "".join(f"  {w:5}" for w in won))
        self._w("  Round: " + "".join(f"  {s:+5}" for s in round_scores))
        self._w("  Total: " + "".join(f"  {s:+5}" for s in total_scores))

    def game_end(self, final_scores: List[int], personas: Dict[int, str]) -> None:
        self._w()
        self._w(f"{'=' * 56}")
        self._w(f"  GAME {self._game_id} OVER")
        ranked = sorted(enumerate(final_scores), key=lambda x: -x[1])
        for rank, (pid, score) in enumerate(ranked, 1):
            self._w(f"  #{rank}  P{pid} ({personas.get(pid, '?')}): {score:+d}")
        self._w(f"{'=' * 56}")
        self._f.close()

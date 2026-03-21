# src/memory/elo_tracker.py

import json
import os
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

K_FACTOR = 32
DEFAULT_ELO = 1000


class EloTracker:
    """
    Tracks ELO ratings per persona across games.

    Multi-player ELO: for each game, every player competes pairwise against
    all others. Deltas are averaged when multiple players share the same persona.

    Stored in data/elo_ratings.json relative to the project root.
    """

    def __init__(self, persistence_path: str = "data/elo_ratings.json"):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path = os.path.join(base_dir, persistence_path)
        self.ratings: Dict = self._load()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_elo(self, persona: str) -> float:
        return self.ratings.get(persona, {}).get("elo", float(DEFAULT_ELO))

    def update_from_game(self, player_results: List[Tuple[str, int]]):
        """
        Update ELO after one completed game.

        player_results: list of (persona, final_score) for every player seat.
        Example: [("forced_zero", 240), ("rational", 180), ("rational", 200), ("rational", 150)]

        Algorithm:
          For each player i, compute the pairwise expected vs. actual score against
          every other player j, then sum the K-weighted deltas.
          When multiple players share a persona, their individual deltas are averaged
          into a single update so the persona ELO moves by the mean performance.
        """
        n = len(player_results)
        if n < 2:
            return

        # Step 1 — compute individual delta for each seat
        per_seat_deltas: List[Tuple[str, float]] = []
        for i, (persona_i, score_i) in enumerate(player_results):
            elo_i = self.get_elo(persona_i)
            delta = 0.0
            for j, (persona_j, score_j) in enumerate(player_results):
                if i == j:
                    continue
                elo_j = self.get_elo(persona_j)
                expected = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))
                if score_i > score_j:
                    actual = 1.0
                elif score_i < score_j:
                    actual = 0.0
                else:
                    actual = 0.5
                delta += K_FACTOR * (actual - expected)
            per_seat_deltas.append((persona_i, delta))

        # Step 2 — average deltas per persona
        persona_total: Dict[str, float] = {}
        persona_count: Dict[str, int] = {}
        for persona, delta in per_seat_deltas:
            persona_total[persona] = persona_total.get(persona, 0.0) + delta
            persona_count[persona] = persona_count.get(persona, 0) + 1

        # Step 3 — apply updates and persist
        for persona, total_delta in persona_total.items():
            avg_delta = total_delta / persona_count[persona]
            old_elo = self.get_elo(persona)
            new_elo = old_elo + avg_delta

            if persona not in self.ratings:
                self.ratings[persona] = {"elo": float(DEFAULT_ELO), "games": 0, "history": []}

            self.ratings[persona]["elo"] = round(new_elo, 1)
            self.ratings[persona]["games"] += 1
            self.ratings[persona]["history"].append(round(new_elo, 1))

            logger.info(f"ELO [{persona}]: {old_elo:.1f} → {new_elo:.1f}  (Δ{avg_delta:+.1f})")

        self._save()

    def get_leaderboard(self) -> str:
        if not self.ratings:
            return "No ELO data yet."
        ranked = sorted(self.ratings.items(), key=lambda x: x[1]["elo"], reverse=True)
        lines = ["=== ELO LEADERBOARD ==="]
        for rank, (persona, data) in enumerate(ranked, 1):
            trend = self._trend_arrow(data["history"])
            lines.append(
                f"  {rank}. {persona.upper():15s} {data['elo']:>7.0f} ELO"
                f"  ({data['games']} games)  {trend}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.ratings, f, indent=2)

    @staticmethod
    def _trend_arrow(history: List[float]) -> str:
        """Return ↑ / ↓ / → based on last 5 games."""
        if len(history) < 2:
            return "→"
        window = history[-5:]
        delta = window[-1] - window[0]
        if delta > 5:
            return "↑"
        if delta < -5:
            return "↓"
        return "→"

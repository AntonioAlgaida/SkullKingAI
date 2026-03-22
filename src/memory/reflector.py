# src/memory/reflector.py

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from src.agents.llm_client import LLMClient
from src.memory.rag_engine import StrategyMemory, FITNESS_WIN, FITNESS_LOSS
from src.memory.counterfactual import CounterfactualSimulator
from src.utils.prompt_loader import PromptLoader
from src.utils.translators import SemanticTranslator
from src.engine.physics import GamePhysics

logger = logging.getLogger(__name__)

# Minimum bid to trigger a success reflection for rational players.
SUCCESS_BID_THRESHOLD = 2

# Minimum round number to trigger a success reflection for forced_zero players.
ZERO_SUCCESS_ROUND_THRESHOLD = 5


class SleepCycleReflector:
    """
    Analyses game traces offline and writes strategic rules to ChromaDB.

    Three reflection types:
      1. Failure   — bid != won  → "what went wrong?"
      2. Success   — bid == won with significant outcome → "what worked?"
      3. Counter   — after a failure, analyse the opponent action that caused it
                     and generate a single combined rule for both sides.

    Performance design:
      - All prompt bundles are loaded once at __init__ (no per-call disk reads).
      - Static content (game rules + persona) → system role for vLLM prefix caching.
      - Dynamic content (play-by-play + task) → user role.
      - All LLM calls within a trace are fired concurrently via asyncio.gather().
    """

    def __init__(self, client: LLMClient, memory: StrategyMemory):
        self.client = client
        self.memory = memory
        _physics        = GamePhysics()
        self._sim       = CounterfactualSimulator(_physics)
        self._trans     = SemanticTranslator(_physics)

        # Cache all static prompt bundles at startup — zero per-call disk I/O
        self._game_intro      = PromptLoader.load("rules", "game_intro")
        self._card_hierarchy  = PromptLoader.load("rules", "card_hierarchy")
        self._trick_mechanics = PromptLoader.load("rules", "trick_mechanics")
        self._persona_cache: Dict[str, str] = {}   # populated on first use per persona

    def _persona(self, persona: str) -> str:
        if persona not in self._persona_cache:
            self._persona_cache[persona] = PromptLoader.get_persona(persona)
        return self._persona_cache[persona]

    def _playing_system(self, persona: str) -> str:
        """Static system prompt for playing-phase reflection (prefix-cached by vLLM)."""
        return (
            "You are a master Skull King tactician performing post-game analysis.\n\n"
            f"[GAME RULES]\n{self._game_intro}\n\n"
            f"{self._card_hierarchy}\n\n"
            f"{self._trick_mechanics}\n\n"
            f"[PERSONA]\n{self._persona(persona)}"
        )

    def _bidding_system(self, persona: str) -> str:
        """Static system prompt for bidding-phase reflection."""
        return (
            "You are a Skull King bidding strategist performing post-game analysis.\n\n"
            f"[GAME RULES]\n{self._game_intro}\n\n"
            f"{self._card_hierarchy}\n\n"
            f"[PERSONA]\n{self._persona(persona)}"
        )

    def _counter_system(self, persona: str) -> str:
        """Static system prompt for counter-strategy reflection."""
        return (
            "You are a Skull King counter-strategy analyst.\n\n"
            f"[GAME RULES]\n{self._game_intro}\n\n"
            f"{self._card_hierarchy}\n\n"
            f"[PERSONA]\n{self._persona(persona)}"
        )

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    async def process_trace(self, trace_path: str, llm_players: Dict[int, str]):
        """
        Loads a trace, builds all reflection prompts synchronously, then fires
        all LLM calls concurrently with asyncio.gather().
        """
        logger.info(f"[SleepCycle] Processing trace: {trace_path}")

        try:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trace: {e}")
            return

        events     = trace_data.get("events", trace_data) if isinstance(trace_data, dict) else trace_data
        rounds_data = self._group_by_round(events)

        # --- Phase 1: build all tasks synchronously (no LLM) ---
        # Each task is (coroutine, persona, phase, metadata, fitness_ids, fitness_delta)
        tasks = []

        for round_num, r_data in rounds_data.items():
            end_event = r_data.get("end_event")
            if not end_event:
                continue

            bids = end_event["bids"]
            won  = end_event["won"]

            for pid, persona in llm_players.items():
                failed  = bids[pid] != won[pid]
                success = bids[pid] == won[pid]

                situation_query      = self._build_situation_query(pid, round_num, r_data, bids, won)
                relevant_play_ids    = self.memory.query_rule_ids(situation_query, persona, "PLAYING")
                relevant_bidding_ids = self.memory.query_rule_ids(situation_query, persona, "BIDDING")
                fitness_delta        = FITNESS_LOSS if failed else FITNESS_WIN
                all_relevant_ids     = relevant_play_ids + relevant_bidding_ids

                starting_hand     = r_data.get("starting_hands",    {}).get(pid, [])
                starting_hand_ids = r_data.get("starting_hand_ids", {}).get(pid, [])

                if failed:
                    # Failure: playing + bidding + counter-strategy
                    sys_p, usr_p = self._build_playing_prompts(
                        pid, persona, round_num, bids[pid], won[pid],
                        r_data["tricks"], starting_hand, starting_hand_ids,
                    )
                    tasks.append((sys_p, usr_p, persona, "PLAYING", round_num,
                                  bids[pid], won[pid], "rule", all_relevant_ids, fitness_delta))

                    if persona.lower() != "forced_zero":
                        sys_b, usr_b = self._build_bidding_prompts(
                            pid, persona, round_num, bids[pid], won[pid], starting_hand
                        )
                        tasks.append((sys_b, usr_b, persona, "BIDDING", round_num,
                                      bids[pid], won[pid], "rule", [], 0.0))

                    sys_c, usr_c = self._build_counter_prompts(
                        pid, persona, round_num, bids, won,
                        r_data["tricks"], llm_players,
                    )
                    if sys_c is not None:
                        tasks.append((sys_c, usr_c, persona, "PLAYING", round_num,
                                      bids[pid], won[pid], "counter", [], 0.0))

                elif success and self._should_reflect_on_success(persona, bids[pid], round_num):
                    sys_s, usr_s = self._build_success_prompts(
                        pid, persona, round_num, bids[pid], won[pid],
                        r_data["tricks"], starting_hand,
                    )
                    tasks.append((sys_s, usr_s, persona, "PLAYING", round_num,
                                  bids[pid], won[pid], "rule", all_relevant_ids, fitness_delta))

        # --- Phase 2: fire all LLM calls concurrently ---
        if not tasks:
            return

        coroutines = [
            self.client.a_generate(usr, system_prompt=sys)
            for sys, usr, *_ in tasks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # --- Phase 3: memorise rules and update fitness ---
        for raw, (sys_p, usr_p, persona, phase, round_num, bid, won_n, rule_prefix,
                  fitness_ids, fitness_delta) in zip(results, tasks):
            if isinstance(raw, Exception):
                logger.error(f"Reflection LLM error ({persona}, {phase}): {raw}")
                continue
            self._save_rule(raw, persona, phase, round_num, bid, won_n, rule_prefix)
            if fitness_ids:
                self.memory.update_fitness(fitness_ids, fitness_delta)

    # ------------------------------------------------------------------ #
    # Prompt builders — return (system_prompt, user_prompt) tuples        #
    # ------------------------------------------------------------------ #

    def _build_playing_prompts(
        self, pid, persona, round_num, bid, won, tricks, starting_hand, starting_hand_ids=None
    ) -> Tuple[str, str]:
        history_text = self._format_trick_history(tricks)
        goal_text    = "win EXACTLY 0 tricks." if persona == "forced_zero" else f"win EXACTLY {bid} tricks."
        hand_str     = ", ".join(starting_hand) if starting_hand else "Unknown"

        if persona == "forced_zero":
            constraints = (
                "1. ZERO TOLERANCE: Winning even 1 trick is total failure.\n"
                "2. Focus on the specific card interaction that caused the accidental win.\n"
                "3. Identify a concrete disposal window that was missed or misused."
            )
        else:
            constraints = (
                "1. PRECISION: Over-winning and under-winning are equally bad.\n"
                "2. Focus on the specific trick where the bid became unachievable.\n"
                "3. Identify whether the failure was a bidding error or a play error."
            )

        counterfactual_section = ""
        if starting_hand_ids:
            critical_trick, critical_idx = self._find_critical_trick_with_index(pid, bid, won, tricks)
            if critical_trick and critical_trick.get("actions"):
                tricks_before = tricks[:critical_idx]
                remaining = self._sim.reconstruct_hand_at_trick(pid, starting_hand_ids, tricks_before)
                legal_alts = self._sim.get_legal_alternatives(
                    pid, None, critical_trick["actions"], remaining
                )
                actual_aid = next((aid for p, aid in critical_trick["actions"] if p == pid), None)
                if actual_aid is not None:
                    actual_won = (critical_trick.get("winner") == pid)
                    alternatives = self._sim.simulate_alternatives(
                        pid, actual_aid, critical_trick["actions"], legal_alts
                    )
                    counterfactual_section = "\n" + self._sim.format_evidence(
                        pid, actual_aid, actual_won, alternatives, self._trans
                    )

        system = self._playing_system(persona)
        user   = f"""\
[FAILURE REPORT]
Player {pid} goal: {goal_text}
Result: bid {bid}, won {won}. FAILED.  Round {round_num}.

[STARTING HAND]
{hand_str}

[PLAY-BY-PLAY]
{history_text}
{counterfactual_section}
[CONSTRAINTS]
{constraints}
4. NO GENERIC ADVICE. Be card-specific.
5. EXPLOIT THE STATE: Reference specific opponent statuses (FULL/STARVING) visible in the plays.
6. USE THE COUNTERFACTUAL: Ground your rule in the engine-verified evidence above.
7. Be concise. One rule. One insight.

[TASK]
Identify the exact trick where the failure became inevitable and what Player {pid} misread.
Output ONE rule starting with [RULE]:
Example: [RULE]: When an opponent is FULL and leads a mid-suit card, they are trying to lose — play your Escape immediately rather than a suit card that might win.
"""
        return system, user

    def _build_bidding_prompts(
        self, pid, persona, round_num, bid, won, starting_hand
    ) -> Tuple[str, str]:
        hand_str = ", ".join(starting_hand) if starting_hand else "Unknown"
        system   = self._bidding_system(persona)
        user     = f"""\
[FAILURE REPORT]
Player {pid} starting hand: {hand_str}
Bid: {bid} | Actually won: {won}
Result: FAILED. The player misjudged hand strength.  Round {round_num}.

[CONSTRAINTS]
1. Focus entirely on hand evaluation at bid time. Why did this hand win {won} tricks instead of {bid}?
2. Identify specific card combinations that are misleading (e.g., "low trumps are not guaranteed wins").
3. Give BIDDING advice only — not play advice.

[TASK]
Write your chain-of-thought reasoning, then output ONE rule starting with [RULE]:
Example: [RULE]: When holding multiple low Black cards (1–5) with no Pirates, reduce your bid by 1 — low trumps rarely survive a full round of competition.
"""
        return system, user

    def _build_success_prompts(
        self, pid, persona, round_num, bid, won, tricks, starting_hand
    ) -> Tuple[str, str]:
        history_text = self._format_trick_history(tricks)
        hand_str     = ", ".join(starting_hand) if starting_hand else "Unknown"

        if persona == "forced_zero":
            success_context = (
                f"Player {pid} bid 0 and won EXACTLY 0 tricks in Round {round_num} "
                f"(worth +{10 * round_num} pts). SUCCESS.\n"
                "Focus on: what sequence of plays allowed dangerous cards to be safely discarded? "
                "What opponents' states were exploited?"
            )
        else:
            success_context = (
                f"Player {pid} bid {bid} and won exactly {won} tricks. SUCCESS.\n"
                "Focus on: which cards were played at the optimal moment? "
                "Were bonuses captured while maintaining bid accuracy? "
                "How were opponent states exploited?"
            )

        system = self._playing_system(persona)
        user   = f"""\
[SUCCESS REPORT]
{success_context}

[STARTING HAND]
{hand_str}

[PLAY-BY-PLAY]
{history_text}

[CONSTRAINTS]
1. Identify the key decision that made success possible — not luck, but skill.
2. Reference specific cards or opponent states that were correctly read.
3. The rule must be actionable and reusable in future games with similar hands.
4. NO GENERIC ADVICE. Be card-specific and situation-specific.

[TASK]
Analyse what Player {pid} did correctly. Write your chain-of-thought, then output ONE rule starting with [RULE]:
Example: [RULE]: When holding a Pirate and the Skull King appears in an opponent's trick, playing the Pirate immediately disposes of it safely — the SK wins, your Pirate loses, and you preserve your zero bid.
"""
        return system, user

    def _build_counter_prompts(
        self,
        failing_pid: int,
        failing_persona: str,
        round_num: int,
        bids: List[int],
        won: List[int],
        tricks: List[Dict],
        llm_players: Dict[int, str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (system, user) for a combined counter-strategy prompt that teaches
        the failing player to defend AND (if a succeeding opponent exists) to exploit.
        Capped at one LLM call regardless of how many opponents were in the trick.
        Returns (None, None) if no critical trick is found.
        """
        critical_trick = self._find_critical_trick(failing_pid, bids[failing_pid], won[failing_pid], tricks)
        if not critical_trick:
            return None, None

        trick_plays = critical_trick.get("plays", [])
        other_pids  = []
        for play_str in trick_plays:
            match = re.match(r"P(\d+) played", play_str)
            if match:
                opid = int(match.group(1))
                if opid != failing_pid:
                    other_pids.append(opid)

        if not other_pids:
            return None, None

        history_text       = self._format_trick_history(tricks)
        critical_plays_text = "\n".join(trick_plays)
        winner_pid         = critical_trick.get("winner", "?")

        opp_context_lines = []
        for opid in other_pids:
            opp_persona = llm_players.get(opid, "unknown")
            opp_bid     = bids[opid] if opid < len(bids) else "?"
            opp_won_n   = won[opid]  if opid < len(won)  else "?"
            opp_status  = self._hunger_label(opp_bid, opp_won_n, round_num, sum(won))
            opp_context_lines.append(
                f"  P{opid} (persona={opp_persona}): bid={opp_bid}, won={opp_won_n}, status={opp_status}"
            )
        opp_context = "\n".join(opp_context_lines)

        # Pick one succeeding opponent to generate an exploit section for
        exploit_pid     = next(
            (opid for opid in other_pids
             if llm_players.get(opid) and bids[opid] == won[opid]),
            None
        )
        exploit_section = ""
        if exploit_pid is not None:
            exploit_section = f"""

[EXPLOIT TASK]
Player {exploit_pid} ({llm_players[exploit_pid]}) succeeded (bid={bids[exploit_pid]}, won={won[exploit_pid]}).
Their action in the critical trick contributed to Player {failing_pid}'s failure.
Also output ONE exploit rule for Player {exploit_pid} starting with [EXPLOIT]:
Example: [EXPLOIT]: When a forced_zero opponent has played multiple Escapes and appears desperate, leading a mid-value suit forces them to either follow suit with a dangerous card or waste an emergency resource.
"""

        system = self._counter_system(failing_persona)
        user   = f"""\
[FAILURE CONTEXT]
Round {round_num}. Player {failing_pid} ({failing_persona}) bid {bids[failing_pid]}, won {won[failing_pid]}. FAILED.

[CRITICAL TRICK — where failure became inevitable]
{critical_plays_text}
Trick winner: P{winner_pid}

[OPPONENT STATES]
{opp_context}

[FULL ROUND PLAY-BY-PLAY]
{history_text}

[DEFEND TASK]
Explain what the opponent did that caused Player {failing_pid}'s failure.
Output ONE counter-strategy rule for Player {failing_pid} starting with [RULE]:
Example: [RULE]: When a FULL rational opponent leads a mid-value suit card (6–9), they are trying to offload tricks — immediately play your lowest legal card to avoid accidentally winning.
{exploit_section}"""
        return system, user

    # ------------------------------------------------------------------ #
    # Rule extraction and memorisation                                     #
    # ------------------------------------------------------------------ #

    def _save_rule(
        self, raw_content: str, persona: str, phase: str,
        round_num: int, bid: int, won: int, rule_prefix: str,
    ):
        """Extract [RULE]: from LLM output and write to ChromaDB."""
        match = re.search(r"\[RULE\]\s*:\s*(.*)", raw_content, re.IGNORECASE)
        if match:
            new_rule = match.group(1).strip()
            tag = rule_prefix.upper()
            logger.info(f"[{persona.upper()}] [{tag}] New rule: {new_rule}")
            self.memory.memorize_rule(new_rule, persona, {
                "round_num": round_num, "bid": bid, "won": won,
                "phase": phase, "type": rule_prefix,
            })
        else:
            logger.warning(
                f"[{persona.upper()}] Failed to extract [RULE] from response:\n{raw_content[:300]}"
            )

        # Also save exploit rule if present in the same response
        exploit_match = re.search(r"\[EXPLOIT\]\s*:\s*(.*)", raw_content, re.IGNORECASE)
        if exploit_match:
            exploit_rule = exploit_match.group(1).strip()
            logger.info(f"[{persona.upper()}] [EXPLOIT] New rule: {exploit_rule}")
            self.memory.memorize_rule(exploit_rule, persona, {
                "round_num": round_num, "bid": bid, "won": won,
                "phase": phase, "type": "exploit",
            })

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _should_reflect_on_success(self, persona: str, bid: int, round_num: int) -> bool:
        if persona == "forced_zero":
            return round_num >= ZERO_SUCCESS_ROUND_THRESHOLD
        return bid >= SUCCESS_BID_THRESHOLD or round_num >= 7

    def _find_critical_trick(self, pid, bid, won, tricks) -> Optional[Dict]:
        result = self._find_critical_trick_with_index(pid, bid, won, tricks)
        return result[0] if result else None

    def _find_critical_trick_with_index(
        self, pid: int, bid: int, won: int, tricks: List[Dict]
    ) -> Tuple[Optional[Dict], int]:
        if won > bid:
            pid_wins = 0
            for i, trick in enumerate(tricks):
                if trick.get("winner") == pid:
                    pid_wins += 1
                    if pid_wins == bid + 1:
                        return trick, i
        else:
            last_miss, last_idx = None, -1
            for i, trick in enumerate(tricks):
                if trick.get("winner") != pid:
                    last_miss, last_idx = trick, i
            return last_miss, last_idx
        return None, -1

    @staticmethod
    def _hunger_label(bid: int, won: int, round_num: int, total_won: int) -> str:
        if bid == -1:
            return "BID_HIDDEN"
        tricks_left = round_num - total_won
        delta = bid - won
        if delta == 0:
            return "FULL"
        if delta < 0:
            return "OVERBOARD"
        if tricks_left > 0 and delta == tricks_left:
            return "STARVING"
        if tricks_left > 0 and 0 < delta < tricks_left:
            return "HUNGRY"
        return "DOOMED"

    @staticmethod
    def _build_situation_query(
        pid: int, round_num: int, r_data: Dict, bids: List[int], won: List[int]
    ) -> str:
        round_stage = "Late Game" if round_num > 7 else ("Mid Game" if round_num > 4 else "Early Game")
        hand = r_data.get("starting_hands", {}).get(pid, [])
        hand_str = ", ".join(str(c) for c in hand[:4]) if hand else "unknown"
        return (
            f"{round_stage}. Round {round_num}. "
            f"Hand: {hand_str}. Bid {bids[pid]}, won {won[pid]}."
        )

    @staticmethod
    def _format_trick_history(tricks: List[Dict]) -> str:
        lines = []
        for i, trick in enumerate(tricks):
            plays  = ", ".join(trick.get("plays", []))
            winner = trick.get("winner", "?")
            lines.append(f"Trick {i + 1}: {plays}  →  WINNER: P{winner}")
        return "\n".join(lines) if lines else "No trick data available."

    def _group_by_round(self, events: List[Dict]) -> Dict[int, Dict]:
        rounds: Dict[int, Dict] = {}
        current_trick: List[str]   = []
        current_trick_actions: List[tuple] = []

        for event in events:
            r = event.get("round")
            if r is None:
                continue

            if r not in rounds:
                rounds[r] = {
                    "tricks":           [],
                    "end_event":        None,
                    "starting_hands":   {},
                    "starting_hand_ids": {},
                }

            if "my_hand" in event and event.get("phase") == "BIDDING":
                pid = event["player"]
                if pid not in rounds[r]["starting_hand_ids"]:
                    rounds[r]["starting_hand_ids"][pid] = list(event["my_hand"])

            if "my_hand" in event and event.get("phase") == "PLAYING":
                pid = event["player"]
                if pid not in rounds[r]["starting_hands"]:
                    rounds[r]["starting_hands"][pid] = [
                        self._trans.translate_card(cid) for cid in event["my_hand"]
                    ]

            if "action_id" in event and event.get("phase") == "PLAYING":
                card_str = event.get("card_text", str(event["action_id"]))
                current_trick.append(f"P{event['player']} played {card_str}")
                current_trick_actions.append((event["player"], event["action_id"]))

            elif event.get("event_type") == "trick_end":
                if current_trick:
                    rounds[r]["tricks"].append({
                        "plays":   current_trick.copy(),
                        "actions": current_trick_actions.copy(),
                        "winner":  event.get("winner"),
                    })
                    current_trick = []
                    current_trick_actions = []

            elif event.get("event_type") == "round_end":
                rounds[r]["end_event"] = event

        return rounds

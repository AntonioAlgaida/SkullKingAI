# src/memory/reflector.py

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
# Bidding 1 and winning 1 is not interesting enough to learn from.
SUCCESS_BID_THRESHOLD = 2

# Minimum round number to trigger a success reflection for forced_zero players.
# Early-round zero bids (small stakes) are not worth learning from.
ZERO_SUCCESS_ROUND_THRESHOLD = 5


class SleepCycleReflector:
    """
    Analyses game traces offline and writes strategic rules to ChromaDB.

    Three reflection types:
      1. Failure   — bid != won  → "what went wrong?"
      2. Success   — bid == won with significant outcome → "what worked?"
      3. Counter   — after a failure, analyse the opponent action that caused it
                     and generate rules for BOTH sides.
    """

    def __init__(self, client: LLMClient, memory: StrategyMemory):
        self.client  = client
        self.memory  = memory
        _physics     = GamePhysics()
        self._sim    = CounterfactualSimulator(_physics)
        self._trans  = SemanticTranslator(_physics)

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def process_trace(self, trace_path: str, llm_players: Dict[int, str]):
        """
        trace_path  : path to a game_trace.json
        llm_players : {player_id: persona}, e.g. {0: "forced_zero", 1: "rational"}
        """
        logger.info(f"[SleepCycle] Processing trace: {trace_path}")

        try:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trace: {e}")
            return

        events = trace_data.get("events", trace_data) if isinstance(trace_data, dict) else trace_data
        rounds_data = self._group_by_round(events)

        for round_num, r_data in rounds_data.items():
            end_event = r_data.get("end_event")
            if not end_event:
                continue

            bids = end_event["bids"]
            won  = end_event["won"]

            for pid, persona in llm_players.items():
                failed  = bids[pid] != won[pid]
                success = bids[pid] == won[pid]

                # --- Fitness credit assignment (offline) ---
                # Query which rules were semantically relevant to this situation
                # BEFORE generating new rules so the new rule isn't self-penalised.
                situation_query = self._build_situation_query(
                    pid, round_num, r_data, bids, won
                )
                relevant_play_ids    = self.memory.query_rule_ids(situation_query, persona, "PLAYING")
                relevant_bidding_ids = self.memory.query_rule_ids(situation_query, persona, "BIDDING")
                fitness_delta = FITNESS_LOSS if failed else FITNESS_WIN

                starting_hand    = r_data.get("starting_hands",    {}).get(pid, [])
                starting_hand_ids = r_data.get("starting_hand_ids", {}).get(pid, [])

                if failed:
                    # --- Reflection type 1: Failure ---
                    self._generate_playing_rule(
                        pid, persona, round_num, bids[pid], won[pid],
                        r_data["tricks"], starting_hand, starting_hand_ids,
                    )
                    if persona.lower() != "forced_zero":
                        self._generate_bidding_rule(pid, persona, round_num, bids[pid], won[pid],
                                                     starting_hand)

                    # --- Reflection type 3: Counter-strategy ---
                    self._generate_counter_strategy_rule(
                        pid, persona, round_num, bids, won,
                        r_data["tricks"], llm_players
                    )

                elif success and self._should_reflect_on_success(persona, bids[pid], round_num):
                    # --- Reflection type 2: Success ---
                    self._generate_success_rule(pid, persona, round_num, bids[pid], won[pid],
                                                r_data["tricks"], starting_hand)

                # Apply fitness update AFTER rule generation (new rules start neutral)
                self.memory.update_fitness(relevant_play_ids + relevant_bidding_ids, fitness_delta)

    # ------------------------------------------------------------------ #
    # Reflection type 1: Failure                                          #
    # ------------------------------------------------------------------ #

    def _generate_playing_rule(self, pid, persona, round_num, bid, won, tricks, starting_hand,
                               starting_hand_ids=None):
        history_text = self._format_trick_history(tricks)
        goal_text = "win EXACTLY 0 tricks." if persona == "forced_zero" else f"win EXACTLY {bid} tricks."

        game_intro      = PromptLoader.load("rules", "game_intro")
        card_hierarchy  = PromptLoader.load("rules", "card_hierarchy")
        trick_mechanics = PromptLoader.load("rules", "trick_mechanics")
        persona_guide   = PromptLoader.get_persona(persona)
        hand_str        = ", ".join(starting_hand) if starting_hand else "Unknown"

        if persona == "forced_zero":
            constraints = (
                "1. ZERO TOLERANCE: Winning even 1 trick is total failure. Never imply 'only one trick' is acceptable.\n"
                "2. Focus on the specific card interaction that caused the accidental win.\n"
                "3. Identify a concrete disposal window that was missed or misused."
            )
        else:
            constraints = (
                "1. PRECISION: Over-winning and under-winning are equally bad.\n"
                "2. Focus on the specific trick where the bid became unachievable.\n"
                "3. Identify whether the failure was a bidding error or a play error."
            )

        # --- Counterfactual simulation at the critical trick ---
        counterfactual_section = ""
        if starting_hand_ids:
            critical_trick, critical_idx = self._find_critical_trick_with_index(pid, bid, won, tricks)
            if critical_trick and critical_trick.get("actions"):
                tricks_before = tricks[:critical_idx]
                remaining = self._sim.reconstruct_hand_at_trick(
                    pid, starting_hand_ids, tricks_before
                )
                legal_alts = self._sim.get_legal_alternatives(
                    pid, None, critical_trick["actions"], remaining
                )
                # Find the player's actual action in this trick
                actual_aid = next(
                    (aid for p, aid in critical_trick["actions"] if p == pid), None
                )
                if actual_aid is not None:
                    actual_won = (critical_trick.get("winner") == pid)
                    alternatives = self._sim.simulate_alternatives(
                        pid, actual_aid, critical_trick["actions"], legal_alts
                    )
                    counterfactual_section = "\n" + self._sim.format_evidence(
                        pid, actual_aid, actual_won, alternatives, self._trans
                    )

        prompt = f"""[SYSTEM]
You are a master Skull King tactician performing post-game analysis.

[RULES]
{game_intro}

{card_hierarchy}

{trick_mechanics}

[PERSONA & GOAL]
{persona_guide}
Player {pid} goal: {goal_text}
Result: bid {bid}, won {won}. FAILED.

[ROUND {round_num} STARTING HAND]
{hand_str}

[PLAY-BY-PLAY]
{history_text}
{counterfactual_section}
[CONSTRAINTS]
{constraints}
4. NO GENERIC ADVICE: Rules like "play low" or "save cards" are useless. Be card-specific.
5. EXPLOIT THE STATE: Reference specific opponent statuses (FULL/STARVING) visible in the plays.
6. USE THE COUNTERFACTUAL: The engine-verified evidence above shows what WOULD have happened.
   Ground your rule in this evidence, not speculation.
7. Be concise. One rule. One insight.

[TASK]
Identify the exact trick where the failure became inevitable and what Player {pid} misread.
Output ONE rule starting with [RULE]:
Example: [RULE]: When an opponent is FULL and leads a mid-suit card, they are trying to lose — play your Escape immediately rather than a suit card that might win.
"""
        self._query_and_save(prompt, persona, round_num, bid, won, phase="PLAYING")

    def _generate_bidding_rule(self, pid, persona, round_num, bid, won, starting_hand):
        bundle   = PromptLoader.get_bidding_bundle(persona)
        hand_str = ", ".join(starting_hand) if starting_hand else "Unknown"

        prompt = f"""[SYSTEM]
You are an AI Strategy Architect for Skull King. Analyse a failed bid and write one strict bidding rule.

[RULES CONTEXT]
{bundle}

[FAILURE REPORT]
Player {pid} starting hand: {hand_str}
Bid: {bid} | Actually won: {won}
Result: FAILED. The player misjudged hand strength.

[CONSTRAINTS]
1. Focus entirely on hand evaluation at bid time. Why did this hand win {won} tricks instead of {bid}?
2. Identify specific card combinations that are misleading (e.g., "low trumps are not guaranteed wins").
3. Give BIDDING advice only — not play advice.

[TASK]
Write your chain-of-thought reasoning, then output ONE rule starting with [RULE]:
Example: [RULE]: When holding multiple low Black cards (1–5) with no Pirates, reduce your bid by 1 — low trumps rarely survive a full round of competition.
"""
        self._query_and_save(prompt, persona, round_num, bid, won, phase="BIDDING")

    # ------------------------------------------------------------------ #
    # Reflection type 2: Success                                          #
    # ------------------------------------------------------------------ #

    def _should_reflect_on_success(self, persona: str, bid: int, round_num: int) -> bool:
        """Decide whether a successful round is interesting enough to learn from."""
        if persona == "forced_zero":
            # Late-game zero bid successes are high-value and hard to execute
            return round_num >= ZERO_SUCCESS_ROUND_THRESHOLD
        else:
            # Rational: non-trivial bid (≥2 tricks) or a high-round success
            return bid >= SUCCESS_BID_THRESHOLD or round_num >= 7

    def _generate_success_rule(self, pid, persona, round_num, bid, won, tricks, starting_hand):
        history_text    = self._format_trick_history(tricks)
        game_intro      = PromptLoader.load("rules", "game_intro")
        card_hierarchy  = PromptLoader.load("rules", "card_hierarchy")
        trick_mechanics = PromptLoader.load("rules", "trick_mechanics")
        persona_guide   = PromptLoader.get_persona(persona)
        hand_str        = ", ".join(starting_hand) if starting_hand else "Unknown"

        if persona == "forced_zero":
            success_context = (
                f"Player {pid} bid 0 and won EXACTLY 0 tricks in Round {round_num} (worth +{10 * round_num} pts). SUCCESS.\n"
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

        prompt = f"""[SYSTEM]
You are a master Skull King tactician extracting winning patterns from a successful round.
Your goal: distil what Player {pid} did RIGHT into a reusable strategic rule.

[RULES]
{game_intro}

{card_hierarchy}

{trick_mechanics}

[PERSONA]
{persona_guide}

[SUCCESS REPORT]
{success_context}

[ROUND {round_num} STARTING HAND]
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
        self._query_and_save(prompt, persona, round_num, bid, won, phase="PLAYING", is_success=True)

    # ------------------------------------------------------------------ #
    # Reflection type 3: Counter-strategy                                 #
    # ------------------------------------------------------------------ #

    def _generate_counter_strategy_rule(
        self,
        failing_pid: int,
        failing_persona: str,
        round_num: int,
        bids: List[int],
        won: List[int],
        tricks: List[Dict],
        llm_players: Dict[int, str],
    ):
        """
        Identify the trick that caused the failure, then generate two rules:
          - One for the failing player: how to defend against this opponent pattern.
          - One for the opponent who "caused" it (if they're an LLM player and succeeded).
        """
        critical_trick = self._find_critical_trick(failing_pid, bids[failing_pid], won[failing_pid], tricks)
        if not critical_trick:
            return

        # Identify opponent players in that critical trick (excluding the failing player)
        trick_plays = critical_trick.get("plays", [])
        other_pid_in_trick = []
        for play_str in trick_plays:
            # Format: "P<id> played <card>"
            match = re.match(r"P(\d+) played", play_str)
            if match:
                other_pid = int(match.group(1))
                if other_pid != failing_pid:
                    other_pid_in_trick.append(other_pid)

        # Only proceed if at least one opponent was in the trick
        if not other_pid_in_trick:
            return

        history_text  = self._format_trick_history(tricks)
        persona_guide = PromptLoader.get_persona(failing_persona)
        game_intro    = PromptLoader.load("rules", "game_intro")
        card_hierarchy = PromptLoader.load("rules", "card_hierarchy")

        critical_plays_text = "\n".join(trick_plays)
        winner_pid = critical_trick.get("winner", "?")

        # Build opponent context string
        opp_context_lines = []
        for opp_pid in other_pid_in_trick:
            opp_persona = llm_players.get(opp_pid, "unknown")
            opp_bid = bids[opp_pid] if opp_pid < len(bids) else "?"
            opp_won = won[opp_pid] if opp_pid < len(won) else "?"
            opp_status = self._hunger_label(opp_bid, opp_won, round_num, sum(won))
            opp_context_lines.append(
                f"  P{opp_pid} (persona={opp_persona}): bid={opp_bid}, won={opp_won}, status={opp_status}"
            )
        opp_context = "\n".join(opp_context_lines)

        # Rule for the failing player: how to read and defend against this opponent behaviour
        failing_prompt = f"""[SYSTEM]
You are a Skull King counter-strategy analyst. A player failed their bid because of a specific opponent action.
Your job: teach the failing player how to READ and DEFEND against this opponent behaviour in the future.

[RULES]
{game_intro}
{card_hierarchy}

[FAILING PLAYER PERSONA]
{persona_guide}

[FAILURE CONTEXT]
Round {round_num}. Player {failing_pid} ({failing_persona}) bid {bids[failing_pid]}, won {won[failing_pid]}. FAILED.

[CRITICAL TRICK — where the failure became inevitable]
{critical_plays_text}
Trick winner: P{winner_pid}

[OPPONENT STATES AT THAT MOMENT]
{opp_context}

[FULL ROUND PLAY-BY-PLAY]
{history_text}

[TASK]
Explain what the opponent did (or was trying to do) that caused Player {failing_pid}'s failure.
Then output ONE counter-strategy rule for Player {failing_pid} starting with [RULE]:
Example: [RULE]: When a FULL rational opponent leads a mid-value suit card (6–9), they are trying to offload tricks — immediately play your lowest legal card to avoid accidentally winning.
"""
        self._query_and_save(failing_prompt, failing_persona, round_num,
                             bids[failing_pid], won[failing_pid], phase="PLAYING",
                             rule_prefix="counter")

        # Rule for the opponent who succeeded — reinforce what worked
        for opp_pid in other_pid_in_trick:
            opp_persona = llm_players.get(opp_pid)
            if opp_persona is None:
                continue  # Not an LLM player — skip
            if bids[opp_pid] != won[opp_pid]:
                continue  # Opponent also failed — don't reinforce a failed pattern

            opp_guide = PromptLoader.get_persona(opp_persona)
            exploit_prompt = f"""[SYSTEM]
You are a Skull King counter-strategy analyst. An opponent action successfully disrupted a rival's bid.
Your job: teach Player {opp_pid} to repeat this pattern deliberately.

[RULES]
{game_intro}
{card_hierarchy}

[PLAYER {opp_pid} PERSONA]
{opp_guide}

[SUCCESS CONTEXT]
Round {round_num}. Player {opp_pid} ({opp_persona}) bid {bids[opp_pid]}, won {won[opp_pid]}. SUCCEEDED.
Their action in the critical trick contributed to Player {failing_pid} ({failing_persona}) failing their bid.

[CRITICAL TRICK]
{critical_plays_text}
Trick winner: P{winner_pid}

[FULL ROUND PLAY-BY-PLAY]
{history_text}

[TASK]
Identify what Player {opp_pid} did that successfully disrupted the opponent.
Output ONE exploit rule starting with [RULE]:
Example: [RULE]: When a forced_zero opponent has played multiple Escapes and appears desperate, leading a mid-value suit forces them to either follow suit with a dangerous card or waste an emergency tool.
"""
            self._query_and_save(exploit_prompt, opp_persona, round_num,
                                 bids[opp_pid], won[opp_pid], phase="PLAYING",
                                 rule_prefix="exploit")

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    def _query_and_save(
        self,
        prompt: str,
        persona: str,
        round_num: int,
        bid: int,
        won: int,
        phase: str,
        is_success: bool = False,
        rule_prefix: str = "rule",
    ):
        """Send prompt to LLM, extract [RULE]:, save to memory."""
        try:
            raw_content = self.client.generate(prompt)
            match = re.search(r"\[RULE\]\s*:\s*(.*)", raw_content, re.IGNORECASE)
            if match:
                new_rule = match.group(1).strip()
                tag = "SUCCESS" if is_success else rule_prefix.upper()
                logger.info(f"[{persona.upper()}] [{tag}] New rule: {new_rule}")
                metadata = {
                    "round_num": round_num,
                    "bid": bid,
                    "won": won,
                    "phase": phase,
                    "type": rule_prefix,
                }
                self.memory.memorize_rule(new_rule, persona, metadata)
            else:
                logger.warning(f"[{persona.upper()}] Failed to extract [RULE] from response:\n{raw_content[:300]}")
        except Exception as e:
            logger.error(f"Reflection failed for {persona}: {e}")

    def _find_critical_trick(
        self, pid: int, bid: int, won: int, tricks: List[Dict]
    ) -> Optional[Dict]:
        """Returns the critical trick dict (without its index)."""
        result = self._find_critical_trick_with_index(pid, bid, won, tricks)
        return result[0] if result else None

    def _find_critical_trick_with_index(
        self, pid: int, bid: int, won: int, tricks: List[Dict]
    ) -> Tuple[Optional[Dict], int]:
        """
        Returns (critical_trick, index_in_tricks_list).
        For over-bid (won > bid): the (bid+1)th trick won by pid — the one that pushed them over.
        For under-bid (won < bid): the last trick in the round where pid did NOT win.
        """
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
        """
        total_won: sum of tricks won by ALL players so far (to compute tricks remaining).
        """
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
        """Builds a RAG query string describing the player's situation this round.
        Used for offline fitness credit assignment."""
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
            plays = ", ".join(trick.get("plays", []))
            winner = trick.get("winner", "?")
            lines.append(f"Trick {i + 1}: {plays}  →  WINNER: P{winner}")
        return "\n".join(lines) if lines else "No trick data available."

    def _group_by_round(self, events: List[Dict]) -> Dict[int, Dict]:
        """Parse the flat event log into structured rounds."""
        rounds: Dict[int, Dict] = {}
        current_trick: List[str] = []
        current_trick_actions: List[tuple] = []   # [(player_id, action_id)]

        for event in events:
            r = event.get("round")
            if r is None:
                continue

            if r not in rounds:
                rounds[r] = {
                    "tricks": [],
                    "end_event": None,
                    "starting_hands": {},      # {pid: [card_name, ...]}  — for LLM prompts
                    "starting_hand_ids": {},   # {pid: [action_id, ...]}  — for counterfactual
                }

            # Capture full starting hand from BIDDING events (player has all cards, IDs intact)
            if "my_hand" in event and event.get("phase") == "BIDDING":
                pid = event["player"]
                if pid not in rounds[r]["starting_hand_ids"]:
                    rounds[r]["starting_hand_ids"][pid] = list(event["my_hand"])

            # Capture text starting hand from first PLAYING event (used in LLM prompts)
            if "my_hand" in event and event.get("phase") == "PLAYING":
                pid = event["player"]
                if pid not in rounds[r]["starting_hands"]:
                    hand_text = [self._trans.translate_card(cid) for cid in event["my_hand"]]
                    rounds[r]["starting_hands"][pid] = hand_text

            # Accumulate trick plays
            if "action_id" in event and event.get("phase") == "PLAYING":
                card_str = event.get("card_text", str(event["action_id"]))
                current_trick.append(f"P{event['player']} played {card_str}")
                current_trick_actions.append((event["player"], event["action_id"]))

            elif event.get("event_type") == "trick_end":
                if current_trick:
                    rounds[r]["tricks"].append({
                        "plays":   current_trick.copy(),
                        "actions": current_trick_actions.copy(),   # raw (pid, aid) tuples
                        "winner":  event.get("winner"),
                    })
                    current_trick = []
                    current_trick_actions = []

            elif event.get("event_type") == "round_end":
                rounds[r]["end_event"] = event

        return rounds

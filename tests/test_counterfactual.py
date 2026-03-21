# tests/test_counterfactual.py
#
# Pytest tests for CounterfactualSimulator.
# Card ID quick reference (from physics.py):
#   Green 1-14  → 0-13    (Green 7 = 6, Green 1 = 0, Green 3 = 2, Green 5 = 4)
#   Purple 1-14 → 14-27   (Purple 3 = 16)
#   Yellow 1-14 → 28-41   (Yellow 5 = 32)
#   Black 1-14  → 42-55   (Black 1 = 42, Black 11 = 52)
#   Pirates (5) → 56-60   (first Pirate = 56)
#   Tigress     → 61
#   Skull King  → 62
#   Mermaids    → 63-64   (first Mermaid = 63)
#   Escapes     → 65-69   (first Escape = 65, second = 66)
#   Loot        → 70-71
#   Kraken      → 72
#   White Whale → 73
#   Tigress as Escape (virtual) → 74

import pytest
from unittest.mock import MagicMock
from src.engine.physics import GamePhysics
from src.memory.counterfactual import CounterfactualSimulator, AlternativeOutcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def physics():
    """Real GamePhysics instance (no network calls, pure deterministic logic)."""
    return GamePhysics()


@pytest.fixture
def sim(physics):
    """CounterfactualSimulator backed by a real GamePhysics instance."""
    return CounterfactualSimulator(physics)


# ---------------------------------------------------------------------------
# Tests: simulate_alternatives
# ---------------------------------------------------------------------------

def test_simulate_alternatives_escape_avoids_win(sim):
    """
    When the lead card is Green 5 and P3 is void in Green, both Black 11
    (which wins via trump) and Escape (which loses) are legal.
    Verify: Escape loses, Black 11 wins — so the simulator correctly
    distinguishes the outcomes of the two alternatives.
    """
    # P0 leads Green 5 (action 4).  P1 plays Escape (65).  P2 plays Escape (66).
    # P3 (pid=3) actually played Black 11 (52) and WON.
    # P3's remaining hand at that point: [52 (Black 11), 65 (Escape)].
    # P3 is void in Green → both cards legal.
    full_trick = [(0, 4), (1, 65), (2, 66), (3, 52)]
    legal_alternatives = [52, 65]  # includes actual card; simulator skips 52

    results = sim.simulate_alternatives(
        player_id=3,
        actual_action_id=52,
        full_trick_actions=full_trick,
        legal_alternatives=legal_alternatives,
    )

    # Only Escape (65) should appear (actual card 52 is excluded)
    assert len(results) == 1
    escape_result = results[0]
    assert escape_result.action_id == 65
    # Escape never wins tricks
    assert escape_result.would_win is False
    assert escape_result.destroyed is False


def test_simulate_alternatives_winning_option_identified(sim):
    """
    Pirate (56) wins over any number card; Escape (65) loses to everything.
    When P3 actually played Escape and lost, the simulator should flag
    the Pirate alternative as would_win=True.
    """
    # P0 leads Escape (65) — special lead, no suit constraint.
    # P1 plays Green 3 (2).  P2 plays Green 7 (6).
    # P3 (pid=3) actually played Escape (65 — note: second escape is 66 to avoid collision).
    # We use action 66 as P3's actual play so it differs from P0's 65.
    full_trick = [(0, 65), (1, 2), (2, 6), (3, 66)]
    # P3 had [66 (Escape), 56 (Pirate)] in hand; legal_alternatives includes both.
    legal_alternatives = [66, 56]

    results = sim.simulate_alternatives(
        player_id=3,
        actual_action_id=66,
        full_trick_actions=full_trick,
        legal_alternatives=legal_alternatives,
    )

    # Only Pirate (56) is an alternative (66 == actual, skipped)
    assert len(results) == 1
    pirate_result = results[0]
    assert pirate_result.action_id == 56
    assert pirate_result.would_win is True


# ---------------------------------------------------------------------------
# Tests: get_legal_alternatives
# ---------------------------------------------------------------------------

def test_get_legal_alternatives_follow_suit(sim):
    """
    When the lead suit is Green and the player holds a Green card,
    they MUST follow suit.  Off-suit number cards are illegal; specials are OK.
    """
    # P0 leads Green 7 (action 6).
    # P1 (pid=1) has: Green 3 (2), Yellow 5 (32), Pirate (56).
    full_trick = [(0, 6), (1, 2)]  # P1's actual play is irrelevant here
    remaining_hand = [2, 32, 56]

    legal = sim.get_legal_alternatives(
        player_id=1,
        actual_action_id=2,
        full_trick_actions=full_trick,
        remaining_hand_ids=remaining_hand,
    )

    assert 2 in legal   # Green 3 — must follow suit
    assert 56 in legal  # Pirate — special, always legal
    assert 32 not in legal  # Yellow 5 — off-suit while holding Green → illegal


def test_get_legal_alternatives_void_in_lead_suit(sim):
    """
    When the player has NO card of the lead suit they are void and may
    play any card freely.
    """
    # P0 leads Green 7 (action 6).
    # P1 has: Yellow 5 (32), Pirate (56) — no Green.
    full_trick = [(0, 6), (1, 32)]
    remaining_hand = [32, 56]

    legal = sim.get_legal_alternatives(
        player_id=1,
        actual_action_id=32,
        full_trick_actions=full_trick,
        remaining_hand_ids=remaining_hand,
    )

    assert set(legal) == {32, 56}


def test_get_legal_alternatives_special_lead_no_constraint(sim):
    """
    When a character card (Skull King) leads the trick, there is no
    suit constraint — every card in hand is legal.
    """
    # P0 leads Skull King (62).
    # P1 has: Green 3 (2), Yellow 5 (32), Pirate (56).
    full_trick = [(0, 62), (1, 2)]
    remaining_hand = [2, 32, 56]

    legal = sim.get_legal_alternatives(
        player_id=1,
        actual_action_id=2,
        full_trick_actions=full_trick,
        remaining_hand_ids=remaining_hand,
    )

    assert set(legal) == {2, 32, 56}


def test_get_legal_alternatives_pirate_lead_no_constraint(sim):
    """
    Pirate lead also imposes no suit constraint — all cards legal.
    """
    # P0 leads Pirate (56).
    # P1 has: Green 3 (2), Black 1 (42).
    full_trick = [(0, 56), (1, 2)]
    remaining_hand = [2, 42]

    legal = sim.get_legal_alternatives(
        player_id=1,
        actual_action_id=2,
        full_trick_actions=full_trick,
        remaining_hand_ids=remaining_hand,
    )

    assert set(legal) == {2, 42}


# ---------------------------------------------------------------------------
# Tests: reconstruct_hand_at_trick
# ---------------------------------------------------------------------------

def test_reconstruct_hand_at_trick_one_trick_played(sim):
    """
    After one trick in which pid=0 played Pirate (56),
    the remaining hand should no longer contain card 56.
    """
    starting_hand = [56, 65, 6]  # Pirate, Escape, Green 7
    tricks_before = [
        {"actions": [(0, 56), (1, 65), (2, 6), (3, 4)]}
    ]

    remaining = sim.reconstruct_hand_at_trick(
        player_id=0,
        starting_hand_ids=starting_hand,
        tricks_before=tricks_before,
    )

    assert 56 not in remaining
    assert 65 in remaining
    assert 6 in remaining


def test_reconstruct_hand_at_trick_tigress_removal(sim):
    """
    When pid=0 played action 74 (Tigress as Escape), the underlying
    card ID 61 must be removed from the reconstructed hand.
    """
    # Starting hand contains card 61 (Tigress) and 65 (Escape).
    starting_hand = [61, 65]
    tricks_before = [
        {"actions": [(0, 74), (1, 65)]}  # P0 used virtual action 74 → card 61
    ]

    remaining = sim.reconstruct_hand_at_trick(
        player_id=0,
        starting_hand_ids=starting_hand,
        tricks_before=tricks_before,
    )

    assert 61 not in remaining
    assert 65 in remaining


def test_reconstruct_hand_no_tricks_before(sim):
    """
    With no prior tricks, the reconstructed hand is identical to the
    starting hand.
    """
    starting_hand = [56, 65, 6]

    remaining = sim.reconstruct_hand_at_trick(
        player_id=0,
        starting_hand_ids=starting_hand,
        tricks_before=[],
    )

    assert remaining == starting_hand


# ---------------------------------------------------------------------------
# Tests: format_evidence
# ---------------------------------------------------------------------------

def test_format_evidence_contains_key_info(sim):
    """
    format_evidence must include 'COUNTERFACTUAL', mention player 3,
    include the actual card name, and describe whether alternatives
    would win or lose.
    """
    # Build a minimal translator mock.
    translator = MagicMock()
    translator.translate_card.return_value = "[Black 11]"

    # One alternative that would LOSE.
    losing_alt = AlternativeOutcome(
        action_id=65,
        card_name="[Escape]",
        would_win=False,
        bonus_pts=0,
        destroyed=False,
    )
    # One alternative that WOULD WIN.
    winning_alt = AlternativeOutcome(
        action_id=56,
        card_name="[Pirate]",
        would_win=True,
        bonus_pts=0,
        destroyed=False,
    )

    evidence = sim.format_evidence(
        player_id=3,
        actual_action_id=52,   # Black 11
        actual_won_trick=True,
        alternatives=[losing_alt, winning_alt],
        translator=translator,
    )

    assert "COUNTERFACTUAL" in evidence
    assert "P3" in evidence
    assert "[Black 11]" in evidence           # actual card from mock translator
    assert "would LOSE" in evidence           # Escape result
    assert "WOULD WIN" in evidence            # Pirate result (upper-case in source)


# ---------------------------------------------------------------------------
# Tests: Kraken / destroyed tricks
# ---------------------------------------------------------------------------

def test_simulate_alternatives_destroyed_trick(sim):
    """
    When the Kraken (action 72) appears in a trick, TrickResult.destroyed
    is True.  simulate_alternatives must propagate destroyed=True into the
    AlternativeOutcome.
    """
    # P0 plays Green 5 (4).  P1 plays Kraken (72).  P2 plays Escape (65).
    # P3 (pid=3) actually played Green 7 (6).
    # Alternative: we test replacing P3's card with Yellow 5 (32).
    full_trick = [(0, 4), (1, 72), (2, 65), (3, 6)]
    legal_alternatives = [6, 32]  # actual=6 will be skipped; 32 is the alternative

    results = sim.simulate_alternatives(
        player_id=3,
        actual_action_id=6,
        full_trick_actions=full_trick,
        legal_alternatives=legal_alternatives,
    )

    assert len(results) == 1
    alt = results[0]
    assert alt.action_id == 32
    assert alt.destroyed is True
    # When destroyed, winner_id is None → would_win must be False
    assert alt.would_win is False

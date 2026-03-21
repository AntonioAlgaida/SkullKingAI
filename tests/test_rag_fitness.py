# tests/test_rag_fitness.py
#
# Pytest tests for fitness tracking in StrategyMemory (src/memory/rag_engine.py).
#
# Every test gets a fresh, isolated ChromaDB via the tmp_path fixture so
# tests cannot interfere with each other or with the production database.

import pytest
from src.memory.rag_engine import StrategyMemory, FITNESS_MAX, FITNESS_MIN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory(tmp_path):
    """
    Isolated StrategyMemory backed by a temporary ChromaDB directory.
    os.path.join(base, abs_path) returns abs_path when it is absolute,
    so passing str(tmp_path / "chroma") goes directly to the temp dir.
    """
    return StrategyMemory(persistence_path=str(tmp_path / "chroma"))


def _base_metadata():
    return {"phase": "PLAYING", "round_num": 3, "bid": 2, "won": 1}


def _add_rule(memory, text="When Black trump is led, escape immediately.", persona="rational"):
    memory.memorize_rule(text, persona, _base_metadata())


def _get_first_rule(memory, persona="rational"):
    col = memory._get_collection(persona)
    data = col.get(include=["metadatas", "documents"])
    if not data["ids"]:
        return None, None
    return data["ids"][0], data["metadatas"][0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memorize_rule_has_fitness_zero(memory):
    """
    A freshly memorised rule must carry fitness=0.0 so it starts neutral
    and must earn positive rank through actual game outcomes.
    """
    _add_rule(memory)
    rule_id, meta = _get_first_rule(memory)
    assert rule_id is not None, "Rule was not stored"
    assert float(meta["fitness"]) == 0.0


def test_update_fitness_win_increases(memory):
    """
    After a winning outcome delta (+0.5), the rule's fitness must increase
    from 0.0 to 0.5.
    """
    _add_rule(memory)
    rule_id, _ = _get_first_rule(memory)
    memory.update_fitness([rule_id], +0.5)
    _, meta = _get_first_rule(memory)
    assert abs(float(meta["fitness"]) - 0.5) < 1e-6


def test_update_fitness_loss_decreases(memory):
    """
    After a losing outcome delta (−0.3), the rule's fitness must decrease
    from 0.0 to −0.3.
    """
    _add_rule(memory)
    rule_id, _ = _get_first_rule(memory)
    memory.update_fitness([rule_id], -0.3)
    _, meta = _get_first_rule(memory)
    assert abs(float(meta["fitness"]) - (-0.3)) < 1e-6


def test_update_fitness_clamped_max(memory):
    """
    Repeatedly applying +0.5 must never push fitness above FITNESS_MAX (5.0).
    """
    _add_rule(memory)
    rule_id, _ = _get_first_rule(memory)
    for _ in range(20):
        memory.update_fitness([rule_id], +0.5)
    _, meta = _get_first_rule(memory)
    assert float(meta["fitness"]) <= FITNESS_MAX


def test_update_fitness_clamped_min(memory):
    """
    Repeatedly applying −0.3 must never push fitness below FITNESS_MIN (−3.0).
    """
    _add_rule(memory)
    rule_id, _ = _get_first_rule(memory)
    for _ in range(20):
        memory.update_fitness([rule_id], -0.3)
    _, meta = _get_first_rule(memory)
    assert float(meta["fitness"]) >= FITNESS_MIN


def test_update_fitness_preserves_other_metadata(memory):
    """
    Updating fitness must not clobber other metadata fields such as
    phase or round_num.
    """
    meta_in = {"phase": "PLAYING", "round_num": 5, "bid": 2, "won": 2}
    memory.memorize_rule("Hold Skull King until Pirate appears.", "rational", meta_in)
    rule_id, _ = _get_first_rule(memory)
    memory.update_fitness([rule_id], +0.5)
    _, meta_out = _get_first_rule(memory)
    assert meta_out["phase"] == "PLAYING"
    assert int(meta_out["round_num"]) == 5


def test_query_rule_ids_returns_matching_ids(memory):
    """
    query_rule_ids with a phase filter must return IDs belonging only to
    rules of that phase, not rules stored for a different phase.
    """
    # Store two PLAYING rules and one BIDDING rule.
    memory.memorize_rule(
        "When holding Skull King in late game, lead aggressively.",
        "rational",
        {"phase": "PLAYING", "round_num": 8, "bid": 3, "won": 3},
    )
    memory.memorize_rule(
        "When holding many escapes, consider playing Skull King early.",
        "rational",
        {"phase": "PLAYING", "round_num": 7, "bid": 2, "won": 2},
    )
    memory.memorize_rule(
        "Bid 0 only when holding five or more escapes.",
        "rational",
        {"phase": "BIDDING", "round_num": 6, "bid": 0, "won": 0},
    )

    # Fetch the PLAYING rule IDs directly so we know their exact IDs.
    col = memory._get_collection("rational")
    all_data = col.get(where={"phase": "PLAYING"}, include=["documents"])
    playing_ids = set(all_data["ids"])
    bidding_data = col.get(where={"phase": "BIDDING"}, include=["documents"])
    bidding_ids = set(bidding_data["ids"])

    returned = memory.query_rule_ids(
        "Skull King late game strategy", "rational", "PLAYING", n_results=5
    )

    # All returned IDs must belong to PLAYING rules, none to BIDDING.
    assert len(returned) > 0
    for rid in returned:
        assert rid in playing_ids, f"ID {rid} is not a PLAYING rule"
        assert rid not in bidding_ids, f"ID {rid} is a BIDDING rule, should not appear"


def test_retrieve_rules_fitness_reranking(memory):
    """
    After storing two rules and boosting one to max fitness, retrieve_rules
    must return at least one result without raising an exception.
    The exact ordering depends on embedding similarity, so we only assert
    non-empty output — the key guarantee is that fitness re-ranking does
    not crash.
    """
    memory.memorize_rule(
        "When you are FULL, play Escape to avoid over-winning.",
        "rational",
        {"phase": "PLAYING", "round_num": 5, "bid": 2, "won": 2},
    )
    memory.memorize_rule(
        "When an opponent is STARVING, do not lead with Black trump.",
        "rational",
        {"phase": "PLAYING", "round_num": 6, "bid": 3, "won": 3},
    )

    # Boost second rule to near-max fitness.
    col = memory._get_collection("rational")
    all_data = col.get(include=["documents"])
    second_id = all_data["ids"][1]
    memory.update_fitness([second_id], FITNESS_MAX)

    state = {
        "phase": "PLAYING",
        "round_num": 6,
        "my_hand": [62],   # Skull King
        "current_player_id": 0,
        "bids": [2, 2, 2, 2],
        "tricks_won": [1, 1, 1, 1],
    }
    result = memory.retrieve_rules(state, "rational")
    assert isinstance(result, str)
    assert len(result) > 0


def test_memorize_rule_deduplication(memory):
    """
    Adding the same rule text twice must result in only one entry in the
    collection (semantic deduplication rejects the second insertion).
    """
    rule_text = "When leading with Black 14, expect to win unless Skull King appears."
    memory.memorize_rule(rule_text, "rational", _base_metadata())
    memory.memorize_rule(rule_text, "rational", _base_metadata())
    col = memory._get_collection("rational")
    assert col.count() == 1


def test_update_fitness_empty_ids(memory):
    """
    Calling update_fitness with an empty list must not raise any exception.
    This is a guard against callers that find no relevant rules.
    """
    memory.update_fitness([], +0.5)  # must be silent and safe

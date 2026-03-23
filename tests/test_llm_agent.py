# tests/test_llm_agent.py

import unittest
from unittest.mock import MagicMock
from src.agents.llm_agent import LLMAgent
from src.utils.translators import SemanticTranslator
from src.engine.physics import GamePhysics

class TestLLMAgent(unittest.TestCase):
    def setUp(self):
        self.physics = GamePhysics()
        self.translator = SemanticTranslator(self.physics)

        # Mock the Client so we don't need LM Studio running for unit tests
        self.mock_client = MagicMock()

        # Use Rational persona so forced-zero shortcuts don't interfere
        self.agent = LLMAgent(
            client=self.mock_client,
            translator=self.translator,
            persona="Rational"
        )

    def test_optimization_single_choice(self):
        """If only 1 move is legal, agent should not query LLM."""
        state = {
            "phase": "PLAYING",
            "legal_actions": [42],
            # ... other state keys irrelevant for this test ...
        }
        action = self.agent.act(state)
        self.assertEqual(action, 42)
        self.mock_client.get_move_with_content.assert_not_called()

    def test_valid_llm_response(self):
        """Test normal flow: LLM returns a legal action."""
        state = {
            "phase": "BIDDING",
            "round_num": 1,
            "current_player_id": 0,
            "my_hand": [10, 20],
            "legal_actions": [0, 1, 2],
            "bids": [-1]*4,
            "tricks_won": [0]*4,
            "graveyard": []
        }

        # Mock LLM returning action 1
        self.mock_client.get_move_with_content.return_value = ("reasoning: bid 1", 1)

        action = self.agent.act(state)
        self.assertEqual(action, 1)

    def test_illegal_move_retry_logic(self):
        """Test that agent catches illegal moves and retries."""
        state = {
            "phase": "PLAYING",
            "round_num": 1,
            "current_player_id": 0,
            "my_hand": [10, 20],
            "legal_actions": [10, 20], # Must play card 10 or 20
            "bids": [0]*4,
            "tricks_won": [0]*4,
            "current_trick": [],
            "graveyard": []
        }

        # Sequence: Returns 99 (Illegal) -> Returns 10 (Legal)
        self.mock_client.get_move_with_content.side_effect = [
            ("ACTION: 99", 99),
            ("ACTION: 10", 10),
        ]

        action = self.agent.act(state)

        self.assertEqual(action, 10)
        # Verify it called client twice
        self.assertEqual(self.mock_client.get_move_with_content.call_count, 2)

if __name__ == '__main__':
    unittest.main()
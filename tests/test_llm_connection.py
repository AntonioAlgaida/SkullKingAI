# tests/test_llm_connection.py

import unittest
from src.agents.llm_client import LLMClient

class TestLLMConnection(unittest.TestCase):
    def setUp(self):
        # Adjust port if your LM Studio is different
        self.client = LLMClient(base_url="http://192.168.1.67:1234/v1")

    def test_parsing_reasoning_model(self):
        """Test if the parser handles the <think> block correctly."""
        # Simulate a raw string response from a DeepSeek-style model
        mock_response = """
        ◁think▷
        The user wants the highest card. 
        Green 14 is higher than Escape.
        I should play Green 14 (ID 0).
        ◁/think▷
        [ACTION]: 0
        """
        parsed_id = self.client._parse_action(mock_response)
        self.assertEqual(parsed_id, 0, "Failed to parse action after thinking block")

    def test_live_generation(self):
        """Test actual connection to LM Studio."""
        prompt = """
        [STATE]
        My Hand: [Green 14], [Escape]
        Legal Moves: ID 0 (Green 14), ID 1 (Escape)
        
        [TASK]
        Pick the highest card.
        Output [ACTION]: <ID>
        """
        
        action_id = self.client.get_move_with_content(prompt)[1]
        print(f"\nLive LLM Selected Action ID: {action_id}")
        self.assertGreaterEqual(action_id, 0)
        
    def test_robust_parsing(self):
        # Case 1: The standard requested format
        self.assertEqual(self.client._parse_action("My choice is [ACTION]: 5"), 5)
        
        # Case 2: Bidding variations
        self.assertEqual(self.client._parse_action("I will Bid 0"), 0)
        self.assertEqual(self.client._parse_action("Bid: 2"), 2)
        
        # Case 3: Case sensitivity and spacing
        self.assertEqual(self.client._parse_action("action : 12"), 12)
        
        # Case 4: Multiple numbers (Take the LAST one)
        # Often LLMs say: "I considered 1, but decided on 0"
        complex_text = "Thinking: 1 is risky. [ACTION]: 0"
        self.assertEqual(self.client._parse_action(complex_text), 0)

        # Case 5: Reasoning model specific
        reasoning_text = "◁think▷ I pick 7 ◁/think▷ Final answer is ACTION: 3"
        self.assertEqual(self.client._parse_action(reasoning_text), 3)

if __name__ == '__main__':
    unittest.main()
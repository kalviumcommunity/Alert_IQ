"""
Unit tests for ConversationManager multi-turn tracking and trimming logic.
"""
import unittest
from src.history_manager import ConversationManager


class TestConversationManager(unittest.TestCase):

    def setUp(self):
        self.system_prompt = "You are Alert_IQ Assistant."
        self.manager = ConversationManager(
            system_prompt=self.system_prompt,
            max_token_budget=150,
            preserve_recent_turns=1
        )

    def test_system_prompt_preserved(self):
        messages = self.manager.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], self.system_prompt)

    def test_multi_turn_accumulation(self):
        self.manager.add_user_message("Query 1")
        self.manager.add_assistant_message("Answer 1")
        messages = self.manager.get_messages()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["content"], "Query 1")
        self.assertEqual(messages[2]["content"], "Answer 1")

    def test_token_counting_overhead(self):
        toks = self.manager.count_total_tokens()
        self.assertGreater(toks, 0)

    def test_budget_enforcement_and_trimming(self):
        # Adding multiple lengthy turns to force eviction
        long_text = "This is a detailed and verbose alert payload with many parameters and stack trace lines. " * 3
        for i in range(5):
            self.manager.add_user_message(f"Turn {i} {long_text}")
            self.manager.add_assistant_message(f"Response {i} {long_text}")

        # The manager should have trimmed older turns
        self.assertGreater(len(self.manager.trim_events), 0)
        messages = self.manager.get_messages()
        # System prompt is always first
        self.assertEqual(messages[0]["role"], "system")
        # Total tokens are bounded
        self.assertLessEqual(self.manager.count_total_tokens(), 150)


if __name__ == "__main__":
    unittest.main()

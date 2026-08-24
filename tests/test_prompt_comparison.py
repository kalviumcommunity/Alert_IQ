"""
Unit tests for prompt variants and comparison framework.
"""
import unittest
from unittest.mock import MagicMock
from src.prompt_comparison import PROMPT_VARIANT_A, PROMPT_VARIANT_B, run_prompt_comparison, TEST_SCENARIOS


class TestPromptComparison(unittest.TestCase):

    def test_prompt_definitions_structure(self):
        # Task 1 & 2 validation: Role, Scope, Constraints, Fallback
        self.assertIn("system_message", PROMPT_VARIANT_A)
        self.assertIn("system_message", PROMPT_VARIANT_B)

        sys_b = PROMPT_VARIANT_B["system_message"]
        self.assertIn("ROLE & OBJECTIVE", sys_b)
        self.assertIn("SCOPE", sys_b)
        self.assertIn("CONSTRAINTS", sys_b)
        self.assertIn("Safety Fallback", sys_b)

    def test_run_prompt_comparison_execution(self):
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [
            "Variant A response 1",
            "Variant B response 1",
            "Variant A response 2",
            "Variant B response 2",
        ]

        results = run_prompt_comparison(mock_client)
        self.assertEqual(len(results), len(TEST_SCENARIOS))
        self.assertIn("Variant A (Vague / Unconstrained)", results[0]["runs"])
        self.assertIn("Variant B (Structured / Role, Scope & Constraints)", results[0]["runs"])


if __name__ == "__main__":
    unittest.main()

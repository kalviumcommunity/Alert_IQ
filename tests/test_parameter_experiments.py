"""
Unit tests for Model Parameter Experiments module.
"""
import unittest
from unittest.mock import MagicMock
from src.parameter_experiments import (
    run_temperature_experiment,
    run_max_tokens_experiment,
    run_additional_parameters_experiment
)


class TestParameterExperiments(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()

    def test_temperature_experiment(self):
        self.mock_client.chat_completion.side_effect = [
            "Factual deterministic response",
            "Balanced response",
            "Creative varied response"
        ]
        results = run_temperature_experiment(self.mock_client)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["temperature"], 0.0)
        self.assertEqual(results[1]["temperature"], 0.7)
        self.assertEqual(results[2]["temperature"], 1.4)
        self.assertIsNotNone(results[0]["response"])

    def test_max_tokens_experiment(self):
        self.mock_client.chat_completion.side_effect = [
            "Step 1: Check memory usage",
            "Step 1: Check memory usage. Step 2: Clear cache.",
            "Step 1: Check memory usage. Step 2: Clear cache. Step 3: Restart leaked process."
        ]
        results = run_max_tokens_experiment(self.mock_client)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["max_tokens_cap"], 25)
        self.assertEqual(results[1]["max_tokens_cap"], 80)
        self.assertEqual(results[2]["max_tokens_cap"], 200)

    def test_additional_parameters_experiment(self):
        self.mock_client.chat_completion.side_effect = [
            "Strict top_p response",
            "Broad top_p response",
            "1. Step 1\n2. Step 2"
        ]
        results = run_additional_parameters_experiment(self.mock_client)
        self.assertIn("top_p_strict_0_1", results)
        self.assertIn("top_p_broad_0_95", results)
        self.assertIn("stop_sequence_response", results)


if __name__ == "__main__":
    unittest.main()

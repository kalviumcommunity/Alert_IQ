"""
Unit tests for Token Counter and Cost Estimation module.
"""
import unittest
from src.token_counter import count_tokens, calculate_cost, analyze_length_token_relationship, PRICING_CATALOG


class TestTokenCounter(unittest.TestCase):

    def test_count_tokens_empty(self):
        self.assertEqual(count_tokens(""), 0)

    def test_count_tokens_samples(self):
        short = "Hello world"
        toks = count_tokens(short)
        self.assertGreaterEqual(toks, 2)

    def test_cost_calculation(self):
        # 1,000,000 input tokens and 1,000,000 output tokens for gemini-2.5-flash
        res = calculate_cost(1_000_000, 1_000_000, "gemini-2.5-flash")
        self.assertAlmostEqual(res["input_cost_usd"], 0.075)
        self.assertAlmostEqual(res["output_cost_usd"], 0.300)
        self.assertAlmostEqual(res["total_cost_usd"], 0.375)

    def test_length_token_relationship(self):
        samples = [
            {"category": "Prose", "description": "English sentence", "text": "This is a simple test sentence."},
            {"category": "Code", "description": "Code snippet", "text": "def test_fn(x, y=10): return x + y"}
        ]
        results = analyze_length_token_relationship(samples)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertGreater(r["character_count"], 0)
            self.assertGreater(r["token_count"], 0)
            self.assertGreater(r["chars_per_token"], 0)


if __name__ == "__main__":
    unittest.main()

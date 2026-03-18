import unittest

from worker_plan_internal.llm_util.model_pricing import (
    register_model_pricing,
    estimate_cost,
    load_pricing_from_llm_config,
    _PRICING_REGISTRY,
)


class TestModelPricing(unittest.TestCase):
    def setUp(self):
        _PRICING_REGISTRY.clear()

    def tearDown(self):
        _PRICING_REGISTRY.clear()

    def test_register_and_exact_match(self):
        register_model_pricing("gpt-5-nano", 0.05, 0.40)
        cost = estimate_cost("gpt-5-nano", input_tokens=1_000_000, output_tokens=0)
        self.assertAlmostEqual(cost, 0.05)

    def test_prefix_match_with_version_suffix(self):
        register_model_pricing("gpt-5-nano", 0.05, 0.40)
        cost = estimate_cost(
            "gpt-5-nano-2025-08-07",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        self.assertAlmostEqual(cost, 0.05 + 0.40)

    def test_thinking_tokens_billed_at_output_rate(self):
        register_model_pricing("claude-sonnet-4-6", 3.00, 15.00)
        cost = estimate_cost(
            "claude-sonnet-4-6",
            input_tokens=1_000_000,
            output_tokens=500_000,
            thinking_tokens=500_000,
        )
        # input: 3.00, output+thinking: 15.00 * 1M = 15.00
        self.assertAlmostEqual(cost, 3.00 + 15.00)

    def test_unknown_model_returns_none(self):
        result = estimate_cost("unknown-model", input_tokens=100, output_tokens=100)
        self.assertIsNone(result)

    def test_load_from_llm_config(self):
        config = {
            "openai-gpt-5-nano": {
                "arguments": {"model": "gpt-5-nano"},
                "pricing": {
                    "input_per_million_tokens": 0.05,
                    "output_per_million_tokens": 0.40,
                },
            },
            "ollama-llama3.1": {
                "arguments": {"model": "llama3.1:latest"},
                "pricing_kind": "free",
            },
        }
        load_pricing_from_llm_config(config)
        self.assertIn("gpt-5-nano", _PRICING_REGISTRY)
        self.assertNotIn("llama3.1:latest", _PRICING_REGISTRY)

    def test_longest_prefix_wins(self):
        register_model_pricing("gpt-5", 1.00, 2.00)
        register_model_pricing("gpt-5-nano", 0.05, 0.40)
        cost = estimate_cost("gpt-5-nano-2025-08-07", input_tokens=1_000_000, output_tokens=0)
        self.assertAlmostEqual(cost, 0.05)

    def test_zero_tokens_returns_zero(self):
        register_model_pricing("gpt-5-nano", 0.05, 0.40)
        cost = estimate_cost("gpt-5-nano", input_tokens=0, output_tokens=0)
        self.assertAlmostEqual(cost, 0.0)

    def test_realistic_gpt5_nano_cost(self):
        """Verify the exact scenario from the bug report: 24485 input, 140366 output."""
        register_model_pricing("gpt-5-nano", 0.05, 0.40)
        cost = estimate_cost(
            "gpt-5-nano-2025-08-07",
            input_tokens=24485,
            output_tokens=140366,
        )
        expected = (24485 * 0.05 + 140366 * 0.40) / 1_000_000
        self.assertAlmostEqual(cost, round(expected, 6))
        self.assertGreater(cost, 0.0)


if __name__ == "__main__":
    unittest.main()

"""
Tests for TextFixer modules

Author: Egon (VoynichLabs), 2026-03-29
PURPOSE: Verify that each TextFixer module correctly strips its target patterns
         without corrupting surrounding text.
"""

import unittest
from .modules import (
    hedge_reducer,
    preamble_stripper,
    formal_reducer,
    disclaimer_stripper,
    apply_modules,
    apply_all_enabled,
)


class TestHedgeReducer(unittest.TestCase):
    """Test hedge_reducer module strips hedging language."""

    def test_i_think(self):
        self.assertEqual(
            hedge_reducer.transform("I think the budget needs revision."),
            "The budget needs revision."
        )

    def test_i_think_that(self):
        self.assertEqual(
            hedge_reducer.transform("I think that the budget needs revision."),
            "The budget needs revision."
        )

    def test_perhaps(self):
        self.assertEqual(
            hedge_reducer.transform("Perhaps we should reconsider the timeline."),
            "We should reconsider the timeline."
        )

    def test_important_to_note(self):
        self.assertEqual(
            hedge_reducer.transform("It's important to note that costs are rising."),
            "Costs are rising."
        )

    def test_its_worth_noting(self):
        self.assertEqual(
            hedge_reducer.transform("It's worth noting that the deadline is firm."),
            "The deadline is firm."
        )

    def test_i_would_suggest(self):
        result = hedge_reducer.transform("I would suggest that we reconsider the approach.")
        self.assertNotIn("I would suggest", result)
        self.assertIn("reconsider", result)

    def test_multiple_hedges(self):
        text = "I think perhaps we should maybe reconsider."
        result = hedge_reducer.transform(text)
        self.assertNotIn("I think", result)
        self.assertNotIn("perhaps", result)
        self.assertNotIn("maybe", result)

    def test_preserves_non_hedge_text(self):
        text = "The budget is $500,000. The timeline is 6 months."
        self.assertEqual(hedge_reducer.transform(text), text)

    def test_case_insensitive(self):
        result = hedge_reducer.transform("PERHAPS the risk is overstated.")
        self.assertNotIn("PERHAPS", result)

    def test_bears_mentioning(self):
        result = hedge_reducer.transform("It bears mentioning that the vendor is unreliable.")
        self.assertNotIn("bears mentioning", result)
        self.assertIn("vendor is unreliable", result)


class TestPreambleStripper(unittest.TestCase):
    """Test preamble_stripper module strips conversational openers."""

    def test_sure(self):
        result = preamble_stripper.transform("Sure, here's the analysis.")
        self.assertTrue(result.startswith("Here"))

    def test_certainly(self):
        self.assertEqual(
            preamble_stripper.transform("Certainly, the project has three phases."),
            "The project has three phases."
        )

    def test_happy_to_help(self):
        result = preamble_stripper.transform("I'd be happy to help with that! The project plan includes...")
        self.assertTrue(result.startswith("The project plan"))

    def test_great_question(self):
        result = preamble_stripper.transform("Great question! The WBS breaks down as follows:")
        self.assertTrue(result.startswith("The WBS"))

    def test_excellent_question(self):
        result = preamble_stripper.transform("Excellent question! Here are the details.")
        self.assertNotIn("Excellent question", result)

    def test_comprehensive_analysis(self):
        result = preamble_stripper.transform("Here is a comprehensive analysis of the risks:")
        self.assertNotIn("Here is a comprehensive", result)

    def test_upon_review(self):
        result = preamble_stripper.transform("Upon careful review, the plan needs adjustment.")
        self.assertNotIn("Upon careful review", result)
        self.assertIn("plan needs adjustment", result)

    def test_preserves_mid_text(self):
        text = "The project has risks. Sure, some are manageable."
        # "Sure" mid-sentence should not be stripped (only at line start)
        result = preamble_stripper.transform(text)
        self.assertIn("Sure", result)


class TestFormalReducer(unittest.TestCase):
    """Test formal_reducer module simplifies vocabulary."""

    def setUp(self):
        formal_reducer.enabled = True

    def tearDown(self):
        formal_reducer.enabled = False

    def test_furthermore(self):
        self.assertEqual(
            formal_reducer.transform("Furthermore, the costs are high."),
            "Also, the costs are high."
        )

    def test_utilize(self):
        self.assertEqual(
            formal_reducer.transform("We should utilize the existing tools."),
            "We should use the existing tools."
        )

    def test_in_order_to(self):
        self.assertEqual(
            formal_reducer.transform("In order to succeed, we need funding."),
            "To succeed, we need funding."
        )

    def test_due_to_the_fact(self):
        self.assertEqual(
            formal_reducer.transform("Due to the fact that costs rose, we adjusted."),
            "Because costs rose, we adjusted."
        )

    def test_leverage_verb(self):
        result = formal_reducer.transform("We should leverage the existing platform.")
        self.assertIn("use", result)

    def test_with_regard_to(self):
        result = formal_reducer.transform("With regard to the budget, we need cuts.")
        self.assertIn("About", result)

    def test_for_the_purpose_of(self):
        result = formal_reducer.transform("For the purpose of testing, we used mock data.")
        self.assertIn("To", result)
        self.assertNotIn("For the purpose of", result)


class TestDisclaimerStripper(unittest.TestCase):
    """Test disclaimer_stripper module removes boilerplate disclaimers."""

    def test_parenthesized_note(self):
        text = "The risk assessment shows three critical areas. (Note: This analysis is not a substitute for professional risk management advice.)"
        result = disclaimer_stripper.transform(text)
        self.assertNotIn("not a substitute", result)
        self.assertIn("three critical areas", result)

    def test_please_consult(self):
        text = "Budget estimate: $2.3M. Please consult a qualified financial professional before proceeding."
        result = disclaimer_stripper.transform(text)
        self.assertNotIn("consult", result)
        self.assertIn("$2.3M", result)

    def test_not_legal_advice(self):
        text = "The governance framework requires annual audits. This is not legal advice."
        result = disclaimer_stripper.transform(text)
        self.assertNotIn("not legal advice", result)
        self.assertIn("annual audits", result)

    def test_does_not_constitute(self):
        text = "Review the contracts carefully. This does not constitute legal advice."
        result = disclaimer_stripper.transform(text)
        self.assertNotIn("does not constitute", result)
        self.assertIn("Review the contracts", result)

    def test_we_recommend_consulting(self):
        text = "The tax structure is complex. We recommend consulting a tax professional."
        result = disclaimer_stripper.transform(text)
        self.assertNotIn("recommend consulting", result)
        self.assertIn("tax structure is complex", result)


class TestApplyModules(unittest.TestCase):
    """Test the module composition API."""

    def test_default_modules(self):
        text = "Certainly, I think the project has risks. It's important to note that the budget is tight."
        result = apply_modules(text)
        self.assertNotIn("Certainly", result)
        self.assertNotIn("I think", result)
        self.assertNotIn("important to note", result)
        self.assertIn("project has risks", result)
        self.assertIn("budget is tight", result)

    def test_specific_modules(self):
        text = "Sure, the budget is $500K."
        result = apply_modules(text, module_ids=['preamble_stripper'])
        self.assertNotIn("Sure", result)
        self.assertIn("$500K", result)

    def test_empty_module_list(self):
        text = "I think perhaps we should reconsider."
        result = apply_modules(text, module_ids=[])
        self.assertEqual(result, text)

    def test_apply_all_enabled(self):
        text = "Certainly! I think the risks are significant."
        result = apply_all_enabled(text)
        self.assertNotIn("Certainly", result)
        self.assertNotIn("I think", result)
        self.assertIn("risks are significant", result)

    def test_real_planexe_output(self):
        """Test against realistic PlanExe pipeline output."""
        text = """Certainly! Here is a comprehensive analysis of the potential risks:

I think the most significant risk is budget overrun. It's important to note that similar projects have exceeded estimates by 30-50%. Perhaps the team should consider a phased approach.

Furthermore, the timeline assumes no regulatory delays. It's worth noting that FDA approval processes can take 12-18 months longer than planned.

(Note: This analysis is for informational purposes only and should not be considered professional advice. Please consult with a qualified project manager before proceeding.)"""

        result = apply_modules(text)

        # Preambles gone
        self.assertNotIn("Certainly!", result)
        self.assertNotIn("Here is a comprehensive", result)

        # Hedges gone
        self.assertNotIn("I think", result)
        self.assertNotIn("important to note", result)
        self.assertNotIn("Perhaps", result)
        self.assertNotIn("worth noting", result)

        # Disclaimer gone
        self.assertNotIn("informational purposes", result)
        self.assertNotIn("Please consult", result)

        # Substance preserved
        self.assertIn("budget overrun", result)
        self.assertIn("30-50%", result)
        self.assertIn("FDA approval", result)
        self.assertIn("12-18 months", result)


class TestHitCounts(unittest.TestCase):
    """Test pattern hit tracking."""

    def test_counts_substitutions(self):
        hedge_reducer.reset_counts()
        hedge_reducer.transform("I think the plan is good. I think the budget is tight.")
        self.assertEqual(hedge_reducer.total_hits, 2)

    def test_tracks_by_rule_index(self):
        hedge_reducer.reset_counts()
        hedge_reducer.transform("I think perhaps we should reconsider.")
        # At least two different pattern indices should have hits
        self.assertGreaterEqual(len(hedge_reducer.hit_counts), 2)

    def test_reset_clears_counts(self):
        hedge_reducer.reset_counts()
        hedge_reducer.transform("I think the plan is solid.")
        self.assertGreater(hedge_reducer.total_hits, 0)
        hedge_reducer.reset_counts()
        self.assertEqual(hedge_reducer.total_hits, 0)
        self.assertEqual(len(hedge_reducer.hit_counts), 0)

    def test_no_match_no_count(self):
        hedge_reducer.reset_counts()
        hedge_reducer.transform("The budget is $500,000.")
        self.assertEqual(hedge_reducer.total_hits, 0)

    def test_counts_accumulate(self):
        hedge_reducer.reset_counts()
        hedge_reducer.transform("I think the plan is good.")
        hedge_reducer.transform("I think the budget is tight.")
        self.assertEqual(hedge_reducer.total_hits, 2)


if __name__ == '__main__':
    unittest.main()

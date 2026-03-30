"""
Tests for TextFixer modules

Author: Egon (VoynichLabs), 2026-03-29
PURPOSE: Verify that each TextFixer module correctly strips its target patterns
         without corrupting surrounding text.
"""

import unittest
from .modules import TextFixer


def _make_fixer() -> TextFixer:
    """Create a TextFixer loaded with default rules."""
    fixer = TextFixer()
    fixer.load(TextFixer.default_rules_path())
    return fixer


class TestHedgeReducer(unittest.TestCase):
    """Test hedge_reducer module strips hedging language."""

    def setUp(self):
        self.fixer = _make_fixer()
        self.module = self.fixer.get_module('hedge_reducer')

    def test_i_think(self):
        self.assertEqual(
            self.module.transform("I think the budget needs revision."),
            "The budget needs revision."
        )

    def test_i_think_that(self):
        self.assertEqual(
            self.module.transform("I think that the budget needs revision."),
            "The budget needs revision."
        )

    def test_perhaps(self):
        self.assertEqual(
            self.module.transform("Perhaps we should reconsider the timeline."),
            "We should reconsider the timeline."
        )

    def test_important_to_note(self):
        self.assertEqual(
            self.module.transform("It's important to note that costs are rising."),
            "Costs are rising."
        )

    def test_its_worth_noting(self):
        self.assertEqual(
            self.module.transform("It's worth noting that the deadline is firm."),
            "The deadline is firm."
        )

    def test_i_would_suggest(self):
        result = self.module.transform("I would suggest that we reconsider the approach.")
        self.assertNotIn("I would suggest", result)
        self.assertIn("reconsider", result)

    def test_multiple_hedges(self):
        text = "I think perhaps we should maybe reconsider."
        result = self.module.transform(text)
        self.assertNotIn("I think", result)
        self.assertNotIn("perhaps", result)
        self.assertNotIn("maybe", result)

    def test_preserves_non_hedge_text(self):
        text = "The budget is $500,000. The timeline is 6 months."
        self.assertEqual(self.module.transform(text), text)

    def test_case_insensitive(self):
        result = self.module.transform("PERHAPS the risk is overstated.")
        self.assertNotIn("PERHAPS", result)

    def test_bears_mentioning(self):
        result = self.module.transform("It bears mentioning that the vendor is unreliable.")
        self.assertNotIn("bears mentioning", result)
        self.assertIn("vendor is unreliable", result)


class TestPreambleStripper(unittest.TestCase):
    """Test preamble_stripper module strips conversational openers."""

    def setUp(self):
        self.fixer = _make_fixer()
        self.module = self.fixer.get_module('preamble_stripper')

    def test_sure(self):
        result = self.module.transform("Sure, here's the analysis.")
        self.assertTrue(result.startswith("Here"))

    def test_certainly(self):
        self.assertEqual(
            self.module.transform("Certainly, the project has three phases."),
            "The project has three phases."
        )

    def test_happy_to_help(self):
        result = self.module.transform("I'd be happy to help with that! The project plan includes...")
        self.assertTrue(result.startswith("The project plan"))

    def test_great_question(self):
        result = self.module.transform("Great question! The WBS breaks down as follows:")
        self.assertTrue(result.startswith("The WBS"))

    def test_excellent_question(self):
        result = self.module.transform("Excellent question! Here are the details.")
        self.assertNotIn("Excellent question", result)

    def test_comprehensive_analysis(self):
        result = self.module.transform("Here is a comprehensive analysis of the risks:")
        self.assertNotIn("Here is a comprehensive", result)

    def test_upon_review(self):
        result = self.module.transform("Upon careful review, the plan needs adjustment.")
        self.assertNotIn("Upon careful review", result)
        self.assertIn("plan needs adjustment", result)

    def test_preserves_mid_text(self):
        text = "The project has risks. Sure, some are manageable."
        result = self.module.transform(text)
        self.assertIn("Sure", result)


class TestFormalReducer(unittest.TestCase):
    """Test formal_reducer module simplifies vocabulary."""

    def setUp(self):
        self.fixer = _make_fixer()
        self.module = self.fixer.get_module('formal_reducer')
        self.module.enabled = True

    def test_furthermore(self):
        self.assertEqual(
            self.module.transform("Furthermore, the costs are high."),
            "Also, the costs are high."
        )

    def test_utilize(self):
        self.assertEqual(
            self.module.transform("We should utilize the existing tools."),
            "We should use the existing tools."
        )

    def test_in_order_to(self):
        self.assertEqual(
            self.module.transform("In order to succeed, we need funding."),
            "To succeed, we need funding."
        )

    def test_due_to_the_fact(self):
        self.assertEqual(
            self.module.transform("Due to the fact that costs rose, we adjusted."),
            "Because costs rose, we adjusted."
        )

    def test_leverage_verb(self):
        result = self.module.transform("We should leverage the existing platform.")
        self.assertIn("use", result)

    def test_with_regard_to(self):
        result = self.module.transform("With regard to the budget, we need cuts.")
        self.assertIn("About", result)

    def test_for_the_purpose_of(self):
        result = self.module.transform("For the purpose of testing, we used mock data.")
        self.assertIn("To", result)
        self.assertNotIn("For the purpose of", result)


class TestDisclaimerStripper(unittest.TestCase):
    """Test disclaimer_stripper module removes boilerplate disclaimers."""

    def setUp(self):
        self.fixer = _make_fixer()
        self.module = self.fixer.get_module('disclaimer_stripper')

    def test_parenthesized_note(self):
        text = "The risk assessment shows three critical areas. (Note: This analysis is not a substitute for professional risk management advice.)"
        result = self.module.transform(text)
        self.assertNotIn("not a substitute", result)
        self.assertIn("three critical areas", result)

    def test_please_consult(self):
        text = "Budget estimate: $2.3M. Please consult a qualified financial professional before proceeding."
        result = self.module.transform(text)
        self.assertNotIn("consult", result)
        self.assertIn("$2.3M", result)

    def test_not_legal_advice(self):
        text = "The governance framework requires annual audits. This is not legal advice."
        result = self.module.transform(text)
        self.assertNotIn("not legal advice", result)
        self.assertIn("annual audits", result)

    def test_does_not_constitute(self):
        text = "Review the contracts carefully. This does not constitute legal advice."
        result = self.module.transform(text)
        self.assertNotIn("does not constitute", result)
        self.assertIn("Review the contracts", result)

    def test_we_recommend_consulting(self):
        text = "The tax structure is complex. We recommend consulting a tax professional."
        result = self.module.transform(text)
        self.assertNotIn("recommend consulting", result)
        self.assertIn("tax structure is complex", result)


class TestApplyModules(unittest.TestCase):
    """Test the TextFixer.apply() composition API."""

    def setUp(self):
        self.fixer = _make_fixer()

    def test_default_modules(self):
        text = "Certainly, I think the project has risks. It's important to note that the budget is tight."
        result = self.fixer.apply(text)
        self.assertNotIn("Certainly", result)
        self.assertNotIn("I think", result)
        self.assertNotIn("important to note", result)
        self.assertIn("project has risks", result)
        self.assertIn("budget is tight", result)

    def test_specific_modules(self):
        text = "Sure, the budget is $500K."
        result = self.fixer.apply(text, module_ids=['preamble_stripper'])
        self.assertNotIn("Sure", result)
        self.assertIn("$500K", result)

    def test_empty_module_list(self):
        text = "I think perhaps we should reconsider."
        result = self.fixer.apply(text, module_ids=[])
        self.assertEqual(result, text)

    def test_apply_all_enabled(self):
        text = "Certainly! I think the risks are significant."
        result = self.fixer.apply(text)
        self.assertNotIn("Certainly", result)
        self.assertNotIn("I think", result)
        self.assertIn("risks are significant", result)

    def test_module_ids(self):
        ids = self.fixer.module_ids()
        self.assertIn('hedge_reducer', ids)
        self.assertIn('preamble_stripper', ids)
        self.assertIn('disclaimer_stripper', ids)
        self.assertIn('formal_reducer', ids)

    def test_enabled_module_ids(self):
        enabled = self.fixer.enabled_module_ids()
        self.assertIn('hedge_reducer', enabled)
        self.assertNotIn('formal_reducer', enabled)

    def test_real_planexe_output(self):
        """Test against realistic PlanExe pipeline output."""
        text = """Certainly! Here is a comprehensive analysis of the potential risks:

I think the most significant risk is budget overrun. It's important to note that similar projects have exceeded estimates by 30-50%. Perhaps the team should consider a phased approach.

Furthermore, the timeline assumes no regulatory delays. It's worth noting that FDA approval processes can take 12-18 months longer than planned.

(Note: This analysis is for informational purposes only and should not be considered professional advice. Please consult with a qualified project manager before proceeding.)"""

        result = self.fixer.apply(text)

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

    def setUp(self):
        self.fixer = _make_fixer()
        self.module = self.fixer.get_module('hedge_reducer')

    def test_counts_substitutions(self):
        self.module.reset_counts()
        self.module.transform("I think the plan is good. I think the budget is tight.")
        self.assertGreaterEqual(self.module.total_hits, 2)

    def test_tracks_by_rule_index(self):
        self.module.reset_counts()
        self.module.transform("I think perhaps we should reconsider.")
        self.assertGreaterEqual(len(self.module.hit_counts), 2)

    def test_reset_clears_counts(self):
        self.module.reset_counts()
        self.module.transform("I think the plan is solid.")
        self.assertGreater(self.module.total_hits, 0)
        self.module.reset_counts()
        self.assertEqual(self.module.total_hits, 0)
        self.assertEqual(len(self.module.hit_counts), 0)

    def test_no_match_leaves_text_unchanged(self):
        self.module.reset_counts()
        text = "The budget is $500,000."
        result = self.module.transform(text)
        self.assertEqual(result, text)

    def test_counts_accumulate(self):
        self.module.reset_counts()
        self.module.transform("I think the plan is good.")
        first_total = self.module.total_hits
        self.module.transform("I think the budget is tight.")
        self.assertGreater(self.module.total_hits, first_total)

    def test_reset_all_counts(self):
        self.module.transform("I think the plan is solid.")
        self.fixer.reset_all_counts()
        for m in self.fixer.all_modules():
            self.assertEqual(m.total_hits, 0)


class TestMultipleRuleFiles(unittest.TestCase):
    """Test loading multiple rule files."""

    def test_load_same_file_twice_overwrites(self):
        fixer = TextFixer()
        fixer.load(TextFixer.default_rules_path())
        count_first = len(fixer.module_ids())
        fixer.load(TextFixer.default_rules_path())
        # Same IDs, so count shouldn't change
        self.assertEqual(len(fixer.module_ids()), count_first)


if __name__ == '__main__':
    unittest.main()

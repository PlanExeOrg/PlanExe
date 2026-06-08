"""
Tests for identify_potential_levers.

These unit tests cover raise_if_no_levers_survived, the guard that turns a
silently-empty lever list into a loud, actionable error. The empty-lever case
is reproduced from a real incident: a plan explicitly about "AI agents" that
banned the word "AI", so the ConstraintChecker rejected every generated lever.

PROMPT> cd worker_plan && python -m pytest worker_plan_internal/lever/tests/test_identify_potential_levers.py -v
"""
import unittest

from worker_plan_internal.lever.identify_potential_levers import raise_if_no_levers_survived


def make_constraint_check(lever_name: str, violated_constraints: list[str]) -> dict:
    """Build one per-lever constraint-check result.

    Mirrors the structure ConstraintChecker writes into potential_levers_raw.json:
    every checked constraint gets an entry; the ones in violated_constraints are
    marked "violated", the rest "satisfied".
    """
    return {
        "lever_name": lever_name,
        "constraint_violations": [
            {
                "constraint_text": text,
                "constraint_classification": "negative",
                "status": "violated",
                "evidence": "the lever references the banned concept",
                "explanation": "the plan is inherently about this concept",
            }
            for text in violated_constraints
        ] + [
            {
                "constraint_text": "Do not use Blockchain",
                "constraint_classification": "negative",
                "status": "satisfied",
                "evidence": "no mention",
                "explanation": "respected",
            }
        ],
    }


class TestRaiseIfNoLeversSurvived(unittest.TestCase):
    def test_does_not_raise_when_levers_survived(self):
        """Happy path: at least one lever survived, so no error."""
        levers_cleaned = [{"lever_id": "abc", "name": "Some lever"}]
        # Even if some constraint checks reported violations, a non-empty
        # result must pass through untouched.
        all_constraint_checks = [make_constraint_check("Some lever", ["Do not use AI"])]
        # Should not raise.
        raise_if_no_levers_survived(levers_cleaned, all_constraint_checks)

    def test_raises_and_names_dominant_constraint(self):
        """Self-contradictory prompt: every lever rejected by 'Do not use AI'."""
        # 25 levers all rejected by the same banned word, plus a couple rejected
        # by a different one — mirroring the real incident's tallies.
        all_constraint_checks = [
            make_constraint_check(f"Lever {i}", ["Do not use AI"]) for i in range(25)
        ] + [
            make_constraint_check(f"Lever {i}", ["Do not use Robots"]) for i in range(25, 29)
        ]

        with self.assertRaises(ValueError) as ctx:
            raise_if_no_levers_survived([], all_constraint_checks)

        message = str(ctx.exception)
        # The dominant constraint must be named so the error is actionable.
        self.assertIn("Do not use AI", message)
        self.assertIn("rejected 25 lever(s)", message)
        # Reports how many levers were generated before filtering.
        self.assertIn("All 29 generated levers", message)
        # Points the user at the real fix.
        self.assertIn("negative constraint contradicts the plan", message)

    def test_dominant_constraint_is_the_most_frequent(self):
        """When several constraints reject levers, the most frequent leads."""
        all_constraint_checks = [
            make_constraint_check(f"Lever {i}", ["Do not use Robots"]) for i in range(3)
        ] + [
            make_constraint_check(f"Lever {i}", ["Do not use AI"]) for i in range(3, 13)
        ]

        with self.assertRaises(ValueError) as ctx:
            raise_if_no_levers_survived([], all_constraint_checks)

        message = str(ctx.exception)
        # "Do not use AI" (10) must appear before "Do not use Robots" (3).
        self.assertLess(message.index("Do not use AI"), message.index("Do not use Robots"))

    def test_raises_generic_error_without_violation_data(self):
        """Empty result with no constraint-check data still fails loud."""
        with self.assertRaises(ValueError) as ctx:
            raise_if_no_levers_survived([], [])
        self.assertIn("produced 0 levers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

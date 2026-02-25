from worker_plan_internal.assume.fermi_sanity_check import validate_quantified_assumptions
from worker_plan_internal.assume.quantified_assumptions import ConfidenceLevel, QuantifiedAssumption


def _build_assumption(**kwargs) -> QuantifiedAssumption:
    defaults = {
        "assumption_id": "test",
        "question": "What is the budget?",
        "claim": "Assumption: We will deliver 5,000,000 USD.",
        "lower_bound": 5_000_000.0,
        "upper_bound": 5_000_000.0,
        "unit": "usd",
        "confidence": ConfidenceLevel.high,
        "evidence": "Assumption: We will deliver 5,000,000 USD.",
        "extracted_numbers": [5_000_000.0],
        "raw_assumption": "Assumption: We will deliver 5,000,000 USD."
    }
    defaults.update(kwargs)
    return QuantifiedAssumption(**defaults)


def test_budget_passes_basic_checks():
    assumption = _build_assumption()
    report = validate_quantified_assumptions([assumption])
    assert report.passed == 1
    assert report.failed == 0
    assert report.total_assumptions == 1


def test_low_confidence_needs_evidence():
    assumption = _build_assumption(
        assumption_id="low-evidence",
        confidence=ConfidenceLevel.low,
        evidence="Low",
    )
    report = validate_quantified_assumptions([assumption])
    assert report.failed == 1
    assert any("Low confidence" in reason for reason in report.entries[0].reasons)


def test_span_ratio_detects_wide_boundaries():
    assumption = _build_assumption(
        assumption_id="wide-range",
        lower_bound=1.0,
        upper_bound=100_000.0,
        claim="Assumption: The project will cost 1 to 100,000 USD.",
        extracted_numbers=[1.0, 100_000.0]
    )
    report = validate_quantified_assumptions([assumption])
    assert any("Range spans" in reason for reason in report.entries[0].reasons)
    assert report.failed == 1

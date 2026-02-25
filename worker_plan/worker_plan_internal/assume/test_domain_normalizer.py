"""Unit tests for DomainNormalizer."""

from worker_plan_internal.assume.quantified_assumptions import (
    QuantifiedAssumption,
    ConfidenceLevel,
)
from worker_plan_internal.assume.domain_normalizer import (
    DomainNormalizer,
    DomainProfile,
)


def test_domain_normalizer_loads_default_profiles():
    """DomainNormalizer initializes with default profiles."""
    normalizer = DomainNormalizer(profiles_yaml_path="/nonexistent/path.yaml")
    assert normalizer.default_profile is not None
    assert normalizer.default_profile.id == "default"


def test_domain_profile_currency_detection():
    """DomainProfile correctly scores currency signals."""
    profile_dict = {
        "id": "carpenter",
        "name": "Carpenter",
        "currency": {"default": "DKK", "aliases": ["kr", "dkk"]},
        "units": {"metric": True, "convert": []},
        "heuristics": {"confidence_keywords": {}},
        "detection": {"currency_signals": ["DKK", "kr"], "unit_signals": [], "keyword_signals": []},
    }
    profile = DomainProfile(profile_dict)
    score = profile.score_match(["DKK"], [], [])
    assert score == 10  # DKK matches currency signal


def test_domain_profile_keyword_detection():
    """DomainProfile scores keyword signals."""
    profile_dict = {
        "id": "carpenter",
        "name": "Carpenter",
        "currency": {"default": "DKK"},
        "units": {"metric": True, "convert": []},
        "heuristics": {"confidence_keywords": {}},
        "detection": {
            "currency_signals": [],
            "unit_signals": [],
            "keyword_signals": ["carpenter", "wood", "materials"],
        },
    }
    profile = DomainProfile(profile_dict)
    score = profile.score_match([], [], ["carpenter", "wood"])
    assert score == 6  # Two keyword matches @ 3 points each


def test_domain_detection_carpenter():
    """Carpenter profile is detected from DKK + metric + material keywords."""
    normalizer = DomainNormalizer(profiles_yaml_path="/nonexistent/path.yaml")

    # Manually add carpenter profile
    carpenter_dict = {
        "id": "carpenter",
        "name": "Carpenter",
        "currency": {"default": "DKK", "aliases": ["kr"]},
        "units": {"metric": True, "convert": [{"from": "sqft", "to": "m2", "factor": 0.092903}]},
        "heuristics": {"confidence_keywords": {"high": ["I've done this"], "medium": [], "low": ["estimate"]}},
        "detection": {"currency_signals": ["DKK"], "unit_signals": ["m2"], "keyword_signals": ["carpenter"]},
    }
    normalizer.profiles["carpenter"] = DomainProfile(carpenter_dict)

    # Test detection
    assumption = QuantifiedAssumption(
        assumption_id="test1",
        question="Cost?",
        claim="Carpenter project in DKK costing 10000 to 15000 for materials in m2.",
        lower_bound=10000,
        upper_bound=15000,
        unit="DKK",
        confidence=ConfidenceLevel.medium,
        evidence="Quote from carpenter",
        extracted_numbers=[10000, 15000],
        raw_assumption="Cost estimate: 10000-15000 DKK",
    )

    domain = normalizer.detect_domain(assumption)
    assert domain.id == "carpenter"


def test_normalize_confidence_per_domain():
    """Confidence is re-assessed based on domain keywords."""
    normalizer = DomainNormalizer(profiles_yaml_path="/nonexistent/path.yaml")

    carpenter_dict = {
        "id": "carpenter",
        "name": "Carpenter",
        "currency": {"default": "DKK"},
        "units": {"metric": True, "convert": []},
        "heuristics": {"confidence_keywords": {"high": ["I've done this"], "medium": ["expect"], "low": ["estimate"]}},
        "detection": {"currency_signals": [], "unit_signals": [], "keyword_signals": []},
    }
    normalizer.profiles["carpenter"] = DomainProfile(carpenter_dict)

    # Low confidence claim with domain keyword
    assumption = QuantifiedAssumption(
        assumption_id="test2",
        question="Duration?",
        claim="Estimate 5 to 7 days.",
        lower_bound=5,
        upper_bound=7,
        unit="days",
        confidence=ConfidenceLevel.low,
        evidence="Rough estimate",
        extracted_numbers=[5, 7],
        raw_assumption="Duration: 5-7 days (estimate)",
    )

    normalized = normalizer.normalize(assumption)
    # Since "estimate" is in low_confidence_words, should stay low
    assert normalized.confidence == ConfidenceLevel.low


def test_unit_conversion():
    """Units are converted to metric."""
    normalizer = DomainNormalizer(profiles_yaml_path="/nonexistent/path.yaml")

    carpenter_dict = {
        "id": "carpenter",
        "name": "Carpenter",
        "currency": {"default": "DKK"},
        "units": {"metric": True, "convert": [{"from": "sqft", "to": "m2", "factor": 0.092903}]},
        "heuristics": {"confidence_keywords": {}},
        "detection": {"currency_signals": [], "unit_signals": [], "keyword_signals": []},
    }
    profile = DomainProfile(carpenter_dict)

    # Convert 100 sqft to m2
    result = normalizer.normalize_unit(100, "sqft", profile)
    assert abs(result - 9.2903) < 0.001


def test_currency_normalization():
    """Currency converts to profile default."""
    normalizer = DomainNormalizer(profiles_yaml_path="/nonexistent/path.yaml")

    carpenter_dict = {
        "id": "carpenter",
        "name": "Carpenter",
        "currency": {"default": "DKK"},
        "units": {"metric": True, "convert": []},
        "heuristics": {"confidence_keywords": {}},
        "detection": {"currency_signals": [], "unit_signals": [], "keyword_signals": []},
    }
    profile = DomainProfile(carpenter_dict)

    norm_val, eur_equiv = normalizer.normalize_currency(10000, "DKK", profile)
    assert norm_val == 10000  # DKK stays as-is
    assert eur_equiv is not None  # EUR equivalent calculated


def test_batch_normalization():
    """Batch normalization processes multiple assumptions."""
    normalizer = DomainNormalizer(profiles_yaml_path="/nonexistent/path.yaml")

    assumptions = [
        QuantifiedAssumption(
            assumption_id="a1",
            question="Q1",
            claim="Budget 5000 to 7000.",
            lower_bound=5000,
            upper_bound=7000,
            unit="USD",
            confidence=ConfidenceLevel.high,
            evidence="Approved",
            extracted_numbers=[5000, 7000],
            raw_assumption="Assumption: 5000-7000",
        ),
        QuantifiedAssumption(
            assumption_id="a2",
            question="Q2",
            claim="Timeline 10 to 14 days.",
            lower_bound=10,
            upper_bound=14,
            unit="days",
            confidence=ConfidenceLevel.medium,
            evidence="Estimate",
            extracted_numbers=[10, 14],
            raw_assumption="Assumption: 10-14 days",
        ),
    ]

    normalized = normalizer.normalize_batch(assumptions)
    assert len(normalized) == 2
    assert normalized[0].assumption_id == "a1"
    assert normalized[1].assumption_id == "a2"

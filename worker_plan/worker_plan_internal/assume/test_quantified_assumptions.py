from worker_plan_internal.assume.quantified_assumptions import (
    ConfidenceLevel,
    QuantifiedAssumptionExtractor,
)


def test_extract_range_and_unit():
    extractor = QuantifiedAssumptionExtractor()
    entries = [
        {
            "question": "What capacity?",
            "assumptions": "Assumption: The solar farm will deliver 50-60 MW of capacity before year two.",
        }
    ]
    assumption = extractor.extract(entries)[0]
    assert assumption.lower_bound == 50.0
    assert assumption.upper_bound == 60.0
    assert assumption.unit == "mw"
    assert assumption.extracted_numbers == [50.0, 60.0]


def test_confidence_detection_handles_low_words():
    extractor = QuantifiedAssumptionExtractor()
    entries = [
        {
            "question": "Timeline",
            "assumptions": "Assumption: We expect roughly 8 months of construction, though delays are possible.",
        }
    ]
    assumption = extractor.extract(entries)[0]
    assert assumption.confidence == ConfidenceLevel.low


def test_extract_handles_missing_numbers():
    extractor = QuantifiedAssumptionExtractor()
    entries = [
        {
            "question": "Safety",
            "assumptions": "Assumption: Construction will follow all standards, no explicit numbers provided.",
        }
    ]
    assumption = extractor.extract(entries)[0]
    assert assumption.lower_bound is None
    assert assumption.upper_bound is None
    assert assumption.extracted_numbers == []

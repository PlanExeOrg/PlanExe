import pytest
from worker_plan_internal.utils.json_repair import repaired_json_str
from pydantic import ValidationError
from worker_plan_internal.lever.enrich_potential_levers import BatchCharacterizationResult

def test_schema_invalid_still_fails():
    invalid = '{"characterizations": [{"lever_id": "x", "synergy_text": "syn", "conflict_text": "conf"}]}'
    repaired = repaired_json_str(invalid)
    with pytest.raises(ValidationError):
        BatchCharacterizationResult.model_validate_json(repaired)


def test_malformed_not_repairable():
    raw = '{ NOT JSON'
    with pytest.raises(ValueError):
        repaired_json_str(raw)

if __name__ == "__main__":
    pytest.main([__file__])

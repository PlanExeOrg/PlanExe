import json

from worker_plan_internal.lever.enrich_potential_levers import normalize_characterizations_json


def make_characterization_payload():
    return {
        "lever_id": "alpha",
        "description": "desc",
        "synergy_text": "syn",
        "conflict_text": "conf"
    }


def test_normalize_alias_levers_to_characterizations():
    alias_payload = {
        "levers": [make_characterization_payload()],
        "metadata": {"note": "alias format"}
    }

    normalized = normalize_characterizations_json(json.dumps(alias_payload))
    parsed = json.loads(normalized)

    assert "characterizations" in parsed
    assert "levers" not in parsed
    assert parsed["characterizations"][0]["lever_id"] == "alpha"
    assert parsed["metadata"]["note"] == "alias format"


def test_normalize_leaves_valid_json_untouched():
    payload = {
        "characterizations": [make_characterization_payload()],
        "metadata": {"note": "already correct"}
    }
    normalized = normalize_characterizations_json(json.dumps(payload))
    parsed = json.loads(normalized)
    assert "characterizations" in parsed
    assert parsed["metadata"]["note"] == "already correct"


def test_normalize_characterizations_renames_id_field():
    payload = {
        "characterizations": [
            {
                "id": "alpha",
                "description": "desc",
                "synergy_text": "syn",
                "conflict_text": "conf"
            }
        ]
    }

    normalized = normalize_characterizations_json(json.dumps(payload))
    parsed = json.loads(normalized)
    entry = parsed["characterizations"][0]
    assert entry["lever_id"] == "alpha"
    assert "id" not in entry

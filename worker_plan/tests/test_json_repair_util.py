"""
Tests for the json_repair_util module.

Exercises JSON extraction and repair on realistic malformed LLM outputs.
"""
import pytest
from worker_plan_internal.llm_util.json_repair_util import extract_json_from_response, repair_json


class TestExtractJsonFromResponse:
    """Tests for extracting JSON from LLM responses with preamble."""

    def test_clean_json_object(self):
        text = '{"key": "value"}'
        assert extract_json_from_response(text) == '{"key": "value"}'

    def test_json_in_markdown_fence(self):
        text = 'Here is the result:\n```json\n{"levers": [1, 2, 3]}\n```'
        assert extract_json_from_response(text) == '{"levers": [1, 2, 3]}'

    def test_json_with_thinking_preamble(self):
        text = '<think>Let me analyze these levers carefully...</think>\n{"lever_id": "abc", "name": "Test"}'
        result = extract_json_from_response(text)
        assert result is not None
        assert '"lever_id"' in result

    def test_json_array(self):
        text = 'The options are:\n["option1", "option2", "option3"]'
        result = extract_json_from_response(text)
        assert result is not None
        assert 'option1' in result

    def test_no_json_content(self):
        text = 'This is just plain text with no JSON at all.'
        assert extract_json_from_response(text) is None

    def test_json_in_plain_code_block(self):
        text = 'Result:\n```\n{"name": "test"}\n```\nDone.'
        assert extract_json_from_response(text) == '{"name": "test"}'


class TestRepairJson:
    """Tests for repairing malformed JSON from LLM responses."""

    def test_valid_json_passes_through(self):
        result = repair_json('{"key": "value", "count": 42}')
        assert result == {"key": "value", "count": 42}

    def test_trailing_comma_in_object(self):
        result = repair_json('{"key": "value", "count": 42,}')
        assert result["key"] == "value"
        assert result["count"] == 42

    def test_trailing_comma_in_array(self):
        result = repair_json('["a", "b", "c",]')
        assert result == ["a", "b", "c"]

    def test_thinking_preamble_then_json(self):
        text = "Let me think about this...\nThe levers are:\n" + '{"levers": [{"name": "test"}]}'
        result = repair_json(text)
        assert "levers" in result
        assert result["levers"][0]["name"] == "test"

    def test_single_quotes(self):
        result = repair_json("{'key': 'value'}")
        assert result["key"] == "value"

    def test_completely_invalid_raises(self):
        with pytest.raises(ValueError, match="Could not parse or repair"):
            repair_json("This is not JSON at all and has no brackets")

    def test_nested_json_with_trailing_commas(self):
        text = '{"outer": {"inner": "value",}, "list": [1, 2,],}'
        result = repair_json(text)
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2]

    def test_markdown_fenced_json_with_errors(self):
        text = 'Here is the output:\n```json\n{"name": "test", "options": ["a", "b",]}\n```'
        result = repair_json(text)
        assert result["name"] == "test"

    def test_lever_like_structure(self):
        """Test with a realistic lever-like JSON structure."""
        text = '''
        {"characterizations": [
            {
                "lever_id": "abc-123",
                "description": "A test lever for evaluation",
                "synergy_text": "Works well with other levers",
                "conflict_text": "May conflict with budget constraints"
            }
        ]}
        '''
        result = repair_json(text)
        assert len(result["characterizations"]) == 1
        assert result["characterizations"][0]["lever_id"] == "abc-123"

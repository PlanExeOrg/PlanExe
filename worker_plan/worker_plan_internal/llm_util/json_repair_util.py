"""
Utility for repairing malformed JSON from LLM responses.

Local LLMs sometimes return JSON with issues like:
- Thinking preamble before the actual JSON
- Trailing commas
- Unclosed brackets
- Single quotes instead of double quotes

This module provides a fallback parser that attempts to extract and repair
valid JSON from messy LLM output.
"""
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Extract JSON content from an LLM response that may contain
    non-JSON preamble (e.g., thinking text, markdown fences).
    
    Returns the extracted JSON string, or None if no JSON found.
    """
    # Try to find JSON in markdown code blocks first
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Try to find a JSON object or array
    # Look for the first { or [ and match to the last } or ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        end_idx = text.rfind(end_char)
        if end_idx == -1 or end_idx <= start_idx:
            continue
        return text[start_idx:end_idx + 1]
    
    return None


def repair_json(text: str) -> Any:
    """
    Attempt to parse JSON from potentially malformed LLM output.
    
    Strategy:
    1. Try standard json.loads first
    2. Extract JSON from response (strip preamble/markdown)
    3. Use json_repair library for structural fixes
    
    Returns the parsed JSON object.
    Raises ValueError if JSON cannot be repaired.
    """
    # Step 1: Try direct parsing
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Step 2: Try extracting JSON from response
    extracted = extract_json_from_response(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    
    # Step 3: Use json_repair library
    try:
        from json_repair import repair_json as _repair_json
        text_to_repair = extracted if extracted else text
        repaired = _repair_json(text_to_repair, return_objects=True)
        if repaired is not None:
            logger.info("Successfully repaired malformed JSON using json_repair")
            return repaired
    except Exception as e:
        logger.warning(f"json_repair failed: {e}")
    
    raise ValueError(f"Could not parse or repair JSON from LLM response (length={len(text)})")

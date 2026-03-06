import json
import re

try:
    import dirtyjson
except ImportError:  # dirtyjson may not be installed yet
    dirtyjson = None

MATCH_JSON = re.compile(r"(\[.*\]|\{.*\})", flags=re.DOTALL)
TRAILING_COMMA_PATTERN = re.compile(r',\s*([}\]])')


def cleanup_json_text(raw_text: str) -> str:
    if not raw_text:
        return raw_text

    match_list = re.search(r'\[.*\]', raw_text, re.DOTALL)
    match_obj = re.search(r'\{.*\}', raw_text, re.DOTALL)

    if match_list and (not match_obj or match_list.start() < match_obj.start()):
        raw_text = match_list.group()
    elif match_obj:
        raw_text = match_obj.group()

    raw_text = re.sub(r'^```json\s*', '', raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r'```$', '', raw_text, flags=re.MULTILINE)
    raw_text = raw_text.strip()

    while True:
        modified = TRAILING_COMMA_PATTERN.sub(r'\1', raw_text)
        if modified == raw_text:
            break
        raw_text = modified

    return raw_text


def parse_tolerant_json(raw_text: str) -> tuple[str, dict]:
    cleaned = cleanup_json_text(raw_text)
    last_error = None
    try:
        parsed = json.loads(cleaned)
        return cleaned, parsed
    except json.JSONDecodeError as exc:
        last_error = exc
    if dirtyjson:
        try:
            parsed = dirtyjson.loads(cleaned)
            return cleaned, parsed
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse JSON: {last_error}")


def repaired_json_str(raw_text: str) -> str:
    cleaned, parsed = parse_tolerant_json(raw_text)
    return json.dumps(parsed)

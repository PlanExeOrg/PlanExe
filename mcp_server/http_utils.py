from typing import Any


def strip_redundant_content(payload: Any) -> tuple[Any, bool]:
    """Remove top-level content when structuredContent is present."""
    if not isinstance(payload, dict):
        return payload, False
    if "structuredContent" not in payload or "content" not in payload:
        return payload, False
    stripped = dict(payload)
    stripped.pop("content", None)
    return stripped, True

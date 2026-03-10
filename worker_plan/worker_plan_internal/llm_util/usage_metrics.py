"""
File-based usage metrics for local runs.

Records per-LLM-call metrics (model, tokens, duration, success/failure) to a JSONL file
in the run output directory. Works without a database — designed for local/offline runs.

Usage:
    from worker_plan_internal.llm_util.usage_metrics import set_usage_metrics_path, record_usage_metric

    # Set once at pipeline start
    set_usage_metrics_path(run_id_dir / ExtraFilenameEnum.USAGE_METRICS_JSONL.value)

    # Called automatically by LLMExecutor._record_attempt_token_metrics()
    record_usage_metric(model="gpt-4", duration=1.23, success=True, input_tokens=100, output_tokens=50)

    # Clear after pipeline completes to avoid stale state
    set_usage_metrics_path(None)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_usage_metrics_path: Optional[Path] = None

# Ordered list: first match wins, so more specific patterns come before broad ones.
_ERROR_CATEGORIES: list[tuple[str, list[str]]] = [
    ("rate_limit", ["rate limit", "429", "too many requests"]),
    ("auth_error", ["unauthorized", "forbidden", "auth", "401", "403"]),
    ("server_error", ["500", "502", "503", "504", "server error", "internal server"]),
    ("timeout", ["timeout", "timed out"]),
    ("empty_response", ["empty", "no response", "none"]),
    ("connection_error", ["connection", "connect", "dns", "network"]),
    ("invalid_json", ["json", "validation error", "pydantic"]),
    ("model_not_found", ["model not found", "not found", "404"]),
]


def classify_error(error_message: str) -> str:
    """Map a raw exception message to a short category string.

    Uses case-insensitive keyword matching against *_ERROR_CATEGORIES*.
    Returns ``"unknown"`` if no pattern matches.
    """
    lowered = error_message.lower()
    for category, keywords in _ERROR_CATEGORIES:
        for keyword in keywords:
            if keyword in lowered:
                return category
    return "unknown"


def set_usage_metrics_path(path: Optional[Path]) -> None:
    """Set the JSONL file path for recording usage metrics."""
    global _usage_metrics_path
    _usage_metrics_path = path


def get_usage_metrics_path() -> Optional[Path]:
    """Get the current JSONL file path for recording usage metrics."""
    return _usage_metrics_path


def record_usage_metric(
    model: str,
    duration_seconds: float,
    success: bool,
    error_message: Optional[str] = None,
    error_id: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    thinking_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> None:
    """Append a single usage metric record to the JSONL file.

    Best-effort: never raises exceptions to avoid blocking the LLM pipeline.
    """
    path = _usage_metrics_path
    if path is None:
        logger.warning("record_usage_metric called but no usage metrics path is set")
        return

    record = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "model": model,
        "duration_seconds": round(duration_seconds, 3),
    }
    if error_message:
        category = classify_error(error_message)
        record["error"] = category
        if category == "unknown":
            record["error_detail"] = error_message[:200]
        if error_id:
            record["error_id"] = error_id
    if input_tokens is not None:
        record["input_tokens"] = input_tokens
    if output_tokens is not None:
        record["output_tokens"] = output_tokens
    if thinking_tokens is not None:
        record["thinking_tokens"] = thinking_tokens
    if cost_usd is not None:
        record["cost_usd"] = cost_usd

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write usage metric: %s", exc)

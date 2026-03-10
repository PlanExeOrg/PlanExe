"""Tests for usage_metrics classify_error() and record_usage_metric()."""

import json
from pathlib import Path

import pytest

from worker_plan_internal.llm_util.usage_metrics import (
    classify_error,
    record_usage_metric,
    set_usage_metrics_path,
)


# ---------- classify_error ----------

@pytest.mark.parametrize(
    "error_message, expected_category",
    [
        # invalid_json
        ("Expecting value: line 1 column 1 (char 0) json decode", "invalid_json"),
        ("validation error for MyModel", "invalid_json"),
        ("pydantic.error_wrappers.ValidationError", "invalid_json"),
        # timeout
        ("Request timeout after 30s", "timeout"),
        ("Connection timed out", "timeout"),
        # empty_response
        ("Empty response from model", "empty_response"),
        ("Response was None", "empty_response"),
        ("Got no response from server", "empty_response"),
        # connection_error
        ("ConnectionRefusedError: [Errno 111]", "connection_error"),
        ("Failed to connect to host", "connection_error"),
        ("DNS resolution failed", "connection_error"),
        ("Network is unreachable", "connection_error"),
        # rate_limit
        ("Rate limit exceeded", "rate_limit"),
        ("Error code: 429", "rate_limit"),
        ("Too many requests, please slow down", "rate_limit"),
        # auth_error
        ("401 Unauthorized", "auth_error"),
        ("403 Forbidden", "auth_error"),
        ("Authentication failed", "auth_error"),
        # server_error
        ("500 Internal Server Error", "server_error"),
        ("502 Bad Gateway", "server_error"),
        ("503 Service Unavailable", "server_error"),
        ("504 Gateway Timeout", "server_error"),
        ("Internal server error occurred", "server_error"),
        # model_not_found
        ("Model not found: gpt-5-turbo", "model_not_found"),
        ("404 Not Found", "model_not_found"),
        # unknown
        ("Something completely unexpected happened", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_error(error_message: str, expected_category: str) -> None:
    assert classify_error(error_message) == expected_category


def test_classify_error_with_llm_chat_error_str() -> None:
    """classify_error should see through LLMChatError wrapping to the root cause."""
    from worker_plan_internal.llm_util.llm_errors import LLMChatError
    cause = ValueError("1 validation error for MyModel")
    err = LLMChatError(cause=cause)
    assert classify_error(str(err)) == "invalid_json"


def test_classify_error_case_insensitive() -> None:
    assert classify_error("RATE LIMIT exceeded") == "rate_limit"
    assert classify_error("Pydantic ValidationError") == "invalid_json"
    assert classify_error("TIMEOUT waiting for response") == "timeout"


# ---------- record_usage_metric with classification ----------

def test_record_usage_metric_writes_classified_error(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="gpt-4",
            duration_seconds=1.5,
            success=False,
            error_message="pydantic validation error: blah blah long message",
        )

        lines = metrics_file.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["error"] == "invalid_json"
        assert record["success"] is False
    finally:
        set_usage_metrics_path(None)


def test_record_usage_metric_no_error_field_when_none(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="gpt-4",
            duration_seconds=0.5,
            success=True,
        )

        record = json.loads(metrics_file.read_text().strip())
        assert "error" not in record
        assert "error_detail" not in record
    finally:
        set_usage_metrics_path(None)


def test_record_usage_metric_unknown_includes_error_detail(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="openrouter-model",
            duration_seconds=19.6,
            success=False,
            error_message="Something completely unexpected happened",
        )

        record = json.loads(metrics_file.read_text().strip())
        assert record["error"] == "unknown"
        assert record["error_detail"] == "Something completely unexpected happened"
    finally:
        set_usage_metrics_path(None)


def test_record_usage_metric_known_category_no_error_detail(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="gpt-4",
            duration_seconds=1.0,
            success=False,
            error_message="Connection refused",
        )

        record = json.loads(metrics_file.read_text().strip())
        assert record["error"] == "connection_error"
        assert "error_detail" not in record
    finally:
        set_usage_metrics_path(None)


def test_record_usage_metric_unknown_truncates_long_error(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        long_msg = "x" * 500
        record_usage_metric(
            model="gpt-4",
            duration_seconds=1.0,
            success=False,
            error_message=long_msg,
        )

        record = json.loads(metrics_file.read_text().strip())
        assert record["error"] == "unknown"
        assert len(record["error_detail"]) == 200
    finally:
        set_usage_metrics_path(None)


# ---------- error_id ----------

def test_record_usage_metric_includes_error_id(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="gpt-4",
            duration_seconds=1.0,
            success=False,
            error_message="Connection refused",
            error_id="abc123def456",
        )

        record = json.loads(metrics_file.read_text().strip())
        assert record["error"] == "connection_error"
        assert record["error_id"] == "abc123def456"
    finally:
        set_usage_metrics_path(None)


def test_record_usage_metric_no_error_id_when_not_provided(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="gpt-4",
            duration_seconds=1.0,
            success=False,
            error_message="Connection refused",
        )

        record = json.loads(metrics_file.read_text().strip())
        assert "error_id" not in record
    finally:
        set_usage_metrics_path(None)


def test_record_usage_metric_no_error_id_on_success(tmp_path: Path) -> None:
    metrics_file = tmp_path / "usage_metrics.jsonl"
    set_usage_metrics_path(metrics_file)
    try:
        record_usage_metric(
            model="gpt-4",
            duration_seconds=0.5,
            success=True,
            error_id="should_not_appear",
        )

        record = json.loads(metrics_file.read_text().strip())
        assert "error_id" not in record
    finally:
        set_usage_metrics_path(None)

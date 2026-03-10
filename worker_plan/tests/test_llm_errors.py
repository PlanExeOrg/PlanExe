"""Tests for LLMChatError."""

from worker_plan_internal.llm_util.llm_errors import LLMChatError


def test_llm_chat_error_carries_cause() -> None:
    cause = RuntimeError("connection refused")
    err = LLMChatError(cause=cause)
    assert err.cause is cause


def test_llm_chat_error_generates_error_id() -> None:
    err = LLMChatError(cause=RuntimeError("x"))
    assert isinstance(err.error_id, str)
    assert len(err.error_id) == 12


def test_llm_chat_error_accepts_custom_error_id() -> None:
    err = LLMChatError(cause=RuntimeError("x"), error_id="abc123")
    assert err.error_id == "abc123"


def test_llm_chat_error_str_contains_cause_and_id() -> None:
    cause = ValueError("invalid json response")
    err = LLMChatError(cause=cause, error_id="deadbeef1234")
    s = str(err)
    assert "deadbeef1234" in s
    assert "invalid json response" in s
    assert "LLM chat interaction failed" in s


def test_llm_chat_error_is_value_error() -> None:
    """LLMChatError extends ValueError for backward compatibility."""
    err = LLMChatError(cause=RuntimeError("x"))
    assert isinstance(err, ValueError)


def test_llm_chat_error_unique_ids() -> None:
    err1 = LLMChatError(cause=RuntimeError("a"))
    err2 = LLMChatError(cause=RuntimeError("b"))
    assert err1.error_id != err2.error_id


def test_llm_chat_error_default_message() -> None:
    err = LLMChatError(cause=RuntimeError("x"))
    assert err.message == "LLM chat interaction failed"
    assert str(err).startswith("LLM chat interaction failed [")


def test_llm_chat_error_custom_message() -> None:
    err = LLMChatError(cause=RuntimeError("timeout"), message="LLM chat interaction 2 failed")
    assert err.message == "LLM chat interaction 2 failed"
    assert str(err).startswith("LLM chat interaction 2 failed [")
    assert "timeout" in str(err)

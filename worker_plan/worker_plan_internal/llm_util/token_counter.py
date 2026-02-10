"""
Extract and count tokens from LLM provider responses.

Supports multiple provider types including OpenAI, OpenRouter, Ollama, and others.
Extracts input_tokens, output_tokens, and thinking_tokens when available.
"""
import logging
from typing import Optional, Any, Dict
from llama_index.core.llms.llm import ChatResponse

logger = logging.getLogger(__name__)

__all__ = ["TokenCount", "extract_token_count"]


class TokenCount:
    """Container for token count information from an LLM response."""

    def __init__(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        raw_usage_data: Optional[Dict[str, Any]] = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens
        self.raw_usage_data = raw_usage_data or {}

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens."""
        return (self.input_tokens or 0) + (self.output_tokens or 0) + (self.thinking_tokens or 0)

    def __repr__(self) -> str:
        return (
            f"TokenCount(input={self.input_tokens}, output={self.output_tokens}, "
            f"thinking={self.thinking_tokens}, total={self.total_tokens})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "raw_usage_data": self.raw_usage_data,
        }


def extract_token_count(response: Any) -> TokenCount:
    """
    Extract token counts from an LLM response.

    Handles multiple response types and providers:
    - llama_index ChatResponse (most common)
    - OpenAI usage objects
    - OpenRouter responses
    - Anthropic responses with cache_usage
    - Generic responses with usage attribute

    Args:
        response: The response object from an LLM call.

    Returns:
        TokenCount object with extracted token information.
    """
    if response is None:
        return TokenCount()

    raw_usage_data = {}
    input_tokens = None
    output_tokens = None
    thinking_tokens = None

    try:
        # Handle llama_index ChatResponse
        if isinstance(response, ChatResponse):
            return _extract_from_chat_response(response)

        # Handle direct usage object (from some OpenAI-like calls)
        if hasattr(response, "usage"):
            return _extract_from_usage_object(response.usage)

        # Handle dict responses (e.g., from structured output)
        if isinstance(response, dict):
            return _extract_from_dict(response)

        # Fallback: try to extract common attributes
        if hasattr(response, "get"):
            # Dict-like interface
            input_tokens = response.get("input_tokens") or response.get("prompt_tokens")
            output_tokens = response.get("output_tokens") or response.get("completion_tokens")
            thinking_tokens = response.get("thinking_tokens") or response.get("cache_creation_input_tokens")

        logger.debug(f"Extracted token counts from response: input={input_tokens}, output={output_tokens}, thinking={thinking_tokens}")

    except Exception as e:
        logger.warning(f"Error extracting token counts from response: {e}")

    return TokenCount(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        raw_usage_data=raw_usage_data,
    )


def _extract_from_chat_response(response: ChatResponse) -> TokenCount:
    """Extract from llama_index ChatResponse."""
    input_tokens = None
    output_tokens = None
    thinking_tokens = None
    raw_usage_data = {}

    # Try to get usage from response object
    if hasattr(response, "raw"):
        raw = response.raw
        if isinstance(raw, dict) and "usage" in raw:
            usage = raw["usage"]
            raw_usage_data = usage.copy() if isinstance(usage, dict) else {}
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            thinking_tokens = usage.get("reasoning_tokens") or usage.get("thinking_tokens")

    # Also check message for usage info
    if hasattr(response, "message") and hasattr(response.message, "usage"):
        usage = response.message.usage
        if hasattr(usage, "prompt_tokens"):
            input_tokens = input_tokens or usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            output_tokens = output_tokens or usage.completion_tokens

    return TokenCount(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        raw_usage_data=raw_usage_data,
    )


def _extract_from_usage_object(usage: Any) -> TokenCount:
    """Extract from a usage object (OpenAI style)."""
    input_tokens = None
    output_tokens = None
    thinking_tokens = None
    raw_usage_data = {}

    try:
        if hasattr(usage, "prompt_tokens"):
            input_tokens = usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            output_tokens = usage.completion_tokens
        if hasattr(usage, "reasoning_tokens"):
            thinking_tokens = usage.reasoning_tokens
        if hasattr(usage, "cache_creation_input_tokens"):
            # Anthropic cache tokens
            thinking_tokens = thinking_tokens or usage.cache_creation_input_tokens

        # Capture raw data
        if hasattr(usage, "__dict__"):
            raw_usage_data = usage.__dict__.copy()
        elif isinstance(usage, dict):
            raw_usage_data = usage.copy()

    except Exception as e:
        logger.debug(f"Error extracting from usage object: {e}")

    return TokenCount(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        raw_usage_data=raw_usage_data,
    )


def _extract_from_dict(response: dict) -> TokenCount:
    """Extract from a dictionary response."""
    input_tokens = None
    output_tokens = None
    thinking_tokens = None

    # Check for usage key
    usage = response.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        thinking_tokens = usage.get("reasoning_tokens") or usage.get("thinking_tokens")
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            raw_usage_data=usage.copy(),
        )

    # Direct keys
    input_tokens = response.get("prompt_tokens") or response.get("input_tokens")
    output_tokens = response.get("completion_tokens") or response.get("output_tokens")
    thinking_tokens = response.get("reasoning_tokens") or response.get("thinking_tokens")

    return TokenCount(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        raw_usage_data=response.copy() if input_tokens or output_tokens or thinking_tokens else {},
    )

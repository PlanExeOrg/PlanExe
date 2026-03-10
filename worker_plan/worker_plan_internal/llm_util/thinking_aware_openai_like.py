"""
ThinkingAwareOpenAILike: OpenAILike subclass that handles thinking tokens gracefully.

When thinking tokens are enabled in LM Studio, models like Qwen put ALL output into
`reasoning_content` and return `content: None`. The parent OpenAILike class only reads
`content` via `from_openai_message()`, so downstream code gets None and crashes.

This subclass overrides `chat()` to check the raw OpenAI response for
`reasoning_content` when `content` is None/empty, and swaps it into the
ChatResponse message so downstream code works unchanged.

When reasoning_content contains a thinking preamble followed by the final JSON answer,
we extract the JSON from the end of the string rather than returning the full thinking
text (which would fail model_validate_json).
"""

import json
import logging
from typing import Any, Sequence
from llama_index.llms.openai_like import OpenAILike

try:
    from llama_index.core.llms.types import ChatResponse, CompletionResponse
except ImportError:
    from llama_index.core.base.llms.types import ChatResponse, CompletionResponse

from llama_index.core.llms import ChatMessage

logger = logging.getLogger(__name__)


def _extract_json_from_thinking(text: str) -> str:
    """
    Try to extract a JSON object from text that may contain a thinking preamble.

    Strategy:
    1. Direct parse (handles short responses where reasoning_content IS the JSON).
    2. Look for </think> tag — content after it is the final answer.
    3. Scan right-to-left for each '{' occurrence and try to parse
       from that position to end-of-string.

    Returns the extracted JSON string, or the original text if nothing works.
    """
    # 1. Direct parse — fast path for short responses
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. </think> tag (some models emit this to separate thinking from answer)
    think_end = text.rfind('</think>')
    if think_end != -1:
        after_think = text[think_end + len('</think>'):].strip()
        if after_think:
            try:
                json.loads(after_think)
                logger.debug("_extract_json_from_thinking: extracted via </think> tag")
                return after_think
            except (json.JSONDecodeError, ValueError):
                pass

    # 3. Scan right-to-left for '{' that leads to a valid JSON parse
    pos = len(text)
    while True:
        pos = text.rfind('{', 0, pos)
        if pos == -1:
            break
        candidate = text[pos:]
        try:
            json.loads(candidate)
            logger.debug(
                "_extract_json_from_thinking: extracted JSON from pos %d (text len %d)",
                pos, len(text)
            )
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass
        pos -= 1  # try the next '{' to the left

    # Nothing worked — return original text (caller will handle the failure)
    logger.warning(
        "_extract_json_from_thinking: could not extract JSON from %d-char text", len(text)
    )
    return text


class ThinkingAwareOpenAILike(OpenAILike):
    """
    OpenAILike subclass that falls back to reasoning_content when content is None.

    Usage in llm_config/<profile>.json:
    {
        "class": "ThinkingAwareOpenAILike",
        "arguments": {
            "model": "qwen/qwen3.5-9b",
            "api_base": "http://127.0.0.1:1234/v1",
            ...
        }
    }
    """

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Chat with thinking token fallback.

        After the parent produces a ChatResponse, checks if content is empty.
        If so, reads reasoning_content from the raw OpenAI response and
        replaces the message content with the extracted JSON (or full text if
        JSON extraction fails).
        """
        response = super().chat(messages, **kwargs)

        # Check if content is None or empty
        content = response.message.content
        if not content or content.strip() == "":
            # Access the raw OpenAI response stored in ChatResponse.raw
            raw = getattr(response, 'raw', None)
            if raw and hasattr(raw, 'choices') and raw.choices:
                raw_message = raw.choices[0].message
                reasoning_content = getattr(raw_message, 'reasoning_content', None)
                if reasoning_content:
                    logger.info(
                        "ThinkingAwareOpenAILike: content was None/empty, "
                        "falling back to reasoning_content (%d chars)",
                        len(reasoning_content)
                    )
                    # Extract JSON from the reasoning_content (may contain thinking preamble)
                    extracted = _extract_json_from_thinking(reasoning_content)
                    if extracted != reasoning_content:
                        logger.info(
                            "ThinkingAwareOpenAILike: extracted JSON (%d chars) from "
                            "reasoning_content (%d chars)",
                            len(extracted), len(reasoning_content)
                        )

                    from llama_index.core.llms import ChatMessage as CM
                    from llama_index.core.base.llms.types import TextBlock

                    new_message = CM(
                        role=response.message.role,
                        blocks=[TextBlock(text=extracted)],
                        additional_kwargs=response.message.additional_kwargs,
                    )
                    response.message = new_message
                else:
                    logger.warning(
                        "ThinkingAwareOpenAILike: content is None/empty and "
                        "no reasoning_content found in raw response."
                    )

        return response

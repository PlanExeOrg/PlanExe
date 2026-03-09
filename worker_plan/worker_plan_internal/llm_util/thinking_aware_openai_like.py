"""
ThinkingAwareOpenAILike: OpenAILike subclass that handles thinking tokens gracefully.

When thinking tokens are enabled in LM Studio, models like Qwen put ALL output into
`reasoning_content` and return `content: None`. The parent OpenAILike class only reads
`content` via `from_openai_message()`, so downstream code gets None and crashes.

This subclass overrides `chat()` to check the raw OpenAI response for
`reasoning_content` when `content` is None/empty, and swaps it into the
ChatResponse message so downstream code works unchanged.
"""

import logging
from typing import Any, Sequence
from llama_index.llms.openai_like import OpenAILike

try:
    from llama_index.core.llms.types import ChatResponse, CompletionResponse
except ImportError:
    from llama_index.core.base.llms.types import ChatResponse, CompletionResponse

from llama_index.core.llms import ChatMessage

logger = logging.getLogger(__name__)


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
        replaces the message content with it.
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
                    # Replace the message content with reasoning_content
                    from llama_index.core.llms import ChatMessage as CM
                    from llama_index.core.base.llms.types import TextBlock
                    
                    new_message = CM(
                        role=response.message.role,
                        blocks=[TextBlock(text=reasoning_content)],
                        additional_kwargs=response.message.additional_kwargs,
                    )
                    response.message = new_message
                else:
                    logger.warning(
                        "ThinkingAwareOpenAILike: content is None/empty and "
                        "no reasoning_content found in raw response."
                    )
        
        return response

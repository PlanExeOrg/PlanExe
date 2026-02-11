"""
Usage:
python -m worker_plan_internal.llm_util.track_activity

IDEA: TrackActivity only tracks when the LLM succeeds, but not when it fails.
I don’t have any interception of the response, so the real reason why it failed is speculation. I have no evidence.
TrackActivity, it would be awesome if it could track whenever the LLM failed and why.
"""
import json
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatStartEvent, LLMChatEndEvent, LLMCompletionStartEvent, LLMCompletionEndEvent, LLMStructuredPredictStartEvent, LLMStructuredPredictEndEvent

logger = logging.getLogger(__name__)

class TrackActivity(BaseEventHandler):
    """
    Troubleshooting what is going on within LlamaIndex.

    - What AI model (LLM/reasoning model/diffusion model/other) was used. 
    - What was the input/output. 
    - When did it start/end.
    - Backtrack of where the inference was called from.
    """
    model_config = {'extra': 'allow'}
    
    def __init__(self, jsonl_file_path: Path, write_to_logger: bool = False) -> None:
        super().__init__()
        if not isinstance(jsonl_file_path, Path):
            raise ValueError(f"jsonl_file_path must be a Path, got: {jsonl_file_path!r}")
        if not isinstance(write_to_logger, bool):
            raise ValueError(f"write_to_logger must be a bool, got: {write_to_logger!r}")
        self.jsonl_file_path = jsonl_file_path
        self.write_to_logger = write_to_logger
    
    def _filter_sensitive_data(self, data: Any) -> Any:
        """Recursively filter out sensitive fields from event data."""
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                if key.lower() in ['api_key']:
                    filtered[key] = "[REDACTED]"
                else:
                    filtered[key] = self._filter_sensitive_data(value)
            return filtered
        elif isinstance(data, list):
            return [self._filter_sensitive_data(item) for item in data]
        else:
            return data

    def _find_usage_dict(self, data: Any) -> Optional[dict]:
        """Search nested structures for a usage dict."""
        if isinstance(data, dict):
            usage = data.get("usage")
            if isinstance(usage, dict):
                return usage
            for value in data.values():
                found = self._find_usage_dict(value)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for item in data:
                found = self._find_usage_dict(item)
                if found is not None:
                    return found
        return None

    def _extract_token_usage(self, event_data: dict) -> Optional[dict]:
        """Extract token usage data from event payloads, if present."""
        try:
            from worker_plan_internal.llm_util.token_counter import extract_token_count
        except Exception as exc:
            logger.debug("Token counter unavailable: %s", exc)
            return None

        candidates = []
        if isinstance(event_data, dict):
            candidates.append(event_data.get("response"))
            candidates.append(event_data.get("output"))
            candidates.append(event_data.get("outputs"))
            response = event_data.get("response") if isinstance(event_data.get("response"), dict) else None
            if response:
                candidates.append(response.get("raw"))
                candidates.append(response.get("usage"))
                raw = response.get("raw")
                if isinstance(raw, dict):
                    candidates.append(raw.get("usage"))
            candidates.append(event_data.get("usage"))
            candidates.append(self._find_usage_dict(event_data))

        for candidate in candidates:
            if candidate is None:
                continue
            token_count = extract_token_count(candidate)
            if any(value is not None for value in [token_count.input_tokens, token_count.output_tokens, token_count.thinking_tokens]):
                return token_count.to_dict()

        return None

    def handle(self, event: Any) -> None:
        if isinstance(event, (LLMChatStartEvent, LLMChatEndEvent, LLMCompletionStartEvent, LLMCompletionEndEvent, LLMStructuredPredictStartEvent, LLMStructuredPredictEndEvent)):
            # Create event record with timestamp and backtrace
            event_data = json.loads(event.model_dump_json())
            filtered_event_data = self._filter_sensitive_data(event_data)
            
            event_record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event.__class__.__name__,
                "event_data": filtered_event_data,
                "backtrace": traceback.format_stack()
            }

            if isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent, LLMStructuredPredictEndEvent)):
                token_usage = self._extract_token_usage(filtered_event_data)
                if token_usage is not None:
                    event_record["token_usage"] = token_usage
            
            # Append to JSONL file
            with open(self.jsonl_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_record) + '\n')
            
            # Write to logger if enabled
            if self.write_to_logger:
                logger.info(f"{event.__class__.__name__}: {event!r}")


if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from enum import Enum
    from pydantic import BaseModel, Field
    from llama_index.core.instrumentation.dispatcher import instrument_tags

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    jsonl_file_path = Path("track_activity.jsonl")
    root = get_dispatcher()
    root.add_event_handler(TrackActivity(jsonl_file_path=jsonl_file_path, write_to_logger=True))

    class CostType(str, Enum):
        cheap = 'cheap'
        medium = 'medium'
        expensive = 'expensive'


    class ExtractDetails(BaseModel):
        location: str = Field(description="Name of the location.")
        cost: CostType = Field(description="Cost of the plan.")
        summary: str = Field(description="What is this about.")


    llm = get_llm("ollama-llama3.1")

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Fill out the details as best you can."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="I want to visit to Mars."
        ),
    ]
    sllm = llm.as_structured_llm(ExtractDetails)

    with instrument_tags({"tag1": "tag1"}):
        response = sllm.chat(messages)
        print(f"response:\n{response!r}")

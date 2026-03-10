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
from worker_plan_api.filenames import ExtraFilenameEnum

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
        self._llm_start_time_by_key: dict[str, datetime] = {}
    
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
        usage = self._find_usage_dict(event_data)
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            thinking_tokens = (
                usage.get("reasoning_tokens")
                or usage.get("thinking_tokens")
                or usage.get("cache_creation_input_tokens")
            )
            total_tokens = usage.get("total_tokens")
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens
                if total_tokens is not None
                else (input_tokens or 0) + (output_tokens or 0) + (thinking_tokens or 0),
                "raw_usage_data": usage,
            }

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

    def _extract_model_name(self, event_data: dict) -> str:
        """Extract model/provider name from event payloads."""
        if not isinstance(event_data, dict):
            return "unknown"

        model = event_data.get("model") or event_data.get("model_name") or event_data.get("llm_model")
        if not model:
            tags = event_data.get("tags")
            if isinstance(tags, dict):
                model = tags.get("llm_model") or tags.get("model") or tags.get("model_name")

        response = event_data.get("response")
        if not model and isinstance(response, dict):
            model = response.get("model") or response.get("model_name") or response.get("model_id") or response.get("llm_model")
            raw = response.get("raw")
            if not model and isinstance(raw, dict):
                model = raw.get("model") or raw.get("model_name") or raw.get("model_id") or raw.get("llm_model")

        provider = event_data.get("provider")
        if not provider and isinstance(response, dict):
            provider = response.get("provider")
            raw = response.get("raw")
            if not provider and isinstance(raw, dict):
                provider = raw.get("provider")

        if model and provider and str(provider) not in str(model):
            return f"{provider}:{model}"
        if model:
            return str(model)
        if provider:
            return str(provider)
        return "unknown"

    def _extract_cost(self, event_data: dict) -> float:
        """Extract cost from usage data if available."""
        usage = self._find_usage_dict(event_data)
        if not isinstance(usage, dict):
            return 0.0

        cost = usage.get("cost")
        if cost is None:
            cost_details = usage.get("cost_details")
            if isinstance(cost_details, dict):
                cost = cost_details.get("upstream_inference_cost") or cost_details.get("total_cost")

        try:
            return float(cost) if cost is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _load_activity_overview(self, path: Path) -> dict:
        if not path.exists():
            return {
                "last_updated": None,
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_thinking_tokens": 0,
                "total_tokens": 0,
                "models": {},
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("models", {})
                return data
        except Exception:
            logger.debug("Failed to read activity overview file: %s", path, exc_info=True)
        return {
            "last_updated": None,
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_thinking_tokens": 0,
            "total_tokens": 0,
            "models": {},
        }

    def _update_activity_overview(self, event_data: dict) -> None:
        token_usage = self._extract_token_usage(event_data) or {}
        cost = self._extract_cost(event_data)
        if not token_usage and cost == 0.0:
            return

        overview_path = self.jsonl_file_path.parent / ExtraFilenameEnum.ACTIVITY_OVERVIEW_JSON.value
        overview = self._load_activity_overview(overview_path)
        model_name = self._extract_model_name(event_data)

        model_stats = overview["models"].setdefault(
            model_name,
            {
                "total_cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            },
        )

        input_tokens = int(token_usage.get("input_tokens") or 0)
        output_tokens = int(token_usage.get("output_tokens") or 0)
        thinking_tokens = int(token_usage.get("thinking_tokens") or 0)
        total_tokens = int(token_usage.get("total_tokens") or input_tokens + output_tokens + thinking_tokens)

        model_stats["total_cost"] = float(model_stats.get("total_cost", 0.0)) + cost
        model_stats["input_tokens"] = int(model_stats.get("input_tokens", 0)) + input_tokens
        model_stats["output_tokens"] = int(model_stats.get("output_tokens", 0)) + output_tokens
        model_stats["thinking_tokens"] = int(model_stats.get("thinking_tokens", 0)) + thinking_tokens
        model_stats["total_tokens"] = int(model_stats.get("total_tokens", 0)) + total_tokens
        model_stats["calls"] = int(model_stats.get("calls", 0)) + 1

        overview["total_cost"] = float(overview.get("total_cost", 0.0)) + cost
        overview["total_input_tokens"] = int(overview.get("total_input_tokens", 0)) + input_tokens
        overview["total_output_tokens"] = int(overview.get("total_output_tokens", 0)) + output_tokens
        overview["total_thinking_tokens"] = int(overview.get("total_thinking_tokens", 0)) + thinking_tokens
        overview["total_tokens"] = int(overview.get("total_tokens", 0)) + total_tokens
        overview["last_updated"] = datetime.now().isoformat()

        try:
            with open(overview_path, "w", encoding="utf-8") as f:
                json.dump(overview, f, indent=2, sort_keys=True)
        except Exception:
            logger.debug("Failed to write activity overview file: %s", overview_path, exc_info=True)

    def _record_token_metrics_row(self, event_data: dict, duration_seconds: Optional[float] = None) -> None:
        """Persist per-event token metrics directly from instrumentation payloads."""
        try:
            from worker_plan_internal.llm_util.token_instrumentation import get_current_task_id, get_current_user_id, get_current_api_key_id
            from worker_plan_internal.llm_util.token_metrics_store import get_token_metrics_store
        except Exception as exc:
            logger.debug("Token metrics store unavailable in TrackActivity: %s", exc)
            return

        task_id = get_current_task_id()
        user_id = get_current_user_id()
        if not task_id:
            return

        token_usage = self._extract_token_usage(event_data) or {}
        cost_usd = self._extract_cost(event_data)
        model_name = self._extract_model_name(event_data)
        upstream_provider, upstream_model = self._split_provider_and_model(model_name)
        if model_name.strip().lower() == "unknown":
            model_name = ""

        has_usage = any(
            token_usage.get(key) is not None
            for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
        )
        if not has_usage and cost_usd == 0.0:
            return

        try:
            store = get_token_metrics_store()
            store.record_token_usage(
                task_id=str(task_id),
                user_id=str(user_id) if user_id else None,
                api_key_id=get_current_api_key_id(),
                llm_model=model_name or (upstream_model or "unknown"),
                upstream_provider=upstream_provider,
                upstream_model=upstream_model,
                input_tokens=token_usage.get("input_tokens"),
                output_tokens=token_usage.get("output_tokens"),
                thinking_tokens=token_usage.get("thinking_tokens"),
                cost_usd=cost_usd if cost_usd != 0.0 else None,
                duration_seconds=duration_seconds,
                success=True,
                raw_usage_data=token_usage.get("raw_usage_data"),
            )
        except Exception:
            logger.debug("Failed to persist token metrics from TrackActivity", exc_info=True)

    def _record_file_usage_metric(self, event_data: dict, token_usage: Optional[dict], duration_seconds: Optional[float]) -> None:
        """Write a usage metric row to usage_metrics.jsonl via the shared module."""
        cost = self._extract_cost(event_data)
        if not token_usage and cost == 0.0:
            return
        from worker_plan_internal.llm_util.usage_metrics import record_usage_metric
        model_name = self._extract_model_name(event_data)
        record_usage_metric(
            model=model_name or "unknown",
            duration_seconds=duration_seconds or 0.0,
            success=True,
            input_tokens=int(token_usage["input_tokens"]) if token_usage and token_usage.get("input_tokens") is not None else None,
            output_tokens=int(token_usage["output_tokens"]) if token_usage and token_usage.get("output_tokens") is not None else None,
            thinking_tokens=int(token_usage["thinking_tokens"]) if token_usage and token_usage.get("thinking_tokens") is not None else None,
            cost_usd=cost or None,
        )

    @staticmethod
    def _split_provider_and_model(model_name: str) -> tuple[Optional[str], Optional[str]]:
        if not model_name:
            return None, None
        normalized = model_name.strip().lower()
        if normalized in {"unknown", "none", "null"}:
            return None, None
        if ":" not in model_name:
            return None, model_name
        provider, model = model_name.split(":", 1)
        return provider or None, model or None

    @staticmethod
    def _parse_event_timestamp(event_data: dict) -> Optional[datetime]:
        if not isinstance(event_data, dict):
            return None
        ts = event_data.get("timestamp")
        if not isinstance(ts, str) or not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _event_match_key(event_data: dict) -> Optional[str]:
        if not isinstance(event_data, dict):
            return None
        span_id = event_data.get("span_id")
        if isinstance(span_id, str) and span_id:
            return f"span:{span_id}"
        event_id = event_data.get("id_")
        if isinstance(event_id, str) and event_id:
            return f"id:{event_id}"
        tags = event_data.get("tags")
        if isinstance(tags, dict):
            llm_exec_id = tags.get("llm_executor_uuid")
            if isinstance(llm_exec_id, str) and llm_exec_id:
                return f"llm_executor_uuid:{llm_exec_id}"
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
                duration_seconds: Optional[float] = None
                match_key = self._event_match_key(filtered_event_data)
                if match_key:
                    start_ts = self._llm_start_time_by_key.pop(match_key, None)
                    end_ts = self._parse_event_timestamp(filtered_event_data)
                    if start_ts and end_ts:
                        duration_seconds = max(0.0, (end_ts - start_ts).total_seconds())

                token_usage = self._extract_token_usage(filtered_event_data)
                if token_usage is not None:
                    event_record["token_usage"] = token_usage
                self._update_activity_overview(filtered_event_data)
                self._record_token_metrics_row(filtered_event_data, duration_seconds=duration_seconds)
                self._record_file_usage_metric(filtered_event_data, token_usage, duration_seconds)
            elif isinstance(event, (LLMChatStartEvent, LLMCompletionStartEvent, LLMStructuredPredictStartEvent)):
                match_key = self._event_match_key(filtered_event_data)
                start_ts = self._parse_event_timestamp(filtered_event_data)
                if match_key and start_ts:
                    self._llm_start_time_by_key[match_key] = start_ts
            
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

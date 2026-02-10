"""
Instrumentation for capturing and storing token metrics from LLM calls.

This module provides decorators and utilities for integrating token counting
into the LLM pipeline without modifying the core LLMExecutor class.
"""
import logging
import functools
from typing import Optional, Callable, Any
from worker_plan_internal.llm_util.token_counter import extract_token_count
from worker_plan_internal.llm_util.token_metrics_store import get_token_metrics_store

logger = logging.getLogger(__name__)

__all__ = ["record_llm_tokens", "get_current_run_id", "set_current_run_id"]

# Thread-local storage for current run_id (set by the pipeline)
_current_run_id: Optional[str] = None


def set_current_run_id(run_id: Optional[str]) -> None:
    """Set the current run ID for token tracking."""
    global _current_run_id
    _current_run_id = run_id
    logger.debug(f"Set current run_id for token tracking: {run_id}")


def get_current_run_id() -> Optional[str]:
    """Get the current run ID for token tracking."""
    return _current_run_id


def record_llm_tokens(
    llm_model: str,
    task_name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> Callable:
    """
    Decorator to record token metrics from an LLM call result.

    This decorator should wrap functions that return LLM responses.
    It extracts token counts from the response and stores them in the database.

    Args:
        llm_model: The LLM model identifier
        task_name: Optional name of the task/stage
        duration_seconds: Optional duration of the call

    Returns:
        A decorator function

    Example:
        @record_llm_tokens("gpt-4", task_name="ReviewPlan")
        def my_llm_function(llm):
            return llm.chat([...])
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                # Try to extract and store token counts
                run_id = get_current_run_id()
                if run_id is None:
                    logger.debug(f"No run_id set for token tracking in {func.__name__}")
                    return result

                try:
                    token_count = extract_token_count(result)
                    store = get_token_metrics_store()

                    success = token_count.total_tokens > 0 or result is not None
                    store.record_token_usage(
                        run_id=run_id,
                        llm_model=llm_model,
                        input_tokens=token_count.input_tokens,
                        output_tokens=token_count.output_tokens,
                        thinking_tokens=token_count.thinking_tokens,
                        duration_seconds=duration_seconds,
                        task_name=task_name or func.__name__,
                        success=success,
                        raw_usage_data=token_count.raw_usage_data if token_count.raw_usage_data else None,
                    )
                except Exception as e:
                    logger.warning(f"Error recording token metrics in {func.__name__}: {e}")

                return result

            except Exception as e:
                logger.error(f"Error in record_llm_tokens decorator for {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


def record_attempt_tokens(
    attempt_index: int,
    llm_model: str,
    duration_seconds: float,
    success: bool,
    error_message: Optional[str] = None,
    response: Optional[Any] = None,
) -> None:
    """
    Record token metrics for an LLMExecutor attempt.

    This is a utility function for recording metrics for individual LLM attempts
    within the LLMExecutor execution loop.

    Args:
        attempt_index: The attempt number (0-indexed)
        llm_model: The LLM model identifier
        duration_seconds: Duration of the attempt in seconds
        success: Whether the attempt succeeded
        error_message: Error message if the attempt failed
        response: The response object from the LLM (to extract tokens)
    """
    run_id = get_current_run_id()
    if run_id is None:
        return

    try:
        store = get_token_metrics_store()
        token_count = extract_token_count(response) if response else None

        store.record_token_usage(
            run_id=run_id,
            llm_model=llm_model,
            input_tokens=token_count.input_tokens if token_count else None,
            output_tokens=token_count.output_tokens if token_count else None,
            thinking_tokens=token_count.thinking_tokens if token_count else None,
            duration_seconds=duration_seconds,
            task_name=f"llm_attempt_{attempt_index}",
            success=success,
            error_message=error_message,
            raw_usage_data=token_count.raw_usage_data if token_count and token_count.raw_usage_data else None,
        )
    except Exception as e:
        logger.warning(f"Error recording attempt tokens for attempt {attempt_index}: {e}")

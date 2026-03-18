"""
Cycle through multiple LLMs, if one fails, try the next one.

I want all LLM invocations to go through this class.

It happens that the json output of the LLM doesn't match the expected schema.
When I inspect the raw response, I can see that the json comes close to the expected schema,
with tiny mistakes here and there. I guess with a more fuzzy json parser than Pydantic,
the json could be extracted.

It happens that an LLM provider is unavailable. Where a model used to be available, and have been removed from the provider.
Or the LLM server has to be started by the developer on the local machine.

Having multiple LLMs available is a good idea, because it increases the chances of success.
If one fails, then the next one may be able to respond.
If all of them fails, then the exception is raised. Exhausted all LLMs.

This is the class that `PlanTask` is using, the root class of all tasks in the pipeline.
Subtasks such as `ReviewPlan` are also using this class to invoke the LLM.

IDEA: Scheduling strategy: randomize the order of LLMs.
IDEA: Scheduling strategy: cycle through the LLM list twice, so there are two chances to succeed.
"""
import time
import logging
import inspect
import typing
import traceback
from uuid import uuid4
from typing import Any, Callable, Optional, List
from dataclasses import dataclass, field
from llama_index.core.llms.llm import LLM
from llama_index.core.instrumentation.dispatcher import instrument_tags
from pydantic import ValidationError
from worker_plan_internal.llm_factory import get_llm
from worker_plan_internal.llm_util.usage_metrics import record_usage_metric

logger = logging.getLogger(__name__)

# Substrings (lowercased) that indicate a transient / retriable error.
_TRANSIENT_PATTERNS: list[str] = [
    "rate limit", "rate_limit", "ratelimit", "429",
    "timeout", "timed out", "connection", "connect",
    "temporarily unavailable", "503", "502", "500",
    "overloaded", "capacity", "try again",
    "server error", "internal error",
    # Empty/null responses from structured LLM calls — the provider returned a
    # successful HTTP 200 but the response body contained no usable content.
    # This is a transient API-side failure, not a model capability failure.
    "nonetype", "'none' object", "none' object is not",
]


def is_transient_error(error: Exception) -> bool:
    """Return True if *error* looks like a transient network / rate-limit issue."""
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in _TRANSIENT_PATTERNS)


def _extract_validation_feedback(error: Exception) -> Optional[str]:
    """Extract structured feedback from a Pydantic ValidationError.

    Returns a human-readable summary of the validation failures, or ``None``
    if *error* is not a validation error.
    """
    # Walk the exception chain — LlamaIndex often wraps the original error.
    current: Optional[BaseException] = error
    for _ in range(100):
        if current is None:
            break
        if isinstance(current, ValidationError):
            lines = [f"Pydantic validation failed with {current.error_count()} error(s):"]
            for err in current.errors():
                loc = " → ".join(str(l) for l in err.get("loc", []))
                lines.append(f"  - {loc}: {err['msg']} (type={err['type']})")
            return "\n".join(lines)
        current = current.__cause__ if current.__cause__ else current.__context__
    return None


@dataclass
class RetryConfig:
    """Controls per-model retry behaviour inside :class:`LLMExecutor`.

    Retries only fire for errors classified as *transient* by
    :func:`is_transient_error`.  Permanent errors (auth failures, invalid
    model, validation errors) immediately fall through to the next model.
    """
    max_retries: int = 2
    base_delay: float = 1.0       # seconds
    max_delay: float = 30.0       # seconds
    backoff_multiplier: float = 2.0

    def delay_for_retry(self, retry_index: int) -> float:
        """Compute delay in seconds for the *retry_index*-th retry (0-based)."""
        delay = self.base_delay * (self.backoff_multiplier ** retry_index)
        return min(delay, self.max_delay)

class PipelineStopRequested(RuntimeError):
    """
    Raised when the pipeline execution is requested to stop by `should_stop_callback` after a task succeeds.

    This exception happens when the user presses Ctrl-C or closes the browser tab,
    so there is no point in continuing wasting resources on a 30 minute task.

    The PlanTask.run() method intercepts the PipelineStopRequested exception and create a the PIPELINE_STOP_REQUESTED_FLAG file,
    signaling that the pipeline was stopped by the user. So in post-mortem, it's fast to determine if the pipeline was stopped with this exception.
    """
    pass

class LLMModelBase:
    def create_llm(self) -> LLM:
        raise NotImplementedError("Subclasses must implement this method")

class LLMModelFromName(LLMModelBase):
    def __init__(self, name: str):
        self.name = name

    def create_llm(self) -> LLM:
        return get_llm(self.name)
    
    def __repr__(self) -> str:
        return f"LLMModelFromName(name='{self.name}')"

    @classmethod
    def from_names(cls, names: list[str]) -> list['LLMModelBase']:
        return [cls(name) for name in names]

class LLMModelWithInstance(LLMModelBase):
    def __init__(self, llm: LLM):
        self.llm = llm

    def create_llm(self) -> LLM:
        return self.llm
    
    def __repr__(self) -> str:
        return f"LLMModelWithInstance(llm={self.llm.__class__.__name__})"

    @classmethod
    def from_instances(cls, llms: list[LLM]) -> list['LLMModelBase']:
        return [cls(llm) for llm in llms]

@dataclass
class LLMAttempt:
    """Stores the result of a single LLM attempt."""
    stage: str
    llm_model: LLMModelBase
    success: bool
    duration: float
    result: Optional[Any] = None
    exception: Optional[Exception] = None

@dataclass
class ShouldStopCallbackParameters:
    """Parameters passed to the should_stop_callback after each attempt."""
    last_attempt: LLMAttempt
    total_duration: float
    attempt_index: int
    total_attempts: int

class LLMExecutor:
    """
    Cycle through multiple LLMs, falling back to the next on failure.
    A callback can be used to abort execution after any attempt.

    When *retry_config* is provided, transient errors (rate-limits, timeouts,
    connection problems) are retried with exponential back-off before moving
    on to the next model.

    When *max_validation_retries* > 0, Pydantic validation errors trigger an
    automatic retry on the **same** model.  Before the retry, structured error
    feedback is stored in :attr:`validation_feedback` so that the caller's
    ``execute_function`` can inspect it and inject the feedback into the prompt.
    """
    def __init__(
        self,
        llm_models: list[LLMModelBase],
        should_stop_callback: Optional[Callable[[ShouldStopCallbackParameters], None]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_validation_retries: int = 0,
    ):
        """
        Args:
            llm_models: A list of LLM models to try.
            should_stop_callback: A function that will be called after each attempt.
                If the callback raises PipelineStopRequested, the execution will be aborted. This is the only exception that is allowed to be raised by the callback, that doesn't indicate a problem.
                If the callback raises any other exception, the execution will be aborted. This indicates a problem with the callback.
                If the callback returns None, the execution will continue.
                If no callback is provided, the execution will continue until all LLMs are exhausted.
            retry_config: Optional retry settings for transient errors.
                When provided, each model is retried up to ``retry_config.max_retries``
                times on transient errors before falling through to the next model.
            max_validation_retries: Number of extra attempts per model on Pydantic
                validation failures.  The structured error feedback is available
                via :attr:`validation_feedback` so the ``execute_function`` can
                include it in the next prompt.
        """
        if not llm_models:
            raise ValueError("No LLMs provided")

        if should_stop_callback is not None and not callable(should_stop_callback):
            raise TypeError("should_stop_callback must be a function that can raise PipelineStopRequested to stop execution")

        self.llm_models = llm_models
        self.should_stop_callback = should_stop_callback
        self.retry_config = retry_config or RetryConfig(max_retries=0)
        self.max_validation_retries = max_validation_retries
        self.attempts: List[LLMAttempt] = []
        self._validation_feedback: Optional[str] = None

    @property
    def validation_feedback(self) -> Optional[str]:
        """Structured description of the last validation error, or ``None``.

        Callers can check this inside their ``execute_function`` to inject
        error feedback into the prompt before a retry attempt.  It is cleared
        before the first attempt and set only on validation-error retries.
        """
        return self._validation_feedback

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def run(self, execute_function: Callable[[LLM], Any]):
        self._validate_execute_function(execute_function)

        # Reset attempts for each new run
        self.attempts = []
        self._validation_feedback = None
        overall_start_time = time.perf_counter()

        for index, llm_model in enumerate(self.llm_models):
            # First attempt with this model.
            attempt = self._try_one_attempt(llm_model, execute_function)
            self.attempts.append(attempt)
            self._check_stop_callback(attempt, overall_start_time, index)

            if attempt.success:
                self._validation_feedback = None
                return attempt.result

            # Retry transient errors with exponential back-off.
            for retry_num in range(self.retry_config.max_retries):
                if attempt.exception is None or not is_transient_error(attempt.exception):
                    break  # permanent error — skip to next model

                delay = self.retry_config.delay_for_retry(retry_num)
                logger.info(
                    f"LLMExecutor: transient error from {llm_model!r}, "
                    f"retrying in {delay:.1f}s (retry {retry_num + 1}/{self.retry_config.max_retries})"
                )
                time.sleep(delay)

                attempt = self._try_one_attempt(llm_model, execute_function)
                self.attempts.append(attempt)
                self._check_stop_callback(attempt, overall_start_time, index)

                if attempt.success:
                    return attempt.result

            # On validation errors, retry the *same* model with error feedback.
            if self.max_validation_retries > 0 and attempt.exception is not None:
                for retry_num in range(self.max_validation_retries):
                    feedback = _extract_validation_feedback(attempt.exception)
                    if feedback is None:
                        break  # not a validation error — skip to next model

                    self._validation_feedback = feedback
                    logger.info(
                        f"LLMExecutor: validation error from {llm_model!r}, "
                        f"retrying with feedback (retry {retry_num + 1}/{self.max_validation_retries})"
                    )

                    attempt = self._try_one_attempt(llm_model, execute_function)
                    self.attempts.append(attempt)
                    self._check_stop_callback(attempt, overall_start_time, index)

                    if attempt.success:
                        self._validation_feedback = None
                        return attempt.result

                self._validation_feedback = None

        # If we get here, all attempts have failed.
        self._raise_final_exception()

    def _validate_execute_function(self, execute_function: Callable[[LLM], Any]) -> None:
        """
        Validate that the execute_function is a function that takes a single LLM parameter.
        It doesn't matter what the return type is or if it doesn't return anything.
        """
        if not callable(execute_function):
            raise TypeError("validate_execute_function1: must be a function that takes a LLM parameter")
        
        # Validate function signature
        sig = inspect.signature(execute_function)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise TypeError("validate_execute_function2: must be a function that takes a single parameter")
        
        try:
            # Use get_type_hints to correctly resolve postponed annotations (strings)
            # This is the key to supporting `from __future__ import annotations`
            type_hints = typing.get_type_hints(execute_function)
            param_name = params[0].name
            param_type = type_hints.get(param_name)
        except (NameError, TypeError) as e:
            # NameError happens if a type hint string can't be resolved.
            # TypeError can happen with complex but invalid type hints.
            raise TypeError(f"Could not resolve type hints for execute_function. Error: {e}")

        if param_type is None:
            # No type hint provided, so we can't validate. Let it pass.
            return

        # Now `param_type` is guaranteed to be a real type object.
        # Use issubclass for the most flexible and correct check.
        if not (isinstance(param_type, type) and issubclass(param_type, LLM)):
            raise TypeError(
                f"validate_execute_function3: must be a function that takes a single parameter of type LLM, "
                f"but got type '{param_type}'"
            )

    def _try_one_attempt(self, llm_model: LLMModelBase, execute_function: Callable[[LLM], Any]) -> LLMAttempt:
        """
        Performs a single, complete attempt with one LLM, returning a detailed result.
        
        Args:
            llm_model: The LLM model to try.
            execute_function: The callback to execute with the llm. The callback must not raise the PipelineStopRequested exception, since that interferes with the `ExecutePipeline.stopped_by_callback` property.

        Returns:
            A detailed result of the attempt.
        """
        attempt_start_time = time.perf_counter()
        try:
            llm = llm_model.create_llm()
        except Exception as e:
            duration = time.perf_counter() - attempt_start_time
            error_id = getattr(e, "error_id", None) or uuid4().hex[:12]
            logger.error(f"LLMExecutor: Error creating LLM {llm_model!r} error_id={error_id!r}: {e!r} traceback: {traceback.format_exc()}")
            self._record_attempt_token_metrics(
                llm_model_name=getattr(llm_model, "name", llm_model.__class__.__name__),
                duration=duration,
                success=False,
                error_message=str(e),
                response=None,
                error_id=error_id,
            )
            return LLMAttempt(stage='create', llm_model=llm_model, success=False, duration=duration, exception=e)

        llm_executor_uuid = str(uuid4())
        try:
            logger.debug(f"LLMExecutor will invoke execute_function. LLM {llm_model!r}. llm_executor_uuid: {llm_executor_uuid!r}")
            with instrument_tags({"llm_executor_uuid": llm_executor_uuid}):
                result = execute_function(llm)
            duration = time.perf_counter() - attempt_start_time
            logger.info(f"LLMExecutor did invoke execute_function. LLM {llm_model!r}. llm_executor_uuid: {llm_executor_uuid!r}. Duration: {duration:.2f} seconds")
            self._record_attempt_token_metrics(
                llm_model_name=getattr(llm_model, "name", llm.__class__.__name__),
                duration=duration,
                success=True,
                error_message=None,
                response=result,
            )
            return LLMAttempt(stage='execute', llm_model=llm_model, success=True, duration=duration, result=result)
        except PipelineStopRequested as e:
            logger.info(f"LLMExecutor: Stopping because the execute_function callback raised PipelineStopRequested: {e!r}")
            raise
        except Exception as e:
            duration = time.perf_counter() - attempt_start_time
            error_id = getattr(e, "error_id", None) or uuid4().hex[:12]
            logger.error(f"LLMExecutor: error when invoking execute_function. LLM {llm_model!r} and llm_executor_uuid: {llm_executor_uuid!r} and error_id: {error_id!r}: {e!r} traceback: {traceback.format_exc()}")
            self._record_attempt_token_metrics(
                llm_model_name=getattr(llm_model, "name", llm.__class__.__name__),
                duration=duration,
                success=False,
                error_message=str(e),
                response=None,
                error_id=error_id,
            )
            return LLMAttempt(stage='execute', llm_model=llm_model, success=False, duration=duration, exception=e)

    def _record_attempt_token_metrics(
        self,
        llm_model_name: str,
        duration: float,
        success: bool,
        error_message: Optional[str],
        response: Optional[Any],
        error_id: Optional[str] = None,
    ) -> None:
        """Best-effort token metrics recording; never blocks LLM execution flow."""
        try:
            from worker_plan_internal.llm_util.token_instrumentation import record_attempt_tokens

            record_attempt_tokens(
                attempt_index=len(self.attempts),
                llm_model=llm_model_name,
                duration_seconds=duration,
                success=success,
                error_message=error_message,
                response=response,
            )
        except Exception as exc:
            logger.debug("Failed to record token metrics for attempt: %s", exc)

        # File-based usage metrics for local runs (no database required).
        # Successful calls are also recorded by TrackActivity via llama_index
        # instrumentation events — when those events fire, TrackActivity
        # writes richer data (token counts, cost, provider info).  However,
        # some LLM backends (e.g. Anthropic) override structured_predict()
        # and bypass self.chat(), so instrumentation events never fire.
        # Recording here as well ensures every call gets at least basic
        # tracking (model, duration, success).  Duplicate rows (one from
        # here, one from TrackActivity) are acceptable — consumers can
        # deduplicate if needed.
        if not success:
            record_usage_metric(
                model=llm_model_name,
                duration_seconds=duration,
                success=False,
                error_message=error_message,
                error_id=error_id,
            )
        else:
            # Try to retrieve token counts captured by an external hook
            # (e.g. the Anthropic httpx usage hook installed by the runner).
            extra_kwargs: dict = {}
            try:
                from worker_plan_internal.llm_util.usage_metrics import pop_captured_llm_usage
                usage, model_id = pop_captured_llm_usage()
                if usage:
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    if input_tokens is not None:
                        extra_kwargs["input_tokens"] = int(input_tokens)
                    if output_tokens is not None:
                        extra_kwargs["output_tokens"] = int(output_tokens)
                    # Estimate cost via model_pricing if we have tokens
                    if input_tokens or output_tokens:
                        try:
                            from worker_plan_internal.llm_util.model_pricing import estimate_cost
                            cost = estimate_cost(
                                model_name=model_id or llm_model_name,
                                input_tokens=int(input_tokens or 0),
                                output_tokens=int(output_tokens or 0),
                            )
                            if cost is not None:
                                extra_kwargs["cost_usd"] = cost
                        except Exception:
                            pass
            except ImportError:
                pass
            record_usage_metric(
                model=llm_model_name,
                duration_seconds=duration,
                success=True,
                **extra_kwargs,
            )

    def _check_stop_callback(self, last_attempt: LLMAttempt, start_time: float, attempt_index: int) -> None:
        """Checks the callback, if it exists, to see if execution should stop."""
        if self.should_stop_callback is None:
            return
        
        parameters = ShouldStopCallbackParameters(
            last_attempt=last_attempt,
            total_duration=time.perf_counter() - start_time,
            attempt_index=attempt_index,
            total_attempts=len(self.llm_models)
        )
        
        try:
            self.should_stop_callback(parameters)
        except PipelineStopRequested as e:
            logger.warning(f"Callback raised PipelineStopRequested. Aborting execution after attempt {attempt_index}: {e}")
            raise

    def _raise_final_exception(self) -> None:
        """Raise the final exception when no attempt succeeds."""
        rows = []
        for attempt_index, attempt in enumerate(self.attempts):
            status = "success" if attempt.success else "failed"
            rows.append(f" - Attempt {attempt_index} with {attempt.llm_model!r} {status} during '{attempt.stage}' stage: {attempt.exception!r}")
        error_summary = "\n".join(rows)
        raise Exception(f"Failed to run. Exhausted all LLMs. Failure summary:\n{error_summary}")

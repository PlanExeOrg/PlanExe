# Proposal: LLMExecutor Retry & Resilience Improvements

**Status:** Implemented
**Author:** Bubba + Egon
**Related PRs:** #198 (DeduplicateLeversTask per-lever decomposition), #220 (implementation)

---

## Background

`llm_executor.py` implements a fallback chain: when one LLM fails, it tries the next configured LLM in sequence. This handles model-level failures well.

Several tasks (e.g., `DeduplicateLeversTask`, `ReviewPlanTask`) had added their own local retry loops on top of `llm_executor.run()`, creating duplicated, inconsistent retry logic scattered across task files.

neoneye's design direction (from PR #198 review): **retry logic belongs in `llm_executor.py`, not in individual tasks.**

---

## Implementation (Option 3: Two-tier resilience)

PR #220 implemented the recommended two-tier approach.

### Tier 1 — `LLMExecutor` handles transient errors

`RetryConfig` dataclass controls per-model retry behaviour:

```python
@dataclass
class RetryConfig:
    max_retries: int = 2          # retries per model (0 = no retries)
    base_delay: float = 1.0       # seconds
    max_delay: float = 30.0       # seconds
    backoff_multiplier: float = 2.0
```

`is_transient_error()` classifies exceptions by matching error strings against known transient patterns (rate limits, timeouts, connection errors, 5xx status codes). Only transient errors trigger retries — permanent errors (auth failures, validation errors, schema mismatches) immediately fall through to the next model.

The retry loop uses exponential backoff: 1s, 2s, 4s, ... capped at `max_delay`.

### Tier 2 — Luigi resume handles task-level failures

If all LLM attempts and retries are exhausted, the task raises and Luigi marks it failed. Re-running the pipeline resumes from the failed task. The operator can swap model config between runs to address systematic failures.

### Integration with related features

- **LLMChatError (#237):** Tasks wrap LLM exceptions in `LLMChatError` with unique `error_id` for traceability. `is_transient_error()` works through the wrapper because `LLMChatError.__str__()` preserves the original cause text.
- **Usage metrics (#110):** Failed attempts are recorded to `usage_metrics.jsonl` with error classification and `error_id`. Successful attempts are recorded by `TrackActivity` instrumentation.
- **Pipeline stop:** `PipelineStopRequested` is never retried — it propagates immediately.

---

## Context Overflow Handling (separate from retries)

`DeduplicateLeversTask` uses compact-history fallback for context window overflow. This is not a retry — it's a prompt restructuring. The distinction matters:

- **Retry:** Same prompt, same schema, try again. Appropriate for transient errors. Handled by `LLMExecutor`.
- **Prompt restructuring:** Different prompt (compacted context), same schema. Appropriate for context overflow. Stays in the task.

---

## Future Work

- Config-driven retry settings via `llm_config/custom.json` (currently hardcoded defaults).
- Remove remaining per-task retry loops as each task is updated (follow-on PRs).

---

## References

- `worker_plan/worker_plan_internal/llm_util/llm_executor.py` — retry logic and `RetryConfig`
- `worker_plan/worker_plan_internal/llm_util/llm_errors.py` — `LLMChatError` with error traceability
- `worker_plan/worker_plan_internal/llm_util/usage_metrics.py` — file-based usage metrics
- `worker_plan/worker_plan_internal/lever/deduplicate_levers.py` — example of per-task compact fallback (PR #198)

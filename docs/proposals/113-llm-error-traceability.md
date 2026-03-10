# Proposal: LLM Error Traceability via Error UUIDs

**Author:** neoneye
**Date:** 2026-03-10
**Status:** Implemented (PR #237)

---

## Problem

When an LLM call fails inside a pipeline task, the exception was caught and re-raised as a generic `ValueError`:

```python
except Exception as e:
    logger.debug(f"LLM chat interaction failed: {e}")
    logger.error("LLM chat interaction failed.", exc_info=True)
    raise ValueError("LLM chat interaction failed.") from e
```

This pattern appeared in **38 files** across the codebase. By the time `LLMExecutor` caught the re-raised `ValueError`, `str(e)` was the fixed string `"LLM chat interaction failed."` — the root cause (timeout, auth error, invalid JSON, etc.) was lost.

PR #236 added `classify_error()` to categorize error strings into short labels for `usage_metrics.jsonl`. But because the original exception was masked, these errors all classified as `"unknown"` with `error_detail: "LLM chat interaction failed."` — defeating the purpose of classification.

The root cause *was* logged via `logger.error(..., exc_info=True)`, but there was no way to correlate a log line with a specific row in `usage_metrics.jsonl`.

---

## Goals

1. **Preserve the root cause** so `classify_error()` can categorize it correctly.
2. **Correlate metrics with logs** so a user can look up the full traceback for any failed metric row.
3. **Keep the change mechanical** — the call sites should all follow the same pattern.
4. **No behaviour change** — callers that catch `ValueError` must continue to work.

---

## Implementation

### 1. `LLMChatError` exception

A dedicated exception that carries structured context, defined in
`worker_plan/worker_plan_internal/llm_util/llm_errors.py`:

```python
import uuid

class LLMChatError(ValueError):
    """Raised when an LLM chat interaction fails.

    Carries the root-cause exception and a unique error_id for
    cross-referencing log entries with usage_metrics.jsonl rows.

    Extends ValueError for backward compatibility with existing
    except ValueError catch sites.
    """
    def __init__(self, cause: Exception, error_id: str | None = None, message: str | None = None):
        self.cause = cause
        self.error_id = error_id or uuid.uuid4().hex[:12]
        self.message = message or "LLM chat interaction failed"
        super().__init__(f"{self.message} [{self.error_id}]: {cause}")
```

Key properties:
- `str(LLMChatError)` includes the root cause, so `classify_error()` categorizes correctly.
- `error_id` is a short UUID (12 hex chars) printed in both the log and the metric row.
- `message` allows callers to distinguish between numbered interactions (e.g. interaction 1 vs 2).
- Extends `ValueError` for backward compatibility with existing catch sites.

### 2. Updated call sites (38 files)

The standard pattern (30 files):

```python
from worker_plan_internal.llm_util.llm_errors import LLMChatError

except Exception as e:
    llm_error = LLMChatError(cause=e)
    logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
    logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
    raise llm_error from e
```

For files with numbered interactions (e.g. `expert_finder.py`, `questions_answers.py`):

```python
except Exception as e:
    llm_error = LLMChatError(cause=e, message="LLM chat interaction 2 failed")
    logger.debug(f"{llm_error.message} [{llm_error.error_id}]: {e}")
    logger.error(f"{llm_error.message} [{llm_error.error_id}]", exc_info=True)
    raise llm_error from e
```

### 3. `error_id` in `usage_metrics.jsonl`

`record_usage_metric()` accepts an optional `error_id` parameter.
`LLMExecutor._record_attempt_token_metrics()` extracts the `error_id` from the exception:

```python
if not success:
    error_id = getattr(exception, "error_id", None) if exception else None
    record_usage_metric(
        model=llm_model_name,
        duration_seconds=duration,
        success=False,
        error_message=error_message,
        error_id=error_id,
    )
```

### 4. Verified output

**Log output:**
```
ERROR LLM chat interaction failed [4c2a64973bcd]: 1 validation error for ...
Traceback (most recent call last):
  ...
```

**usage_metrics.jsonl row:**
```json
{"timestamp": "2026-03-10T19:50:18.821350", "success": false, "model": "openrouter-gemini-2.0-flash-001", "duration_seconds": 5.391, "error": "invalid_json", "error_id": "4c2a64973bcd"}
```

A user can now `grep 4c2a64973bcd` in the logs to find the full traceback.

---

## Files changed

| File | Change |
|---|---|
| `worker_plan/worker_plan_internal/llm_util/llm_errors.py` | **New.** `LLMChatError` exception class with `cause`, `error_id`, `message`. |
| `worker_plan/worker_plan_internal/llm_util/usage_metrics.py` | Added `error_id` parameter to `record_usage_metric()`. |
| `worker_plan/worker_plan_internal/llm_util/llm_executor.py` | Extract `error_id` from exception, pass to `record_usage_metric()`. |
| 36 files with `raise ValueError("LLM chat interaction failed.")` | Mechanical replacement with `LLMChatError`. |
| 2 files with `raise ValueError("LLM chat interaction N failed.")` | Replaced with `LLMChatError(cause=e, message=...)`. |
| `worker_plan/tests/test_llm_errors.py` | **New.** 8 tests for `LLMChatError`. |
| `worker_plan/tests/test_usage_metrics.py` | Added tests for `error_id` field and `LLMChatError` integration. |

---

## Out of scope

- Structured logging (JSON log format) — useful but a separate concern.
- Retry-level error tracking — `LLMExecutor` already tracks per-attempt results.
- Persisting error IDs to the database for cloud runs — can be added later via `TrackActivity`.

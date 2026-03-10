# Proposal: Usage Metrics for Local Runs (No Database Required)

**Author:** Bubba (VoynichLabs)  
**Date:** 2026-03-08  
**Status:** Implemented (PR #219)

---

## Problem

PlanExe has solid token-counting infrastructure (`token_counter.py`, `token_instrumentation.py`,
`token_metrics_store.py`), but it only activates when a `PLANEXE_TASK_ID` is set and a database
connection is available. For local CLI runs — the primary use case for self-hosted users — **zero
usage data is recorded**.

This is not specific to LM Studio. Any local inference server (Ollama, llama.cpp server, vLLM,
LMDeploy, Jan, etc.) has the same problem: performance metrics live only in the server's UI or
stdout, not in the plan output.

As a result, users cannot answer basic questions:

- How many tokens did this plan consume?
- How long did each task take?
- Which tasks were the heaviest?
- How does Qwen 9B compare to Qwen 35B on the same prompt?

---

## Solution

Write a lightweight `usage_metrics.json` file into the run output directory at the end of every
pipeline run. No database. No task ID. No web stack. Works for any OpenAI-compatible backend.

### What to capture (per LLM call, from `response.raw.usage`)

| Field | Source | Notes |
|---|---|---|
| `input_tokens` | `usage.prompt_tokens` | Already extracted by `extract_token_count()` |
| `output_tokens` | `usage.completion_tokens` | Already extracted |
| `duration_seconds` | `time.perf_counter()` delta | Already measured in `LLMExecutor._try_one_attempt()` |
| `llm_model` | config key | Already available in `LLMAttempt` |
| `task_name` | Luigi task class name | Available in `PlanTask.run()` |
| `success` | `LLMAttempt.success` | Already in `LLMAttempt` |

### Output file: `{run_id_dir}/usage_metrics.json`

```json
{
  "run_id_dir": "/path/to/run",
  "generated_at": "2026-03-08T17:45:00Z",
  "summary": {
    "total_input_tokens": 412847,
    "total_output_tokens": 98341,
    "total_duration_seconds": 2847.3,
    "total_calls": 63,
    "successful_calls": 63,
    "failed_calls": 0,
    "avg_tokens_per_second": 38.7
  },
  "calls": [
    {
      "task_name": "IdentifyPurposeTask",
      "llm_model": "lmstudio-qwen35b",
      "input_tokens": 1243,
      "output_tokens": 412,
      "duration_seconds": 14.2,
      "tokens_per_second": 29.0,
      "success": true,
      "timestamp": "2026-03-08T12:28:14Z"
    }
  ]
}
```

---

## Implementation

### Option A — File-based accumulator in `LLMExecutor` (minimal change)

Extend `_record_attempt_token_metrics()` in `llm_executor.py` to also append to a local JSON
file when `PLANEXE_TASK_ID` is not set (i.e., always for CLI runs):

```python
def _record_attempt_token_metrics(self, ...):
    # existing DB path (unchanged)
    try:
        from worker_plan_internal.llm_util.token_instrumentation import record_attempt_tokens
        record_attempt_tokens(...)
    except Exception as exc:
        logger.debug("Failed to record token metrics for attempt: %s", exc)

    # NEW: always write to file if run_id_dir is available
    run_id_dir = os.environ.get("RUN_ID_DIR")
    if run_id_dir:
        _append_usage_record(run_id_dir, {
            "llm_model": llm_model_name,
            "duration_seconds": duration,
            "success": success,
            "error_message": error_message,
            "token_count": extract_token_count(response).to_dict() if response else None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
```

`_append_usage_record()` would be a thread-safe JSON-lines appender (one record per line, atomic
rename on flush) so concurrent Luigi workers don't corrupt the file.

### Option B — Summary writer in `run_plan_pipeline.py`

At the end of the pipeline run, walk all output JSON files and reconstruct timing from
`start_time.json` + task completion timestamps. Less accurate (no per-call token counts unless
providers embed them in task outputs) but zero changes to the hot path.

### Recommended: Option A

Option A captures real per-call token counts from the API response `usage` object. The
`extract_token_count()` function already handles all providers (OpenAI, Anthropic, OpenRouter,
Ollama, LM Studio). No new dependencies. The `RUN_ID_DIR` env var is already required to run the
pipeline — it's always set.

---

## Provider Coverage

| Provider | `usage` in response | Notes |
|---|---|---|
| LM Studio | ✅ | `prompt_tokens`, `completion_tokens` in every response |
| Ollama | ✅ | `prompt_eval_count`, `eval_count` (needs mapping) |
| llama.cpp server | ✅ | Same as LM Studio |
| OpenAI | ✅ | Standard |
| OpenRouter | ✅ | Plus `cost` field |
| Anthropic | ✅ | `input_tokens`, `output_tokens` |
| vLLM | ✅ | OpenAI-compatible |
| Jan | ✅ | OpenAI-compatible |

All local inference servers that implement the OpenAI `/v1/chat/completions` spec return
`usage` in the response body. `extract_token_count()` already handles the field name
variations.

---

## Why Not Just Use `PLANEXE_TASK_ID`?

The existing DB path requires:

1. A running database (web app stack)
2. `PLANEXE_TASK_ID` env var set to a valid UUID from the DB
3. `database_api.planexe_db_singleton` importable

None of these are present in a bare CLI run. The file-based approach requires only `RUN_ID_DIR`
(already mandatory) and standard library (`json`, `os`, `threading`).

---

## What This Enables

- **Model comparison**: Run the same prompt on Qwen 9B vs 35B, compare `usage_metrics.json`
  — total tokens, tok/s, task-level breakdown.
- **Regression detection**: Task X used 2K tokens last week, now uses 20K — prompt bloat caught.
- **Hardware benchmarking**: Share `summary.avg_tokens_per_second` alongside run results.
- **Cost estimation**: For cloud providers (OpenRouter, OpenAI), multiply by pricing from config.
- **Self-documenting runs**: The output directory is self-contained — no external DB query needed
  to understand what happened.

---

## Implementation Notes (PR #219)

The final implementation diverges from Option A above in several ways:

### File format: JSONL, not JSON

`usage_metrics.jsonl` uses one JSON object per line (append-only). This avoids the need for
atomic rename or in-memory accumulation — each LLM call appends a single line. Thread-safe
by nature since each write is a short append to a file handle.

### Recording source: llama_index instrumentation, not LLMExecutor

Successful calls are recorded by `TrackActivity` (the llama_index `BaseEventHandler`) which
receives the actual `ChatResponse` with full token counts, cost, and `provider:model` info.
`LLMExecutor._record_attempt_token_metrics()` only records **failures**, since instrumentation
end events are not emitted when the LLM call fails.

This was necessary because `execute_function(llm)` returns the processed result (a Pydantic
model or string), not the raw `ChatResponse`. The instrumentation layer is the only place
with access to the real response.

### Model field includes provider

The `model` field contains the full `provider:model` string (e.g.
`Google AI Studio:google/gemini-2.0-flash-001`), matching `activity_overview.json`.

### Example output

```json
{"timestamp": "2026-03-10T13:36:48.250446", "success": true, "model": "Google AI Studio:google/gemini-2.0-flash-001", "duration_seconds": 4.879, "input_tokens": 5316, "output_tokens": 643, "cost_usd": 0.0007888}
{"timestamp": "2026-03-10T13:36:53.554864", "success": true, "model": "Google:google/gemini-2.0-flash-001", "duration_seconds": 5.237, "input_tokens": 8877, "output_tokens": 562, "cost_usd": 0.0011125}
```

### Key files

| File | Role |
|------|------|
| `worker_plan/worker_plan_internal/llm_util/usage_metrics.py` | Core module: `set_usage_metrics_path()`, `record_usage_metric()` |
| `worker_plan/worker_plan_internal/llm_util/track_activity.py` | Records successful calls via `_record_file_usage_metric()` |
| `worker_plan/worker_plan_internal/llm_util/llm_executor.py` | Records failed calls only |
| `worker_plan/worker_plan_internal/plan/run_plan_pipeline.py` | Sets/clears metrics path around pipeline execution |
| `worker_plan/worker_plan_api/filenames.py` | `USAGE_METRICS_JSONL` constant |

### Resolved open questions

1. **Ollama field mapping**: Handled by `extract_token_count()` and `TrackActivity._extract_token_usage()` which already support multiple field name variations.
2. **Thread safety**: JSONL append-per-line is safe for concurrent Luigi workers.
3. **Thinking tokens**: Recorded when available (e.g. from OpenRouter reasoning models). `null` for providers that don't expose them.
4. **Retention on resume**: Appended. A resumed run adds only the new calls alongside the restored snapshot's existing metrics.

---

## Follow-up: Error Classification (PR #236)

Raw Python exception strings in the `"error"` field were verbose and could leak raw model output.
PR #236 added `classify_error()` to replace them with short category labels:

| Category | Matches on |
|---|---|
| `invalid_json` | "json", "validation error", "pydantic" |
| `timeout` | "timeout", "timed out" |
| `empty_response` | "empty", "none", "no response" |
| `connection_error` | "connection", "connect", "dns", "network" |
| `rate_limit` | "rate limit", "429", "too many requests" |
| `auth_error` | "auth", "401", "403", "unauthorized", "forbidden" |
| `server_error` | "500", "502", "503", "504", "server error", "internal server" |
| `model_not_found` | "model not found", "not found", "404" |
| `unknown` | fallback — raw message preserved in `error_detail` (truncated to 200 chars) |

---

## Follow-up: Error Traceability (PR #237)

All 38 call sites that raised `ValueError("LLM chat interaction failed.")` were converted to
`LLMChatError` (see [proposal 113](113-llm-error-traceability.md)). This:

- **Preserves the root cause** in `str(LLMChatError)`, so `classify_error()` categorizes correctly
  (previously all were `"unknown"`).
- **Adds `error_id`** (12-char UUID) to both log output and the JSONL row, enabling cross-referencing.

Example failure row after both PRs:

```json
{"timestamp": "2026-03-10T19:50:18.821350", "success": false, "model": "openrouter-gemini-2.0-flash-001", "duration_seconds": 5.391, "error": "invalid_json", "error_id": "4c2a64973bcd"}
```

A user can `grep 4c2a64973bcd` in the logs to find the full traceback.

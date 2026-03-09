# Proposal: Usage Metrics for Local Runs (No Database Required)

**Author:** Bubba (VoynichLabs)  
**Date:** 2026-03-08  
**Status:** Proposal

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

## Open Questions

1. **Ollama field mapping**: `prompt_eval_count` / `eval_count` vs `prompt_tokens` /
   `completion_tokens` — should `extract_token_count()` add an Ollama-specific branch, or
   should the Ollama LLM adapter normalize field names before returning?

2. **Thread safety**: Luigi runs tasks in parallel workers. The appender needs an
   `fcntl.flock()` or atomic rename approach to avoid corruption under concurrent writes.

3. **Thinking tokens for local models**: Qwen 3 thinking tokens are stripped before the
   response reaches llama_index (LM Studio does not expose them separately). The
   `thinking_tokens` field will be `null` for local runs — acceptable for now.

4. **Retention**: Should `usage_metrics.json` be overwritten on resume (Luigi skip-completed
   behavior) or appended? Appending is safer — a resumed run adds only the new calls.

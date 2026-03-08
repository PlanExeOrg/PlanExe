# Proposal: LLMExecutor Retry & Resilience Improvements

**Status:** Draft  
**Author:** Bubba + Egon  
**Related PRs:** #198 (DeduplicateLeversTask per-lever decomposition)

---

## Background

`llm_executor.py` currently implements a fallback chain: when one LLM fails, it tries the next configured LLM in sequence. This handles model-level failures well.

Several tasks (e.g., `DeduplicateLeversTask`, `ReviewPlanTask`) have added their own local retry loops on top of `llm_executor.run()`. This creates duplicated, inconsistent retry logic scattered across task files.

neoneye's design direction (from PR #198 review): **retry logic belongs in `llm_executor.py`, not in individual tasks.**

---

## Problems with the Current Approach

1. **Duplicated retry logic** — Each task that needs retries implements its own loop with its own backoff, attempt counts, and error handling.
2. **Hardcoded attempt counts** — `max_retries = 3` appears in task files, not configurable without code changes.
3. **Inconsistent error handling** — Some tasks retry on all exceptions, others on specific ones. No shared policy.
4. **No coordination with Luigi resume** — If a task fails mid-pipeline, Luigi can resume from the last successful task. But per-call retries inside a task don't benefit from Luigi's resume mechanism.

---

## Proposed Improvements

### Option 1: Config-driven retry in `llm_executor.py`

Add retry configuration to the LLM config file (e.g., `llm_config/custom.json`):

```json
{
  "models": [...],
  "retry": {
    "max_attempts": 3,
    "backoff_seconds": 2,
    "retry_on": ["timeout", "rate_limit", "connection_error"]
  }
}
```

`LLMExecutor.run()` reads this config and wraps each model attempt with the specified retry policy. Tasks call `llm_executor.run()` once — no local retry loops needed.

**Pros:** Centralized, configurable, consistent.  
**Cons:** Requires distinguishing retriable errors (transient) from non-retriable ones (schema failure, bad prompt). Schema failures should NOT be retried blindly — they need a different prompt/schema, not the same call again.

---

### Option 2: Luigi resume as the resilience mechanism

Instead of retrying individual LLM calls, lean into Luigi's existing task-level resume:

- If a task fails, Luigi marks it failed and stops.
- Re-running the pipeline resumes from the last successful task (Luigi checks output file existence).
- The failed task retries from scratch with the same or updated config.

This is already documented in MEMORY.md ("Luigi Resume Pattern") and used in practice.

**How to make this work better:**
- Ensure all tasks write partial progress to output files where possible (already true for most tasks).
- Add a `--retry-failed` flag to the pipeline runner that re-runs only failed tasks.
- Tasks should be idempotent: same input + same config = same output.

**Pros:** No new retry code. Uses battle-tested Luigi infrastructure. Failures are explicit and logged.  
**Cons:** Each retry requires a full pipeline re-run invocation. Not suitable for per-call transient errors (network blip, brief rate limit).

---

### Option 3: Two-tier resilience (recommended)

Combine both approaches:

**Tier 1 — `llm_executor.py` handles transient errors:**
- Short retry (2–3 attempts) for network/timeout errors only.
- Configurable via LLM config file.
- Does NOT retry schema/validation failures.

**Tier 2 — Luigi resume handles task-level failures:**
- If all LLM attempts fail (schema error, model unavailable), task raises and Luigi marks it failed.
- Re-run pipeline → Luigi resumes from failed task.
- Operator can swap model config between runs to address systematic failures.

**Tasks themselves:** No local retry loops. One `llm_executor.run()` call per operation. Compact-history fallback (as in `DeduplicateLeversTask`) is task-specific logic for context management, not a retry mechanism.

---

## Context Overflow Handling (separate from retries)

`DeduplicateLeversTask` introduced compact-history fallback for context window overflow. This is not a retry — it's a prompt restructuring. The distinction matters:

- **Retry:** Same prompt, same schema, try again. Appropriate for transient errors.
- **Prompt restructuring:** Different prompt (compacted context), same schema. Appropriate for context overflow.

This distinction should be preserved in any future `llm_executor` design. Prompt restructuring logic stays in the task; transient error retries move to `llm_executor`.

---

## Recommended Next Steps

1. Add retry config schema to `llm_config` (transient errors only, configurable attempts + backoff).
2. Implement in `LLMExecutor.run()` with error classification (transient vs. structural).
3. Document Luigi resume as the standard recovery path for structural failures.
4. Remove per-task retry loops as each task is updated (follow-on PRs).

---

## References

- `worker_plan/worker_plan_internal/llm_util/llm_executor.py` — current fallback chain implementation
- `worker_plan/worker_plan_internal/lever/deduplicate_levers.py` — example of per-task compact fallback (PR #198)
- Luigi resume pattern — documented in project MEMORY.md
- `SPEED_VS_DETAIL` config — precedent for pipeline-level behavioral config

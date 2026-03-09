# Luigi Resume Enhancements — Proposal

**Date:** 2026-03-07  
**Author:** Bubba (Mac Mini agent)  
**Context:** Written from direct experience running overnight pipeline recovery sessions with Qwen 3.5-35B-A3B on a Mac Mini M4 Pro. Every suggestion here is grounded in a concrete friction point hit during real runs.

---

## Background

Luigi's file-based task completion tracking is one of PlanExe's most valuable properties for local model runs. When a 60-task pipeline fails at task 30, Luigi resumes from task 30 — not task 1. For long runs on slow local hardware, this is the difference between a 4-hour retry and a 4-minute one.

But the current resume flow has rough edges that require manual intervention, log-watching, and tribal knowledge. This proposal describes targeted enhancements to make resume-driven iteration faster, safer, and automatable.

---

## 1. Webhook / Event Hooks on Task Completion and Failure

### The problem

Right now, monitoring a running pipeline means tailing a log file and parsing freeform text every 60 seconds. Agents and humans alike poll `log.txt` to find out if a task passed or failed. There is no push notification.

### What we want

A lightweight event hook system that fires on task state changes:

- `task.started` — task began executing
- `task.completed` — task wrote its output file and marked done
- `task.failed` — task raised an exception or exhausted retries
- `pipeline.completed` — all tasks finished successfully
- `pipeline.failed` — pipeline terminated with at least one failure

### Proposed interface

In the run config or environment, specify a webhook URL:

```
PLANEXE_WEBHOOK_URL=http://localhost:9000/planexe-events
```

On each event, POST a JSON payload:

```json
{
  "event": "task.failed",
  "task": "PreProjectAssessmentTask",
  "run_id_dir": "/path/to/run",
  "timestamp_utc": "2026-03-06T21:55:00Z",
  "error_summary": "1 validation error for ExpertDetails — combined_summary: Field required",
  "attempt": 1,
  "model": "lmstudio-qwen3.5-35b-a3b"
}
```

### Why this matters

An agent monitoring a pipeline via webhook can react in seconds instead of polling every minute. It can detect failure, inspect the error, apply a fix, and resume — without ever reading a log file. This is the foundation for autonomous overnight pipeline repair.

A Discord webhook makes this immediately useful: post task completions and failures directly to a channel, with structured data, without any polling infrastructure.

---

## 2. Task Invalidation CLI

### The problem

To re-run a completed task, you currently delete its output file manually. This requires knowing which file corresponds to which task, finding it in the run directory, and deleting it without accidentally removing dependent outputs. There is no guard against invalidating a task that has already-complete downstream dependents that would then also need to be re-run.

### What we want

A CLI command:

```bash
planexe invalidate <TaskName> [--run-dir /path/to/run] [--cascade]
```

Behavior:

- Without `--cascade`: deletes only the output file(s) for the named task. On next resume, only that task re-runs.
- With `--cascade`: deletes output files for the named task AND all downstream tasks that depend on it. Useful after fixing a schema or prompt that affects interpretation further down the chain.
- Prints what would be deleted before deleting (dry-run first).

### Example

```
$ planexe invalidate SelectScenarioTask --run-dir ./run/Qwen_Clean_v1
Would delete:
  run/Qwen_Clean_v1/002-17-selected_scenario_raw.json
  run/Qwen_Clean_v1/002-18-selected_scenario.json
  run/Qwen_Clean_v1/002-19-scenarios.md
Proceed? [y/N]
```

### Why this matters

Tonight we needed to re-run `SelectScenarioTask` after applying a fix. Without knowing exactly which files to delete, the safe move is to delete all files from that task number onward — which means re-running 40+ already-complete tasks. A targeted invalidation command makes surgical retries possible.

---

## 3. Plan File Hot-Editing with Downstream Invalidation

### The problem

The input plan (`001-2-plan.txt`) is locked in at run start. If a user wants to refine the plan description mid-run — clarify scope, correct a factual error, tighten the framing — there is no supported path. The only option is start a new run from scratch.

### What we want

A mechanism to edit the plan file and selectively invalidate downstream tasks:

```bash
planexe edit-plan --run-dir ./run/Qwen_Clean_v1
# opens plan.txt in $EDITOR
# after save, asks: "Invalidate all tasks? [Y/n]"
# or: "Which tasks to invalidate? (comma-separated, or 'all')"
```

Alternatively: a `--invalidate-from <TaskName>` flag that cascades from a specific task boundary, allowing early tasks that don't use the plan text directly to be preserved.

### Why this matters

On local hardware, a full re-run can take 4–6 hours. If a user notices a plan description issue at task 20, they currently have to restart from scratch. Targeted invalidation from the point where the plan text first materially influences output would save hours.

---

## 4. Pipeline Status Command

### The problem

There is no quick way to ask "where is this pipeline right now?" without parsing `log.txt`. The run directory contains output files, but mapping file names to task names and status requires knowing the file-naming convention.

### What we want

```bash
planexe status --run-dir ./run/Qwen_Clean_v1
```

Output:

```
Run: Qwen_Clean_v1
Model: lmstudio-qwen3.5-35b-a3b
Started: 2026-03-06 19:47 UTC

✅ DONE  (23 tasks)
  SetupTask, StartTimeTask, RedlineGateTask, ...

⏳ RUNNING (1 task)
  PreProjectAssessmentTask — started 21:53 UTC

❌ FAILED (1 task)
  PreProjectAssessmentTask — "1 validation error for ExpertDetails"

⬜ PENDING (38 tasks)
  IdentifyRisksTask, CreateWBSLevel3Task, ...
```

### Why this matters

Agents and humans checking in on an overnight run need this information immediately. Currently it requires log-parsing expertise. A status command makes the pipeline observable without tribal knowledge.

---

## 5. Per-Task Timeout Configuration

### The problem

LM Studio's `request_timeout` is set globally in the model config JSON. Some tasks (e.g., `PreProjectAssessmentTask` with multiple expert sub-calls) consistently take longer than others and hit the global timeout. Raising the global timeout to accommodate slow tasks means slow failure detection for tasks that genuinely hang.

### What we want

Per-task timeout overrides in the model config or a separate task config file:

```json
{
  "task_timeouts": {
    "PreProjectAssessmentTask": 900,
    "CreateWBSLevel3Task": 1200,
    "default": 600
  }
}
```

### Why this matters

`PreProjectAssessmentTask` runs multiple expert sub-calls sequentially. On Qwen 3.5-35B-A3B, each expert call takes 2–4 minutes. A 600-second global timeout is right for most tasks but causes `ReadTimeout` failures on this specific task. The fix tonight was a full LM Studio restart — not ideal at 10pm when a pipeline is running unattended.

---

## 6. Structured Failure Log

### The problem

When a task fails, the error goes into `log.txt` as a Python traceback mixed with INFO/DEBUG lines. Extracting the failure signature (task name, model, error type, missing fields, attempt number) requires log-parsing.

### What we want

On any task failure, append a structured entry to `failures.jsonl` in the run directory:

```json
{
  "timestamp_utc": "2026-03-06T21:55:22Z",
  "task": "PreProjectAssessmentTask",
  "model": "lmstudio-qwen3.5-35b-a3b",
  "attempt": 1,
  "error_type": "ValidationError",
  "missing_fields": ["combined_summary", "go_no_go_recommendation"],
  "invalid_fields": [],
  "raw_error": "1 validation error for ExpertDetails...",
  "run_id_dir": "/path/to/run"
}
```

### Why this matters

Structured failure data enables:
- Automated retry decisions (is this truncation or a model capability gap?)
- Cross-run comparison (does the same task fail on different models?)
- PR evidence (attach `failures.jsonl` excerpt to PRs as proof)
- Model scorecards (which models fail at which tasks?)

Tonight, diagnosing each failure required reading a Python traceback and manually extracting the task name, model, and missing fields. A `failures.jsonl` file would have made that instant.

---

## Implementation Priority

Ordered by value-to-effort ratio:

| Priority | Feature | Effort | Value |
|----------|---------|--------|-------|
| 1 | Structured failure log (`failures.jsonl`) | Low | High |
| 2 | Pipeline status command | Medium | High |
| 3 | Task invalidation CLI | Medium | High |
| 4 | Webhook / event hooks | Medium | High |
| 5 | Per-task timeout config | Low | Medium |
| 6 | Plan hot-editing + cascade | High | Medium |

Items 1 and 5 are config/logging changes with minimal blast radius. Items 2–4 are new CLI surface area but don't touch pipeline logic. Item 6 requires careful dependency graph analysis.

---

## Decision Request

Approve this roadmap and we will implement each item as a separate focused PR, starting with structured failure logging (item 1) and the task invalidation CLI (item 3).

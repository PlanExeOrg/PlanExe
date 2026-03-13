# PlanExe MCP Interface Specification (v1.0)

## 1. Purpose

### 1.1 What is PlanExe

PlanExe is a service that generates **strategic project-plan drafts** from a natural-language prompt. You describe a large goal (e.g. open a clinic, launch a product, build a moon base)—the kind of project that in reality takes months or years. PlanExe produces a structured draft with 20+ sections: steps, documents, and deliverables. The plan is not executable in its current form; it is a draft to refine and act on. Creating a plan is a long-running operation (100+ LLM inference calls): create a plan with a prompt, poll status, then download the HTML report and zip when done.

### 1.2 What kind of plan does it create

The plan is a **project plan**: a DAG of steps (Luigi pipeline stages) that produce artifacts including a Gantt chart, risk analysis, and other project management deliverables. The main output is a self-contained interactive HTML report (~700KB) with collapsible sections, interactive Gantt charts, and embedded JavaScript. The report contains 20+ sections including executive summary, investor pitch, project plan with SMART criteria, strategic decision analysis, scenario comparison, assumptions with expert review, governance structure, SWOT analysis, team role profiles, simulated expert criticism, work breakdown structure, plan review, Q&A, premortem with failure scenarios, self-audit checklist, and adversarial premise attacks. There is also a zip file containing all intermediary pipeline files (md, json, csv) that fed the report. Plan quality depends on prompt quality; use the example_prompts tool to see the baseline before calling plan_create.

#### 1.2.1 Agent-facing summary (for server instructions / tool descriptions)

Implementors should expose the following to agents so they understand what PlanExe does:

- **What:** PlanExe turns a plain-English goal into a strategic project-plan draft (20+ sections) in ~10–20 min. Sections include executive summary, interactive Gantt charts, investor pitch, SWOT, governance, team profiles, work breakdown, scenario comparison, expert criticism, and adversarial sections (premortem, self-audit, premise attacks) that stress-test the plan. The output is a draft to refine, not an executable or final document — but it surfaces hard questions the prompter may not have considered.
- **Required interaction order:** Call `example_plans` (optional) and `example_prompts` first. Optional before `plan_create`: call `model_profiles` to inspect profile guidance and available models in each profile. Then complete a non-tool step: formulate a detailed prompt as flowing prose (not structured markdown), typically ~300-800 words, using the examples as a baseline; include objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria; get user approval. Only after approval, call `plan_create`. Then poll `plan_status` (about every 5 minutes); use `plan_download` (mcp_local helper) or `plan_file_info` (mcp_cloud tool) when complete (`pending`/`processing` = keep polling, `completed` = download now, `failed` = terminal error, `stopped` = user called plan_stop). If a plan fails or is stopped before completing all steps, call `plan_resume` to continue from where it left off without discarding completed work. Use `plan_retry` for a full restart (plan must be in failed or stopped state). Both accept the `plan_id` and optional `model_profile` (default `baseline`). To stop, call `plan_stop` with the `plan_id` from `plan_create`.
- **Output:** Self-contained interactive HTML report (~700KB) with collapsible sections and interactive Gantt charts — open in a browser. The zip contains the intermediary pipeline files (md, json, csv) that fed the report.

### 1.3 Scope of this document

This document specifies a Model Context Protocol (MCP) interface for PlanExe that enables AI agents and client UIs to:

1. Create and run long-running plan generation workflows.
2. Receive real-time progress updates (plan status, log output).
3. List, read, and edit artifacts produced in an output directory.
4. Stop and resume execution with Luigi-aware incremental recomputation.

The interface is designed to support:

- interactive "build systems" behavior (like make / bazel),
- resumable DAG execution (Luigi),
- deterministic artifact management.

---

## 2. Goals

### 2.1 Functional goals

- Plan-based orchestration: each run is associated with a plan ID.
- Long-running execution: starts asynchronously; clients poll or subscribe to events.
- Artifact-first workflow: outputs are exposed as file-like artifacts.
- Stop / Resume with minimal recompute:
  - on resume, only invalidated downstream stages regenerate.
- Progress reporting:
  - progress_percentage
- Editable artifacts:
  - user edits a generated file
  - pipeline continues from that point, producing dependent outputs

### 2.2 Non-functional goals

- Idempotency: repeated tool calls should not corrupt state.
- Observability: logs, state transitions, and artifacts must be inspectable.
- Concurrency safety: prevent conflicting writes and illegal resume patterns.
- Extensibility: future versions can add plan graph browsing, caching backends, exports.
- Agent-native: the interface is designed for autonomous AI agents as the primary consumer. See the [Autonomous Agent Guide](autonomous_agent_guide.md) for the complete agent workflow.

---

## 3. Non-goals

- Defining PlanExe's internal plan schema, content format, or prompt strategy.
- Providing remote code execution inside artifacts.
- Implementing a full Luigi UI clone in MCP v1 (optional later).
- Guaranteeing ETA estimates (allowed but must be optional / best-effort).

### 3.1 MCP tools vs MCP tasks ("Run as task")

The MCP specification defines two different mechanisms:

- **MCP tools** (e.g. plan_create, plan_status, plan_stop, plan_retry, plan_resume): the server exposes named tools; the client calls them and receives a response. PlanExe's interface is **tool-based**: the agent calls plan_create → receives plan_id → polls plan_status → optionally calls plan_resume or plan_retry on failed/stopped → uses plan_file_info (and optionally plan_download via mcp_local). This document specifies those tools.
- **MCP tasks protocol** ("Run as task" in some UIs): a separate mechanism where the client can run a tool "as a task" using RPC methods such as tasks/run, tasks/get, tasks/result, tasks/cancel, tasks/list, so the tool runs in the background and the client polls for results.

PlanExe **does not** use or advertise the MCP tasks protocol. Implementors and clients should use the **tools only**. Do not enable "Run as task" for PlanExe; many clients (e.g. Cursor) and the Python MCP SDK do not support the tasks protocol properly. Intended flow: optionally call `example_plans`; call `example_prompts`; optionally call `model_profiles`; perform the non-tool prompt drafting/approval step; call `plan_create`; poll `plan_status`; if failed or stopped call `plan_resume` to continue or `plan_retry` for a full restart (optional); then call `plan_file_info` (or `plan_download` via mcp_local) when completed.

---

## 4. System Model

### 4.1 Core entities

#### Plan

A long-lived container for a PlanExe project run.

**Key properties**

- plan_id: UUID returned by plan_create for that plan. Each plan_create returns a new UUID. Use that exact UUID for all MCP calls; do not substitute ids from other services.
- output_dir: artifact root namespace for plan
- config: immutable run configuration (models, runtime limits, Luigi params)
- created_at, updated_at

#### Execution

A single execution attempt inside a plan.

**Key properties**

- state: pending | processing | completed | failed | stopped
- progress_percentage: computed progress percentage (float)
- started_at, ended_at

#### Artifact

A file-like output managed by PlanExe.

**Key properties**

- path: path relative to plan output root
- size, updated_at
- content_type: text/markdown, text/html, application/json, etc.
- sha256: content hash for optimistic locking and invalidation

#### Event

A typed message emitted during execution for UI/agent consumption.

**Key properties**

- cursor: ordering token
- ts: timestamp
- type: event type
- data: event payload

---

## 5. State Machine

### 5.1 PlanItem.state values

The public MCP `state` field is aligned with `PlanItem.state`:

- pending (queued, waiting for a worker)
- processing (picked up by a worker)
- completed
- failed
- stopped (user called `plan_stop`)

### 5.2 Allowed transitions

- pending → processing when picked up by a worker
- processing → completed via normal success
- processing → failed via error
- processing → stopped via `plan_stop`
- failed → pending when `plan_retry` or `plan_resume` is accepted
- stopped → pending when `plan_retry` or `plan_resume` is accepted

### 5.3 Invalid transitions

- completed → processing (new run must be triggered by creating a new plan)
- processing → processing is not a state transition on the same plan; create separate plans for parallel work.

---

## 6. MCP Tools (v1 Required)

All tool names below are normative.

### 6.1 example_prompts

**Call this first.** Returns example prompts that define the baseline for what a good prompt looks like. Do not call plan_create yet. Correct flow: optionally call `example_plans`; call this tool; optionally call `model_profiles`; then complete a non-tool step (draft and approve a detailed prompt, typically ~300-800 words); only then call `plan_create`. If you call `plan_create` before formulating and approving a prompt, the resulting plan will be lower quality than it could be.

Write the prompt as flowing prose, not structured markdown with headers or bullet lists. Weave technical specs, constraints, and targets naturally into sentences. Include banned words/approaches and governance structure inline. Typical length: 300–800 words. The examples demonstrate this prose style — match their tone and density.

**Request:** no parameters (empty object).

**Response:**

```json
{
  "samples": ["prompt text 1", "prompt text 2", "..."],
  "message": "..."
}
```

---

### 6.1.1 model_profiles

Optional helper tool to discover valid `model_profile` choices and currently available models without relying on internal config knowledge.
Profiles with zero available models are omitted from the returned `profiles` array.
If no models are available in any profile, the tool returns `isError=true` with `error.code = MODEL_PROFILES_UNAVAILABLE`.

**Request:** no parameters (empty object).

**Response (shape)**

```json
{
  "default_profile": "baseline",
  "profiles": [
    {
      "profile": "baseline",
      "title": "Baseline",
      "summary": "Cheap and fast; recommended default when creating a plan.",
      "model_count": 5,
      "models": [
        {
          "key": "openrouter-gpt-oss-20b",
          "provider_class": "OpenRouter",
          "model": "openai/gpt-oss-20b",
          "priority": 0
        }
      ]
    }
  ],
  "message": "..."
}
```

Use the returned `profile` values directly in `plan_create.model_profile`.

---

### 6.2 plan_create

**Call only after example_prompts and after the non-tool drafting/approval step.** Start creating a new plan with the approved prompt.

**Request**

**Schema**

```json
{
  "type": "object",
  "properties": {
    "prompt": { "type": "string" },
    "model_profile": {
      "type": "string",
      "enum": ["baseline", "premium", "frontier", "custom"],
      "default": "baseline"
    },
    "user_api_key": { "type": "string" }
  },
  "required": ["prompt"]
}
```

**Example**

```json
{
  "prompt": "string",
  "model_profile": "baseline",
  "user_api_key": "pex_..."
}
```

**Prompt quality**

The `prompt` parameter should be a detailed description of what the plan should cover. Good prompts are typically 300-800 words and include:

- Objective
- Scope
- Constraints
- Timeline
- Stakeholders
- Budget/resources
- Success criteria

Write as flowing prose, not structured markdown. Include banned approaches, governance preferences, and phasing inline. Short one-liners (e.g., "Construct a bridge") tend to produce poor output because they lack context for the planning pipeline. Important details are location, budget, time frame.

**Counterexamples: when NOT to use PlanExe**

Use a normal single LLM response (not PlanExe) for one-shot micro-tasks. PlanExe runs a heavy multi-step planning pipeline and is best for substantial project planning.

- Bad (do not send to plan_create): "Give me a 5-point checklist for launching a coffee shop."
- Better non-PlanExe action: ask the LLM directly for a checklist.
- Better PlanExe prompt: "Create a 12-month strategic launch plan for a coffee shop in Austin with budget caps, lease milestones, hiring plan, permits, supply chain, marketing channels, risk register, governance, and success KPIs."

- Bad (do not send to plan_create): "Summarize this text in 6 bullets."
- Better non-PlanExe action: use direct summarization in the chat model.

- Bad (invalid assumption): "Run only the risk-register part of PlanExe."
- Rule: PlanExe pipeline execution is fixed end-to-end. Callers cannot choose internal step subsets.
- Better PlanExe prompt: request a full plan where risk analysis is one required deliverable.

- Bad (do not send to plan_create): "Rewrite this email to sound professional."
- Better non-PlanExe action: use direct rewriting in the chat model.

**Optional**

- model_profile: LLM profile (`baseline` | `premium` | `frontier` | `custom`). If unsure, call `model_profiles` first.
- user_api_key: user API key for credits and attribution (if your deployment requires it).

Clients can call the MCP tool **example_prompts** to retrieve example prompts. Use these as examples for plan_create; they can also call plan_create with any prompt—short prompts produce less detailed plans.

For the full catalog file:

- `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl` — JSONL with `id`, `prompt`, optional `tags`, and optional `mcp_example` (true = curated for MCP).

**Response**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "created_at": "2026-01-14T12:34:56Z"
}
```

**Important**

- plan_id is a UUID returned by plan_create. Use this exact UUID for plan_status/plan_stop/plan_retry/plan_file_info (and plan_download when using mcp_local).

**Behavior**

- Must be idempotent only if client supplies an optional client_request_id (optional extension).
- Plan config is immutable after creation in v1.
- By default, repeated `plan_create` calls produce new plans (new `plan_id`s).

---

### 6.3 plan_status

Returns plan status and progress. Used for progress bars and UI states. **Polling interval:** call at reasonable intervals only (e.g. every 5 minutes); plan generation typically takes 10–20 minutes (baseline profile) and may take longer on higher-quality profiles.

**Request**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1"
}
```

**Input**

- plan_id: UUID returned by plan_create. Use it to reference the plan being created.

**Caller contract (state meanings)**

- `pending`: queued and waiting for a worker. Keep polling.
- `processing`: picked up by a worker and in progress. Keep polling.
- `completed`: terminal success. Download artifacts now.
- `failed`: terminal error. Do not keep polling for completion.
- `stopped`: user called `plan_stop`. Consider `plan_resume` to continue or `plan_retry` to restart.

**Terminal states**

- `completed`, `failed`, `stopped`

**Response**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "state": "processing",
  "progress_percentage": 62.0,
  "steps_completed": 19,
  "steps_total": 30,
  "current_step": "SWOT Analysis",
  "timing": {
    "started_at": "2026-01-14T12:35:10Z",
    "elapsed_sec": 512
  },
  "files_count": 42,
  "files": [
    {
      "path": "019-swot_analysis.md",
      "updated_at": "2026-01-14T12:43:11Z"
    }
  ]
}
```

**Failure diagnostics**

When `state` is `"failed"`, the response includes a consolidated `error` dict to help agents understand the failure and recommend recovery:

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "state": "failed",
  "progress_percentage": 53.0,
  "error": {
    "failure_reason": "generation_error",
    "failed_step": "016-expert_criticism",
    "message": "LLM provider returned 503",
    "recoverable": true
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `error.failure_reason` | string (nullable) | Failure category: `generation_error`, `worker_error`, `inactivity_timeout`, `internal_error`, `version_mismatch`. |
| `error.failed_step` | string (nullable) | Pipeline step that was active at time of failure (e.g. `"016-expert_criticism"`). |
| `error.message` | string (nullable) | Human-readable error message (max 256 chars). |
| `error.recoverable` | boolean (nullable) | `true` → suggest `plan_resume`; `false` → suggest `plan_retry`. |

These fields are `null` for legacy plans created before failure diagnostics were added. The `error` dict is absent for non-failed states.

**Agent decision guidance:**

- `recoverable: true` → call `plan_resume` to continue from the failed step.
- `recoverable: false` → call `plan_retry` for a clean restart.
- `failure_reason: "version_mismatch"` → always use `plan_retry` (snapshot is from a different pipeline version).

**Notes**

- progress_percentage must be a float within [0,100].
- `files` contains the most recent 10 files (sorted by pipeline step number). `files_count` gives the total number of files produced. This lets agents see what the pipeline just produced rather than always seeing the same early files.

---

### 6.4 plan_stop

Requests the plan generation to stop. Pass the **plan_id** (the UUID returned by plan_create). Call `plan_stop` with that plan_id.

**Request**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1"
}
```

**Input**

- plan_id: UUID returned by plan_create. Use this same UUID when calling plan_stop to request the plan to stop.

**Response**

```json
{
  "state": "processing",
  "stop_requested": true
}
```

**Required semantics**

- Must stop workers cleanly where possible.
- Must persist enough Luigi state to resume incrementally.

---

### 6.5 plan_retry

Retries a plan that is currently in `failed` state.

**Request**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "model_profile": "baseline"
}
```

**Input**

- plan_id: UUID of a failed plan.
- model_profile: optional (`baseline` | `premium` | `frontier` | `custom`), default `baseline`.

**Response**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "state": "pending",
  "model_profile": "baseline",
  "retried_at": "2026-02-24T15:20:00Z"
}
```

**Required semantics**

- Only failed plans are retryable.
- On success, the same plan_id is reset to `pending` and requeued.
- Prior artifacts for that plan are cleared before requeue.

**Error behavior**

- Unknown plan_id: `PLAN_NOT_FOUND` (`isError=true`).
- Plan not failed: `PLAN_NOT_FAILED` (`isError=true`).

---

### 6.6 plan_resume

Resume a failed plan without discarding completed intermediary files. Plan generation restarts from the first incomplete step, skipping all steps that already produced output files.

Use `plan_resume` when `plan_status` shows `failed` or `stopped` and plan generation was interrupted before completing all steps (network drop, timeout, `plan_stop`, worker crash). For a full restart or to change `model_profile`, use `plan_retry` instead.

**Request**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "model_profile": "baseline"
}
```

**Input**

- plan_id: UUID of a failed plan.
- model_profile: optional (`baseline` | `premium` | `frontier` | `custom`), default `baseline`.

**Response**

```json
{
  "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "state": "pending",
  "model_profile": "baseline",
  "resume_count": 1,
  "resumed_at": "2026-03-09T15:20:00Z"
}
```

**Required semantics**

- The MCP tool only accepts plans in `failed` state. However, the underlying Luigi mechanism is more general: Luigi skips any task whose output file already exists and re-executes any task whose output file is missing. This means a completed plan can be partially re-run by deleting `999-pipeline_complete.txt` and the output files of the tasks you want to regenerate — Luigi will re-execute those tasks and all their downstream dependents. The MCP API does not yet expose this capability; it is available when running the pipeline locally via `run_plan_pipeline.py`.
- On success, the same plan_id is reset to `pending` and requeued.
- Prior artifacts are **preserved** — the worker restores the output directory from the stored zip snapshot.
- `resume_count` tracks how many times the plan has been resumed.

**When to use plan_resume vs plan_retry**

| Scenario | Use |
|----------|-----|
| Interrupted before completing all steps (network drop, timeout, stop, crash) | `plan_resume` |
| Full restart from scratch | `plan_retry` |
| Change model_profile for a fresh start | `plan_retry` |
| Continue where you left off, preserving completed work | `plan_resume` |

**Pipeline compatibility caveat**

Resuming restores the output directory from a stored zip snapshot and relies on Luigi's file-based step skipping: if an output file exists, the step is skipped. This works reliably when the server code matches the code that produced the snapshot. It can break when the server has changed between the original plan generation and the resume attempt:

- **Renamed or removed filenames** (`FilenameEnum` in `filenames.py`): if a step's output filename changes, Luigi won't find the old file and will re-execute the step — but downstream steps that already completed may reference the old filename's content, producing inconsistencies.
- **New steps added**: new steps may expect upstream outputs that don't exist in the snapshot.
- **Changed step dependencies**: if the DAG wiring changes, Luigi may execute steps in a different order or skip steps that should re-execute.
- **Changed JSON schemas**: resumed steps may read snapshot files whose internal structure no longer matches what the current code expects.

In practice, resuming a plan that failed minutes or hours ago on the same deployment is safe. Resuming a plan that is weeks or months old — after the server has been updated with pipeline changes — may produce errors or inconsistent output. When in doubt, use `plan_retry` for a clean restart.

**Pipeline version check**

A `PIPELINE_VERSION` integer constant (in `worker_plan_api/pipeline_version.py`) is stamped into the plan's `parameters` on every `plan_create`, `plan_retry`, and `plan_resume`. On `plan_resume`, the stored version is compared to the current constant using strict equality. If they differ, resume is rejected with error code `PIPELINE_VERSION_MISMATCH` and the caller should use `plan_retry` for a clean restart. Legacy plans with no stored version are treated as incompatible.

Bump `PIPELINE_VERSION` whenever the pipeline changes in a way that would break resume from an older snapshot (renamed filenames, new tasks, changed DAG wiring, changed JSON schemas). Finer-grained approaches (manifest, per-task schema versions) can be layered on later if needed.

**Error behavior**

- Unknown plan_id: `PLAN_NOT_FOUND` (`isError=true`).
- Plan not failed: `PLAN_NOT_RESUMABLE` (`isError=true`).
- Pipeline version mismatch: `PIPELINE_VERSION_MISMATCH` (`isError=true`). Use `plan_retry` instead.

---

### 6.7 Download flow (plan_download vs plan_file_info)

**If your client exposes plan_download** (e.g. mcp_local): use it to save the report or zip locally; it calls plan_file_info under the hood, then fetches and writes to the local save path (e.g. PLANEXE_PATH).

**If you only have plan_file_info** (e.g. direct connection to mcp_cloud): call it with plan_id and artifact ("report" or "zip"); use the returned download_url to fetch the file (e.g. GET with API key if configured).

**plan_file_info input**

- plan_id: UUID returned by plan_create. Use it to download the created plan.
- artifact: "report" or "zip" (default "report").

**plan_download local path behavior (mcp_local)**

- Save directory is `PLANEXE_PATH`.
- If `PLANEXE_PATH` is unset, save to current working directory.
- If `PLANEXE_PATH` points to a file (not a directory), return an error.
- Filenames are `<plan_id>-030-report.html` or `<plan_id>-run.zip`.
- If a filename already exists, append `-1`, `-2`, ... before extension.
- Successful responses include `saved_path`.

**plan_file_info URL behavior (mcp_cloud)**

- `download_url` is an absolute URL where the requested artifact can be downloaded.

---

## 7. Targets

### 7.1 Standard targets

The following targets MUST be supported:

- build_plan
- validate_plan
- build_plan_and_validate

Targets map to Luigi "final tasks".

---

## 8. Concurrency & Locking

### 8.1 Client-side concurrency guidance

The server does not enforce a global limit on how many plans a client can create.
Concurrency is a client-side coordination concern.

Recommended practice for MCP clients:

- Start with 1 active plan.
- If needed, increase to 2 plans in parallel.
- Going beyond 4 parallel plans is usually hard to track; avoid unless necessary.

Additional semantics:

- Every `plan_create` call creates a new independent plan with a new `plan_id`.
- `plan_retry` and `plan_resume` reuse the existing failed or stopped `plan_id` (they do not create a new plan id).
- The server does not deduplicate “same prompt” requests into a single shared plan.
- Keep your own plan registry/client state if you run multiple plans concurrently.

---

## 9. Error Model

### 9.1 Error object shape

Tool errors return:

- `error.code`: stable machine-readable string
- `error.message`: human-readable message
- `error.details`: optional object

Example:

```json
{
  "error": {
    "code": "PLAN_NOT_FOUND",
    "message": "Plan not found: <plan_id>"
  }
}
```

### 9.2 isError behavior

- `plan_create`, `plan_status`, `plan_stop`, `plan_retry`, `plan_resume`: unknown/invalid requests return `isError=true` with `error`.
- `model_profiles`: returns `isError=true` with `MODEL_PROFILES_UNAVAILABLE` when no models are available in any profile.
- `plan_file_info`: uses mixed behavior:
  - returns `{}` (not an error) while artifacts are not ready.
  - may return `{"error": ...}` with `isError=false` for terminal artifact-level problems.
  - returns `isError=true` for unknown plan_id (`PLAN_NOT_FOUND`).
- `mcp_local` may return proxy/transport failures as `REMOTE_ERROR` and local download write failures as `DOWNLOAD_FAILED`.

### 9.3 Minimal code contract (current)

Cloud/core tool codes:

- `INVALID_TOOL`: unknown MCP tool name.
- `INTERNAL_ERROR`: uncaught server error.
- `PLAN_NOT_FOUND`: plan_id not found.
- `PLAN_NOT_FAILED`: plan_retry called for a plan that is not in failed or stopped state.
- `PLAN_NOT_RESUMABLE`: plan_resume called for a plan that is not in failed or stopped state.
- `PIPELINE_VERSION_MISMATCH`: plan_resume snapshot was created by a different pipeline version; use plan_retry instead.
- `INVALID_USER_API_KEY`: provided user_api_key is invalid.
- `USER_API_KEY_REQUIRED`: deployment requires user_api_key for plan_create.
- `INSUFFICIENT_CREDITS`: caller account has no credits for plan_create.
- `MODEL_PROFILES_UNAVAILABLE`: model_profiles found zero available models across all profiles.
- `generation_failed`: plan_file_info report path when plan ended in failed.
- `content_unavailable`: plan_file_info cannot read requested artifact bytes.

Local proxy specific codes:

- `REMOTE_ERROR`: mcp_local could not call mcp_cloud (network/HTTP/protocol layer failure).
- `DOWNLOAD_FAILED`: mcp_local could not write/download artifact to local filesystem.

### 9.4 Caller handling guidance

- Retry with backoff:
  - `INTERNAL_ERROR`
  - `REMOTE_ERROR`
  - `content_unavailable` (short retry window)
- Do not retry unchanged request:
  - `INVALID_USER_API_KEY`
  - `USER_API_KEY_REQUIRED`
  - `INSUFFICIENT_CREDITS`
  - `INVALID_TOOL`
- For `PLAN_NOT_FAILED`: call `plan_retry` only after `plan_status.state` is `failed` or `stopped`.
- For `PLAN_NOT_RESUMABLE`: call `plan_resume` only after `plan_status.state` is `failed` or `stopped`.
- For `PIPELINE_VERSION_MISMATCH`: the snapshot is incompatible with the current pipeline; use `plan_retry` for a clean restart.
- For `PLAN_NOT_FOUND`: verify plan_id source and stop polling that id.
- For `generation_failed`: treat as terminal failure and surface plan progress_message to user.

---

## 10. Security & Isolation

### 10.1 Sandbox constraints

- All artifacts must live under plan-scoped storage.
- Artifact URIs must not permit path traversal.

### 10.2 Access control

At minimum:

- plan must be scoped to a user identity (metadata.user_id)
- callers without permission must receive PERMISSION_DENIED

### 10.3 Sensitive data handling

- logs may include model prompts/responses → treat logs as sensitive artifacts
- allow a config option to redact prompt content in event streaming

### 10.4 Authentication mode

- MCP authentication is API-key header based.
- Clients should send `X-API-Key: pex_...` on MCP requests.
- OAuth is not supported for the MCP API.

---

## 11. Performance Requirements

### 11.1 Responsiveness

- plan_status must return within < 250ms under normal load.

### 11.2 Large artifacts

- server SHOULD impose max read size per call (e.g., 2–10MB)

---

## 12. Observability Requirements

The server MUST persist:

- run lifecycle events
- stop reasons
- failure tracebacks as artifacts (e.g., run_error.json)
- luigi execution logs (run.log)

---

## 13. Reference UI Integration Contract

To match your UI behavior:

**Progress bars**

Use:

- plan_status.progress_percentage
- or progress_updated events

---

## 14. Compatibility & Versioning

### 14.1 Versioning strategy

- MCP server exposes: planexe.version = "1.0"
- breaking changes require major bump

### 14.2 Forward compatibility

Clients must ignore unknown fields and unknown event types.

---

## 15. Testing Strategy

### 15.1 Contract tests (required)

- Start/stop/resume loops
- Invalid transition errors
- Event cursor monotonicity

### 15.2 Determinism tests (recommended)

- Given same inputs + same edits, ensure same downstream artifacts unless models are stochastic
- If models are stochastic, test pipeline correctness, not identical bytes

### 15.3 Load tests (recommended)

- multiple plans concurrently, one run each
- event streaming stability under heavy log output

---

## 16. Future Extensions (MCP Resources)

PlanExe is artifact-first, and MCP already has a native concept for that: resources. Today artifacts are exposed via download_url or via proxy download + saved_path. Future versions SHOULD expose artifacts as MCP resources so clients can fetch them via standard resource reads (and treat PlanExe as a first-class MCP server rather than a thin API wrapper).

**Proposed resource identifiers**

- planexe://plan/<plan_id>/report
- planexe://plan/<plan_id>/zip

**Recommended resource metadata**

- mime type (content_type)
- size (bytes)
- sha256 (content hash)
- generated_at (UTC timestamp)

**Notes**

- Resources can be backed by existing HTTP endpoints internally; the MCP resource read returns the bytes + metadata.
- This enables richer MCP client UX (preview, caching, validation) without custom tool calls.

---

## 17. Future Tools (High-Leverage, Low-Complexity)

The following tools remove common UX friction without expanding the core model.

### ~~17.1 plan_list~~ (IMPLEMENTED)

Implemented. Returns up to 50 plans newest-first, each with plan_id, state, progress_percentage, created_at, and prompt_excerpt. Default limit: 10.

### 17.2 plan_wait

Blocking helper that polls internally until the plan completes or times out. Returns the final plan_status payload plus suggested next steps.

**Notes**

- Inputs: plan_id, timeout_sec (optional), poll_interval_sec (optional).
- Outputs: same as plan_status + next_steps (string or list).

### 17.3 plan_get_latest

Simplest recovery: return the most recently created plan for the caller.

**Notes**

- Useful for single-user / single-session flows.
- Should be scoped to the caller/user_id when available.

### 17.4 plan_logs_tail (optional)

Return the tail of recent log lines for troubleshooting failures.

**Notes**

- Inputs: plan_id, max_lines (optional), since_cursor (optional).
- Useful when plan_status shows failed but no context.

---

## Appendix A — Example End-to-End Flow

**Create plan**

```json
{ "prompt": "..." }
```

**Start run**

```json
{ "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }
```

**Stop**

```json
{ "plan_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }
```

---

## Appendix B — Optional v1.1 Extensions

If you want richer Luigi integration later:

- planexe.plan.graph (nodes + edges + states)
- planexe.plan.invalidate (rerun subtree)
- planexe.export.bundle (zip all artifacts)
- planexe.validate.only (audit without regeneration)
- planexe.plan.archive (freeze plan)

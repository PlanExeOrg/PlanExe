# PlanExe MCP Interface Specification (v1.0)

## 1. Purpose

### 1.1 What is PlanExe

PlanExe is a service that generates **rough-draft project plans** from a natural-language prompt. You describe a large goal (e.g. open a clinic, launch a product, build a moon base)—the kind of project that in reality takes months or years. PlanExe produces a structured draft: steps, documents, and deliverables. The plan is not executable in its current form; it is a draft to refine and act on. Creating a plan is a long-running task (100+ LLM inference calls): create a task with a prompt, poll status, then download the HTML report and zip when done.

### 1.2 What kind of plan does it create

The plan is a **project plan**: a DAG of steps (Luigi tasks) that produce artifacts including a Gantt chart, risk analysis, and other project management deliverables. The main output is a large HTML file (approx 700KB) containing many sections. There is also a zip file containing all intermediary files (md, json, csv). Plan quality depends on prompt quality; use the prompt_examples tool to see the baseline before calling task_create.

#### 1.2.1 Agent-facing summary (for server instructions / tool descriptions)

Implementors should expose the following to agents so they understand what PlanExe does:

- **What:** PlanExe turns a plain-English goal into a structured strategic-plan draft (executive summary, Gantt, risk register, governance, etc.) in ~15–20 min. The plan is a draft to refine, not an executable or final document.
- **Required interaction order:** Step 1 — Call prompt_examples to fetch example prompts. Step 2 — Formulate a good prompt (use examples as a baseline; similar structure; get user approval). Step 3 — Only then call task_create with the approved prompt. Then poll task_status; use task_download or task_file_info when complete. To stop, call task_stop with the task_id from task_create.
- **Output:** Large HTML report (~700KB) and optional zip of intermediate files (md, json, csv).

### 1.3 Scope of this document

This document specifies a Model Context Protocol (MCP) interface for PlanExe that enables AI agents and client UIs to:

1. Create and run long-running plan generation workflows.
2. Receive real-time progress updates (task status, log output).
3. List, read, and edit artifacts produced in an output directory.
4. Stop and resume execution with Luigi-aware incremental recomputation.

The interface is designed to support:

- interactive "build systems" behavior (like make / bazel),
- resumable DAG execution (Luigi),
- deterministic artifact management.

---

## 2. Goals

### 2.1 Functional goals

- Task-based orchestration: each run is associated with a task ID.
- Long-running execution: starts asynchronously; clients poll or subscribe to events.
- Artifact-first workflow: outputs are exposed as file-like artifacts.
- Stop / Resume with minimal recompute:
  - on resume, only invalidated downstream tasks regenerate.
- Progress reporting:
  - progress_percentage
- Editable artifacts:
  - user edits a generated file
  - pipeline continues from that point, producing dependent outputs

### 2.2 Non-functional goals

- Idempotency: repeated tool calls should not corrupt state.
- Observability: logs, state transitions, and artifacts must be inspectable.
- Concurrency safety: prevent conflicting writes and illegal resume patterns.
- Extensibility: future versions can add task graph browsing, caching backends, exports.

---

## 3. Non-goals

- Defining PlanExe's internal plan schema, content format, or prompt strategy.
- Providing remote code execution inside artifacts.
- Implementing a full Luigi UI clone in MCP v1 (optional later).
- Guaranteeing ETA estimates (allowed but must be optional / best-effort).

### 3.1 MCP tools vs MCP tasks ("Run as task")

The MCP specification defines two different mechanisms:

- **MCP tools** (e.g. task_create, task_status, task_stop): the server exposes named tools; the client calls them and receives a response. PlanExe's interface is **tool-based**: the agent calls task_create → receives task_id → polls task_status → uses task_download. This document specifies those tools.
- **MCP tasks protocol** ("Run as task" in some UIs): a separate mechanism where the client can run a tool "as a task" using RPC methods such as tasks/run, tasks/get, tasks/result, tasks/cancel, tasks/list, so the tool runs in the background and the client polls for results.

PlanExe **does not** use or advertise the MCP tasks protocol. Implementors and clients should use the **tools only**. Do not enable "Run as task" for PlanExe; many clients (e.g. Cursor) and the Python MCP SDK do not support the tasks protocol properly. The intended flow is: Step 1 — call prompt_examples; Step 2 — formulate a good prompt (user approval); Step 3 — call task_create; then poll task_status and call task_download when complete.

---

## 4. System Model

### 4.1 Core entities

#### Task

A long-lived container for a PlanExe project run.

**Key properties**

- task_id: UUID returned by task_create for that task. Each task_create returns a new UUID. Use that exact UUID for all MCP calls; do not substitute ids from other services.
- output_dir: artifact root namespace for task
- config: immutable run configuration (models, runtime limits, Luigi params)
- created_at, updated_at

#### Run

A single execution attempt inside a task (e.g., after a resume).

**Key properties**

- state: running | stopped | completed | failed
- progress_percentage: computed progress percentage (float)
- started_at, ended_at

#### Artifact

A file-like output managed by PlanExe.

**Key properties**

- path: path relative to task output root
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

### 5.1 Task states

Tasks may exist independent of active runs.

- created: task initialized, no run started
- active: at least one run exists, may be running or stopped
- archived: optional; immutable, no new runs allowed

### 5.2 Run states

- running
- stopping (optional transitional state)
- stopped (user stopped, resumable)
- completed
- failed (resumable depending on failure type)

### 5.3 Allowed transitions

- running → stopped via task_stop
- running → completed via normal success
- running → failed via error

**Invalid**

- completed → running (new run must be triggered by creating a new task)
- running → running (no concurrent runs in v1)

---

## 6. MCP Tools (v1 Required)

All tool names below are normative.

### 6.1 prompt_examples

**Step 1 — Call this first.** Returns example prompts that define the baseline for what a good prompt looks like. Do not call task_create yet. Correct flow: Step 1 — call this tool to fetch examples. Step 2 — Formulate a good prompt (use examples as a baseline; similar structure; get user approval). Step 3 — Only then call task_create with the approved prompt. If you call task_create before formulating and approving a prompt, the resulting plan will be lower quality than it could be.

**Request:** no parameters (empty object).

**Response:**

```json
{
  "samples": ["prompt text 1", "prompt text 2", "..."],
  "message": "..."
}
```

---

### 6.2 task_create

**Step 3 — Call only after prompt_examples (Step 1) and after you have formulated a good prompt and got user approval (Step 2).** Start creating a new plan with the approved prompt. speed_vs_detail modes: 'all' runs the full pipeline with all details (slower, higher token usage/cost). 'fast' runs the full pipeline with minimal work per step (faster, fewer details), useful to verify the pipeline is working. 'ping' runs the pipeline entrypoint and makes a single LLM call to verify the worker_plan_database is processing tasks and can reach the LLM.

**Request**

**Schema**

```json
{
  "type": "object",
  "properties": {
    "prompt": { "type": "string" },
    "speed_vs_detail": {
      "type": "string",
      "enum": ["ping", "fast", "all"],
      "default": "ping"
    }
  },
  "required": ["prompt"]
}
```

**Example**

```json
{
  "prompt": "string",
  "speed_vs_detail": "ping"
}
```

**Prompt quality**

The `prompt` parameter should be a detailed description of what the plan should cover. Good prompts are typically 300–800 words and include:

- Clear context: background, constraints, and goals
- Specific requirements: budget, timeline, location, or technical constraints
- Success criteria: what "done" looks like
- Banned words or approaches (if any)

Short one-liners (e.g., "Construct a bridge") tend to produce poor output because they lack context for the planning pipeline. Important details are location, budget, time frame.

Clients can call the MCP tool **prompt_examples** to retrieve example prompts. Use these as examples for task_create; they can also call task_create with any prompt—short prompts produce less detailed plans.

For the full catalog file:

- `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl` — JSONL with `id`, `prompt`, optional `tags`, and optional `mcp_example` (true = curated for MCP).

**Response**

```json
{
  "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "created_at": "2026-01-14T12:34:56Z"
}
```

**Important**

- task_id is a UUID returned by task_create. Use this exact UUID for task_status/task_stop/task_download.

**Behavior**

- Must be idempotent only if client supplies an optional client_request_id (optional extension).
- Task config is immutable after creation in v1.

---

### 6.3 task_status

Returns run status and progress. Used for progress bars and UI states. **Polling interval:** call at reasonable intervals only (e.g. every 5 minutes); plan generation takes 15–20+ minutes and frequent polling is unnecessary.

**Request**

```json
{
  "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1"
}
```

**Input**

- task_id: UUID returned by task_create. Use it to reference the plan being created.

**Response**

```json
{
  "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "state": "running",
  "progress_percentage": 62.0,
  "timing": {
    "started_at": "2026-01-14T12:35:10Z",
    "elapsed_sec": 512
  },
  "files": [
    {
      "path": "plan.md",
      "updated_at": "2026-01-14T12:43:11Z"
    }
  ]
}
```

**Notes**

- progress_percentage must be a float within [0,100].

---

### 6.4 task_stop

Requests the plan generation to stop. Pass the **task_id** (the UUID returned by task_create). This is a normal MCP tool call: call task_stop with that task_id.

**Request**

```json
{
  "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1"
}
```

**Input**

- task_id: UUID returned by task_create. Use this same UUID when calling task_stop to request the run to stop.

**Response**

```json
{
  "state": "stopped"
}
```

**Required semantics**

- Must stop workers cleanly where possible.
- Must persist enough Luigi state to resume incrementally.

---

### 6.5 Download flow (task_download vs task_file_info)

**If your client exposes task_download** (e.g. mcp_local): use it to save the report or zip locally; it calls task_file_info under the hood, then fetches and writes to the local save path (e.g. PLANEXE_PATH).

**If you only have task_file_info** (e.g. direct connection to mcp_cloud): call it with task_id and artifact ("report" or "zip"); use the returned download_url to fetch the file (e.g. GET with API key if configured).

**task_file_info input**

- task_id: UUID returned by task_create. Use it to download the created plan.
- artifact: "report" or "zip" (default "report").

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

### 8.1 Single active run per task

In v1, tasks MUST enforce:

- at most one run in running state.

---

## 9. Error Model

Errors MUST return:

- code: stable machine-readable
- message: human-readable
- details: optional

**Example:**

```json
{
  "error": {
    "code": "RUN_ALREADY_ACTIVE",
    "message": "A run is currently active for this task.",
    "details": { "run_id": "run_0001" }
  }
}
```

### 9.1 Required error codes

- TASK_NOT_FOUND
- RUN_NOT_FOUND
- RUN_ALREADY_ACTIVE
- RUN_NOT_ACTIVE
- INVALID_TARGET
- INVALID_ARTIFACT_URI
- CONFLICT
- PERMISSION_DENIED
- RUNNING_READONLY
- INTERNAL_ERROR

---

## 10. Security & Isolation

### 10.1 Sandbox constraints

- All artifacts must live under task-scoped storage.
- Artifact URIs must not permit path traversal.

### 10.2 Access control

At minimum:

- task must be scoped to a user identity (metadata.user_id)
- callers without permission must receive PERMISSION_DENIED

### 10.3 Sensitive data handling

- logs may include model prompts/responses → treat logs as sensitive artifacts
- allow a config option to redact prompt content in event streaming

---

## 11. Performance Requirements

### 11.1 Responsiveness

- task_status must return within < 250ms under normal load.

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

- task_status.progress_percentage
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

- multiple tasks concurrently, one run each
- event streaming stability under heavy log output

---

## 16. Future Extensions (MCP Resources)

PlanExe is artifact-first, and MCP already has a native concept for that: resources. Today artifacts are exposed via download_url or via proxy download + saved_path. Future versions SHOULD expose artifacts as MCP resources so clients can fetch them via standard resource reads (and treat PlanExe as a first-class MCP server rather than a thin API wrapper).

**Proposed resource identifiers**

- planexe://task/<task_id>/report
- planexe://task/<task_id>/zip

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

### 17.1 task_list (or task_recent)

Return a short list of recent tasks so agents can recover if they lost a task_id.

**Notes**

- Default limit: 5–10 tasks.
- Include task_id, created_at, state, and prompt summary.

### 17.2 task_wait

Blocking helper that polls internally until the task completes or times out. Returns the final task_status payload plus suggested next steps.

**Notes**

- Inputs: task_id, timeout_sec (optional), poll_interval_sec (optional).
- Outputs: same as task_status + next_steps (string or list).

### 17.3 task_get_latest

Simplest recovery: return the most recently created task for the caller.

**Notes**

- Useful for single-user / single-session flows.
- Should be scoped to the caller/user_id when available.

### 17.4 task_logs_tail (optional)

Return the tail of recent log lines for troubleshooting failures.

**Notes**

- Inputs: task_id, max_lines (optional), since_cursor (optional).
- Useful when task_status shows failed but no context.

---

## Appendix A — Example End-to-End Flow

**Create task**

```json
{ "prompt": "..." }
```

**Start run**

```json
{ "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }
```

**Stop**

```json
{ "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }
```

---

## Appendix B — Optional v1.1 Extensions

If you want richer Luigi integration later:

- planexe.task.graph (nodes + edges + states)
- planexe.task.invalidate (rerun subtree)
- planexe.export.bundle (zip all artifacts)
- planexe.validate.only (audit without regeneration)
- planexe.task.archive (freeze task)

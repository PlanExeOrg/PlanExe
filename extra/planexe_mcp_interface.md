PlanExe MCP Interface Specification (v2.0)

1. Purpose

This document specifies a Model Context Protocol (MCP) interface for PlanExe that enables AI agents and client UIs to:
	1.	Create and run long-running plan generation workflows.
	2.	Receive real-time progress updates (task status, log output).
	3.	List, read, and edit artifacts produced in an output directory.
	4.	Stop and resume execution with Luigi-aware incremental recomputation.

The interface is designed to support:
	•	interactive “build systems” behavior (like make / bazel),
	•	resumable DAG execution (Luigi),
	•	deterministic artifact management.

⸻

2. Goals

2.1 Functional goals
	•	Task-based orchestration: each run is associated with a task ID.
	•	Long-running execution: starts asynchronously; clients poll or subscribe to events.
	•	Artifact-first workflow: outputs are exposed as file-like artifacts.
	•	Stop / Resume with minimal recompute:
	•	on resume, only invalidated downstream tasks regenerate.
	•	Progress reporting:
	•	progress_percentage
	•	Editable artifacts:
	•	user edits a generated file
	•	pipeline continues from that point, producing dependent outputs

2.2 Non-functional goals
	•	Idempotency: repeated tool calls should not corrupt state.
	•	Observability: logs, state transitions, and artifacts must be inspectable.
	•	Concurrency safety: prevent conflicting writes and illegal resume patterns.
	•	Extensibility: future versions can add task graph browsing, caching backends, exports.

⸻

3. Non-goals
	•	Defining PlanExe’s internal plan schema, content format, or prompt strategy.
	•	Providing remote code execution inside artifacts.
	•	Implementing a full Luigi UI clone in MCP v1 (optional later).
	•	Guaranteeing ETA estimates (allowed but must be optional / best-effort).

⸻

4. System Model

4.1 Core entities

Task
A long-lived container for a PlanExe project run.

Key properties
	•	task_id: stable unique identifier (UUID, matches TaskItem.id)
	•	output_dir: artifact root namespace for task
	•	config: immutable run configuration (models, runtime limits, Luigi params)
	•	created_at, updated_at

Run
A single execution attempt inside a task (e.g., after a resume).

Key properties
	•	state: running | stopped | completed | failed
	•	progress_percentage: computed progress percentage (float)
	•	started_at, ended_at

Artifact
A file-like output managed by PlanExe.

Key properties
	•	path: path relative to task output root
	•	size, updated_at
	•	content_type: text/markdown, text/html, application/json, etc.
	•	sha256: content hash for optimistic locking and invalidation

Event
A typed message emitted during execution for UI/agent consumption.

Key properties
	•	cursor: ordering token
	•	ts: timestamp
	•	type: event type
	•	data: event payload

⸻

5. State Machine

5.1 Task states

Tasks may exist independent of active runs.
	•	created: task initialized, no run started
	•	active: at least one run exists, may be running or stopped
	•	archived: optional; immutable, no new runs allowed

5.2 Run states
	•	running
	•	stopping (optional transitional state)
	•	stopped (user stopped, resumable)
	•	completed
	•	failed (resumable depending on failure type)

5.3 Allowed transitions
	•	running → cancelled via tasks/cancel
	•	running → completed via normal success
	•	running → failed via error

Invalid
	•	completed → running (new run must be triggered by creating a new task)
	•	running → running (no concurrent runs in v1)

⸻

6. MCP Tools (v2 Required)

All tool names below are normative.

6.1 plan_generate

Generate a new plan. This tool supports MCP task augmentation (Run as task).

Request

Schema

{
  "type": "object",
  "properties": {
    "idea": { "type": "string" },
    "speed_vs_detail": {
      "type": "string",
      "enum": ["ping", "fast", "all"],
      "default": "ping"
    },
    "idempotency_key": { "type": "string" }
  },
  "required": ["idea"]
}

Example

{
  "idea": "string",
  "speed_vs_detail": "ping",
  "idempotency_key": "optional-key"
}

Response (sync)

{
  "task_id": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1",
  "status": "completed",
  "progress_percentage": 100.0,
  "report": {
    "content_type": "text/html; charset=utf-8",
    "sha256": "…",
    "download_size": 1234,
    "download_url": "https://…"
  }
}

Response (task-augmented)

If the client supplies `task` in the MCP `tools/call` params, the server returns
a CreateTaskResult containing a Task object (see MCP Tasks Protocol).

Behavior
	•	Idempotency is supported via `idempotency_key` (dedupe by user_id + key).
	•	Task config is immutable after creation.

⸻

6.2 Legacy wrappers (optional)

For backwards compatibility, these remain available but are not task-augmentable:
	•	task_create
	•	task_status
	•	task_stop

⸻

7. MCP Tasks Protocol

Servers declare tasks capability during initialization. Supported methods:
	•	tasks/get
	•	tasks/result
	•	tasks/cancel
	•	(optional) tasks/list

Task statuses follow MCP:
	•	working
	•	completed
	•	failed
	•	cancelled
	•	input_required

Tasks include a TTL (milliseconds since creation). Expired, terminal tasks may be
removed during cleanup sweeps.

8. Targets

8.1 Standard targets

The following targets MUST be supported:
	•	build_plan
	•	validate_plan
	•	build_plan_and_validate

Targets map to Luigi “final tasks”.

10. Concurrency & Locking

10.1 Single active run per task

In v1, tasks MUST enforce:
	•	at most one run in running state.

⸻

11. Error Model

Errors MUST return:
	•	code: stable machine-readable
	•	message: human-readable
	•	details: optional

Example:

{
  "error": {
    "code": "RUN_ALREADY_ACTIVE",
    "message": "A run is currently active for this task.",
    "details": { "run_id": "run_0001" }
  }
}

11.1 Required error codes
	•	TASK_NOT_FOUND
	•	RUN_NOT_FOUND
	•	RUN_ALREADY_ACTIVE
	•	RUN_NOT_ACTIVE
	•	INVALID_TARGET
	•	INVALID_ARTIFACT_URI
	•	CONFLICT
	•	PERMISSION_DENIED
	•	RUNNING_READONLY
	•	INTERNAL_ERROR

⸻

12. Security & Isolation

12.1 Sandbox constraints
	•	All artifacts must live under task-scoped storage.
	•	Artifact URIs must not permit path traversal.

12.2 Access control

At minimum:
	•	task must be scoped to a user identity (metadata.user_id)
	•	callers without permission must receive PERMISSION_DENIED

12.3 Sensitive data handling
	•	logs may include model prompts/responses → treat logs as sensitive artifacts
	•	allow a config option to redact prompt content in event streaming

⸻

13. Performance Requirements

13.1 Responsiveness
	•	tasks/get must return within < 250ms under normal load.

13.2 Large artifacts
	•	server SHOULD impose max read size per call (e.g., 2–10MB)

⸻

14. Observability Requirements

The server MUST persist:
	•	run lifecycle events
	•	stop reasons
	•	failure tracebacks as artifacts (e.g., run_error.json)
	•	luigi execution logs (run.log)

⸻

15. Reference UI Integration Contract

To match your UI behavior:

Progress bars

Use:
	•	tasks/get.status + progress_percentage
	•	or progress_updated events

⸻

16. Compatibility & Versioning

16.1 Versioning strategy
	•	MCP server exposes: planexe.version = "1.0"
	•	breaking changes require major bump

16.2 Forward compatibility

Clients must ignore unknown fields and unknown event types.

⸻

17. Testing Strategy

17.1 Contract tests (required)
	•	Start/stop/resume loops
	•	Invalid transition errors
	•	Event cursor monotonicity

17.2 Determinism tests (recommended)
	•	Given same inputs + same edits, ensure same downstream artifacts unless models are stochastic
	•	If models are stochastic, test pipeline correctness, not identical bytes

17.3 Load tests (recommended)
	•	multiple tasks concurrently, one run each
	•	event streaming stability under heavy log output

⸻

18. Future Extensions (MCP Resources)

PlanExe is artifact-first, and MCP already has a native concept for that: resources.
Today artifacts are exposed via download_url or via proxy download + saved_path.
Future versions SHOULD expose artifacts as MCP resources so clients can fetch them
via standard resource reads (and treat PlanExe as a first-class MCP server rather
than a thin API wrapper).

Proposed resource identifiers
	•	planexe://task/<task_id>/report
	•	planexe://task/<task_id>/zip

Recommended resource metadata
	•	mime type (content_type)
	•	size (bytes)
	•	sha256 (content hash)
	•	generated_at (UTC timestamp)

Notes
	•	Resources can be backed by existing HTTP endpoints internally; the MCP
		resource read returns the bytes + metadata.
	•	This enables richer MCP client UX (preview, caching, validation) without
		custom tool calls.

⸻

19. Future Tools (High-Leverage, Low-Complexity)

The following tools remove common UX friction without expanding the core model.

19.1 tasks/list
Return a short list of recent tasks so agents can recover if they lost a task_id.

Notes
	•	Default limit: 5–10 tasks.
	•	Include taskId, createdAt, status, and prompt summary.

19.2 task_wait
Blocking helper that polls internally until the task completes or times out.
Returns the final tasks/result payload plus suggested next steps.

Notes
	•	Inputs: task_id, timeout_sec (optional), poll_interval_sec (optional).
	•	Outputs: same as tasks/result + next_steps (string or list).

19.3 task_get_latest
Simplest recovery: return the most recently created task for the caller.

Notes
	•	Useful for single-user / single-session flows.
	•	Should be scoped to the caller/user_id when available.

19.4 task_logs_tail (optional)
Return the tail of recent log lines for troubleshooting failures.

Notes
	•	Inputs: task_id, max_lines (optional), since_cursor (optional).
	•	Useful when tasks/get shows failed but no context.

⸻

Appendix A — Example End-to-End Flow

Create task (task-augmented tool call)

tools/call { name: "plan_generate", arguments: { "idea": "..." }, task: { "ttl": 43200000 } }

Check status

tasks/get { "taskId": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }

Wait for result

tasks/result { "taskId": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }

Cancel

tasks/cancel { "taskId": "5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1" }

⸻

Appendix B — Optional v1.1 Extensions

If you want richer Luigi integration later:
	•	planexe.task.graph (nodes + edges + states)
	•	planexe.task.invalidate (rerun subtree)
	•	planexe.export.bundle (zip all artifacts)
	•	planexe.validate.only (audit without regeneration)
	•	planexe.task.archive (freeze task)

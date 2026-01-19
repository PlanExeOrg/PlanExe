PlanExe MCP Interface Specification (v1.0)

1. Purpose

This document specifies a Model Context Protocol (MCP) interface for PlanExe that enables AI agents and client UIs to:
	1.	Create and run long-running plan generation workflows.
	2.	Receive real-time progress updates (phases, task status, log output).
	3.	List, read, and edit artifacts produced in an output directory.
	4.	Stop and resume execution with Luigi-aware incremental recomputation.

The interface is designed to support:
	•	interactive “build systems” behavior (like make / bazel),
	•	resumable DAG execution (Luigi),
	•	deterministic artifact management.

⸻

2. Goals

2.1 Functional goals
	•	Session-based orchestration: each run is associated with a session ID.
	•	Long-running execution: starts asynchronously; clients poll or subscribe to events.
	•	Artifact-first workflow: outputs are exposed as file-like artifacts.
	•	Stop / Resume with minimal recompute:
	•	on resume, only invalidated downstream tasks regenerate.
	•	Fine-grained progress reporting:
	•	overall progress
	•	phase
	•	current Luigi task (or logical task)
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

Session
A long-lived container for a PlanExe project run.

Key properties
	•	session_id: stable unique identifier
	•	output_dir: artifact root namespace for session
	•	config: immutable run configuration (models, runtime limits, Luigi params)
	•	created_at, updated_at

Run
A single execution attempt inside a session (e.g., after a resume).

Key properties
	•	run_id: monotonic per session (run_0001, run_0002…)
	•	state: running | stopped | completed | failed
	•	phase: high-level stage (e.g., generating_plan, validating, exporting)
	•	progress: computed progress metrics
	•	started_at, ended_at

Artifact
A file-like output managed by PlanExe.

Key properties
	•	artifact_uri: stable URI
	•	path: path relative to session output root
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

5.1 Session states

Sessions may exist independent of active runs.
	•	created: session initialized, no run started
	•	active: at least one run exists, may be running or stopped
	•	archived: optional; immutable, no new runs allowed

5.2 Run states
	•	running
	•	stopping (optional transitional state)
	•	stopped (user stopped, resumable)
	•	completed
	•	failed (resumable depending on failure type)

5.3 Allowed transitions
	•	running → stopped via planexe_stop
	•	running → completed via normal success
	•	running → failed via error

Invalid
	•	completed → running (new run must be triggered by creating a new session)
	•	running → running (no concurrent runs in v1)

⸻

6. MCP Tools (v1 Required)

All tool names below are normative.

6.1 planexe.session.create

Creates a new session and output namespace.

Request

{
  "idea": "string",
  "config": {
    "planner_model": "string",
    "validator_model": "string",
    "max_runtime_sec": 7200,
    "luigi": {
      "workers": 4,
      "scheduler": "local"
    }
  },
  "metadata": {
    "user_id": "string",
    "tags": ["string"]
  }
}

Response

{
  "session_id": "pxe_2026_01_14__abcd1234",
  "created_at": "2026-01-14T12:34:56Z"
}

Behavior
	•	Must be idempotent only if client supplies an optional client_request_id (optional extension).
	•	Session config is immutable after creation in v1.

⸻

6.2 planexe_status

Returns run status and progress. Used for progress bars and UI states.

Request

{
  "session_id": "pxe_..."
}

Response

{
  "session_id": "pxe_...",
  "state": "running",
  "phase": "generating_plan",
  "progress": {
    "overall": 0.62,
    "current_task": {
      "name": "WriteDetailedPlan",
      "pct": 0.33
    }
  },
  "timing": {
    "started_at": "2026-01-14T12:35:10Z",
    "elapsed_sec": 512
  },
  "latest_artifacts": [
    {
      "path": "plan.md",
      "updated_at": "2026-01-14T12:43:11Z"
    }
  ],
  "warnings": ["string"]
}

Notes
	•	progress.overall must be within [0,1].
	•	phase must be stable and enumerable.

⸻

6.3 planexe_stop

Stops the active run.

Request

{
  "session_id": "pxe_...",
  "mode": "graceful"
}

Response

{
  "state": "stopped"
}

Required semantics
	•	Must stop workers cleanly where possible.
	•	Must persist enough Luigi state to resume incrementally.

⸻

7. Targets and Phases

8.1 Standard targets

The following targets MUST be supported:
	•	build_plan
	•	validate_plan
	•	build_plan_and_validate

Targets map to Luigi “final tasks”.

8.2 Standard phases

Phases MUST be enumerable and stable (for UI progress bars).

Recommended phases:
	•	initializing
	•	generating_plan
	•	validating
	•	exporting
	•	finalizing

⸻

10. Concurrency & Locking

10.1 Single active run per session

In v1, sessions MUST enforce:
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
    "message": "A run is currently active for this session.",
    "details": { "run_id": "run_0001" }
  }
}

11.1 Required error codes
	•	SESSION_NOT_FOUND
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
	•	All artifacts must live under session-scoped storage.
	•	Artifact URIs must not permit path traversal.

12.2 Access control

At minimum:
	•	session must be scoped to a user identity (metadata.user_id)
	•	callers without permission must receive PERMISSION_DENIED

12.3 Sensitive data handling
	•	logs may include model prompts/responses → treat logs as sensitive artifacts
	•	allow a config option to redact prompt content in event streaming

⸻

13. Performance Requirements

13.1 Responsiveness
	•	planexe_status must return within < 250ms under normal load.

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
	•	planexe_status.progress.overall
	•	planexe_status.phase
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
	•	multiple sessions concurrently, one run each
	•	event streaming stability under heavy log output

⸻

Appendix A — Example End-to-End Flow

Create session

planexe.session.create({ "idea": "...", "config": {...} })

Start run

planexe_status({ "session_id": "pxe_..." })

Stop

planexe_stop({ "session_id": "pxe_...", "mode": "graceful" })

⸻

Appendix B — Optional v1.1 Extensions

If you want richer Luigi integration later:
	•	planexe.task.graph (nodes + edges + states)
	•	planexe.task.invalidate (rerun subtree)
	•	planexe.export.bundle (zip all artifacts)
	•	planexe.validate.only (audit without regeneration)
	•	planexe.session.archive (freeze session)

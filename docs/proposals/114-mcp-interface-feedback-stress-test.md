# Proposal 114: MCP Interface Issues from 10-Plan Agent Stress Test

**Date:** 2026-03-11
**Status:** Proposal (I1, I2, I4 implemented; remainder open)
**Source:** Feedback from Claude (Opus 4.6) after three stress-testing sessions (13 plans total) operated through Claude Code CLI.
**Scope:** MCP interface design, observability, and lifecycle management. Does not cover plan content quality or pipeline output improvements.

---

## 1. Context

Three stress-testing sessions were conducted through Claude Code CLI using the PlanExe MCP interface:

- **Session 1** (10 plans): Surfaced core MCP interface friction points — state ambiguity, missing diagnostics, lifecycle gaps.
- **Session 2** (2 plans, with stop/resume cycling): Validated the `stopped` state implementation and identified remaining gaps.
- **Session 3** (1 plan on remote server): First use of `planexeremote` alongside `planexelocal`. Exposed local-vs-remote behavioral differences and prompted a fundamental correction on SSE's role for agents.

The testing surfaced concrete MCP interface friction points that affect agent operators — issues with state management, observability, lifecycle hygiene, and workflow ergonomics. This proposal catalogues the MCP-specific issues and maps them against existing proposals.

---

## 2. Issues

### I1 — `failed` state conflates user-stop and actual failure

**Status:** Implemented (Option A — dedicated `PlanState.stopped` DB state). Supersedes PR #244's Option B (`stop_reason` field).

**Problem:** When `plan_stop` is called, the plan transitions to `failed`. When a worker crashes, it also transitions to `failed`. The agent operator cannot distinguish between:
- User-initiated stop (nothing went wrong)
- Actual failure (network drop, model error, worker crash)

This matters because the correct response differs: user stop suggests `plan_resume`; actual failure may need `plan_retry` or investigation.

**Overlap:** Proposal 87 (plan_resume) acknowledged this in §4 — now updated to reflect the implementation.

**Options:**

| Option | Description | Breaking change? |
|--------|-------------|-----------------|
| A. New `stopped` state | `plan_stop` → `stopped`, worker crash → `failed` | Yes (new enum value in API responses) |
| B. `stop_reason` field | Keep `failed` but add `stop_reason`: `"user_requested"`, `"worker_crash"`, `"timeout"`, `"model_error"`, `null` | No |

Option B is less disruptive and can be added to `plan_status` without changing state enum values.

**Implemented approach:** Option A — a dedicated `PlanState.stopped` enum value (value 5). `plan_stop` now transitions plans to the `stopped` state instead of `failed`. The `stop_reason` field introduced in PR #244 (Option B) has been removed — the state itself communicates intent. `plan_retry` and `plan_resume` accept both `failed` and `stopped` states. DB migration adds `'stopped'` to the PostgreSQL enum type (both `taskstate` for pre-rename databases and `planstate` for fresh databases — the Python class was renamed from `TaskState` to `PlanState` in proposal 74 but the PostgreSQL type name was not changed). The worker's post-pipeline finalization also transitions to `stopped` (not `failed`) when `stop_requested` is true.

**Affected files:** `database_api/model_planitem.py`, `mcp_cloud/db_setup.py`, `worker_plan_database/app.py`, `mcp_cloud/db_queries.py`, `mcp_cloud/handlers.py`, `mcp_cloud/sse.py`, `mcp_cloud/tool_models.py`, `mcp_cloud/schemas.py`, `mcp_local/planexe_mcp_local.py`, `frontend_multi_user/src/app.py`, `frontend_multi_user/src/planexe_modelviews.py`, `frontend_multi_user/templates/plan_iframe.html`, `frontend_multi_user/templates/run_via_database.html`, `frontend_multi_user/templates/index.html`, `frontend_multi_user/templates/account.html`, `mcp_cloud/tests/test_plan_status_tool.py`, `docs/mcp/planexe_mcp_interface.md`, `docs/mcp/autonomous_agent_guide.md`, `docs/proposals/87-plan-resume-mcp-tool.md`, `docs/proposals/111-promising-directions.md`.

---

### I2 — No failure diagnostics in `plan_status`

**Status:** Implemented. Four new nullable columns on `PlanItem` (`failure_reason`, `failed_step`, `error_message`, `recoverable`) populated by the worker on failure and surfaced in `plan_status` responses inside a consolidated `error` dict. Diagnostics are reset on retry/resume.

**Priority: High — identified as the single biggest observability gap.**

**Problem:** When a plan fails, `plan_status` returns `state: "failed"` with no `failure_reason`, `error_message`, or `failed_step` field. The agent can only tell the user "it failed" without explaining why.

During the stress test, Plan 1 (20f1cfac) stalled at 5.5% with zero diagnostic information. The operator had no way to determine if it was a content refusal, a model error, or a worker crash.

**Overlap:** Proposal 113 (LLM Error Traceability) preserves errors in usage metrics logs but does not surface them in `plan_status` responses to the MCP consumer.

**Implemented response shape (on failure):**

```json
{
  "state": "failed",
  "error": {
    "failure_reason": "generation_error",
    "failed_step": "016-expert_criticism",
    "message": "LLM provider returned 503",
    "recoverable": true
  }
}
```

The `recoverable` boolean lets the agent immediately suggest `plan_resume` (transient/recoverable) vs `plan_retry` (fundamental/non-recoverable) without guessing.

**`failure_reason` values:**
- `generation_error` — pipeline step failed (recoverable)
- `worker_error` — unhandled exception in worker (recoverable)
- `inactivity_timeout` — stop flag detected without user request (recoverable)
- `internal_error` — pipeline completed but no report produced (not recoverable)
- `version_mismatch` — resume attempted with incompatible pipeline version (not recoverable)

**Implementation:** Four dedicated nullable columns on `PlanItem`: `failure_reason`, `failed_step`, `error_message`, `recoverable`. The column was originally named `last_error` and renamed to `error_message` to match the consolidated `error.message` field in the MCP response. Migration functions in all three services (mcp_cloud, worker_plan_database, frontend_multi_user) check column existence via `sqlalchemy.inspect` before attempting the rename — this avoids noisy PostgreSQL ERROR logs on restarts where the rename already succeeded. The worker populates them via `_update_failure_diagnostics()` at three failure paths: normal pipeline failure, unhandled exception, and version mismatch on resume. The `plan_status` handler consolidates these into a single `error` dict (with keys `failure_reason`, `failed_step`, `message`, `recoverable`) only when `state == "failed"` — non-failed plans omit the `error` field entirely. Diagnostics are cleared on `plan_retry` and `plan_resume`. The frontend displays diagnostics in the failure trace panel and the admin bulk-fail form uses the renamed field.

**Affected files:** `database_api/model_planitem.py`, `mcp_cloud/db_setup.py`, `mcp_cloud/db_queries.py`, `mcp_cloud/handlers.py`, `mcp_cloud/tool_models.py`, `mcp_cloud/schemas.py`, `mcp_cloud/tests/test_plan_status_tool.py`, `mcp_cloud/README.md`, `mcp_local/README.md`, `worker_plan_database/app.py`, `frontend_multi_user/src/app.py`, `frontend_multi_user/src/planexe_modelviews.py`, `frontend_multi_user/templates/admin/bulk_change_to_failed.html`, `frontend_multi_user/templates/plan_iframe.html`, `docs/mcp/planexe_mcp_interface.md`, `docs/mcp/mcp_details.md`, `docs/mcp/autonomous_agent_guide.md`.

---

### I3 — No plan deletion or cleanup

**Problem:** Stopped and failed plans persist in `plan_list` indefinitely. After 10 plans, the list becomes noisy. There is no way to clean up test runs, abandoned plans, or experiments.

**Proposed solutions (pick one or both):**

| Tool | Behavior |
|------|----------|
| `plan_delete` | Permanently remove a plan from the user's list. Only allowed for terminal states (`completed`, `failed`, `stopped`). |
| `plan_archive` | Soft-delete: plan is hidden from `plan_list` but remains in DB for auditing. |

`plan_archive` is safer for a billing system where plan records may need to be retained.

**Affected files:** `mcp_cloud/db_queries.py`, `mcp_cloud/handlers.py`, `mcp_cloud/schemas.py`, `mcp_cloud/tool_models.py`.

---

### I4 — No idempotency guard on `plan_create`

**Status:** Implemented (PR #242).

**Problem:** If the user double-clicks or the agent accidentally submits twice, two identical plans are created. Each `plan_create` call generates a new `plan_id` with no server-side deduplication.

**Original proposal:** Add an optional `request_id` (client-generated UUID) parameter to `plan_create`. The server checks if a plan with the same `request_id` already exists for this user. This is a standard idempotency pattern (Stripe, AWS, etc.).

**Why `request_id` was not adopted:** The primary MCP consumers are LLM agents. LLMs cannot generate UUIDs natively — they would need an extra tool call just to produce the idempotency key, adding friction to the exact workflow it's meant to protect. A mechanism that requires client cooperation is a poor fit when the clients are language models.

**Implemented approach:** Automatic server-side dedup. Before inserting a new plan, `_create_plan_sync` queries for an existing `pending`/`processing` plan matching `(user_id, prompt, model_profile)` created within a configurable time window (default 10 minutes, env `PLANEXE_DEDUP_WINDOW_SECONDS`). If found, the existing plan is returned with `deduplicated: true` instead of creating a new one. No schema migration needed — uses existing columns. Set `PLANEXE_DEDUP_WINDOW_SECONDS=0` to disable.

**Known limitation:** There is a TOCTOU race — if two identical requests arrive concurrently and both pass the dedup check before either commits, a duplicate plan is created. This is accepted; the cost is wasted tokens for one extra plan, which is not worth the complexity of a database-level lock or migration to fix.

**Affected files:** `mcp_cloud/db_queries.py` (`_find_recent_duplicate_plan`, `_create_plan_sync`), `mcp_cloud/tool_models.py` (`deduplicated` field on `PlanCreateOutput`), `mcp_cloud/schemas.py` (description and `idempotentHint`), `mcp_cloud/app.py` (re-export).

---

### I5 — SSE is the wrong mechanism for MCP agents

**Status:** Implemented and reverted. A `monitor` boolean parameter was added to `plan_create`, `plan_retry`, and `plan_resume` using the MCP SDK's `ctx.report_progress()` and `ctx.info()` APIs. The server sent `notifications/progress` and `notifications/message` correctly, but v4 real-world testing revealed that Claude Code — PlanExe's primary MCP consumer — silently drops both notification types. Claude Code only supports `list_changed` notifications (for refreshing tool/resource definitions). Since no current MCP client can consume the notifications, the implementation was reverted as unused code.

**v4 real-world testing (March 2026):** A `monitor=true` parameter was implemented and tested. Claude Code tested `monitor=true` on `plan_resume`. The resume succeeded and the plan completed, but zero notifications were received. Investigation showed that Claude Code only supports `list_changed` MCP notifications (for refreshing tool/resource definitions). It does not handle `notifications/progress` or `notifications/message` — they are silently dropped. The server was sending notifications correctly; the gap is on the client side. The implementation was reverted as unused code — no MCP client can consume it today. See `docs/proposals/mcp-interface-perception-v4.md` for the full agent feedback.

**Corrected priority ranking (from v4 agent feedback):**

1. Polling `plan_status` — reliable, works everywhere, recommended default
2. `plan_wait` blocking tool (see I8) — would eliminate polling for agents without notification support
3. MCP notifications — correct long-term solution, blocked by client support (revisit when clients add `notifications/progress` handling)
4. SSE — only useful for non-agent consumers (browser UIs, dashboards)
5. HTTP webhooks — only useful for server-to-server

**Updated in v3:** The original framing ("SSE events lack structured data") missed the deeper problem. SSE is designed for real-time UI clients, not turn-based agents.

**Problem:** The agent's SSE monitoring pattern across all 13 plans was:
```
curl -N -s <sse_url> 2>&1 | tail -5   # run in background
```
The agent never reads SSE event content. It waits for `curl` to exit (connection close) and treats that as a "plan finished" signal — abusing a data stream as a binary trigger. When the remote SSE stream dropped at ~48% progress, the agent assumed the plan was done, checked `plan_status`, and discovered it was still running. The failure wasn't in SSE — it was in the pattern.

Even if SSE events contained rich structured progress data, a turn-based MCP agent wouldn't see them until the stream closes, because `curl` runs in the background and output is only read on task completion. MCP agents cannot "watch" a stream in real time.

**What the agent previously asked for (wrong):**
- "Richer SSE events with progress data" — useless, since event content is never read
- "Webhooks" — HTTP callbacks, but CLI agents have no endpoint to register
- "SSE heartbeat pings" — would only help if the pattern itself were sound

**What would actually help MCP agents (corrected in v3, re-corrected in v4):**

The v3 priority ranking assumed MCP clients would surface progress notifications. Real-world testing in v4 showed this is not the case (Claude Code silently drops them). See corrected priority ranking above.

1. **Polling `plan_status` (works everywhere):** The primary method. Reliable across all clients and transports.

2. **`plan_wait` blocking tool:** See I8. Would eliminate polling for agents without notification support.

3. **MCP notifications:** Correct long-term solution, but blocked by client support. Revisit when MCP clients add `notifications/progress` and `notifications/message` handling.

4. **SSE (for non-MCP real-time clients only):** Keep SSE for browser UIs, streaming dashboards, and CLI scripts that can consume events in real time. Do not recommend to MCP agents.

**Overlap:** Proposal 70 §5.1 (SSE progress streaming) is implemented but serves the wrong consumer type for agent use cases.

**Affected files:** `mcp_cloud/sse.py`, `mcp_cloud/schemas.py` (tool description update to stop recommending SSE to agents), `mcp_cloud/handlers.py` (MCP notification support).

---

### I6 — Download URL expiry (15 minutes) is too short

**Problem:** When a user takes time reviewing results before downloading, the signed download URL expires silently. Calling `plan_file_info` again works, but the 15-minute window creates unnecessary friction.

**Proposed fix:** Extend default TTL from 15 to 30–60 minutes. Make it configurable via `PLANEXE_DOWNLOAD_TOKEN_TTL_SECONDS` environment variable.

**Affected files:** `mcp_cloud/download_tokens.py` (change default TTL constant, add env var override).

---

### I7 — No stalled-plan detection

**Problem:** Plan 1 (20f1cfac) stayed at 5.5% for multiple status checks with no indication of whether it was still working, stuck, or had silently failed. There is no `stalled_since` timestamp or heartbeat indicator in `plan_status`.

**Overlap:** Proposal 87 §8 proposes `current_task` and `last_llm_call_at` fields — these would partially address this, but there is no explicit stall detection.

**Proposed additions to `plan_status`:**

| Field | Description |
|-------|-------------|
| `last_progress_at` | ISO 8601 timestamp of the last progress update. Enables the agent to compute time-since-last-progress and decide whether to wait, resume, or retry. |
| `last_llm_call_at` | ISO 8601 timestamp of the most recent LLM call. A gap > 5 minutes with no progress change is a strong stall signal. |

**Affected files:** `worker_plan_database/app.py` (write timestamps on progress), DB model (new timestamp columns), `mcp_cloud/handlers.py` (include in response).

---

### I8 — No `plan_wait` tool for agents without shell access

**Problem:** The recommended SSE monitoring pattern (`curl -N <sse_url>`) requires Bash tool access. Agents running in pure MCP client environments (no shell) can only poll `plan_status`. A blocking `plan_wait` tool would let these agents simply wait for completion.

**Proposed tool:**

```
plan_wait(plan_id, timeout_seconds=1200): Blocks until the plan reaches a terminal state
(completed, failed, stopped) or the timeout expires. Returns the final plan_status response.
```

**Implementation:** Server-side long-poll using the existing SSE infrastructure. The handler subscribes to the plan's state changes and returns when terminal or timeout.

**Consideration:** Long-running HTTP requests may conflict with proxy/load-balancer timeouts. Default timeout should be ≤ 20 minutes. May need to be implemented as repeated short polls on the server side with a single response.

**Affected files:** `mcp_cloud/handlers.py`, `mcp_cloud/schemas.py`, `mcp_cloud/tool_models.py`.

---

### I9 — Prompt iteration has no memory

**Problem:** Each `plan_create` is independent. When iterating on a prompt (e.g. refining "The Game" movie remake across 4 variations), there is no way to link plans or diff their prompts. The agent must manually track the relationship between plans.

**Proposed fix:** Add an optional `parent_plan_id` parameter to `plan_create`. When set:
- The new plan is linked to its parent in the DB
- `plan_list` can optionally filter by lineage
- `plan_status` includes `parent_plan_id` in the response

This enables prompt iteration tracking without changing existing behavior for plans created without a parent.

**Affected files:** `mcp_cloud/tool_models.py` (new optional field), `mcp_cloud/db_queries.py` (store parent reference), DB model (new nullable column).

---

### I10 — Silent partial failures in completed plans

**Problem:** A plan can reach `completed` with sections that are empty or stub-quality (e.g. 6/8 experts returning no feedback in the expert criticism step). There is no signal in `plan_status` or `plan_file_info` that the output has quality gaps. The agent has to read individual files to discover this. `completed` means "all 110 steps ran" — not "all sections produced quality output."

This is a trust gap: the agent cannot confidently tell the user "your plan is ready" without caveats, because it has no visibility into per-section quality.

**Proposed addition:** A `quality_summary` in the completed plan status or file info:

```json
{
  "sections_complete": 108,
  "sections_partial": 2,
  "partial_details": [
    {"step": "016-expert_criticism", "note": "2/8 experts provided feedback"}
  ]
}
```

**Implementation path:** The worker already knows which steps produced output. A post-pipeline validation pass could check key sections for minimum output quality (e.g. non-empty, expected structure present) and write a summary to the DB. `plan_status` surfaces it when `state == "completed"`.

**Affected files:** `worker_plan_database/app.py` (quality validation pass), DB model (quality summary column), `mcp_cloud/handlers.py` (include in completed plan_status response).

---

### I11 — No server identity in responses

**New in v3.** Surfaced during the first session using both `planexelocal` and `planexeremote`.

**Problem:** The agent cannot programmatically determine which server backend it's connected to. When the user ran `/mcp` and reconnected, the active server changed from local to remote without any signal — the only clue was the tool name prefix changing from `mcp__planexelocal__*` to `mcp__planexeremote__*`. If both servers had the same prefix, the switch would be invisible.

This matters because:
- Plans created on local can't be accessed from remote (different plan_id namespaces)
- SSE monitoring patterns that work on local may fail on remote
- Speed expectations differ (local ~13 min, remote ~28 min for baseline plans)

**Proposed fix:** `plan_status` could include a `server` or `endpoint` field so the agent always knows which backend it's talking to. Alternatively, `plan_create` could return a `server_info` object with capabilities:

```json
{
  "server_info": {
    "server_id": "planexe-remote-prod",
    "sse_reliable": false,
    "files_visible_during_processing": false,
    "expected_speed": "slow"
  }
}
```

**Affected files:** `mcp_cloud/handlers.py`, `mcp_cloud/tool_models.py`, `mcp_local/planexe_mcp_local.py`.

---

### I12 — No files visibility during remote processing

**New in v3.** Surfaced during the first remote plan.

**Problem:** `plan_status` on the remote server returns `files_count: 0` while processing. On local, files appear incrementally. The incremental file list on local was useful for:
- Confirming the plan is actually producing output (not just incrementing step count)
- Seeing which pipeline sections have completed
- Early reading of intermediate files if curious

Losing this on remote makes the plan feel more opaque. The inconsistency also means agent workflows that depend on file visibility during processing will break silently when switching servers.

**Proposed fix:** Align remote behavior with local — populate the files list incrementally during processing. If this isn't feasible due to remote storage architecture, document the difference in the tool description.

**Affected files:** `worker_plan_database/app.py` (remote file list population), `mcp_cloud/schemas.py` (document behavioral difference if not fixed).

---

### I13 — SSE reliability differs between local and remote

**New in v3.** The SSE stream on local was 100% reliable across 12 plans and multiple stop/resume cycles. On remote, the stream dropped at ~48% on the first plan. Possible causes: proxy timeout, load balancer idle timeout, or different SSE implementation.

While I5 argues SSE is the wrong tool for agents regardless of reliability, this inconsistency also affects non-agent consumers (browser UIs, dashboards) that would rely on SSE for real-time progress. If SSE is kept for these consumers, it needs to work reliably on remote too.

**Proposed fix:** Investigate and fix the remote SSE connection stability. Common causes: reverse proxy `proxy_read_timeout` too low, load balancer idle timeout, missing SSE keepalive pings.

**Affected files:** Infrastructure/deployment configuration (not application code).

---

## 3. Cross-Reference with Existing Proposals

| Issue | Existing Proposal | Gap |
|-------|-------------------|-----|
| I1 (stopped vs failed) | 87 §4 (deferred) | **Implemented** (Option A — dedicated `PlanState.stopped`) |
| I2 (failure diagnostics) | 113 (logs only) | **Implemented** — surfaced to MCP consumer |
| I3 (plan_delete) | None | New |
| I4 (idempotency) | None | **Implemented** (PR #242) |
| I5 (SSE wrong for agents) | 70 §5.1 (basic SSE done) | SSE serves wrong consumer type; MCP notifications needed |
| I6 (download TTL) | 70 §4 (tokens done) | TTL too short, not configurable |
| I7 (stall detection) | 87 §8 (partial) | No explicit stall timestamps |
| I8 (plan_wait) | None | New |
| I9 (prompt iteration) | None | New |
| I10 (silent partial failures) | None | New |
| I11 (server identity) | None | New (v3) |
| I12 (remote files visibility) | None | New (v3) |
| I13 (remote SSE reliability) | 70 §5.1 | SSE drops on remote; local is fine |

---

## 4. Overlap with Proposal 70 (MCP Interface Evaluation and Roadmap)

Several items here refine or extend issues already tracked in Proposal 70's checklist:

| This Proposal | Proposal 70 Item | Relationship |
|---------------|-------------------|-------------|
| I2 (failure diagnostics) | 70 §5.6 (pipeline stage names) | Complementary — 70 focuses on progress UX, I2 focuses on failure UX |
| I5 (SSE wrong for agents) | 70 §5.1 (SSE implemented) | Correction — SSE serves wrong consumer type for agents; MCP notifications needed |
| I6 (download TTL) | 70 (signed tokens done) | Refinement — increase TTL |
| I13 (remote SSE reliability) | 70 §5.1 (SSE implemented) | Bug — SSE drops on remote but works on local |

If accepted, I1–I4 and I7–I13 should be added to Proposal 70's quick-win checklist as new line items.

---

## 5. Suggested Priority

| Priority | Issues | Rationale |
|----------|--------|-----------|
| ~~P1~~ | ~~I2 (failure diagnostics)~~ | **Implemented**. Four DB columns populated by worker, surfaced in `plan_status`, displayed in frontend. |
| ~~P1~~ | ~~I1 (stopped vs failed)~~ | **Implemented** (Option A). Dedicated `PlanState.stopped` enum value — `plan_stop` transitions to `stopped`, not `failed`. |
| P1 | I5 (SSE wrong for agents) | MCP notifications should replace SSE as the recommended agent completion mechanism. Highest-impact agent UX improvement. |
| P2 | I7 (stall detection) | Prevents agents from waiting indefinitely on stuck plans. |
| P2 | I6 (download TTL) | Low effort, reduces friction. |
| P2 | I3 (plan_delete) | Hygiene for multi-plan sessions. |
| ~~P3~~ | ~~I4 (idempotency)~~ | **Implemented** (PR #242). Server-side auto-dedup on `(user_id, prompt, model_profile)` within a time window. |
| P2 | I10 (silent partial failures) | Agent cannot trust `completed` means quality output. Undermines confidence in the entire workflow. |
| P2 | I12 (remote files visibility) | Behavioral inconsistency between servers breaks agent workflows silently. |
| P2 | I8 (plan_wait) | Upgraded from P3: validated as the correct fallback mechanism when MCP notifications aren't available. |
| P3 | I11 (server identity) | Nice-to-have for multi-server awareness. |
| P3 | I9 (prompt iteration) | Nice-to-have for iteration workflows. |
| P3 | I13 (remote SSE reliability) | Only matters if SSE is kept for non-agent consumers. |

---

## 6. Agent Perception (after 13 plans across three sessions)

**Overall rating: 8.5/10** (unchanged from session 2). The remote server works — same workflow, same tool contract — but exposed new friction that local hid.

### What works well

1. **Tool descriptions are best-in-class.** The agent operates PlanExe without external documentation, solely from tool descriptions. They specify call order, state contract, timing expectations, error codes, and troubleshooting guidance.
2. **State machine is now clean.** Every state has a single obvious next action: `pending` → wait, `processing` → poll, `completed` → download, `stopped` → suggest resume, `failed` → investigate. No state requires guessing.
3. **Remote server works without configuration changes.** The same workflow (create → monitor → status → download) works on both local and remote servers without any adaptation. Same contract, different infrastructure — good API design.
4. **Resume preserves progress.** After stopping at step 12/110, resume continued from step 12, not step 1. The `resume_count` field makes the history visible without the agent needing to track it.
5. **Deduplication prevents accidents.** Same prompt + model_profile within a short window returns the existing plan with `deduplicated=true`.
6. **Progress reporting is honest.** The tool description explicitly warns that `progress_percentage` is not linear and shouldn't be used for time estimates.

### The stop/resume cycle

Session 2 tested: create → stop → resume → stop → resume → complete. At no point was the agent confused about state or what to suggest. The `resume_count` field confirmed history without shadow state. The mark of a good state machine — the agent doesn't need to maintain its own bookkeeping.

### Local vs remote: side-by-side comparison (new in session 3)

| Aspect | planexelocal | planexeremote |
|--------|-------------|---------------|
| Speed (baseline, 110 steps) | ~13-14 minutes | ~28 minutes |
| Files list in plan_status | Populated (shows files as created) | Empty (`files_count: 0` during processing) |
| SSE reliability | 100% across 12 plans + stop/resume | Stream dropped at ~48% on first plan |
| Download URLs | `http://192.168.1.40:8001/...` | `https://mcp.planexe.org/...` |
| Files produced | 172 (parasomnia) | 198 (Delhi water) |

**Trust level — Local:** High. Every feature works as documented. SSE is reliable. State machine is clean. Files are visible during processing.

**Trust level — Remote:** Medium-high. The core workflow works. But SSE dropped once, files aren't visible during processing, and it's 2x slower. The agent would use it confidently for plan creation and download, but would poll `plan_status` instead of relying on SSE.

### Correction: SSE is the wrong tool for agents

Session 3 forced a fundamental re-evaluation of how the agent used SSE. The agent never read SSE event content — it ran `curl` in the background and treated connection close as a "done" signal. This pattern worked reliably on local but failed on remote when the stream dropped at ~48%. The deeper issue: turn-based MCP agents cannot watch real-time streams. MCP notifications over the existing connection are the correct mechanism (see I5).

### Remaining trust gaps

1. **`completed` ≠ quality output.** `completed` means "all steps ran," not "all sections produced quality output." The interface doesn't distinguish between these yet (see I10).
2. **No server identity.** The agent cannot programmatically determine which backend it's talking to, which matters for setting speed expectations and knowing whether plans are accessible cross-server (see I11).
3. **Remote opacity.** No files visible during remote processing makes the plan feel like a black box (see I12).

### Evolution across sessions

| Aspect | Session 1 | Session 2 | Session 3 | Direction |
|--------|-----------|-----------|-----------|-----------|
| Stop state | `failed` (ambiguous) | `stopped` (clear) | `stopped` (clear) | Fixed |
| Resume tracking | No history | `resume_count` field | `resume_count` field | Fixed |
| Tool descriptions | Excellent | Updated with `stopped` state | Identical across servers | Stable |
| Failure diagnostics | Missing | Still missing | **Implemented** | Fixed |
| Silent partial failures | Not surfaced | Not surfaced | Not surfaced | Unchanged |
| Plan cleanup | No delete | No delete | No delete | Unchanged |
| SSE understanding | "Reliable completion detector" | "Reliable completion detector" | "Wrong tool for agents" | Corrected |
| Server awareness | Single server | Single server | No server identity signal | New gap |
| Files during processing | Visible | Visible | Not visible (remote) | New gap |

### Session totals

| # | Plan ID | Description | Server | Time | Files |
|---|---------|-------------|--------|------|-------|
| 1–10 | (various) | Session 1 plans | local | varies | varies |
| 11 | ff5488dc | Riemann Hypothesis Bonn | local | ~13 min | 179 |
| 12 | 6d0c1d80 | Parasomnia facility Bonn | local | ~14 min (with 2 stop/resume) | 172 |
| 13 | d623f577 | Delhi water purification | remote | ~28 min | 198 |

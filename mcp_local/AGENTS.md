# mcp_local agent instructions

Scope: local MCP proxy script that runs on the user's machine and forwards tool calls
to mcp_cloud, a MCP server running in the cloud, over HTTP.

## Interaction model
- The local proxy exposes MCP tools over stdio and forwards requests to mcp_cloud
  using `PLANEXE_URL` (defaults to the hosted `/mcp` endpoint).
- Supported tools: `example_plans`, `example_prompts`, `model_profiles`, `plan_create`, `plan_status`, `plan_stop`, `plan_retry`, `plan_download`.
- `plan_download` calls the remote `plan_file_info` tool to obtain a download URL,
  then downloads the artifact to `PLANEXE_PATH` on the local machine.
- `plan_create` visible input schema includes `prompt`, optional `model_profile`, and optional `user_api_key`.
- Use `model_profiles` to help agents select `plan_create.model_profile` without relying on internal file knowledge.
- Keep workflow wording explicit that prompt drafting + user approval is a non-tool step before `plan_create`.
- Keep concurrency wording explicit: each `plan_create` call creates a new `plan_id`; no global per-client concurrency cap is enforced server-side.

## Public state contract
- `plan_status.state` must use exactly: `pending`, `processing`, `completed`, `failed`.
- Caller contract:
  - `pending`/`processing`: keep polling.
  - `completed`: download is ready.
  - `failed`: terminal error.
- Do not use legacy public names such as `running`, `stopping`, or `stopped`.
- Do not expose internal implementation symbols (for example `PlanState.pending`) in
  model-facing text; use plain public strings.
- Troubleshooting guidance to keep aligned with cloud docs/instructions:
  - `pending` for longer than 5 minutes likely means queued but not picked up by worker.
  - `processing` with no output-file changes for longer than 20 minutes likely means stalled/failed execution.
  - Report both cases at `https://github.com/PlanExeOrg/PlanExe/issues`.

## plan_stop semantics
- `plan_stop` is a stop request/acknowledgement, not a separate lifecycle state.
- Return payload should include current public `state` plus `stop_requested`.

## Constraints
- Do not add dependencies outside the existing runtime (stdlib + `mcp`).
- Keep remote requests compatible with both:
  - HTTP wrapper (`/mcp/tools/call`)
  - Streamable MCP JSON-RPC (`/mcp`)
- Ensure all tool responses include structured content when an output schema is defined.
- Keep local proxy error semantics documented and stable (`REMOTE_ERROR`, `DOWNLOAD_FAILED`) and pass through cloud error payloads unchanged when possible.
- Tool-surface split must remain explicit:
  - local exposes `plan_download`.
  - cloud exposes `plan_file_info`.
  - do not expose `plan_file_info` as a local tool name.
- **Run as task**: Do not advertise the MCP **tasks** protocol (tasks/get, tasks/result, tasks/cancel, tasks/list) or add tool-level "Run as task" support. PlanExe’s interface is tool-based only (plan_create → plan_status → plan_download). The MCP tasks protocol is a different, client-driven feature; Cursor and the Python MCP SDK do not support it properly, so we keep tools-only for compatibility.

## External-facing naming
- All input/output fields use `plan_id` (not `task_id`) and `plans` (not `tasks`).
- Error codes use `PLAN_NOT_FOUND` and `PLAN_NOT_FAILED` (not `TASK_*`).
- Internal code and database table names may still reference `task_item` (legacy); do
  not expose these in tool schemas, descriptions, or user-facing messages.

## plan_status latency
- `plan_status` on the cloud side resolves file lists by trying local DB/disk sources
  first, then falling back to a worker HTTP call (30-second timeout).
- If the worker is unreachable and no local data exists yet (e.g., plan is still
  `processing`), `plan_status` may take up to 30 seconds to respond.
- The local proxy has its own retry logic with exponential backoff for transient
  server errors. Combined with a slow `plan_status`, a single poll can take
  significantly longer than usual — callers should use generous timeouts.

## Env vars
- `PLANEXE_URL`: Base URL for mcp_cloud (e.g., `http://localhost:8001/mcp`).
- `PLANEXE_MCP_API_KEY`: API key forwarded to remote as custom header `X-API-Key`.
- `PLANEXE_PATH`: Local directory where downloads are saved.
  - Must be a directory.
  - Created automatically when missing.
  - Defaults to current working directory when unset.
  - Saved filename pattern: `<plan_id>-<artifact_basename>` with numeric suffixes on collisions.

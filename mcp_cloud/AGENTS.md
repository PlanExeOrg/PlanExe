# mcp_cloud agent instructions

Scope: Model Context Protocol (MCP) server that provides a standardized interface
for AI agents and developer tools to interact with PlanExe. Communicates with
`worker_plan_database` via the shared Postgres database (`database_api` models).

## Module map

| File | Purpose | Key exports |
|------|---------|-------------|
| `app.py` | Re-export facade for backward compatibility | 60+ symbols from all sub-modules |
| `http_server.py` | FastAPI HTTP wrapper: routes, middleware stack, FastMCP mount, SSE endpoint | `app` (FastAPI instance) |
| `handlers.py` | MCP tool handlers dispatched by `TOOL_HANDLERS` dict | `handle_plan_create`, `handle_plan_status`, etc. |
| `db_setup.py` | Flask + SQLAlchemy init, constants, request DTOs | `app` (Flask), `db`, `mcp_cloud_server`, `PLANEXE_SERVER_INSTRUCTIONS` |
| `db_queries.py` | Sync DB queries (run via `asyncio.to_thread()`) | `_create_plan_sync`, `_get_plan_status_snapshot_sync`, `get_plan_state_mapping` |
| `tool_models.py` | Pydantic input/output models for all 9 tools | `PlanCreateInput`, `PlanCreateOutput`, `PlanStatusOutput`, etc. |
| `schemas.py` | Auto-generate JSON Schema from tool_models via `.model_json_schema()` | `TOOL_DEFINITIONS` (list of `ToolDefinition` dataclasses) |
| `auth.py` | API key SHA-256 hashing + user resolution | `_hash_user_api_key`, `_resolve_user_from_api_key` |
| `download_tokens.py` | Signed HMAC-SHA256 download tokens + URL builders | `generate_download_token`, `validate_download_token`, `_get_download_base_url` |
| `sse.py` | SSE stream generator with connection tracking | `plan_progress_stream`, `_track_sse_connection`, `SSEConnectionLimitError` |
| `worker_fetchers.py` | Fetch artifacts (report, zip) from worker HTTP / local FS / DB | `fetch_artifact_from_worker_plan`, `fetch_user_downloadable_zip` |
| `model_profiles.py` | Load LLM models per profile from `llm_config/*.json` | `_get_model_profiles_sync` |
| `example_prompts.py` | Load example prompts from catalog or built-in fallbacks | `_load_mcp_example_prompts` |
| `zip_utils.py` | Zip extraction, legacy sanitization, SHA-256 | `list_files_from_zip_snapshot`, `_sanitize_legacy_zip_snapshot` |
| `http_utils.py` | Strip redundant `content` when `structuredContent` exists | `strip_redundant_content` |
| `dotenv_utils.py` | Load `.env` from mcp_cloud/ or repo root | `load_planexe_dotenv` |
| `config.py` | Flask config (`SQLALCHEMY_TRACK_MODIFICATIONS = False`) | — |

## Import graph

```
app.py (facade — re-exports everything)
├── db_setup.py (Flask, DB, constants)
├── auth.py (key hashing, user lookup)
├── db_queries.py (sync queries; uses db_setup.app context)
├── zip_utils.py (zip ops; uses db_setup.app context)
├── worker_fetchers.py (HTTP + local + DB fallbacks; uses zip_utils)
├── model_profiles.py (profile loading from llm_config/)
├── download_tokens.py (HMAC tokens, context var for base URL)
├── example_prompts.py (catalog loading)
├── schemas.py (auto-generated from tool_models.py)
└── handlers.py (tool handlers; imports all above)

http_server.py (top-level entry point)
├── app.py (re-export facade)
├── tool_models.py (type annotations for FastAPI)
├── auth.py (validate_api_key_secret)
├── download_tokens.py (validate_download_token_secret)
├── sse.py (SSE implementation)
└── http_utils.py (content stripping)
```

## Guidelines
- Keep database access wired through `database_api.planexe_db_singleton.db`;
  do not create new engine/session instances here.
- Preserve the startup sequence in `mcp_cloud/db_setup.py`:
  `.env` loading, logging setup, Flask app config, then `db.init_app(app)`.
  `app.py` is a thin re-export facade; actual init happens in `db_setup.py`.
- Maintain the DB connection logic:
  - Prefer `SQLALCHEMY_DATABASE_URI` when set.
  - Otherwise build from `PLANEXE_POSTGRES_*` (see root `AGENTS.md` for keys).
- MCP tools must follow the specification in `docs/mcp/planexe_mcp_interface.md`:
  - Task management maps to `PlanItem` records (each task = one PlanItem).
  - Events are queried from `EventItem` database records.
- Use the PlanItem UUID as the MCP `plan_id`.
- Public task state contract:
  - `plan_status.state` must use exactly: `pending`, `processing`, `completed`, `failed`.
  - These values correspond 1:1 with `database_api.model_planitem.PlanState`.
  - Do not use legacy public names like `running`, `stopping`, or `stopped` for `plan_status`.
  - Do not expose internal symbol/class names (for example `PlanState.pending`, `PlanItem.state`) in model-facing tool descriptions; use plain public state strings.
- Download contract:
  - `track_activity.jsonl` is internal-only (`PlanItem.run_track_activity_jsonl`).
  - Downloadable zip artifacts must never include `track_activity.jsonl`.
  - Serve new layout snapshots directly; sanitize only legacy/fallback zips.
- `plan_stop` contract:
  - `plan_stop` does not create a separate lifecycle state.
  - Return current public `state` plus `stop_requested` to acknowledge stop-flag request.
- Forbidden imports: `worker_plan.app`, `worker_plan_internal`, `frontend_*`,
  `open_dir_server`.

## Async/sync boundary

All handlers in `handlers.py` are `async`. All database operations in `db_queries.py`
are sync (require Flask app context). Bridge pattern:

```python
result = await asyncio.to_thread(_sync_db_function, args)
```

`db_queries.py` functions check `has_app_context()` and open one if needed, so they
work from both sync and async callers.

## Context variables

Two `contextvars.ContextVar` instances are set per-request by the middleware in
`http_server.py` and cleared in the `finally` block:

- `_download_base_url_ctx` (in `download_tokens.py`): Base URL for building download
  and SSE URLs. Set from request origin for `/mcp` and `/sse/` paths. Read by
  `_get_download_base_url()` — used in `handlers.py` to build `download_url` and
  `sse_url` fields.
- `_authenticated_user_api_key_ctx` (in `http_server.py`): Authenticated user's raw
  API key. Injected into `plan_create` and `plan_list` arguments by FastMCP wrappers.

## Schema auto-generation

`schemas.py` generates JSON Schema from `tool_models.py` Pydantic classes:

1. Define input/output models in `tool_models.py` using `BaseModel` + `Field()`
2. `schemas.py` calls `.model_json_schema()` on each class
3. Wraps output schemas in `oneOf` for error/success variants
4. Packages into `ToolDefinition` dataclasses in `TOOL_DEFINITIONS` list
5. `handlers.py` returns `TOOL_DEFINITIONS` to MCP clients via `handle_list_tools()`

When adding a field to a tool response: update `tool_models.py` model → schemas
auto-update → no manual schema changes needed.

## plan_create contract
- Expose `model_profiles` as the discovery tool for profile selection.
- `model_profiles` must report profile guidance and currently available models after class whitelist filtering.
- Keep workflow wording explicit that prompt drafting + user approval is a non-tool step before `plan_create`.
- Keep concurrency wording explicit: each `plan_create` call creates a new `plan_id`; no global per-client concurrency cap is enforced server-side.
- Visible input schema is intentionally limited to:
  - `prompt`
  - `model_profile` (`baseline`, `premium`, `frontier`, `custom`)
  - `user_api_key` (optional)
- Billing attribution: when `user_api_key` is provided, `handle_plan_create`
  resolves the user and sets `metadata["user_id"]` (UUID) and
  `metadata["api_key_id"]` on the PlanItem. Without `user_api_key` (local dev
  with `PLANEXE_MCP_REQUIRE_AUTH=false`), the plan gets `user_id="admin"` and
  no `api_key_id` — billing is skipped and per-key stats won't show usage.
  This is by design; pass `user_api_key` to enable credit tracking via MCP.
- Auth-disabled key resolution: when `PLANEXE_MCP_REQUIRE_AUTH=false`,
  `_validate_api_key` still resolves any provided API key for attribution
  (`last_used_at`, per-key billing) — it just never rejects unauthenticated
  requests. This sets `_authenticated_user_api_key_ctx` so `plan_create` and
  `plan_retry` can inject the key for billing attribution.
- Plan retry attribution: `handle_plan_retry` resolves caller identity from
  `user_api_key` and passes `caller_metadata` to `_retry_failed_plan_sync`,
  which updates `plan.user_id` and `plan.api_key_id`. Old incremental billing
  entries are archived (`usage_billing_progress` → `usage_billing_settled`),
  not deleted, so the previous key's credit history is preserved.

## HTTP middleware stack

The middleware in `http_server.py` processes requests in this order:

1. **`_NormalizeMcpPath`** (ASGI middleware): Rewrites `/mcp` → `/mcp/` at scope level
   (avoids 307 redirect that breaks Smithery)
2. **`CORSMiddleware`** (FastAPI built-in): Added first, handles OPTIONS preflight
3. **`enforce_api_key`** (HTTP middleware via `BaseHTTPMiddleware`): Auth, body size,
   rate limiting, context var setup
   - Paths requiring auth: `/mcp`, `/download`
   - **`/sse/` is intentionally excluded** — `BaseHTTPMiddleware` pipes the response
     body through an internal `anyio.MemoryObjectStream`; for long-lived SSE streams
     this keeps the middleware's task-group alive indefinitely and starves concurrent
     requests. The SSE endpoint handles auth inline instead.
   - Download tokens are self-authenticating (signed HMAC, no API key needed)
   - Sets `_download_base_url_ctx` for `/mcp` paths
   - Strips redundant `content` from `/mcp` JSON responses on the way out

## MCP Protocol
- The server communicates over stdio (standard input/output) following the MCP protocol.
- Tools are registered via `@mcp_cloud.list_tools()` and handled via `@mcp_cloud.call_tool()`.
- All tool responses must be JSON-serializable and follow the error model in the spec.
- Keep tool error codes/docs aligned with actual runtime payloads (for example `PLAN_NOT_FOUND`, `INVALID_USER_API_KEY`, `USER_API_KEY_REQUIRED`, `INSUFFICIENT_CREDITS`, `generation_failed`, `content_unavailable`, `INTERNAL_ERROR`).
- Event cursors use format `cursor_{event_id}` for incremental polling.
- **Run as task**: We expose MCP **tools** only (plan_create, plan_status, plan_stop, etc.), not the MCP **tasks** protocol (tasks/get, tasks/result, etc.). Do not advertise the tasks capability or add "Run as task" support; the spec and clients (e.g. Cursor) are aligned on tools-only.

## Authentication Policy
- PlanExe MCP cloud authentication is API-key header based.
- Canonical client header is `X-API-Key: pex_...`.
- API key sources (checked in order): `Authorization: Bearer {key}`, `X-API-Key` header, `api_key` query param.
- Validation order: `PLANEXE_MCP_API_KEY` (shared secret) → `UserApiKey` table lookup (SHA-256 hash).
- OAuth is not supported for the MCP API. Do not document, imply, or advertise OAuth support.
- In docs and user-facing error/help text, instruct clients to use `X-API-Key` custom headers.
- Keep the auth split used for connector health checks:
  - Unauthenticated discovery/handshake is allowed for:
    - MCP methods: `initialize`, `notifications/initialized`, `tools/list`, `prompts/list`, `prompts/get`, `resources/list`, `resources/templates/list`, `ping`
    - Probe compatibility: `GET/HEAD/POST /mcp`, `GET/HEAD /mcp/`, and `GET /mcp/tools`
  - `tools/call` without API key is allowed **only** for free setup tools:
    - `example_plans`
    - `example_prompts`
    - `model_profiles`
  - All other tool invocations (for example `plan_create`) must remain API-key protected.
- Keep auth-denial logging explicit (`Auth rejected: ...`) with method/path/user-agent and parsed JSON-RPC methods to make Railway debugging easier.

## SSE endpoint

`GET /sse/plan/{plan_id}` streams real-time plan progress as Server-Sent Events.
This is a complementary REST endpoint, **not** an MCP transport replacement.

Implementation in `sse.py`:
- `plan_progress_stream(plan_id, disconnect_event)`: Async generator that polls
  `_get_plan_status_snapshot_sync()` every ~3s and yields SSE-formatted events.
- Deduplication: only emits `status` events when `state` or `progress_percentage` changes.
- Heartbeat every ~20s of silence (keeps reverse proxies alive).
- `complete` event on terminal state (completed/failed), then generator closes.
- 60-minute absolute timeout.
- Connection tracking via `_track_sse_connection(client_id)` async context manager:
  per-client limit (5) and server-wide limit (200). Raises `SSEConnectionLimitError` (HTTP 429).

Auth: `/sse/` paths require API key (handled inline in the endpoint, NOT via the
`enforce_api_key` middleware — see HTTP middleware stack section for rationale).
URL building: `handlers.py` adds `sse_url` to `plan_create` and `plan_status` responses
using `_get_download_base_url()` from `download_tokens.py`.

Event types: `status`, `heartbeat`, `complete`, `error`.

Config env vars: `PLANEXE_SSE_POLL_INTERVAL`, `PLANEXE_SSE_HEARTBEAT_INTERVAL`,
`PLANEXE_SSE_MAX_DURATION`, `PLANEXE_SSE_MAX_CONNECTIONS`, `PLANEXE_SSE_MAX_TOTAL_CONNECTIONS`.

## Download token system

`download_tokens.py` generates signed, time-limited HMAC-SHA256 tokens scoped to
a specific plan + artifact. Token format: `{expiry_unix_ts}.{hmac_hex}`.

- HMAC message: `plan_id:filename:expiry` — prevents token reuse across artifacts.
- Secret priority: `PLANEXE_DOWNLOAD_TOKEN_SECRET` → `PLANEXE_API_KEY_SECRET` → per-process random (dev only).
- Default TTL: 900 seconds (15 minutes), configurable via `PLANEXE_DOWNLOAD_TOKEN_TTL`.
- Tokens are stateless (no server-side storage or revocation tracking).

The same `_get_download_base_url()` function is used to build both `download_url`
(in `plan_file_info`) and `sse_url` (in `plan_create`/`plan_status`).

## Download URL environment behavior
- `plan_file_info.download_url` and `sse_url` are built from `PLANEXE_MCP_PUBLIC_BASE_URL` when set.
- If `PLANEXE_MCP_PUBLIC_BASE_URL` is unset in HTTP mode, use request host/scheme
  (via `_download_base_url_ctx` context var set in middleware).
- If no public base URL is available, `download_url` and `sse_url` may be absent;
  document this and guide operators to set `PLANEXE_MCP_PUBLIC_BASE_URL`.

## HTTP Compatibility and Crawler Endpoints
- Keep `/mcp` -> `/mcp/` redirect behavior for slashless clients/probers.
- Keep CORS headers on early error responses (401/403/429/etc.) so browser inspectors do not fail with opaque CORS errors.
- Keep `PLANEXE_MCP_CORS_ORIGINS` parsing tolerant to quoted CSV and JSON-array env formats.
- Keep `GET /robots.txt` available (200) for crawler health checks and metadata discovery.
- FastMCP session lifecycle lines like `Terminating session: None` are expected informational logs; do not treat them as application failures solely based on Railway's log-level labeling.

## mcp_local integration
- `mcp_local` runs on the user's machine and forwards tool calls to this server over HTTP.
- It targets either:
  - the HTTP wrapper endpoint (`/mcp/tools/call`), or
  - the streamable MCP JSON-RPC endpoint (`/mcp`).
- Tool-surface split must stay explicit:
  - `mcp_cloud` exposes `plan_file_info` (not `plan_download`).
  - `mcp_local` exposes `plan_download` and implements it via cloud `plan_file_info`.
- `plan_file_info` provides download metadata that `mcp_local` uses to download
  artifacts via `/download/{plan_id}/...`.

## Troubleshooting guidance (caller-facing text)
- Keep guidance aligned across server instructions and tool descriptions:
  - `pending` for longer than 5 minutes usually means queued but not picked up by worker.
  - `processing` with no output-file changes for longer than 20 minutes usually means stalled/failed execution.
  - In both cases, direct users to report issues at `https://github.com/PlanExeOrg/PlanExe/issues`.

## MCP Registry metadata
- Registry metadata for this server lives at `mcp_cloud/server.json`.
- Keep `server.json` aligned with deployed behavior:
  - `remotes[].url` must point at the production MCP endpoint.
  - required auth headers must match the server auth policy (`X-API-Key`).
- Publish with `mcp-publisher` from the `mcp_cloud/` directory so it picks up this file.

## PlanItem deferred columns
- `PlanItem.generated_report_html`, `run_zip_snapshot`, and `run_track_activity_jsonl`
  are declared with `deferred()` so they are **not** loaded by default ORM queries.
- **column_property and deferred()**: `deferred()` descriptors do not support SQL
  operators (`.isnot()`, etc.) at class-definition time — they return `NotImplemented`.
  The `has_*` column properties are therefore defined **after** the class body using
  `PlanItem.__table__.c` (raw Column objects). Type hints inside the class use
  `TYPE_CHECKING` guards so pyright can see them without confusing SQLAlchemy.
- **Session lifetime**: When accessing a deferred column, the ORM instance must still
  be bound to an active session. `get_plan_by_id()` may open a temporary app context
  that closes before the caller touches the deferred attribute, detaching the instance.
  Use `_load_plan_column()` (in `zip_utils.py`) to load a PlanItem and read a deferred
  attribute inside a single app context.
- **Read-only queries**: Prefer explicit column selection (`db.session.query(Col1, Col2, …)`)
  over loading full ORM objects for read-only endpoints like `plan_status` and `plan_list`.
  This avoids accidentally triggering deferred loads and reduces data transfer.

## Worker HTTP fallback ordering
- When resolving file lists or artifacts, try fast local sources first:
  1. DB zip snapshot (`list_files_from_zip_snapshot` / `fetch_file_from_zip_snapshot`)
  2. Local run directory (`list_files_from_local_run_dir`)
  3. Worker HTTP (`fetch_file_list_from_worker_plan` / `fetch_artifact_from_worker_plan`)
- The worker HTTP call has a 30-second timeout. If it runs first, every poll blocks
  for 30 seconds when the worker is unreachable — even when the data is already in the DB.

## Testing
- Automated tests exist under `mcp_cloud/tests/`.
- Tests require Docker dependencies (flask_sqlalchemy, mcp, database_api) — they
  run inside the Docker container, not from the host Python directly.
- If you change MCP tool behavior, state mapping, or tool surface, update/add unit
  tests close to the changed logic.
- Test pattern: `unittest.TestCase` + `unittest.mock.patch` on `mcp_cloud.handlers.*`
  and `mcp_cloud.sse.*` to mock DB calls without a real database.
- Run focused tests from repo root, for example:
  - `python -m unittest mcp_cloud.tests.test_tool_surface_consistency`
  - `python -m unittest mcp_cloud.tests.test_plan_status_tool`
  - `python -m pytest mcp_cloud/tests/test_sse.py -v`

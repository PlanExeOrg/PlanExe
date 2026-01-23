# mcp_cloud agent instructions

Scope: Model Context Protocol (MCP) server that provides a standardized interface
for AI agents and developer tools to interact with PlanExe. Communicates with
`worker_plan_database` via the shared Postgres database (`database_api` models).

## Guidelines
- Keep database access wired through `database_api.planexe_db_singleton.db`;
  do not create new engine/session instances here.
- Preserve the startup sequence in `mcp_cloud/app.py`:
  `.env` loading, logging setup, Flask app config, then `db.init_app(app)`.
- Maintain the DB connection logic:
  - Prefer `SQLALCHEMY_DATABASE_URI` when set.
  - Otherwise build from `PLANEXE_POSTGRES_*` (see root `AGENTS.md` for keys).
- MCP tools must follow the specification in `extra/planexe_mcp_interface.md`:
  - Task management maps to `TaskItem` records (each task = one TaskItem).
  - Events are queried from `EventItem` database records.
- Use the TaskItem UUID as the MCP `task_id`.
- Forbidden imports: `worker_plan.app`, `worker_plan_internal`, `frontend_*`,
  `open_dir_server`.

## MCP Protocol
- The server communicates over stdio (standard input/output) following the MCP protocol.
- Tools are registered via `@mcp_cloud.list_tools()` and handled via `@mcp_cloud.call_tool()`.
- All tool responses must be JSON-serializable and follow the error model in the spec.
- Event cursors use format `cursor_{event_id}` for incremental polling.

## mcp_local integration
- `mcp_local` runs on the user's machine and forwards tool calls to this server over HTTP.
- It targets either:
  - the HTTP wrapper endpoint (`/mcp/tools/call`), or
  - the streamable MCP JSON-RPC endpoint (`/mcp`).
- `task_file_info` provides download metadata that `mcp_local` uses to download
  artifacts via `/download/{task_id}/...`.

## Testing
- No automated tests currently. If you change MCP tool behavior or database mappings,
  add a unit test close to the logic when feasible and run `python test.py` from
  repo root.

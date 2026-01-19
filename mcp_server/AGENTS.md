# mcp_server agent instructions

Scope: Model Context Protocol (MCP) server that provides a standardized interface
for AI agents and developer tools to interact with PlanExe. Communicates with
`worker_plan_database` via the shared Postgres database (`database_api` models).

## Guidelines
- Keep database access wired through `database_api.planexe_db_singleton.db`;
  do not create new engine/session instances here.
- Preserve the startup sequence in `mcp_server/app.py`:
  `.env` loading, logging setup, Flask app config, then `db.init_app(app)`.
- Maintain the DB connection logic:
  - Prefer `SQLALCHEMY_DATABASE_URI` when set.
  - Otherwise build from `PLANEXE_POSTGRES_*` (see root `AGENTS.md` for keys).
- MCP tools must follow the specification in `extra/mcp-spec1.md`:
  - Task management maps to `TaskItem` records (each task = one TaskItem).
  - Events are queried from `EventItem` database records.
- Keep task ID format consistent: `pxe_{YYYY_MM_DD}__{short_uuid}` stored
  in `TaskItem.parameters._mcp_task_id`.
- Forbidden imports: `worker_plan.app`, `worker_plan_internal`, `frontend_*`,
  `open_dir_server`.

## MCP Protocol
- The server communicates over stdio (standard input/output) following the MCP protocol.
- Tools are registered via `@mcp_server.list_tools()` and handled via `@mcp_server.call_tool()`.
- All tool responses must be JSON-serializable and follow the error model in the spec.
- Event cursors use format `cursor_{event_id}` for incremental polling.

## Testing
- No automated tests currently. If you change MCP tool behavior or database mappings,
  add a unit test close to the logic when feasible and run `python test.py` from
  repo root.

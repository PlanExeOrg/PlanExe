# PlanExe agent instructions

Scope: repo-level guardrails for PlanExe services and shared packages.
Always check the package-level `AGENTS.md` for file-specific rules
(use `rg --files -g 'AGENTS.md'` if you are unsure).

## Repo architecture map
- `database_api`: shared SQLAlchemy models for DB-backed services.
- `worker_plan/worker_plan_api`: shared API types/helpers (must stay lightweight).
- `worker_plan`: FastAPI service that runs the pipeline.
- `frontend_single_user`: Gradio UI (local mode).
- `frontend_multi_user`: Flask UI (hosted mode) + Postgres.
- `open_dir_server`: local host opener (security critical).
- `worker_plan_database`: DB-backed worker that polls tasks.
- `mcp_cloud`: MCP stdio server + HTTP wrapper; primary cloud deployment, secondary Docker setup for advanced users, tertiary venv workflow for developers; bridges MCP tools to PlanExe DB/worker_plan.
- `mcp_local`: local MCP proxy that forwards tool calls to `mcp_cloud` and downloads artifacts.

## Shared contracts
- Keep `worker_plan` HTTP endpoints and response shapes backward compatible.
- Preserve shared SQLAlchemy models in `database_api` (nullable defaults for new columns).
- Run directory naming defaults live in `worker_plan/worker_plan_api/generate_run_id.py`.
  Current defaults: single-user uses timestamped IDs; multi-user can use UUIDs.
  Verify in code before changing run-id formats.
- Keep prompt catalog UUIDs stable when used as defaults; they live in
  `worker_plan/worker_plan_api/prompt/data/*.jsonl`.

## Hard rules (agent safety)
- Do not add real API keys or passwords to `.env`, `.env.*`, or `llm_config.json`.
- Do not change run-dir validation or path-allowlist logic in `open_dir_server/app.py`
  unless explicitly instructed.
- Shared packages (`database_api`, `worker_plan_api`) must not import service apps
  (`frontend_*`, `worker_plan_database`, `open_dir_server`, `worker_plan.app`).
- If a service needs shared logic, move it into a shared package rather than
  importing across service boundaries.

## Environment variables
- Canonical env keys live in `.env.docker-example`, `.env.developer-example`,
  and `worker_plan/worker_plan_api/planexe_dotenv.py`. Read those before adding
  or renaming env vars.

## Cross-service conventions
- `.env` and `llm_config.json` are expected in the repo root for Docker setups.
- Use `PLANEXE_*` env vars; prefer existing defaults when adding new ones.
- Do not assume ports; read `docker-compose.yml` or the service env defaults
  (`PLANEXE_*_PORT`) before any manual verification.

## Docker notes
- `PLANEXE_POSTGRES_PORT` changes the host port mapping only; containers still
  connect to Postgres on 5432.
- The `Open Output Dir` button requires the host `open_dir_server` and
  `PLANEXE_OPEN_DIR_SERVER_URL` (OS-specific URLs live in `docs/docker.md`).
- Keep `PLANEXE_HOST_RUN_DIR` consistent with run dir mounts so outputs land in
  the expected host folder.

## Documentation sync
- When changing Docker services, env defaults, or port mappings, update
  `docker-compose.yml`, `docker-compose.md`, and `docs/docker.md` together.
- When changing single-user quickstart or LLM env requirements (e.g.
  `OPENROUTER_API_KEY`), update `docs/getting_started.md`.
- When changing local dev startup steps or the test command, update
  `docs/install_developer.md`.

## Testing strategy
- Prefer unit tests over manual curl/server checks.
- Run `python test.py` from repo root for existing coverage.
- If you change logic without tests, add a unit test close to the code.
- Do not start services or run curl checks unless explicitly requested.

## Coding standards
- Type hints: add to all public function signatures.
- Async/sync: FastAPI code can be `async`; Flask routes must stay sync.
- Error handling: do not use bare `except:`; log stack traces.
- Logging: use `logging` (avoid `print()` in service code).
- Formatting/linting: no repo-wide formatter is configured; follow existing
  file style and do not add new lint tools unless requested.

## Python version
- Canonical version is defined in each package `pyproject.toml`.

## Dependencies
- `worker_plan` and `frontend_multi_user`: add deps in their `pyproject.toml`.
- `frontend_single_user` and `worker_plan_database`: add deps in `requirements.txt`.
- Do not `pip install` ad-hoc without recording the dependency in the
  package manifest.

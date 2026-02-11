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
- For README links, prefer absolute `https://docs.planexe.org/...` URLs so GitHub
  readers land on the published docs site.

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

## Strategic Alignment (The PlanExe Mindset)
We are evolving from a "Plan Generator" (rough drafts) to a "Plan Execution Engine" (enterprise-grade).
New features and proposals should align with this shift:

1.  **Trust & Rigor:** AI output is untrusted by default.
    *   Favor features that verify, cite, and stress-test claims (e.g., Evidence Ledgers, Monte Carlo sims).
    *   Avoid features that just "generate more text" without validation.
2.  **Execution Focus:** A plan is a living tool, not a static PDF.
    *   Favor features that monitor reality vs. plan (e.g., drift monitors, readiness gates).
    *   Ensure data structures support dynamic updates (databases over documents).
3.  **Scale:** We are building for users who manage 100+ plans.
    *   Favor features that automate sorting, ranking, and routing (e.g., Elo ranking, automated dispatch).

## Ecosystem & OpenClaw
PlanExe is designed to be used by **both humans and autonomous agents**.
*   **OpenClaw** (an open-source agent framework) is a first-class citizen.
*   We support MCP (Model Context Protocol) to allow bots to natively call PlanExe tools.
*   When designing APIs, assume the caller might be a Raspberry Pi running OpenClaw, not just a human with a browser.

Docker Compose for PlanExe
==========================

TL;DR
-----
- Services: `database_postgres` (DB on `${PLANEXE_POSTGRES_PORT:-5432}`), `frontend_single_user` (UI on 7860), `worker_plan` (API on 8000), `frontend_multi_user` (UI on `${PLANEXE_FRONTEND_MULTIUSER_PORT:-5001}`), plus DB workers (`worker_plan_database` and `worker_plan_database_1/2/3`), and `mcp_cloud` (MCP interface, stdio); `frontend_single_user` waits for the worker to be healthy and `frontend_multi_user` waits for Postgres health.
- Shared host files: `.env` and `llm_config.json` mounted read-only; `./run` bind-mounted so outputs persist; `.env` is also loaded via `env_file`.
- Postgres defaults to user/db/password `planexe`; override via env or `.env`; data lives in the `database_postgres_data` volume.
- Env defaults live in `docker-compose.yml` but can be overridden in `.env` or your shell (URLs, timeouts, run dirs, optional auth and opener URL).
- `develop.watch` syncs code/config for `worker_plan` and `frontend_single_user`; rebuild with `--no-cache` after big moves or dependency changes; restart policy is `unless-stopped`.

Quickstart (run from repo root)
-------------------------------
- Up (single user): `docker compose up worker_plan frontend_single_user`.
- Up (multi user): `docker compose up frontend_multi_user database_postgres worker_plan worker_plan_database_1 worker_plan_database_2 worker_plan_database_3`.
- Up (MCP server): `docker compose up mcp_cloud` (requires `database_postgres` to be running).
- Down: `docker compose down` (add `--remove-orphans` if stray containers linger).
- Rebuild clean: `docker compose build --no-cache database_postgres worker_plan frontend_single_user frontend_multi_user worker_plan_database worker_plan_database_1 worker_plan_database_2 worker_plan_database_3 mcp_cloud`.
- UI: single user -> http://localhost:7860; multi user -> http://localhost:5001 after the stack is up.
- MCP: configure your MCP client to connect to the `mcp_cloud` container via stdio.
- Logs: `docker compose logs -f worker_plan` or `... frontend_single_user` or `... mcp_cloud`.
- One-off inside a container: `docker compose run --rm worker_plan python -m worker_plan_internal.fiction.fiction_writer` (use `exec` if already running).
- Ensure `.env` and `llm_config.json` exist; copy `.env.docker-example` to `.env` if you need a starter.

Why compose (escaping dependency hell)
--------------------------------------
- Dependency hell: when one Python package requires version A of a dependency while another requires version B (or a different Python), so `pip` cannot satisfy everything in one environment; the resolver loops, pins conflict, or installs a set that breaks another part of the app. System-level deps (libssl) can also clash, and "fixes" often mean uninstalling or downgrading unrelated packages.
- I want to experiment with the `uv` package manager; to try it, install `uv` during the image build and replace the `pip install ...` lines with `uv pip install ...`. Compose keeps that change isolated per service so it doesn’t spill onto the other containers or host Python.
- Compose solves this by isolating environments per service: each image pins its own base Python, OS libs, and `requirements.txt`, so the frontend and worker no longer fight over versions.
- Builds are reproducible: the `Dockerfile` installs a clean env from scratch, so you avoid ghosts from previous virtualenvs or globally-installed wheels.
- If a dependency change fails, you can rebuild from zero or switch base images without nuking your host Python setup.

What compose sets up
--------------------
- Reusable local stack with consistent env/paths under `/app` in each container.
- Shared run dir: `PLANEXE_RUN_DIR=/app/run` in the containers, bound to `${PLANEXE_HOST_RUN_DIR:-${PWD}/run}` on the host so outputs persist.
- Postgres data volume: `database_postgres_data` keeps the database files outside the repo tree.

Service: `database_postgres` (Postgres DB)
------------------------------------------
- Purpose: Storage in a Postgres database for future queue + event logging work; exposes `${PLANEXE_POSTGRES_PORT:-5432}` on the host mapped to 5432 in the container.
- Build: `database_postgres/Dockerfile` (uses the official Postgres image).
- Env defaults: `PLANEXE_POSTGRES_USER=planexe`, `PLANEXE_POSTGRES_PASSWORD=planexe`, `PLANEXE_POSTGRES_DB=planexe`, `PLANEXE_POSTGRES_PORT=5432` (override with env/.env).
- Data/health: data in the named volume `database_postgres_data`; healthcheck uses `pg_isready`.

### Port conflict with local Postgres

The default PostgreSQL port is 5432. On developer machines, this port is often already occupied by a local PostgreSQL installation:
- **macOS**: Postgres.app, Homebrew PostgreSQL, or pgAdmin's bundled server
- **Linux**: System PostgreSQL installed via apt/yum/dnf
- **Windows**: PostgreSQL installer, pgAdmin, or other database tools

If port 5432 is in use, Docker will fail to start `database_postgres` with a "port already in use" error.

**Solution**: Set `PLANEXE_POSTGRES_PORT` to a different value before starting:
```bash
export PLANEXE_POSTGRES_PORT=5433
docker compose up
```

**Important**: This only affects the HOST port mapping (how you access Postgres from your machine, e.g., via DBeaver or `psql`). Inside Docker, containers always communicate with each other on the internal port 5432—this is hardcoded and not affected by `PLANEXE_POSTGRES_PORT`.

Service: `frontend_single_user` (single user UI)
------------------------------------
- Purpose: Single user Gradio UI; waits for a healthy worker and serves on port 7860. Does not use database.
- Build: `frontend_single_user/Dockerfile`.
- Env defaults: `PLANEXE_WORKER_PLAN_URL=http://worker_plan:8000`, timeout, server host/port, optional password, optional `PLANEXE_OPEN_DIR_SERVER_URL` for the host opener.
- Volumes: mirrors the worker (`.env` ro, `llm_config.json` ro, `run/` rw) so both share config and outputs.
- Watch: sync frontend code, shared API code in `worker_plan/`, and config files; rebuild on `worker_plan/pyproject.toml`; restart on compose edits.

Service: `frontend_multi_user` (multi user UI)
------------------------------------------
- Purpose: Multi-user Flask UI with admin views (tasks/events/nonce/workers) backed by Postgres.
- Build: `frontend_multi_user/Dockerfile`.
- Env defaults: DB host `database_postgres`, port `5432`, db/user/password `planexe` (follows `PLANEXE_POSTGRES_*`); admin credentials must be provided via `PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME`/`PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD` (compose will fail if missing); container listens on fixed port `5000`, host maps `${PLANEXE_FRONTEND_MULTIUSER_PORT:-5001}`.
- Health: depends on `database_postgres` health; its own healthcheck hits `/healthcheck` on port 5000.

Service: `worker_plan` (pipeline API)
-------------------------------------
- Purpose: runs the PlanExe pipeline and exposes the API on port 8000; the frontend depends on its health.
- Build: `worker_plan/Dockerfile`.
- Env: `PLANEXE_CONFIG_PATH=/app`, `PLANEXE_RUN_DIR=/app/run`, `PLANEXE_HOST_RUN_DIR=${PWD}/run`, `PLANEXE_WORKER_RELAY_PROCESS_OUTPUT=true`.
- Health: `http://localhost:8000/healthcheck` checked via the compose healthcheck.
- Volumes: `.env` (ro), `llm_config.json` (ro), `run/` (rw).
- Watch: sync `worker_plan/` into `/app/worker_plan`, rebuild on `worker_plan/pyproject.toml`, restart on compose edits.

Service: `worker_plan_database` (DB-backed worker)
--------------------------------------------------
- Purpose: polls `TaskItem` rows in Postgres, marks them processing, runs the PlanExe pipeline, and writes progress/events back to the DB; no HTTP port exposed.
- Build: `worker_plan_database/Dockerfile` (ships `worker_plan` code, shared `database_api` models, and this worker subclass).
- Depends on: `database_postgres` health.
- Env defaults: derives `SQLALCHEMY_DATABASE_URI` from `PLANEXE_POSTGRES_HOST|PORT|DB|USER|PASSWORD` (fallbacks to `database_postgres` + `planexe/planexe` on 5432); `PLANEXE_CONFIG_PATH=/app`, `PLANEXE_RUN_DIR=/app/run`; MachAI confirmation URLs default to `https://example.com/iframe_generator_confirmation` for both `PLANEXE_IFRAME_GENERATOR_CONFIRMATION_PRODUCTION_URL` and `PLANEXE_IFRAME_GENERATOR_CONFIRMATION_DEVELOPMENT_URL` (override with real endpoints).
- Volumes: `.env` (ro), `llm_config.json` (ro), `run/` (rw for pipeline output).
- Entrypoint: `python -m worker_plan_database.app` (runs the long-lived poller loop).
- Multiple workers: compose defines `worker_plan_database_1/2/3` with `PLANEXE_WORKER_ID` set to `1/2/3`. Start the trio with:
  - `docker compose up -d worker_plan_database_1 worker_plan_database_2 worker_plan_database_3`
  - (Use `worker_plan_database` alone if you want a single unnumbered worker.)

Service: `mcp_cloud` (MCP interface)
--------------------------------------
- Purpose: Model Context Protocol (MCP) server that provides a standardized interface for AI agents and developer tools to interact with PlanExe. Communicates with `worker_plan_database` via the shared Postgres database.
- Build: `mcp_cloud/Dockerfile` (ships shared `database_api` models and the MCP server implementation).
- Depends on: `database_postgres` health.
- Env defaults: derives `SQLALCHEMY_DATABASE_URI` from `PLANEXE_POSTGRES_HOST|PORT|DB|USER|PASSWORD` (fallbacks to `database_postgres` + `planexe/planexe` on 5432); `PLANEXE_CONFIG_PATH=/app`, `PLANEXE_RUN_DIR=/app/run`; `PLANEXE_MCP_PUBLIC_BASE_URL=http://localhost:8001` for report download URLs.
- Volumes: `run/` (rw for artifact access).
- Entrypoint: `python -m mcp_cloud.app` (runs the MCP server over stdio).
- Communication: the server communicates over stdio (standard input/output) following the MCP protocol. Configure your MCP client to connect to this container. The container runs with `stdin_open: true` and `tty: true` to enable stdio communication.
- MCP tools: implements the specification in `extra/planexe_mcp_interface.md` including session management, artifact operations, and event streaming.

Usage notes
-----------
- Ports: host `8000->worker_plan`, `7860->frontend_single_user`, `${PLANEXE_FRONTEND_MULTIUSER_PORT:-5001}->frontend_multi_user`, `PLANEXE_POSTGRES_PORT (default 5432)->database_postgres`; change mappings in `docker-compose.yml` if needed.
- `.env` must exist before `docker compose up`; it is both loaded and mounted read-only. Same for `llm_config.json`. If missing, start from `.env.docker-example`.
- Host opener: set `PLANEXE_OPEN_DIR_SERVER_URL` so the frontend can reach your host opener service (see `extra/docker.md` for OS-specific URLs and optional `extra_hosts` on Linux).
- To relocate outputs, set `PLANEXE_HOST_RUN_DIR` (or edit the bind mount) to another host path.
- Database: connect on `localhost:${PLANEXE_POSTGRES_PORT:-5432}` with `planexe/planexe` by default; data persists via the `database_postgres_data` volume.

Example: running stack
----------------------

Snapshot from `docker compose ps` on a live stack with two numbered DB workers; your timestamps, ports, and container names may differ:

```
PROMPT> docker compose ps                                                 
NAME                     IMAGE                            COMMAND                  SERVICE                  CREATED          STATUS                   PORTS
database_postgres        planexe-database_postgres        "docker-entrypoint.s…"   database_postgres        8 hours ago      Up 8 hours (healthy)     0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp
frontend_multi_user      planexe-frontend_multi_user      "python /app/fronten…"   frontend_multi_user      8 hours ago      Up 2 minutes (healthy)   0.0.0.0:5001->5000/tcp, [::]:5001->5000/tcp
worker_plan              planexe-worker_plan              "uvicorn worker_plan…"   worker_plan              2 minutes ago    Up 2 minutes (healthy)   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp
worker_plan_database_1   planexe-worker_plan_database_1   "python -m worker_pl…"   worker_plan_database_1   15 seconds ago   Up 13 seconds            
worker_plan_database_2   planexe-worker_plan_database_2   "python -m worker_pl…"   worker_plan_database_2   15 seconds ago   Up 13 seconds  
```

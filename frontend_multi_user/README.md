# Frontend multi user

Flask-based multi-user UI for PlanExe. Runs in Docker, uses Postgres (defaults to the `database_postgres` service), and only needs the lightweight `worker_plan_api` helpers (no full `worker_plan` install).

## Quickstart with Docker
- Ensure `.env` and the `llm_config/` directory exist in the repo root (they are mounted into the container).
- `docker compose up frontend_multi_user`
- Open http://localhost:${PLANEXE_FRONTEND_MULTIUSER_PORT:-5001}/ (container listens on 5000). Health endpoint: `/healthcheck`.

## Config (env)
- `PLANEXE_FRONTEND_MULTIUSER_DB_HOST|PORT|NAME|USER|PASSWORD`: Postgres target (defaults follow `database_postgres` / `planexe` values).
- `PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME` / `PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD`: Admin login for the UI; must be set (service fails to start if missing).
- `PLANEXE_FRONTEND_MULTIUSER_HOST`: bind address inside the container (default 0.0.0.0).
- `PLANEXE_FRONTEND_MULTIUSER_PORT`: Flask port inside the container (default 5000).
- `PLANEXE_FRONTEND_MULTIUSER_DEBUG`: set `true` to enable Flask debug.
- `PLANEXE_API_KEY_SHOW_ONCE`: set `true` to only show API secrets once after creation (default: secrets are always visible with a copy button).
- `PLANEXE_CONFIG_PATH`: defaults to `/app` so PlanExe picks up `.env` + `llm_config/` that compose mounts.

## Run locally with a venv

For a faster edit/run loop without Docker. Work from inside `frontend_multi_user` so its dependencies stay isolated:

```bash
cd frontend_multi_user
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
export PYTHONPATH=$PWD/..:$PWD/../worker_plan:$PYTHONPATH
python src/app.py
```

Run `deactivate` when you are done with the venv.

The `PYTHONPATH` makes `worker_plan_api` and `database_api` importable without installing the full `worker_plan` package (which has fragile dependencies in `worker_plan_internal`).

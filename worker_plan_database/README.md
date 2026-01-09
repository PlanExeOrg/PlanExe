# Worker plan database

Subclass of the `worker_plan` service that runs the PlanExe pipeline with a Postgres database.

- Polls `TaskItem` rows, marks them processing, and runs the pipeline.
- Reports state/progress back to the DB and posts confirmations to MachAI.
- Uses the same `worker_plan_internal` code as `worker_plan`, plus the shared `database_api` models.
- Configure MachAI confirmation endpoints with `PLANEXE_IFRAME_GENERATOR_CONFIRMATION_PRODUCTION_URL` and `PLANEXE_IFRAME_GENERATOR_CONFIRMATION_DEVELOPMENT_URL` (both are required; the worker fails fast if missing).

## Docker usage
- Build/run single worker: `docker compose up --build worker_plan_database`
- Run three workers (each with `PLANEXE_WORKER_ID=1/2/3`): `docker compose up -d worker_plan_database_1 worker_plan_database_2 worker_plan_database_3`
- Reads `SQLALCHEMY_DATABASE_URI` when provided, otherwise builds one from:
  - `PLANEXE_POSTGRES_HOST|PORT|DB|USER|PASSWORD`
  - falls back to the `database_postgres` service defaults (`planexe/planexe` on port 5432)
- Logs stream to stdout with [12-factor style logging](https://12factor.net/logs). Configure with `PLANEXE_LOG_LEVEL` (defaults to `INFO`).
- Volumes mounted in compose: `./run` (pipeline output), `.env`, `llm_config.json`
- Entrypoint: `python -m worker_plan_database.app`

## Run locally with a venv

For a faster edit/run loop without Docker. Work from inside `worker_plan_database` so its dependencies stay isolated:

```bash
cd worker_plan_database
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ../worker_plan
pip install -r requirements.txt
export PYTHONPATH=$PWD/..:$PYTHONPATH
export PLANEXE_WORKER_ID=local
python -m worker_plan_database.app
```

Run `deactivate` when you are done with the venv.

The `PYTHONPATH` addition allows imports of `database_api` and `worker_plan_database` modules. The `pyrightconfig.json` configures the same paths for editor/IDE support.

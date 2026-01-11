# Worker plan service

This directory hosts the shared `worker_plan_internal` package that generates plans.

- `worker_plan_internal/`: core planning logic.
- `worker_plan_api/`: shared types (e.g., filenames) used by both the worker and frontend.

## Run locally with a venv

For a faster edit/run loop without Docker. Work from inside `worker_plan` so its dependencies stay isolated (they may be incompatible with `frontend_single_user`). Use Python 3.13 — several native wheels (pydantic-core, orjson, tiktoken, greenlet, jiter) do not yet publish for 3.14 and will fail to build.

```bash
cd worker_plan
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
export PYTHONPATH=$PWD/..:$PYTHONPATH
python -m worker_plan.app
```

The app reads configuration from the `.env` file (located in the project root or `PLANEXE_CONFIG_PATH`). Host and port default to `localhost:8000` and can be overridden via `PLANEXE_WORKER_HOST` and `PLANEXE_WORKER_PORT`.

The frontend can then point at `http://localhost:8000` via `PLANEXE_WORKER_PLAN_URL`.

If you hit `ModuleNotFoundError: No module named 'worker_plan'`, ensure you:
- are in `PlanExe/worker_plan` (not a subfolder)
- ran `pip install -e .` in this venv without errors
- exported `PYTHONPATH=$PWD/..:$PYTHONPATH` before starting uvicorn (the package lives one level up when your CWD is `worker_plan`)

If you must stay on Python 3.14, expect source builds and potential failures; exporting `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `pip install -e .` may allow wheels to build, but 3.13 is recommended for a smooth setup.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PLANEXE_WORKER_HOST` | `0.0.0.0` | Host address the worker binds to (only when running via `python -m worker_plan.app`). |
| `PLANEXE_WORKER_PORT` | `8000` | Port the worker listens on (only when running via `python -m worker_plan.app`). |
| `PLANEXE_RUN_DIR` | `run` | Directory under which run output folders are created. |
| `PLANEXE_HOST_RUN_DIR` | *(unset)* | Optional host path base returned in `display_run_dir` to hint where runs live on the host. |
| `PLANEXE_CONFIG_PATH` | `.` | Working directory for the pipeline; used as the `cwd` when spawning `worker_plan_internal.plan.run_plan_pipeline`. |
| `PLANEXE_WORKER_RELAY_PROCESS_OUTPUT` | `false` | When `true`, pipe pipeline stdout/stderr to the worker logs instead of suppressing them. |
| `PLANEXE_PURGE_ENABLED` | `false` | Enable the background scheduler that purges old run directories. |
| `PLANEXE_PURGE_MAX_AGE_HOURS` | `1` | Maximum age (hours) of runs to delete when purging (scheduler and manual default). |
| `PLANEXE_PURGE_INTERVAL_SECONDS` | `3600` | How often the purge scheduler runs when enabled. |
| `PLANEXE_PURGE_RUN_PREFIX` | `PlanExe_` | Only purge runs whose IDs start with this prefix. |
| `PLANEXE_LOG_LEVEL` | `INFO` | Sets the console log level for the worker API and the pipeline process. Accepted values are the standard logging levels (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). |

`PLANEXE_LOG_LEVEL` affects both the FastAPI worker and the spawned pipeline logs written to stdout. File logs in `run/<id>/log.txt` always include DEBUG and above.

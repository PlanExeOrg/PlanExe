# Frontend Single User

This directory contains the PlanExe single-user Gradio frontend.

## Run locally with a venv

For a faster edit/run loop without Docker. Work from inside `frontend_single_user` so its dependencies stay isolated (they may be incompatible with `worker_plan`):

```bash
cd frontend_single_user
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$PWD/../worker_plan:$PYTHONPATH
export PLANEXE_OPEN_DIR_SERVER_URL=http://localhost:5100
export PLANEXE_WORKER_PLAN_URL=http://localhost:8000
python app.py
```

Then open http://localhost:7860 (or your `PLANEXE_GRADIO_SERVER_PORT`). Run `deactivate` when you are done with the venv.

If you prefer to install the shared API package instead of using `PYTHONPATH`, run `pip install -e ../worker_plan` (this will bring the worker dependencies into the same venv).

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PLANEXE_WORKER_PLAN_URL` | `http://worker_plan:8000` | Base URL for `worker_plan` service the UI calls. |
| `PLANEXE_WORKER_PLAN_TIMEOUT` | `30` | HTTP timeout (seconds) for `worker_plan` requests. |
| `PLANEXE_GRADIO_SERVER_NAME` | `0.0.0.0` | Host/interface Gradio binds to. |
| `PLANEXE_GRADIO_SERVER_PORT` | `7860` | Port Gradio listens on. |
| `PLANEXE_PASSWORD` | *(unset)* | Optional password to protect the UI (`user` / `<value>`). Leave unset for local development without auth. |
| `PLANEXE_OPEN_DIR_SERVER_URL` | *(unset)* | URL of the host opener service for “Open Output Dir”; leave unset to hide the button. |

## Password

Leave `PLANEXE_PASSWORD` unset when running PlanExe on your own computer.

However when running in the cloud, here you may want password protection.

Set `PLANEXE_PASSWORD` to turn on Gradio’s basic auth. Example:

```bash
export PLANEXE_PASSWORD=123
docker compose up
```

Then open the app and log in with username `user` and password `123`.

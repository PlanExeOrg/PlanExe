# Host Open Dir Server

## Why this exists
- Docker containers cannot launch host applications (e.g., macOS Finder) because they are isolated from the host OS.
- The Gradio frontend runs in a container and cannot run `open`, `xdg-open`, or `start` on the host.
- This small FastAPI service runs **on the host** and receives a path from the frontend, then asks the host OS to open that path.

## Prerequisites
- Python 3.10+ on the host (outside Docker).

## Setup (virtual environment)
```bash
eval "$(python setup_env.py)"
cd open_dir_server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Configuration
Environment variables (`PLANEXE_` prefixed):
- `PLANEXE_OPEN_DIR_SERVER_HOST` (default `127.0.0.1`)
- `PLANEXE_OPEN_DIR_SERVER_PORT` (default `5100`)
- `PLANEXE_HOST_RUN_DIR`: required; only allow opening paths under this directory.

Frontend configuration:
- Set `PLANEXE_OPEN_DIR_SERVER_URL` so the container can reach the host service (e.g., `http://host.docker.internal:5100`).

## Start the server
From `open_dir_server`:
```bash
eval "$(python setup_env.py)"
cd open_dir_server
source .venv/bin/activate
python app.py
```
The service will listen on `PLANEXE_OPEN_DIR_SERVER_HOST:PLANEXE_OPEN_DIR_SERVER_PORT`.

## Stop the server
- Press `Ctrl+C` in the terminal where it is running.

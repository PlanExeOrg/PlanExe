# PlanExe uses Docker

I'm no fan of Docker and I tried to avoid it.
I first tried dependency hell, and it was fragile. Incompatible packages, preventing installing/upgrading things.
I had to choose between dependency hell or docker hell.
Now I'm going with Docker. Hopefully it turns out to be less fragile.

## Basic lifecycle
- Stop everything: `docker compose down`
- Build fresh (no cache) after code moves: `docker compose build --no-cache database_postgres worker_plan frontend_single_user frontend_multi_user`
- Start services: `docker compose up`
- Stop services (leave images): `docker compose down`
- Build fresh and start services: `docker compose build --no-cache database_postgres worker_plan frontend_single_user frontend_multi_user && docker compose up`

## While developing
- Live rebuild/restart on changes: `docker compose watch` (requires Docker Desktop 4.28+).  
  If watch misses changes after file moves, rerun the no-cache build above.
- View logs: 
  - `docker compose logs -f worker_plan`
  - `docker compose logs -f frontend_single_user`
  - `docker compose logs -f frontend_multi_user`

## Run individual files
- Rebuild the worker image when code or data files change: `docker compose build --no-cache worker_plan`.
- Run a one-off module inside the worker image (same deps/env as the API):  
  `docker compose run --rm worker_plan python -m worker_plan_internal.fiction.fiction_writer` (swap the module path as needed). If containers are already up, use `docker compose exec worker_plan python -m ...` instead.
- For host Ollama access, set `base_url` in `llm_config.json` to `http://host.docker.internal:11434` (default Ollama port). On Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]` under `worker_plan` if that hostname is missing, or use your bridge IP.
- Ensure required env vars (e.g., `DEFAULT_LLM`) are available via `.env` or your shell before running the command.

## Troubleshooting
- If the pipeline stops immediately with missing module errors, rebuild with `--no-cache` so new files are inside the images.
- If you change environment variables (e.g., `PLANEXE_WORKER_RELAY_PROCESS_OUTPUT`), restart: `docker compose down` then `docker compose up`.
- If `frontend_multi_user` can't start because host port 5000 is busy, map it elsewhere: `export PLANEXE_FRONTEND_MULTIUSER_PORT=5001` (or another free port) before `docker compose up`.
- To clean out containers, network, and orphans: `docker compose down --remove-orphans`.
- To reclaim disk space when builds start failing with `No space left on device`:
  - See current usage: `docker system df`
  - Aggressively prune (images, caches, networks not in use): `docker system prune -a`
    - Expect a confirmation prompt; this removed ~37 GB here by deleting unused images and build cache.
  - If needed, prune build cache separately: `docker builder prune`

### Port 5432 already in use (Postgres conflict)

If `database_postgres` fails to start with a "port already in use" error, another PostgreSQL is likely running on your machine. This is common on developer machines where you have:
- **macOS**: Postgres.app (a popular menu-bar Postgres), Homebrew PostgreSQL (`brew install postgresql`), or pgAdmin's bundled server
- **Linux**: System PostgreSQL installed via `apt install postgresql` or similar
- **Windows**: PostgreSQL installer, pgAdmin, or other database tools

**Solution**: Set `PLANEXE_POSTGRES_PORT` to a different value:
```bash
export PLANEXE_POSTGRES_PORT=5433
docker compose up
```

This only affects the HOST port (how you access Postgres from your machine). Inside Docker, containers always connect to each other on port 5432—this is hardcoded and unaffected by `PLANEXE_POSTGRES_PORT`.

To make this permanent, add to your `.env` file:
```
PLANEXE_POSTGRES_PORT=5433
```

When connecting from your host machine (e.g., DBeaver, `psql`), use the port you set:
```bash
psql -h localhost -p 5433 -U planexe -d planexe
```

## Environment notes
- The worker exports logs to stdout when `PLANEXE_WORKER_RELAY_PROCESS_OUTPUT=true` (set in `docker-compose.yml`).
- Shared volumes: `./run` is mounted into both services; `.env` and `llm_config.json` are mounted read-only. Ensure they exist on the host before starting.***
- Database: Postgres runs in `database_postgres` and listens on host `${PLANEXE_POSTGRES_PORT:-5432}` mapped to container `5432`; data is persisted in the named volume `database_postgres_data`.
- Multiuser UI: binds to container port `5000`, exposed on host `${PLANEXE_FRONTEND_MULTIUSER_PORT:-5001}`.
- MCP server downloads: set `PLANEXE_MCP_PUBLIC_BASE_URL` so clients receive a reachable `/download/...` URL (defaults to `http://localhost:8001` in compose).

## Host opener (Open Output Dir)
Because Docker containers cannot launch host apps, the `Open Output Dir` button needs a host-side service.

Set these environment variables before starting:
- `PLANEXE_OPEN_DIR_SERVER_URL` so the container can reach the host opener:
  - macOS/Windows (Docker Desktop): `http://host.docker.internal:5100`
  - Linux: `http://172.17.0.1:5100` (or add `host.docker.internal` pointing to the bridge IP).
- `PLANEXE_HOST_RUN_DIR`: optional; defaults to `PlanExe/run` on the host. Set an absolute path if you relocate the run directory.

1) Start host opener **before** Docker (on the host):
```bash
cd open_dir_server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
2) Provide `PLANEXE_OPEN_DIR_SERVER_URL` via your shell env, `.env`, or docker compose environment for `frontend_single_user`.

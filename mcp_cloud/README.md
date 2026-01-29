# PlanExe MCP Cloud - Experimental, likely to be changed a lot!

Model Context Protocol (MCP) interface for PlanExe. Implements the MCP specification defined in `docs/planexe_mcp_interface.md`.

## Overview

mcp_cloud provides a standardized MCP interface for PlanExe's plan generation workflows. It connects to `worker_plan_database` via the shared Postgres database (`database_api` models).

## Features

- **Task Management**: Create and stop plan generation tasks
- **Progress Tracking**: Real-time status and progress updates
- **File Metadata**: Get report/zip metadata and download URLs

## Run as task (MCP tasks protocol)

MCP has two ways to run long-running work: **tools** (what we use) and the **tasks** protocol ("Run as task" in some UIs). PlanExe uses **tools only**: `task_create`, `task_status`, `task_stop`, `task_download`. The agent creates a task, polls status, then downloads; that is the intended flow per `docs/planexe_mcp_interface.md`. We do not advertise or implement the MCP tasks protocol (tasks/get, tasks/result, etc.). Clients like Cursor do not support it properly—use the tools directly.

## Client Choice Guide

- **Use `mcp_cloud` directly (HTTP)**: If you are running in the cloud or you do
  not need files saved to the local filesystem.
- **Use `mcp_local` (proxy)**: Recommended when you want artifacts downloaded to
  your local disk (`PLANEXE_PATH`). The proxy forwards MCP calls to this server
  and handles file downloads locally.
- **Recommended flow**: Docker (`mcp_cloud`) → `mcp_local` → MCP client (LM Studio/Claude).

## Docker Usage (Recommended)

Build and run mcp_cloud with HTTP endpoints:

```bash
docker compose up --build mcp_cloud
```

mcp_cloud exposes HTTP endpoints on port `8001` (or `${PLANEXE_MCP_HTTP_PORT}`). Set `PLANEXE_MCP_API_KEY` in your `.env` file or environment to enable API key authentication.

### Connecting via HTTP/URL

After starting with Docker, configure your MCP client (e.g., LM Studio) to connect via HTTP:

**Local Docker (development):**

```json
{
  "mcpServers": {
    "planexe": {
      "url": "http://localhost:8001/mcp",
      "headers": {
        "X-API-Key": "your-api-key-here"
      }
    }
  }
}
```

**Railway/Cloud deployment:**

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://your-app.up.railway.app/mcp",
      "headers": {
        "X-API-Key": "your-api-key-here"
      }
    }
  }
}
```

**Alternative header format** (also supported):

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://your-app.up.railway.app/mcp",
      "headers": {
        "API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Set `PLANEXE_MCP_API_KEY` to the same value you use in `Authorization: Bearer <key>` or `X-API-Key`.

### Available HTTP Endpoints

- `POST /mcp` - Main MCP JSON-RPC endpoint (recommended)
- `POST /mcp/tools/call` - Alternative HTTP wrapper for tool calls
- `GET /mcp/tools` - List available tools
- `GET /healthcheck` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)

## Environment Variables

### HTTP Server Configuration

- `PLANEXE_MCP_API_KEY`: **Required for production**. API key for authentication. Clients can provide `Authorization: Bearer <key>` or `X-API-Key`.
- `PLANEXE_MCP_HTTP_HOST`: HTTP server host (default: `127.0.0.1`). Use `0.0.0.0` to bind all interfaces (containers/cloud).
- `PLANEXE_MCP_HTTP_PORT`: HTTP server port (default: `8001`). Railway will override with `PORT` env var.
- `PLANEXE_MCP_PUBLIC_BASE_URL`: Public base URL for report download links (default unset; clients can use the connected base URL).
- `PORT`: Railway-provided port (takes precedence over `PLANEXE_MCP_HTTP_PORT`)
- `PLANEXE_MCP_CORS_ORIGINS`: Comma-separated list of allowed origins (default: `http://localhost,http://127.0.0.1`).
- `PLANEXE_MCP_MAX_BODY_BYTES`: Max request size for `POST /mcp/tools/call` (default: `1048576`).
- `PLANEXE_MCP_RATE_LIMIT`: Max requests per window for `POST /mcp/tools/call` (default: `60`).
- `PLANEXE_MCP_RATE_WINDOW_SECONDS`: Rate limit window in seconds (default: `60`).

### Database Configuration

mcp_cloud uses the same database configuration as other PlanExe services:

- `SQLALCHEMY_DATABASE_URI`: Full database connection string (takes precedence)
- `PLANEXE_POSTGRES_HOST`: Database host (default: `database_postgres`)
- `PLANEXE_POSTGRES_PORT`: Database port (default: `5432`)
- `PLANEXE_POSTGRES_DB`: Database name (default: `planexe`)
- `PLANEXE_POSTGRES_USER`: Database user (default: `planexe`)
- `PLANEXE_POSTGRES_PASSWORD`: Database password (default: `planexe`)
- `PLANEXE_WORKER_PLAN_URL`: URL of the worker_plan HTTP service (default: `http://worker_plan:8000`)

## MCP Tools

See `docs/planexe_mcp_interface.md` for full specification. Available tools:

- `prompt_examples` - Return example prompts. Use these as examples for task_create.
- `task_create` - Create a new task
- `task_status` - Get task status and progress
- `task_stop` - Stop an active task
- `task_file_info` - Get file metadata for report or zip

Note: `task_download` is a synthetic tool provided by `mcp_local`, not by this server. If your client exposes `task_download`, use it to save the report or zip locally; otherwise use `task_file_info` to get `download_url` and fetch the file yourself.

**Tip**: Call `prompt_examples` to get example prompts to use with task_create. The catalog is the same as in the frontends (`worker_plan.worker_plan_api.PromptCatalog`). When running with `PYTHONPATH` set to the repo root (e.g. stdio setup), the catalog is loaded automatically; otherwise built-in examples are returned.

Download flow: call `task_file_info` to obtain the `download_url`, then fetch the
report via `GET /download/{task_id}/030-report.html` (API key required if configured).

## Debugging tools

Use the MCP Inspector to verify tool registration and output schemas.

Everything reference (stdio):

```bash
npx @modelcontextprotocol/inspector --transport stdio npx -y @modelcontextprotocol/server-everything
```

Steps:
- Click "Connect"
- Click "Tools"
- Click "List Tools"

PlanExe MCP (HTTP):

```bash
npx @modelcontextprotocol/inspector --transport http --server-url http://localhost:8001/mcp
```

Steps:
- Click "Connect"
- Click "Tools"
- Click "List Tools"

## Architecture

mcp_cloud maps MCP concepts to PlanExe's database models:

- **Task** → `TaskItem` (each task corresponds to a TaskItem)
- **Run** → Execution of a TaskItem by `worker_plan_database`
- **Report** → HTML report fetched from `worker_plan` via HTTP API

mcp_cloud reads task state and progress from the database, and fetches artifacts from `worker_plan` via HTTP instead of accessing the run directory directly. This allows mcp_cloud to work without mounting the run directory, making it compatible with Railway and other cloud platforms that don't support shared volumes across services.

## Connecting via stdio (Advanced / Contributor Mode)

For local development, you can run mcp_cloud over stdio instead of HTTP. This is
useful for testing but requires local Python + Postgres setup. For most users, the
recommended flow is Docker (server) + `mcp_local` (client).

### Setup

1. Install dependencies in a virtual environment:

```bash
cd mcp_cloud
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Ensure the database is accessible. If using Docker for the database:

```bash
# From repo root, ensure database_postgres is running
docker compose up -d database_postgres
```

3. Set environment variables (create a `.env` file in the repo root or export them):

```bash
export PLANEXE_POSTGRES_HOST=localhost
export PLANEXE_POSTGRES_PORT=5432  # Or your mapped port (e.g., 5433 if you set PLANEXE_POSTGRES_PORT)
export PLANEXE_POSTGRES_DB=planexe
export PLANEXE_POSTGRES_USER=planexe
export PLANEXE_POSTGRES_PASSWORD=planexe
```

   **Note**: The `PYTHONPATH` environment variable in the LM Studio config (see below) ensures that the `database_api` module can be imported. Make sure the path points to the PlanExe repository root (where `database_api/` is located).

### LM Studio Configuration

Add the following to your LM Studio MCP servers configuration file:

```json
{
  "mcpServers": {
    "planexe": {
      "command": "/absolute/path/to/PlanExe/mcp_cloud/.venv/bin/python",
      "args": [
        "-m",
        "mcp_cloud.app"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/PlanExe",
        "PLANEXE_POSTGRES_HOST": "localhost",
        "PLANEXE_POSTGRES_PORT": "5432",
        "PLANEXE_POSTGRES_DB": "planexe",
        "PLANEXE_POSTGRES_USER": "planexe",
        "PLANEXE_POSTGRES_PASSWORD": "planexe"
      }
    }
  }
}
```

**Important**: Replace `/absolute/path/to/PlanExe` with the actual absolute path to your PlanExe repository on your system.

**Example** (if PlanExe is at `/absolute/path/to/PlanExe`):

```json
{
  "mcpServers": {
    "planexe": {
      "command": "/absolute/path/to/PlanExe/mcp_cloud/.venv/bin/python",
      "args": [
        "-m",
        "mcp_cloud.app"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/PlanExe",
        "PLANEXE_POSTGRES_HOST": "localhost",
        "PLANEXE_POSTGRES_PORT": "5432",
        "PLANEXE_POSTGRES_DB": "planexe",
        "PLANEXE_POSTGRES_USER": "planexe",
        "PLANEXE_POSTGRES_PASSWORD": "planexe"
      }
    }
  }
}
```

**Using Docker** (more complex, but keeps dependencies isolated):

You can use `docker compose exec` to run mcp_cloud:

```json
{
  "mcpServers": {
    "planexe": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "/absolute/path/to/PlanExe/docker-compose.yml",
        "exec",
        "-T",
        "mcp_cloud",
        "python",
        "-m",
        "mcp_cloud.app"
      ]
    }
  }
}
```

Note: This requires the `mcp_cloud` container to be running (`docker compose up -d mcp_cloud`).

### Troubleshooting

**Connection issues:**
- Ensure the database is running and accessible at the configured host/port
- Check that the `PYTHONPATH` in the LM Studio config points to the PlanExe repository root (containing `database_api/`, `mcp_cloud/`, etc.)
- Verify the Python interpreter path in the `command` field is correct and points to the venv Python

**Import errors:**
- If you see `ModuleNotFoundError: No module named 'database_api'`, check that `PYTHONPATH` is set correctly
- If you see `ModuleNotFoundError: No module named 'mcp'`, ensure you've installed the requirements: `pip install -r requirements.txt`

**Database connection errors:**
- Verify Postgres is running: `docker compose ps database_postgres`
- Check the port mapping: if you set `PLANEXE_POSTGRES_PORT=5433`, use `5433` in your env vars, not `5432`
- Test connection: `psql -h localhost -p 5432 -U planexe -d planexe` (or your port)

**Path issues:**
- Always use absolute paths in LM Studio config, not relative paths
- On Windows, use forward slashes in the config JSON (e.g., `C:/Users/...`) or escaped backslashes

## Development

Run locally for testing:

```bash
cd mcp_cloud
source .venv/bin/activate  # If not already activated
export PYTHONPATH=$PWD/..:$PYTHONPATH
python -m mcp_cloud.app
```

## Railway Deployment

See `railway.md` for Railway-specific deployment instructions. The server automatically detects Railway's `PORT` environment variable and binds to it.

## Notes

- mcp_cloud communicates with `worker_plan_database` indirectly via the database for task management.
- Artifacts are fetched from `worker_plan` via HTTP instead of accessing the run directory directly. This avoids needing a shared volume mount, making it compatible with Railway and other cloud platforms.
- For artifacts:
  - `report.html` is fetched efficiently via the dedicated `/runs/{run_id}/report` endpoint
  - Other files are fetched by downloading the run zip and extracting the file (less efficient but works without additional endpoints)
- Artifact writes are not yet supported via HTTP (would require a write endpoint in `worker_plan`).
- Artifact writes are rejected while a run is active (strict policy per spec).
- Task IDs use the TaskItem UUID (e.g., `5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1`).
- **Security**: Always set `PLANEXE_MCP_API_KEY` in production deployments to prevent unauthorized access.

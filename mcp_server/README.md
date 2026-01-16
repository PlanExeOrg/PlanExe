# PlanExe MCP Server

Model Context Protocol (MCP) interface for PlanExe. Implements the MCP specification defined in `extra/mcp-spec1.md`.

## Overview

This MCP server provides a standardized interface for AI agents and developer tools to interact with PlanExe's plan generation workflows. It communicates with `worker_plan_database` via the shared Postgres database (`database_api` models).

## Features

- **Session Management**: Create, start, stop, and resume plan generation sessions
- **Progress Tracking**: Real-time status and progress updates
- **Artifact Management**: List, read, and write artifacts (plans, reports, etc.)
- **Event Streaming**: Subscribe to execution events

## Docker Usage (Recommended)

Build and run the MCP server with HTTP endpoints:

```bash
docker compose up --build mcp_server
```

The MCP server exposes HTTP endpoints on port `8001` (or `${PLANEXE_MCP_HTTP_PORT}`). Set `PLANEXE_MCP_API_KEY` in your `.env` file or environment to enable API key authentication.

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

Set `PLANEXE_MCP_API_KEY` to the same value you use in `Authorization: Bearer <key>`, `X-API-Key`, or `?api_key=`.

### Available HTTP Endpoints

- `POST /mcp` - Main MCP endpoint (recommended)
- `POST /mcp/tools/call` - Alternative endpoint for tool calls
- `GET /mcp/tools` - List available tools
- `GET /healthcheck` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)

## Environment Variables

### HTTP Server Configuration

- `PLANEXE_MCP_API_KEY`: **Required for production**. API key for authentication. Clients can provide `Authorization: Bearer <key>`, `X-API-Key`, or `?api_key=`.
- `PLANEXE_MCP_HTTP_HOST`: HTTP server host (default: `127.0.0.1`). Use `0.0.0.0` to bind all interfaces (containers/cloud).
- `PLANEXE_MCP_HTTP_PORT`: HTTP server port (default: `8001`). Railway will override with `PORT` env var.
- `PORT`: Railway-provided port (takes precedence over `PLANEXE_MCP_HTTP_PORT`)
- `PLANEXE_MCP_CORS_ORIGINS`: Comma-separated list of allowed origins (default: `http://localhost,http://127.0.0.1`).
- `PLANEXE_MCP_MAX_BODY_BYTES`: Max request size for `POST /mcp/tools/call` (default: `1048576`).
- `PLANEXE_MCP_RATE_LIMIT`: Max requests per window for `POST /mcp/tools/call` (default: `60`).
- `PLANEXE_MCP_RATE_WINDOW_SECONDS`: Rate limit window in seconds (default: `60`).

### Database Configuration

The MCP server uses the same database configuration as other PlanExe services:

- `SQLALCHEMY_DATABASE_URI`: Full database connection string (takes precedence)
- `PLANEXE_POSTGRES_HOST`: Database host (default: `database_postgres`)
- `PLANEXE_POSTGRES_PORT`: Database port (default: `5432`)
- `PLANEXE_POSTGRES_DB`: Database name (default: `planexe`)
- `PLANEXE_POSTGRES_USER`: Database user (default: `planexe`)
- `PLANEXE_POSTGRES_PASSWORD`: Database password (default: `planexe`)
- `PLANEXE_WORKER_PLAN_URL`: URL of the worker_plan HTTP service (default: `http://worker_plan:8000`)

## MCP Tools

See `extra/mcp-spec1.md` for full specification. Available tools:

- `planexe.session.create` - Create a new session
- `planexe.session.start` - Start execution
- `planexe.session.status` - Get run status and progress
- `planexe.session.stop` - Stop active run
- `planexe.session.resume` - Resume execution
- `planexe.artifact.list` - List artifacts
- `planexe.artifact.read` - Read an artifact
- `planexe.report.read` - Read the generated report (FilenameEnum.REPORT), chunked by default
- `planexe.artifact.write` - Write/edit an artifact
- `planexe.session.events` - Get incremental events

## Architecture

The MCP server maps MCP concepts to PlanExe's database models:

- **Session** → `TaskItem` (each session corresponds to a TaskItem)
- **Run** → Execution of a TaskItem by `worker_plan_database`
- **Artifacts** → Files fetched from `worker_plan` via HTTP API

The server reads task state and progress from the database, and fetches artifacts from `worker_plan` via HTTP instead of accessing the run directory directly. This allows the MCP server to work without mounting the run directory, making it compatible with Railway and other cloud platforms that don't support shared volumes across services.

## Connecting via stdio (Local Development)

For local development, you can run the MCP server over stdio instead of HTTP. This is useful for testing but requires local Python setup.

### Setup

1. Install dependencies in a virtual environment:

```bash
cd mcp_server
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
export PLANEXE_RUN_DIR=./run
```

   **Note**: The `PYTHONPATH` environment variable in the LM Studio config (see below) ensures that the `database_api` module can be imported. Make sure the path points to the PlanExe repository root (where `database_api/` is located).

### LM Studio Configuration

Add the following to your LM Studio MCP servers configuration file:

```json
{
  "mcpServers": {
    "planexe": {
      "command": "/absolute/path/to/PlanExe/mcp_server/.venv/bin/python",
      "args": [
        "-m",
        "mcp_server.app"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/PlanExe",
        "PLANEXE_POSTGRES_HOST": "localhost",
        "PLANEXE_POSTGRES_PORT": "5432",
        "PLANEXE_POSTGRES_DB": "planexe",
        "PLANEXE_POSTGRES_USER": "planexe",
        "PLANEXE_POSTGRES_PASSWORD": "planexe",
        "PLANEXE_RUN_DIR": "/absolute/path/to/PlanExe/run"
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
      "command": "/absolute/path/to/PlanExe/mcp_server/.venv/bin/python",
      "args": [
        "-m",
        "mcp_server.app"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/PlanExe",
        "PLANEXE_POSTGRES_HOST": "localhost",
        "PLANEXE_POSTGRES_PORT": "5432",
        "PLANEXE_POSTGRES_DB": "planexe",
        "PLANEXE_POSTGRES_USER": "planexe",
        "PLANEXE_POSTGRES_PASSWORD": "planexe",
        "PLANEXE_RUN_DIR": "/absolute/path/to/PlanExe/run"
      }
    }
  }
}
```

**Using Docker** (more complex, but keeps dependencies isolated):

You can use `docker compose exec` to run the MCP server:

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
        "mcp_server",
        "python",
        "-m",
        "mcp_server.app"
      ]
    }
  }
}
```

Note: This requires the `mcp_server` container to be running (`docker compose up -d mcp_server`).

### Troubleshooting

**Connection issues:**
- Ensure the database is running and accessible at the configured host/port
- Check that the `PYTHONPATH` in the LM Studio config points to the PlanExe repository root (containing `database_api/`, `mcp_server/`, etc.)
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
cd mcp_server
source .venv/bin/activate  # If not already activated
export PYTHONPATH=$PWD/..:$PYTHONPATH
python -m mcp_server.app
```

## Railway Deployment

See `railway.md` for Railway-specific deployment instructions. The server automatically detects Railway's `PORT` environment variable and binds to it.

## Notes

- The MCP server communicates with `worker_plan_database` indirectly via the database for session/task management.
- Artifacts are fetched from `worker_plan` via HTTP instead of accessing the run directory directly. This avoids needing a shared volume mount, making it compatible with Railway and other cloud platforms.
- For artifacts:
  - `report.html` is fetched efficiently via the dedicated `/runs/{run_id}/report` endpoint
  - Other files are fetched by downloading the run zip and extracting the file (less efficient but works without additional endpoints)
- Artifact writes are not yet supported via HTTP (would require a write endpoint in `worker_plan`).
- Artifact writes are rejected while a run is active (strict policy per spec).
- Session IDs use the format `pxe_{YYYY_MM_DD}__{short_uuid}` for compatibility.
- **Security**: Always set `PLANEXE_MCP_API_KEY` in production deployments to prevent unauthorized access.

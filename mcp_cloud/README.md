# PlanExe MCP Cloud

Model Context Protocol (MCP) interface for PlanExe. Implements the MCP specification defined in [docs/mcp/planexe_mcp_interface.md](https://docs.planexe.org/mcp/planexe_mcp_interface/).

## Overview

mcp_cloud provides a standardized MCP interface for PlanExe's plan generation workflows. It connects to `worker_plan_database_{n}` via the shared Postgres database (`database_api` models).

## Features

- **Task Management**: Create and stop plan generation tasks
- **Progress Tracking**: Real-time status and progress updates via polling or SSE
- **Real-Time SSE Streaming**: Subscribe to live plan progress instead of polling
- **File Metadata**: Get report/zip metadata and signed download URLs
- **Signed Download Tokens**: Time-limited, artifact-scoped HMAC tokens for secure file delivery

## Run as task (MCP tasks protocol)

MCP has two ways to run long-running work: **tools** (what we use) and the **tasks** protocol ("Run as task" in some UIs). PlanExe uses **tools only**: `example_plans`, `example_prompts`, `model_profiles`, `plan_create`, `plan_status`, `plan_stop`, `plan_retry`, `plan_file_info` (or `plan_download` via `mcp_local`). The agent creates a task, polls status, retries on failed when needed, then downloads; that is the intended flow per `docs/mcp/planexe_mcp_interface.md`. We do not advertise or implement the MCP tasks protocol (tasks/get, tasks/result, etc.). Clients like Cursor do not support it properly—use the tools directly.
Workflow clarity: prompt drafting + user approval is a non-tool step between setup tools and `plan_create`.

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
docker compose up
```

Important: `mcp_cloud` enqueues plans and `worker_plan_database_{n}` executes them.
If no `worker_plan_database*` service is running, `plan_create` returns a plan id but the plan will not progress.

mcp_cloud exposes HTTP endpoints on port `8001` (or `${PLANEXE_MCP_HTTP_PORT}`). Authentication is controlled by `PLANEXE_MCP_REQUIRE_AUTH`:
- `false`: no API key needed (local docker default).
- `true`: provide a valid `X-API-Key`.
Accepted keys are (1) UserApiKey from home.planexe.org (`pex_...`), or (2) `PLANEXE_MCP_API_KEY` if set (for dev or shared secret).
OAuth is not supported for the MCP API.
When auth is enabled, MCP handshake/discovery calls (`initialize`, `notifications/initialized`, `tools/list`, `prompts/list`, `resources/list`, `resources/templates/list`, `ping`, `GET /mcp/tools`, and probe traffic to `/mcp` for redirect/handshake compatibility) are intentionally allowed without API key for connector health checks. In addition, `tools/call` is open without API key only for `example_plans`, `example_prompts`, and `model_profiles`; all other tool calls remain protected.

### Connecting via HTTP/URL

After starting with Docker, configure your MCP client (e.g., LM Studio) to connect via HTTP:

**Remote MCP:**

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://mcp.planexe.org/mcp",
      "headers": {
        "X-API-Key": "your-api-key-here"
      }
    }
  }
}
```

Use a UserApiKey from [home.planexe.org](https://home.planexe.org/), or set `PLANEXE_MCP_API_KEY` to a shared secret for local/dev use.

**Running MCP in docker on localhost:**

```json
{
  "mcpServers": {
    "planexe": {
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

### Available HTTP Endpoints

- `POST /mcp` - Main MCP JSON-RPC endpoint (Streamable HTTP; may use SSE for streaming)
- `GET /mcp/tools` - **List tools (JSON). No SSE required.** Use this if your client reports "SSE error" when connecting to `/mcp`.
- `POST /mcp/tools/call` - **Call a tool (JSON). No SSE required.**
- `GET /sse/plan/{plan_id}` - **SSE endpoint for real-time plan progress** (see [SSE section](#real-time-progress-via-sse) below)
- `GET /download/{plan_id}/{filename}?token=...` - Serve report or zip artifact (signed token or API key required)
- `GET /healthcheck` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)
- `GET /robots.txt` - Crawler rules for public metadata discovery

### Discovery / `.well-known` Endpoints

The `/.well-known/` prefix is an [IETF standard (RFC 8615)](https://www.rfc-editor.org/rfc/rfc8615) for machine-readable metadata. Automated systems (registries, crawlers, AI agents) fetch these to discover what the server offers without performing a full handshake.

- **`GET /.well-known/mcp/server-card.json`** — MCP Server Card ([SEP-1649](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1649)). Lets MCP registries (Smithery, etc.) discover the server's name, description, transport type, capabilities, and auth requirements in a single JSON fetch — no MCP handshake needed.
- **`GET /.well-known/glama.json`** — Glama ownership verification. When registering at [glama.ai](https://glama.ai), their crawler fetches this to confirm the server maintainer (contains a maintainer email).

### "SSE error" or "no Server-SSE stream" from the client

Some MCP clients (e.g. OpenClaw/mcporter) connect by doing a **GET** to the server URL and expect a **Server-Sent Events (SSE)** stream (`Content-Type: text/event-stream`). That is the **Streamable HTTP** transport. This server mounts FastMCP at `/mcp`; **GET /mcp** returns a **307 redirect** to `/mcp/`, and the Streamable HTTP handshake may not match what the client expects, so the client reports "SSE error" or "could not fetch … no SSE stream".

**You do not need SSE for tools.** MCP over HTTP can use plain JSON:

- **List tools:** `GET http://<host>:8001/mcp/tools` → returns `{"tools": [...]}` (JSON).
- **Call a tool:** `POST http://<host>:8001/mcp/tools/call` with body `{"tool": "plan_create", "arguments": {"prompt": "…"}}` → returns JSON.

If your client only supports Streamable HTTP and fails on `/mcp`, you have two options:

1. **Point the client at the JSON API** if it allows a separate "tools list" URL: use `GET /mcp/tools` for listing and `POST /mcp/tools/call` for calls (no SSE).
2. **Use baseUrl with trailing slash** (e.g. `http://192.168.1.10:8001/mcp/`) so the client does not follow a redirect; whether that fixes SSE depends on how the client and FastMCP do the Streamable HTTP handshake.

## Environment Variables

### HTTP Server Configuration

- `PLANEXE_MCP_REQUIRE_AUTH`: Require API keys for `/mcp` and `/download` (server default: `true`; `docker-compose.yml` overrides to `false` for local docker convenience).
- `PLANEXE_MCP_API_KEY`: Optional shared secret for auth. When auth is enabled, clients can use this key instead of a UserApiKey. For production with user accounts, keys from home.planexe.org (UserApiKey) are validated against the database.
- `PLANEXE_MCP_HTTP_HOST`: HTTP server host (default: `127.0.0.1`). Use `0.0.0.0` to bind all interfaces (containers/cloud).
- `PLANEXE_MCP_HTTP_PORT`: HTTP server port (default: `8001`). Railway will override with `PORT` env var.
- `PLANEXE_MCP_PUBLIC_BASE_URL`: Public base URL for report/zip download links in `plan_file_info` (e.g. `http://192.168.1.40:8001`). When set, `download_url` is built from this value. When unset, the HTTP server uses the request’s host (scheme + authority), so clients connecting at `http://192.168.1.40:8001/mcp/` get download URLs like `http://192.168.1.40:8001/download/...` instead of localhost. If clients still see localhost in download URLs (e.g. behind a proxy), set this env var explicitly in `.env`.
- `PORT`: Railway-provided port (takes precedence over `PLANEXE_MCP_HTTP_PORT`)
- `PLANEXE_MCP_CORS_ORIGINS`: Comma-separated list of allowed origins. When unset, uses `*` (all origins) so browser-based tools like the MCP Inspector can connect. If you set it (e.g. for a specific frontend), include `http://localhost:6274` and `http://127.0.0.1:6274` for the Inspector.
- `PLANEXE_MCP_MAX_BODY_BYTES`: Max request size for `POST /mcp/tools/call` (default: `1048576`).
- `PLANEXE_MCP_RATE_LIMIT`: Max requests per window for `POST /mcp/tools/call` (default: `60`).
- `PLANEXE_MCP_RATE_WINDOW_SECONDS`: Rate limit window in seconds (default: `60`).
- `PLANEXE_DOWNLOAD_TOKEN_SECRET`: HMAC secret for signed download tokens. Falls back to `PLANEXE_API_KEY_SECRET`, then a per-process random secret (development only; tokens invalidated on restart).
- `PLANEXE_DOWNLOAD_TOKEN_TTL`: Token time-to-live in seconds (default: `900`, i.e. 15 minutes).

### SSE Configuration

- `PLANEXE_SSE_POLL_INTERVAL`: How often the SSE stream polls the database for changes in seconds (default: `3.0`).
- `PLANEXE_SSE_HEARTBEAT_INTERVAL`: Seconds of silence before a heartbeat event is sent (default: `20.0`).
- `PLANEXE_SSE_MAX_DURATION`: Maximum SSE stream lifetime in seconds (default: `3600`, i.e. 1 hour).
- `PLANEXE_SSE_MAX_CONNECTIONS`: Maximum concurrent SSE connections per client IP (default: `5`).
- `PLANEXE_SSE_MAX_TOTAL_CONNECTIONS`: Maximum concurrent SSE connections server-wide (default: `200`).

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

See `docs/mcp/planexe_mcp_interface.md` for full specification. Available tools:

- `example_plans` - Return curated example plans with download links for reports and zip bundles.
- `example_prompts` - Return example prompts. Use these as examples for plan_create.
- `model_profiles` - List profile options and currently available models in each profile.
- `plan_create` - Create a new plan (returns plan_id as UUID; may require user_api_key for credits)
- `plan_status` - Get plan status and progress
- `plan_stop` - Stop an active plan
- `plan_retry` - Retry a failed plan with the same plan_id (optional model_profile, default baseline)
- `plan_file_info` - Get file metadata for report or zip

`plan_status` caller contract:
- `pending` / `processing`: keep polling.
- `completed`: terminal success, download is ready.
- `failed`: terminal error.
- If `failed`, call `plan_retry` to requeue the same plan id.

Concurrency semantics:
- Each `plan_create` call creates a new `plan_id`.
- `plan_retry` reuses the same failed `plan_id`.
- Server does not enforce a global one-plan-at-a-time cap per client.
- Client should track plan ids explicitly when running plans in parallel.

Minimal error contract:
- Tool errors use `{"error":{"code","message","details?"}}`.
- Common codes: `PLAN_NOT_FOUND`, `PLAN_NOT_FAILED`, `INVALID_USER_API_KEY`, `USER_API_KEY_REQUIRED`, `INSUFFICIENT_CREDITS`, `INTERNAL_ERROR`, `generation_failed`, `content_unavailable`.
- `plan_file_info` may return `{}` while output is not ready (not an error payload).

Note: `plan_download` is a synthetic tool provided by `mcp_local`, not by this server. If your client exposes `plan_download`, use it to save the report or zip locally; otherwise use `plan_file_info` to get `download_url` and fetch the file yourself.

> **Breaking change (v2026-02-26):** External-facing field names were renamed from `task_id` → `plan_id`, `tasks` → `plans`, and error codes from `TASK_NOT_FOUND` → `PLAN_NOT_FOUND`, `TASK_NOT_FAILED` → `PLAN_NOT_FAILED`.

**Tip**: Call `example_plans` to preview example output, then call `example_prompts` to get example prompts to use with plan_create, then call `model_profiles` to choose `model_profile` based on current runtime availability. The prompt catalog is the same as in the frontends (`worker_plan.worker_plan_api.PromptCatalog`). When running with `PYTHONPATH` set to the repo root (e.g. stdio setup), the catalog is loaded automatically; otherwise built-in examples are returned.

Download flow: call `plan_file_info` to obtain the `download_url`, then fetch the
report via `GET /download/{plan_id}/030-report.html` (API key required if configured).
If `download_url` is missing, configure `PLANEXE_MCP_PUBLIC_BASE_URL` so the server can emit a reachable absolute URL.

## Real-Time Progress via SSE

Plans take 10-20 minutes to generate. Instead of polling `plan_status` every few minutes, clients can subscribe to a Server-Sent Events (SSE) stream for real-time push updates.

**SSE is optional.** The `plan_status` polling workflow remains fully supported. SSE is a complementary REST endpoint on the FastAPI server, not an MCP transport replacement.

### How to connect

`plan_create` and `plan_status` responses include an `sse_url` field when available:

```json
{
  "plan_id": "abc-123",
  "created_at": "2026-03-01T14:00:00Z",
  "sse_url": "https://mcp.planexe.org/sse/plan/abc-123"
}
```

No API key is required — the plan_id UUID is unguessable and serves as the access token:

```bash
curl -N https://mcp.planexe.org/sse/plan/abc-123
```

### Event types

The stream emits standard SSE events:

```
retry: 5000

event: status
data: {"plan_id":"abc-123","state":"processing","progress_percentage":45.0,"elapsed_sec":120.3}

event: heartbeat
data: {"timestamp":"2026-03-01T14:32:00Z"}

event: complete
data: {"plan_id":"abc-123","state":"completed","progress_percentage":100.0,"elapsed_sec":720.5}
```

| Event | When | Action |
|-------|------|--------|
| `status` | State or progress changes | Update UI / log progress |
| `heartbeat` | Every ~20s of silence | Keeps proxies/load balancers alive; ignore in application logic |
| `complete` | Terminal state (`completed` or `failed`) | Stream closes automatically; proceed to `plan_file_info` |
| `error` | Plan not found or timeout | Stream closes; check error code |

### Behavior

- The stream polls the database every ~3 seconds but only sends `status` events when state or progress actually changes (deduplication).
- A `retry: 5000` hint tells the client to reconnect after 5 seconds if the connection drops.
- The stream closes automatically on terminal states or after 60 minutes (configurable via `PLANEXE_SSE_MAX_DURATION`).
- Connection limits: 5 per client IP, 200 server-wide. Exceeding returns HTTP 429.

### Verifying SSE works

The SSE endpoint is a plain REST endpoint — test it with `curl`, not the MCP Inspector. The MCP Inspector's "SSE" transport option is for the MCP protocol transport, which is unrelated to plan progress SSE.

**Quick smoke test** (no worker or real plan needed):

```bash
curl -N http://localhost:8001/sse/plan/00000000-0000-0000-0000-000000000000
```

Expected output — an error event followed by the stream closing:

```
retry: 5000

event: error
data: {"code":"PLAN_NOT_FOUND","message":"Plan not found: 00000000-0000-0000-0000-000000000000"}
```

**End-to-end test** (requires a running worker):

1. Create a plan:

```bash
curl -s -X POST http://localhost:8001/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "plan_create", "arguments": {"prompt": "Open a small coffee shop in Copenhagen"}}' \
  | python3 -m json.tool
```

2. Check that the response includes `sse_url`:

```json
{
  "plan_id": "abc-123",
  "created_at": "2026-03-01T14:00:00Z",
  "sse_url": "http://localhost:8001/sse/plan/abc-123"
}
```

3. Connect to the SSE stream (use the `plan_id` from step 1):

```bash
curl -N http://localhost:8001/sse/plan/abc-123
```

4. You should see events streaming in as the plan progresses:

```
retry: 5000

event: status
data: {"plan_id":"abc-123","state":"pending","progress_percentage":0.0,"elapsed_sec":5.2}

event: status
data: {"plan_id":"abc-123","state":"processing","progress_percentage":12.0,"elapsed_sec":30.1}

event: complete
data: {"plan_id":"abc-123","state":"completed","progress_percentage":100.0,"elapsed_sec":720.5}
```

If auth is enabled, add `-H "X-API-Key: pex_..."` to both curl commands.

**Verifying via the MCP Inspector**: Use **Streamable HTTP** transport (not SSE) pointed at `http://localhost:8001/mcp/`. Call `plan_create` — the response JSON should include the `sse_url` field. Then use `curl -N` on that URL in a separate terminal.

## Debugging with the MCP Inspector

Use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) to verify tool registration, authentication, and output schemas.

> **Trailing slash required.** The server mounts at `/mcp` which redirects to `/mcp/`.
> Always use `/mcp/` (with trailing slash) in the inspector URL to avoid a 307 redirect
> that crashes `node-fetch` in older inspector versions.

### Local (no authentication)

```bash
npx @modelcontextprotocol/inspector --transport http --server-url http://localhost:8001/mcp/
```

Steps:
- Click "Connect"
- Click "Tools"
- Click "List Tools"

### Production (with API key authentication)

When auth is enabled, the inspector must send the key with every request.
Do not use the inspector OAuth flow for PlanExe MCP.

```bash
npx @modelcontextprotocol/inspector --transport http --server-url https://mcp.planexe.org/mcp/
```

Steps:
1. In the inspector UI, expand **"Authentication"** in the left sidebar
2. Select **Custom Headers**
3. Add header **X-API-Key** with your API key value (e.g. `pex_...`)
4. Click **"Connect"**
5. Click **"Tools"** then **"List Tools"** to verify

The inspector forwards this custom header to the remote server.

**CORS errors:** If you see "CORS preflight response did not succeed" or "status
code: 400" in the browser console when connecting to a deployed MCP server:
1. Redeploy mcp_cloud with the latest changes (OPTIONS preflight is exempt from
   API key, explicit OPTIONS handler, permissive CORS headers).
2. Ensure `PLANEXE_MCP_CORS_ORIGINS` on the deployed server either is unset
   (allows all origins) or includes `http://localhost:6274` and
   `http://127.0.0.1:6274`.
3. If the error persists, the 400 may come from a **proxy or CDN** (Railway,
   Cloudflare, nginx). Ensure OPTIONS requests are forwarded to the app and not
   blocked. Some platforms require explicit CORS or OPTIONS configuration.

### Skipping proxy authentication (development only)

The inspector proxy itself also requires a session token. To disable that during
local development:

```bash
DANGEROUSLY_OMIT_AUTH=true npx @modelcontextprotocol/inspector --transport http --server-url https://mcp.planexe.org/mcp/
```

This only disables the local inspector-proxy token check. The remote server still
still requires API key authentication when `PLANEXE_MCP_REQUIRE_AUTH=true` (UserApiKey or PLANEXE_MCP_API_KEY).

### Everything reference (stdio)

Sanity-check the inspector itself against the reference server:

```bash
npx @modelcontextprotocol/inspector --transport stdio npx -y @modelcontextprotocol/server-everything
```

Steps:
- Click "Connect"
- Click "Tools"
- Click "List Tools"

## Architecture

mcp_cloud maps MCP concepts to PlanExe's database models:

- **Task** → `PlanItem` (each task corresponds to a PlanItem)
- **Run** → Execution of a PlanItem by `worker_plan_database`
- **Report** → HTML report fetched from `worker_plan` via HTTP API

mcp_cloud reads task state and progress from the database, and fetches artifacts from `worker_plan` via HTTP instead of accessing the run directory directly. This allows mcp_cloud to work without mounting the run directory, making it compatible with Railway and other cloud platforms that don't support shared volumes across services.

### Module overview

| Module | Purpose |
|--------|---------|
| `app.py` | Import facade — re-exports symbols from all sub-modules for backward compatibility |
| `http_server.py` | FastAPI HTTP wrapper: routes, middleware (auth, rate limiting, CORS, body size), FastMCP integration, SSE endpoint |
| `handlers.py` | MCP tool request handlers (plan_create, plan_status, etc.) dispatched via `TOOL_HANDLERS` dict |
| `db_setup.py` | Flask app + SQLAlchemy initialization, server constants, request DTOs, `PLANEXE_SERVER_INSTRUCTIONS` |
| `db_queries.py` | Sync database queries designed to run in `asyncio.to_thread()` from async handlers |
| `tool_models.py` | Pydantic models for tool input/output schemas |
| `schemas.py` | Auto-generates JSON Schema from `tool_models.py` via `.model_json_schema()`; defines `TOOL_DEFINITIONS` |
| `auth.py` | API key hashing (SHA-256) and user resolution against the database |
| `download_tokens.py` | Signed, time-limited HMAC-SHA256 download tokens and URL builders |
| `sse.py` | Server-Sent Events stream generator with connection tracking and deduplication |
| `worker_fetchers.py` | Fetch artifacts from worker_plan HTTP, local filesystem, or database (with fallback ordering) |
| `model_profiles.py` | Load and expose available LLM models per profile (baseline/premium/frontier/custom) |
| `example_prompts.py` | Load example prompts from the prompt catalog (or built-in fallbacks) |
| `zip_utils.py` | Zip extraction, sanitization (remove `track_activity.jsonl` from legacy zips), SHA-256 |
| `http_utils.py` | Strip redundant `content` field when `structuredContent` exists |
| `dotenv_utils.py` | Load `.env` from mcp_cloud/ or repo root |
| `config.py` | Minimal Flask config (`SQLALCHEMY_TRACK_MODIFICATIONS = False`) |

### Request flow

```
Client → FastAPI (http_server.py)
         ├─ Middleware: CORS → path normalization → auth → body size → rate limit
         ├─ /mcp/*  → FastMCP (Streamable HTTP / JSON-RPC)
         │           → handlers.py → db_queries.py (via asyncio.to_thread)
         ├─ /mcp/tools/call → REST tool dispatch → handlers.py
         ├─ /sse/plan/{id} → sse.py (streaming response)
         ├─ /download/{id}/{file} → worker_fetchers.py (artifact delivery)
         └─ /, /healthcheck, /.well-known/* → static info
```

### Artifact resolution order

When fetching plan files (reports, zips), `worker_fetchers.py` tries sources in priority order:

1. **DB zip snapshot** (`PlanItem.run_zip_snapshot`) — fastest, always available for completed plans
2. **Local run directory** (`PLANEXE_RUN_DIR/{run_id}/`) — low latency if co-located
3. **Worker HTTP** (`PLANEXE_WORKER_PLAN_URL/runs/{run_id}/...`) — remote fallback (30s timeout)

This layered approach avoids blocking on the worker HTTP API when the data is already available locally or in the database.

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
- Plan IDs use the PlanItem UUID (e.g., `5e2b2a7c-8b49-4d2f-9b8f-6a3c1f05b9a1`).
- **Security**: Authentication is configurable. For production, set `PLANEXE_MCP_REQUIRE_AUTH=true` and use UserApiKey validation (optionally with `PLANEXE_MCP_API_KEY` as a shared secret).

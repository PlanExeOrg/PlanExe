# Railway Configuration for `mcp_cloud`

Deploy mcp_cloud (PlanExe MCP cloud service) to Railway as an HTTP service.

## Required Environment Variables

Set these in your Railway project:

```
PLANEXE_MCP_API_KEY="your-secret-api-key-here"
SQLALCHEMY_DATABASE_URI="postgresql://user:password@host:5432/dbname"
PLANEXE_WORKER_PLAN_URL="http://your-worker-plan-service:8000"
```

Or, if not using `SQLALCHEMY_DATABASE_URI`, configure Postgres connection separately:

```
PLANEXE_POSTGRES_HOST="your-postgres-host"
PLANEXE_POSTGRES_PORT="5432"
PLANEXE_POSTGRES_DB="planexe"
PLANEXE_POSTGRES_USER="planexe"
PLANEXE_POSTGRES_PASSWORD="your-password"
PLANEXE_WORKER_PLAN_URL="http://your-worker-plan-service:8000"
```

## Optional Environment Variables

```
PLANEXE_MCP_HTTP_HOST="0.0.0.0"  # Use to bind all interfaces (default is 127.0.0.1)
PLANEXE_MCP_HTTP_PORT="8001"      # Default (Railway will override with PORT env var)
PLANEXE_WORKER_PLAN_URL="http://your-worker-plan-service:8000"  # URL of worker_plan service
PLANEXE_MCP_CORS_ORIGINS="https://your-frontend.example.com"
PLANEXE_MCP_MAX_BODY_BYTES="1048576"
PLANEXE_MCP_RATE_LIMIT="60"
PLANEXE_MCP_RATE_WINDOW_SECONDS="60"
```

## Railway-Specific Notes

- Railway automatically provides a `PORT` environment variable. The server will use it if set, otherwise defaults to `8001`.
- Set `PLANEXE_MCP_API_KEY` to enable API key authentication. Clients can send it via `Authorization: Bearer` or `X-API-Key`.
- The server exposes an HTTP endpoint at `/mcp` (or `/mcp/tools/call`) for tool invocations.
- Use Railway's Postgres addon or connect to an external Postgres database via `SQLALCHEMY_DATABASE_URI`.
- Set `PLANEXE_WORKER_PLAN_URL` to point to your `worker_plan` service. Artifacts are fetched from `worker_plan` via HTTP instead of using a shared volume mount.

## Client Configuration

After deployment, configure your MCP client (e.g., LM Studio) with:

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://your-railway-app.up.railway.app/mcp",
      "headers": {
        "X-API-Key": "your-secret-api-key-here"
      }
    }
  }
}
```

Replace `https://your-railway-app.up.railway.app` with your Railway deployment URL.

## Health Check

The service exposes a health check endpoint at `/healthcheck` that Railway can use for monitoring.

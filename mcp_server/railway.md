# Railway Configuration for `mcp_server`

Deploy the PlanExe MCP Server to Railway as an HTTP service.

## Required Environment Variables

Set these in your Railway project:

```
PLANEXE_MCP_API_KEY="your-secret-api-key-here"
SQLALCHEMY_DATABASE_URI="postgresql://user:password@host:5432/dbname"
PLANEXE_RUN_DIR="/app/run"
```

Or, if not using `SQLALCHEMY_DATABASE_URI`, configure Postgres connection separately:

```
PLANEXE_POSTGRES_HOST="your-postgres-host"
PLANEXE_POSTGRES_PORT="5432"
PLANEXE_POSTGRES_DB="planexe"
PLANEXE_POSTGRES_USER="planexe"
PLANEXE_POSTGRES_PASSWORD="your-password"
PLANEXE_RUN_DIR="/app/run"
```

## Optional Environment Variables

```
PLANEXE_MCP_HTTP_HOST="0.0.0.0"  # Default
PLANEXE_MCP_HTTP_PORT="8001"      # Default (Railway will override with PORT env var)
```

## Railway-Specific Notes

- Railway automatically provides a `PORT` environment variable. The server will use it if set, otherwise defaults to `8001`.
- Set `PLANEXE_MCP_API_KEY` to enable API key authentication. Clients must provide this key in the `X-API-Key` header.
- The server exposes an HTTP endpoint at `/mcp/tools/call` for tool invocations.
- Use Railway's Postgres addon or connect to an external Postgres database via `SQLALCHEMY_DATABASE_URI`.

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

# Railway Configuration for `mcp_cloud`

Deploy mcp_cloud (PlanExe MCP cloud service) to Railway as an HTTP service.

```
PLANEXE_POSTGRES_PASSWORD="${{shared.PLANEXE_POSTGRES_PASSWORD}}"
PLANEXE_MCP_HTTP_HOST="0.0.0.0"
```

## Required Environment Variables

`SQLALCHEMY_DATABASE_URI` is the full Postgres connection string. Use Railway's
variable reference (e.g. `${{Postgres.DATABASE_URL}}`) to point at your Railway
Postgres addon. If this is not set, the code falls back to
`PLANEXE_POSTGRES_*` env vars with a default host of `database_postgres` --
a Docker Compose service name that **does not resolve on Railway** and will
cause connection failures.

## Optional Environment Variables

```
PLANEXE_MCP_CORS_ORIGINS="https://your-frontend.example.com"
PLANEXE_MCP_MAX_BODY_BYTES="1048576"
PLANEXE_MCP_RATE_LIMIT="60"
PLANEXE_MCP_RATE_WINDOW_SECONDS="60"
```

Or, if not using `SQLALCHEMY_DATABASE_URI`, configure Postgres individually
(all have defaults, but the defaults target Docker Compose, not Railway):

```
PLANEXE_POSTGRES_HOST="your-postgres-host.railway.internal"
PLANEXE_POSTGRES_PORT="5432"
PLANEXE_POSTGRES_DB="planexe"
PLANEXE_POSTGRES_USER="planexe"
PLANEXE_POSTGRES_PASSWORD="your-password"
```

## Railway `PORT` variable -- do not override

Railway **automatically injects** a `PORT` environment variable (e.g. `8080`)
and routes incoming HTTPS traffic to that port. The server **must** listen on
this port.

The port priority in `http_server.py` is:

```
PORT  >  PLANEXE_MCP_HTTP_PORT  >  8001 (default)
```

On Railway, `PORT` always wins. This means:

- **Do not** set `PLANEXE_MCP_HTTP_PORT` in Railway -- it is ignored because
  Railway's `PORT` takes priority, and setting it only adds confusion.
- **Do not** set `PORT` manually -- Railway manages it.
- **Do not** remove the `PORT` check from the code -- it is required for
  Railway (and other PaaS platforms like Heroku, Render, Google Cloud Run).
- `PLANEXE_MCP_HTTP_PORT` is still useful for **local development** and
  **Docker Compose** where Railway's `PORT` is absent.

| Env var               | Railway           | Docker Compose / Local |
|-----------------------|-------------------|------------------------|
| `PORT` (auto-injected)| Used (e.g. 8080) | Not set                |
| `PLANEXE_MCP_HTTP_PORT`| Ignored          | Used (default: 8001)   |

Similarly, `PLANEXE_MCP_HTTP_HOST` defaults to `127.0.0.1` in the code, but
Railway requires `0.0.0.0` to accept external traffic. Set it in Railway:

```
PLANEXE_MCP_HTTP_HOST="0.0.0.0"
```

## Troubleshooting: 502 errors

If you get a 502 from Railway but `/healthcheck` returns 200:

1. **Check the database connection.** The `/healthcheck` endpoint does not touch
   the database, so it will pass even if Postgres is unreachable. Look for
   `SQLALCHEMY_DATABASE_URI not set. Using Postgres defaults` in the deploy log
   -- if you see `host: 'database_postgres'`, the DB host is wrong for Railway.
   Set `SQLALCHEMY_DATABASE_URI` to fix this.

2. **Check the port.** Verify the deploy log shows the same port Railway
   injected. Look for `Starting PlanExe MCP HTTP server on 0.0.0.0:XXXX` and
   confirm `XXXX` matches Railway's `PORT`. If the app listens on a different
   port, Railway's proxy cannot reach it.

3. **Check startup timing.** Railway may send traffic before the app finishes
   initializing. The `healthcheckTimeout` in `railway.toml` is set to 100s to
   allow for slow startups.

## Client Configuration

After deployment, configure your MCP client (e.g., Cursor, Claude, LM Studio) with:

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://mcp.planexe.org/mcp",
      "headers": {
        "X-API-Key": "your-secret-api-key-here"
      }
    }
  }
}
```

Replace `https://mcp.planexe.org` with your Railway deployment URL.

## Health Check

The service exposes `/healthcheck` that Railway uses for monitoring
(configured in `railway.toml`). This is a lightweight endpoint that does
**not** verify database connectivity.

## Domain

Configure a `Custom Domain` named `mcp.planexe.org` that points to Railway.
Railway's reverse proxy handles TLS termination and routes traffic to the
app's `PORT`.

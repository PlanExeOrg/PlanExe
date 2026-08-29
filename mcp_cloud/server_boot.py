"""
PlanExe MCP Cloud — server bootstrap

Configuration constants, MCP / FastAPI creation, and application lifespan.
This module is the canonical source for all env-var-derived settings.
"""
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

# Early startup logging — gated behind PLANEXE_DEBUG_STARTUP so Railway
# deployments stay quiet unless explicitly opted in.
_DEBUG_STARTUP = os.environ.get("PLANEXE_DEBUG_STARTUP", "").strip().lower() in ("1", "true", "yes")


def _startup_log(msg: str) -> None:
    """Emit a startup breadcrumb to stderr when PLANEXE_DEBUG_STARTUP is set."""
    if _DEBUG_STARTUP:
        print(f"[startup] {msg}", file=sys.stderr, flush=True)


_startup_log("server_boot.py: begin imports")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

_startup_log("server_boot.py: 3rd-party imports done")

from mcp_cloud.dotenv_utils import load_planexe_dotenv
_dotenv_loaded, _dotenv_paths = load_planexe_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
if not _dotenv_loaded:
    logger.warning(
        "No .env file found; searched: %s",
        ", ".join(str(path) for path in _dotenv_paths),
    )

_startup_log("server_boot.py: about to import mcp_cloud.app (triggers db_setup)")
from mcp_cloud.app import (
    PLANEXE_SERVER_INSTRUCTIONS,
    TOOL_DEFINITIONS,
)
_startup_log("server_boot.py: mcp_cloud.app imported OK")
from mcp_cloud.auth import validate_api_key_secret
from mcp_cloud.download_tokens import validate_download_token_secret
_startup_log("server_boot.py: auth + download_tokens imported OK")

# ---------------------------------------------------------------------------
# Configuration constants (all derived from environment variables)
# ---------------------------------------------------------------------------
SERVER_VERSION = "1.0.1"

REQUIRED_API_KEY = os.environ.get("PLANEXE_MCP_API_KEY")

HTTP_HOST = os.environ.get("PLANEXE_MCP_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("PORT") or os.environ.get("PLANEXE_MCP_HTTP_PORT", "8001"))
MAX_BODY_BYTES = int(os.environ.get("PLANEXE_MCP_MAX_BODY_BYTES", "1048576"))
RATE_LIMIT_REQUESTS = int(os.environ.get("PLANEXE_MCP_RATE_LIMIT", "60"))
RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("PLANEXE_MCP_RATE_WINDOW_SECONDS", "60"))
DOWNLOAD_RATE_LIMIT_REQUESTS = int(os.environ.get("PLANEXE_MCP_DOWNLOAD_RATE_LIMIT", "10"))
DOWNLOAD_RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("PLANEXE_MCP_DOWNLOAD_RATE_WINDOW_SECONDS", "60"))
GLAMA_MAINTAINER_EMAIL = os.environ.get(
    "PLANEXE_MCP_GLAMA_MAINTAINER_EMAIL",
    "neoneye@gmail.com",
).strip()


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s=%r. Using default=%s", name, raw_value, default)
    return default


AUTH_REQUIRED = _parse_bool_env("PLANEXE_MCP_REQUIRE_AUTH", default=True)

if AUTH_REQUIRED:
    validate_api_key_secret()
    validate_download_token_secret()


def _split_csv_env(value: Optional[str]) -> list[str]:
    if not value:
        return []
    raw = value.strip()
    if not raw:
        return []

    values: list[str]
    if raw.startswith("[") and raw.endswith("]"):
        # Allow JSON array format in env var, e.g. ["https://a", "https://b"].
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values = [str(item) for item in parsed]
        else:
            values = raw.split(",")
    else:
        values = raw.split(",")

    origins: list[str] = []
    for item in values:
        normalized = str(item).strip()
        if not normalized:
            continue
        # Be tolerant to copy-pasted shell quoting in deployment env values.
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        if normalized:
            origins.append(normalized)
    return origins


CORS_ORIGINS = _split_csv_env(os.environ.get("PLANEXE_MCP_CORS_ORIGINS"))
if not CORS_ORIGINS:
    if AUTH_REQUIRED:
        # Production default: only allow known PlanExe origins.
        # Override via PLANEXE_MCP_CORS_ORIGINS if additional origins are needed.
        CORS_ORIGINS = [
            "https://mcp.planexe.org",
            "https://home.planexe.org",
        ]
    else:
        # Dev mode: allow any origin so browser-based tools (e.g. MCP Inspector
        # at localhost:6274) can connect without extra configuration.
        CORS_ORIGINS = ["*"]
        logger.info("CORS wildcard enabled (PLANEXE_MCP_REQUIRE_AUTH=false)")


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------
_startup_log("server_boot.py: creating MCP server")

mcp_server = MCPServer(
    name="planexe-mcp-server",
    instructions=PLANEXE_SERVER_INSTRUCTIONS,
    version=SERVER_VERSION,
)

# Tool registration and MCP prompts are applied by route_registration.
from mcp_cloud.route_registration import register_routes, register_tools_and_prompts
register_tools_and_prompts(mcp_server)

# Transport settings live on streamable_http_app() rather than the server.
#
# transport_security is set explicitly because the SDK turns on DNS rebinding
# protection by itself when host is a loopback address, restricting Host and
# Origin to localhost.  This server is reached over the LAN in development
# (see mcp_cloud/README.md) and through the Railway proxy in production, so the
# implicit localhost-only allowlist would reject legitimate clients.  Access is
# controlled by the API-key middleware and the CORS origins configured above.
mcp_http_app = mcp_server.streamable_http_app(
    streamable_http_path="/",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def _get_mcp_server(request: Request) -> MCPServer:
    mcp_server = getattr(request.app.state, "mcp_server", None)
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="mcp_cloud not initialized")
    return mcp_server


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    from mcp_cloud.middleware import _sweep_rate_buckets
    app.state.mcp_server = mcp_server
    stop_event = asyncio.Event()
    sweeper_task = asyncio.create_task(_sweep_rate_buckets(stop_event))
    try:
        async with mcp_server.session_manager.run():
            yield
    finally:
        stop_event.set()
        sweeper_task.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper_task


# ---------------------------------------------------------------------------
# FastAPI app creation + middleware wiring
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PlanExe – AI Project Planning",
    description="MCP server that generates strategic project-plan drafts from a natural-language prompt",
    version=SERVER_VERSION,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
    allow_headers=["*"],  # Allow any header (e.g. X-API-Key) for CORS preflight
)

# Register all routes (must happen before mounting the MCP sub-app).
register_routes(app, mcp_server, _get_mcp_server)

# Mount the Streamable HTTP MCP endpoint AFTER the explicit /mcp/tools and
# /mcp/tools/call routes so that those routes take priority.  Starlette checks
# routes in registration order; if the mount were first it would shadow the
# REST endpoints with a 404 from the sub-app.
app.mount("/mcp", mcp_http_app)

# Apply middleware from middleware.py.
from mcp_cloud.middleware import apply_middleware
apply_middleware(app)

_startup_log("server_boot.py: app assembled OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting PlanExe MCP Cloud server on {HTTP_HOST}:{HTTP_PORT}")
    if AUTH_REQUIRED:
        logger.info(
            "Authentication required: UserApiKey (from home.planexe.org) or PLANEXE_MCP_API_KEY"
        )
    else:
        logger.warning("Authentication disabled via PLANEXE_MCP_REQUIRE_AUTH=false")

    uvicorn.run("mcp_cloud.server_boot:app", host=HTTP_HOST, port=HTTP_PORT, reload=False)

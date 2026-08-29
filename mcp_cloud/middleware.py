"""
PlanExe MCP Cloud — HTTP middleware

CORS helpers, API-key authentication, rate limiting, body-size enforcement,
and the main ``enforce_api_key`` middleware function.
"""
import asyncio
import contextvars
import json
import logging
from collections import defaultdict, deque
from time import monotonic
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from mcp_cloud.http_utils import strip_redundant_content
from mcp_cloud.app import (
    clear_download_base_url,
    set_download_base_url,
    validate_download_token,
    _resolve_user_from_api_key,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import configuration from server_boot — use module-level import (NOT
# ``from … import``) so that runtime attribute lookups always reflect the
# current value.  This is critical for test-time patching: tests mutate
# ``http_server.CORS_ORIGINS`` etc., and the re-export shim sets the
# attribute on server_boot's namespace, so reading ``_boot.CORS_ORIGINS``
# inside a function picks up the patched value.
# ---------------------------------------------------------------------------
import mcp_cloud.server_boot as _boot


# ---------------------------------------------------------------------------
# Auth policy sets
# ---------------------------------------------------------------------------
PUBLIC_JSONRPC_METHODS_NO_AUTH = {
    "initialize",
    "notifications/initialized",
    "tools/list",
    "prompts/list",
    "prompts/get",
    "resources/list",
    "resources/templates/list",
    "ping",
}
PUBLIC_TOOL_CALLS_NO_AUTH = {
    "example_plans",
    "example_prompts",
    "model_profiles",
}


# ---------------------------------------------------------------------------
# CORS helpers
# ---------------------------------------------------------------------------
def _allowed_cors_origin(request: Request) -> Optional[str]:
    origin = request.headers.get("origin")
    if not origin:
        return None
    if "*" in _boot.CORS_ORIGINS:
        return "*"
    if origin in _boot.CORS_ORIGINS:
        return origin
    return None


def _append_cors_headers(request: Request, response: Response) -> Response:
    """Ensure browser clients receive CORS headers even on early error responses."""
    allow_origin = _allowed_cors_origin(request)
    if not allow_origin:
        return response

    headers = response.headers
    headers.setdefault("Access-Control-Allow-Origin", allow_origin)
    headers.setdefault("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")

    request_headers = request.headers.get("access-control-request-headers")
    if request_headers:
        headers.setdefault("Access-Control-Allow-Headers", request_headers)
    else:
        headers.setdefault("Access-Control-Allow-Headers", "*")

    if allow_origin != "*":
        existing_vary = headers.get("Vary")
        if existing_vary:
            vary_values = [item.strip() for item in existing_vary.split(",") if item.strip()]
            if "Origin" not in vary_values:
                headers["Vary"] = f"{existing_vary}, Origin"
        else:
            headers["Vary"] = "Origin"
    return response


# ---------------------------------------------------------------------------
# JSON-RPC request inspection
# ---------------------------------------------------------------------------
def _extract_jsonrpc_methods_from_payload(payload: Any) -> list[str]:
    methods: list[str] = []
    if isinstance(payload, dict):
        method = payload.get("method")
        if isinstance(method, str):
            methods.append(method)
        return methods

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                method = item.get("method")
                if isinstance(method, str):
                    methods.append(method)
    return methods


async def _extract_jsonrpc_methods_from_request(request: Request) -> list[str]:
    """Best-effort extraction of JSON-RPC method names from request body."""
    try:
        body = await request.body()
    except Exception:
        return []
    if not body:
        return []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    return _extract_jsonrpc_methods_from_payload(payload)


def _extract_jsonrpc_tools_call_names(payload: Any) -> list[str]:
    names: list[str] = []
    entries: list[Any]
    if isinstance(payload, dict):
        entries = [payload]
    elif isinstance(payload, list):
        entries = payload
    else:
        return names

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("method") != "tools/call":
            continue
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        name = params.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _extract_rest_tools_call_name(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    return tool if isinstance(tool, str) else None


# ---------------------------------------------------------------------------
# Public request detection
# ---------------------------------------------------------------------------
async def _is_public_mcp_request_without_auth(request: Request) -> bool:
    """Allow unauthenticated MCP handshake/discovery calls."""
    path = request.url.path
    method = request.method.upper()

    # Keep slashless /mcp compatible for probing and redirect handling.
    if path == "/mcp" and method in {"GET", "HEAD", "POST"}:
        return True
    if path == "/mcp/" and method in {"GET", "HEAD"}:
        return True

    # Public HTTP JSON endpoint for tool introspection.
    if path == "/mcp/tools" and method == "GET":
        return True

    # REST MCP tools call endpoint: expose only free setup/discovery tools.
    if path == "/mcp/tools/call" and method == "POST":
        try:
            payload = json.loads((await request.body()) or b"")
        except json.JSONDecodeError:
            return False
        tool = _extract_rest_tools_call_name(payload)
        return tool in PUBLIC_TOOL_CALLS_NO_AUTH

    # Streamable HTTP endpoint: allow lightweight discovery methods and free setup tools.
    if path != "/mcp/" or method != "POST":
        return False

    try:
        payload = json.loads((await request.body()) or b"")
    except json.JSONDecodeError:
        return False

    methods = _extract_jsonrpc_methods_from_payload(payload)
    if not methods:
        return False
    if all(item in PUBLIC_JSONRPC_METHODS_NO_AUTH for item in methods):
        return True

    if all(item == "tools/call" for item in methods):
        names = _extract_jsonrpc_tools_call_names(payload)
        if names and all(name in PUBLIC_TOOL_CALLS_NO_AUTH for name in names):
            return True
    return False


# ---------------------------------------------------------------------------
# Auth error handling & logging
# ---------------------------------------------------------------------------
async def _make_jsonrpc_auth_error(request: Request, detail: str) -> JSONResponse:
    """Build a JSON-RPC error response for auth failures on /mcp/.

    Returning a plain HTTP 401/403 from the middleware causes the MCP SDK to
    interpret it as an OAuth challenge, triggering a confusing
    ``/.well-known/oauth-authorization-server`` discovery that fails with 404.
    By wrapping the error in a JSON-RPC envelope with HTTP 200, the SDK shows
    the message directly to the user.
    """
    request_id = None
    try:
        body = await request.body()
        if body:
            payload = json.loads(body)
            if isinstance(payload, dict):
                request_id = payload.get("id")
            elif isinstance(payload, list) and payload:
                request_id = payload[0].get("id") if isinstance(payload[0], dict) else None
    except Exception:
        pass
    return JSONResponse(
        status_code=200,
        content={
            "jsonrpc": "2.0",
            "error": {"code": -32001, "message": detail},
            "id": request_id,
        },
    )


async def _log_auth_rejection(request: Request, reason: str) -> None:
    methods = await _extract_jsonrpc_methods_from_request(request)
    logger.info(
        "Auth rejected: reason=%s method=%s path=%s client=%s ua=%s jsonrpc_methods=%s",
        reason,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        request.headers.get("user-agent", ""),
        methods,
    )


# ---------------------------------------------------------------------------
# API key extraction & validation
# ---------------------------------------------------------------------------
_rate_lock = asyncio.Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_download_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_authenticated_user_api_key_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "authenticated_user_api_key", default=None
)


def _normalize_api_key_value(raw_value: Optional[str]) -> Optional[str]:
    """Strip common copy-paste artefacts from an API key value.

    Handles cases where clients paste the full header line (e.g. 'X-API-Key: pex_…')
    or include a Bearer/token scheme prefix (e.g. 'Bearer pex_…'), or wrap the
    value in quotes.
    """
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None

    lower = value.lower()

    # Strip common header-name prefixes (copy-paste of the full header line).
    for prefix in ("x-api-key:", "api-key:", "api_key:", "authorization:"):
        if lower.startswith(prefix):
            value = value[len(prefix):].strip()
            lower = value.lower()
            break

    # Strip scheme prefixes.
    if lower.startswith("bearer "):
        value = value[7:].strip()
    elif lower.startswith("token "):
        value = value[6:].strip()
    elif " " in value:
        # Generic "<scheme> <token>" – keep the last segment.
        value = value.rsplit(" ", 1)[-1].strip()

    # Strip matching surrounding quotes.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()

    return value or None


def _extract_api_key(request: Request) -> Optional[str]:
    # Prefer Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    normalized_auth = _normalize_api_key_value(auth_header)
    if normalized_auth:
        return normalized_auth

    # Fall back to explicit API-key headers (case-insensitive via Starlette).
    for header_name in ("X-API-Key", "API-Key", "API_KEY", "X_API_KEY"):
        value = _normalize_api_key_value(request.headers.get(header_name))
        if value:
            return value

    # Fall back to query parameter (e.g. Smithery passes ?X-API-Key=...).
    for param_name in ("X-API-Key", "api_key", "api-key"):
        value = _normalize_api_key_value(request.query_params.get(param_name))
        if value:
            return value

    return None


async def _validate_api_key(request: Request) -> Optional[JSONResponse]:
    """Return an error response if API key validation fails.
    Accepts: (1) valid UserApiKey from DB, or (2) PLANEXE_MCP_API_KEY if set.
    Authentication can be disabled with PLANEXE_MCP_REQUIRE_AUTH=false.
    """
    provided_key = _extract_api_key(request)

    if not _boot.AUTH_REQUIRED:
        # Auth disabled — still resolve the key for attribution (last_used_at,
        # per-key billing).  If a key IS provided but invalid, reject: the
        # caller clearly intends to authenticate, so silently ignoring a bad
        # key is worse than telling them.
        if provided_key:
            user = await asyncio.to_thread(_resolve_user_from_api_key, provided_key)
            if user:
                _authenticated_user_api_key_ctx.set(provided_key)
            else:
                await _log_auth_rejection(request, reason="invalid_api_key_local")
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "Invalid API key. "
                            "Remove the X-API-Key header for local access, "
                            "or get a valid key at https://home.planexe.org/"
                        )
                    },
                )
        return None

    if not provided_key:
        await _log_auth_rejection(request, reason="missing_api_key")
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Missing API key. Create an API key at https://home.planexe.org/"
            },
        )

    # Accept PLANEXE_MCP_API_KEY (shared secret) if configured
    if _boot.REQUIRED_API_KEY and provided_key == _boot.REQUIRED_API_KEY:
        _authenticated_user_api_key_ctx.set(None)
        return None

    # Accept valid UserApiKey from database (pex_... keys from home.planexe.org)
    user = await asyncio.to_thread(_resolve_user_from_api_key, provided_key)
    if user:
        _authenticated_user_api_key_ctx.set(provided_key)
        return None

    await _log_auth_rejection(request, reason="invalid_api_key")
    return JSONResponse(status_code=403, content={"detail": "Invalid API key. Verify your key or create an account at https://home.planexe.org/"})


def _get_authenticated_user_api_key() -> Optional[str]:
    """Return the current request's authenticated UserApiKey, if available."""
    try:
        return _authenticated_user_api_key_ctx.get()
    except LookupError:
        return None


def _client_identifier(request: Request) -> str:
    api_key = _extract_api_key(request)
    if api_key:
        return f"key:{api_key}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
async def _enforce_rate_limit(request: Request) -> Optional[JSONResponse]:
    if _boot.RATE_LIMIT_REQUESTS <= 0:
        return None
    # Apply to the legacy JSON-call endpoint and to the Streamable HTTP /mcp endpoint.
    if request.url.path not in ("/mcp/tools/call", "/mcp", "/mcp/"):
        return None

    identifier = _client_identifier(request)
    now = monotonic()
    async with _rate_lock:
        bucket = _rate_buckets[identifier]
        while bucket and now - bucket[0] > _boot.RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= _boot.RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )
        bucket.append(now)
    return None


async def _enforce_download_rate_limit(request: Request) -> Optional[JSONResponse]:
    if _boot.DOWNLOAD_RATE_LIMIT_REQUESTS <= 0:
        return None
    if not request.url.path.startswith("/download"):
        return None

    identifier = _client_identifier(request)
    now = monotonic()
    async with _rate_lock:
        bucket = _download_rate_buckets[identifier]
        while bucket and now - bucket[0] > _boot.DOWNLOAD_RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= _boot.DOWNLOAD_RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Download rate limit exceeded"},
            )
        bucket.append(now)
    return None


async def _sweep_rate_buckets(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_boot.RATE_LIMIT_WINDOW_SECONDS)
        except asyncio.TimeoutError:
            pass
        now = monotonic()
        async with _rate_lock:
            for key in list(_rate_buckets):
                bucket = _rate_buckets[key]
                while bucket and now - bucket[0] > _boot.RATE_LIMIT_WINDOW_SECONDS:
                    bucket.popleft()
                if not bucket:
                    del _rate_buckets[key]
            for key in list(_download_rate_buckets):
                bucket = _download_rate_buckets[key]
                while bucket and now - bucket[0] > _boot.DOWNLOAD_RATE_LIMIT_WINDOW_SECONDS:
                    bucket.popleft()
                if not bucket:
                    del _download_rate_buckets[key]


# ---------------------------------------------------------------------------
# Body size enforcement
# ---------------------------------------------------------------------------
async def _enforce_body_size(request: Request) -> Optional[JSONResponse]:
    if request.method != "POST":
        return None
    if request.url.path not in ("/mcp/tools/call", "/mcp/"):
        return None

    content_length = request.headers.get("content-length")
    if not content_length:
        # Streamable HTTP (/mcp/) may use chunked encoding without Content-Length.
        # Only require it on the REST endpoint.
        if request.url.path == "/mcp/tools/call":
            return JSONResponse(
                status_code=411,
                content={"detail": "Length Required"},
            )
        return None

    try:
        if int(content_length) > _boot.MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid Content-Length header"},
        )
    return None


# ---------------------------------------------------------------------------
# Download token & request origin helpers
# ---------------------------------------------------------------------------
def _request_origin(request: Request) -> str:
    """Return externally visible scheme+host, honoring reverse-proxy headers."""
    parsed = urlparse(str(request.base_url))

    # Railway/reverse proxies terminate TLS and forward the original scheme/host.
    forwarded_proto = (request.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip().lower()
    forwarded_host = (request.headers.get("X-Forwarded-Host") or "").split(",")[0].strip()

    scheme = forwarded_proto if forwarded_proto in {"http", "https"} else parsed.scheme
    netloc = forwarded_host or request.headers.get("Host") or parsed.netloc
    return f"{scheme}://{netloc}"


def _has_valid_download_token(request: Request) -> bool:
    """Return True when the request carries a valid signed download token.

    Expected URL shape: /download/{plan_id}/{filename}?token=...
    The token is validated against the plan_id and filename so it cannot be
    reused for a different artifact.
    """
    token = request.query_params.get("token")
    if not token:
        return False
    parts = request.url.path.strip("/").split("/")
    # parts == ["download", "{plan_id}", "{filename}"]
    if len(parts) != 3 or parts[0] != "download":
        return False
    plan_id, filename = parts[1], parts[2]
    return validate_download_token(token, plan_id, filename)


# ---------------------------------------------------------------------------
# ASGI path normalization middleware
# ---------------------------------------------------------------------------
class _NormalizeMcpPath:
    """ASGI middleware: rewrite ``/mcp`` → ``/mcp/`` at the scope level.

    Smithery (and possibly other registries) POST to ``/mcp`` but refuse to
    follow 307 redirects.  By rewriting the path *before* routing, the mounted
    MCP sub-app receives the request directly — no HTTP redirect needed.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Main HTTP middleware
# ---------------------------------------------------------------------------
async def enforce_api_key(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # OPTIONS (CORS preflight) must not require auth; browser does not send custom headers
    if request.method != "OPTIONS" and (
        request.url.path.startswith("/mcp") or request.url.path.startswith("/download")
    ):
        is_public = await _is_public_mcp_request_without_auth(request)

        is_mcp_streamable = request.url.path in ("/mcp", "/mcp/")

        async def _check_auth() -> Optional[Response]:
            """Validate API key; wrap errors as JSON-RPC for Streamable HTTP."""
            err = await _validate_api_key(request)
            if not err:
                return None
            if is_mcp_streamable:
                detail = (err.body or b"").decode(errors="replace")
                try:
                    detail = json.loads(detail).get("detail", detail)
                except Exception:
                    pass
                err = await _make_jsonrpc_auth_error(request, detail)
            return _append_cors_headers(request, err)

        # Even for public/discovery methods (initialize, tools/list, etc.),
        # validate the API key if one was provided.  This lets callers discover
        # a bad key at connection time instead of on the first paid tool call.
        if is_public and _extract_api_key(request):
            blocked = await _check_auth()
            if blocked:
                return blocked

        if not is_public:
            # /download with a valid signed token is self-authenticating — no API key needed.
            is_tokenized_download = (
                request.url.path.startswith("/download")
                and _has_valid_download_token(request)
            )
            if not is_tokenized_download:
                blocked = await _check_auth()
                if blocked:
                    return blocked

    error_response = await _enforce_body_size(request)
    if error_response:
        return _append_cors_headers(request, error_response)

    error_response = await _enforce_rate_limit(request)
    if error_response:
        return _append_cors_headers(request, error_response)

    error_response = await _enforce_download_rate_limit(request)
    if error_response:
        return _append_cors_headers(request, error_response)

    if request.url.path.startswith("/mcp"):
        set_download_base_url(_request_origin(request))
    try:
        response = await call_next(request)
    finally:
        _authenticated_user_api_key_ctx.set(None)
        if request.url.path.startswith("/mcp"):
            clear_download_base_url()
    if request.url.path.startswith("/mcp"):
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            body = getattr(response, "body", None)
            if body:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    return response
                stripped_payload, changed = strip_redundant_content(payload)
                if changed:
                    headers = dict(response.headers)
                    headers.pop("content-length", None)
                    return JSONResponse(
                        status_code=response.status_code,
                        content=stripped_payload,
                        headers=headers,
                        background=response.background,
                    )
    return _append_cors_headers(request, response)


# ---------------------------------------------------------------------------
# Public API: called by server_boot to wire middleware into the FastAPI app
# ---------------------------------------------------------------------------
def apply_middleware(app: Any) -> None:
    """Register middleware on the FastAPI app. Called by server_boot after routes are mounted."""
    app.middleware("http")(enforce_api_key)
    # Rewrite /mcp -> /mcp/ at the ASGI level so clients that refuse to follow
    # 307 redirects (e.g. Smithery) still reach the mounted MCP app.
    # Added last so it becomes the outermost middleware (runs first).
    app.add_middleware(_NormalizeMcpPath)

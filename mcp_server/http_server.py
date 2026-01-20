"""
HTTP server wrapper for PlanExe MCP Server

Provides HTTP/JSON endpoints for MCP tool calls with API key authentication.
Supports deployment to Railway and other cloud platforms.
"""
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from time import monotonic
from typing import Annotated, Any, Awaitable, Callable, Literal, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ContentBlock, TextContent

from mcp_server.http_utils import strip_redundant_content

# Load .env file early
from mcp_server.dotenv_utils import load_planexe_dotenv
_dotenv_loaded, _dotenv_paths = load_planexe_dotenv()

# Configure logging
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

# Import MCP tool handlers from app.py
from mcp_server.app import (
    REPORT_CONTENT_TYPE,
    REPORT_FILENAME,
    fetch_artifact_from_worker_plan,
    handle_task_create,
    handle_task_status,
    handle_task_stop,
    handle_report_read,
    resolve_task_for_task_id,
)

# API key validation
REQUIRED_API_KEY = os.environ.get("PLANEXE_MCP_API_KEY")
if not REQUIRED_API_KEY:
    logger.warning(
        "PLANEXE_MCP_API_KEY not set. API key authentication disabled (not recommended for production)"
    )

HTTP_HOST = os.environ.get("PLANEXE_MCP_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("PORT") or os.environ.get("PLANEXE_MCP_HTTP_PORT", "8001"))
MAX_BODY_BYTES = int(os.environ.get("PLANEXE_MCP_MAX_BODY_BYTES", "1048576"))
RATE_LIMIT_REQUESTS = int(os.environ.get("PLANEXE_MCP_RATE_LIMIT", "60"))
RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("PLANEXE_MCP_RATE_WINDOW_SECONDS", "60"))


def _split_csv_env(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


CORS_ORIGINS = _split_csv_env(os.environ.get("PLANEXE_MCP_CORS_ORIGINS"))
if not CORS_ORIGINS:
    CORS_ORIGINS = ["http://localhost", "http://127.0.0.1"]

_rate_lock = asyncio.Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _extract_api_key(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        parts = auth_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            if token:
                return token
    header_key = request.headers.get("X-API-Key") or request.headers.get("API_KEY")
    if header_key:
        return header_key
    query_key = request.query_params.get("api_key")
    if query_key:
        return query_key
    return None


def _validate_api_key(request: Request) -> Optional[JSONResponse]:
    """Return an error response if API key validation fails."""
    if not REQUIRED_API_KEY:
        return None

    provided_key = _extract_api_key(request)
    if not provided_key:
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Missing API key. Use Authorization: Bearer <key>, X-API-Key, or ?api_key=."
            },
        )

    if provided_key != REQUIRED_API_KEY:
        return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

    return None


def _client_identifier(request: Request) -> str:
    api_key = _extract_api_key(request)
    if api_key:
        return f"key:{api_key}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


async def _enforce_rate_limit(request: Request) -> Optional[JSONResponse]:
    if RATE_LIMIT_REQUESTS <= 0:
        return None
    if request.url.path != "/mcp/tools/call":
        return None

    identifier = _client_identifier(request)
    now = monotonic()
    async with _rate_lock:
        bucket = _rate_buckets[identifier]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )
        bucket.append(now)
    return None


async def _sweep_rate_buckets(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=RATE_LIMIT_WINDOW_SECONDS)
        except asyncio.TimeoutError:
            pass
        now = monotonic()
        async with _rate_lock:
            for key in list(_rate_buckets):
                bucket = _rate_buckets[key]
                while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
                    bucket.popleft()
                if not bucket:
                    del _rate_buckets[key]


async def _enforce_body_size(request: Request) -> Optional[JSONResponse]:
    if request.method != "POST" or request.url.path != "/mcp/tools/call":
        return None

    content_length = request.headers.get("content-length")
    if not content_length:
        return JSONResponse(
            status_code=411,
            content={"detail": "Length Required"},
        )

    try:
        if int(content_length) > MAX_BODY_BYTES:
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


# Request/Response models
class MCPToolCallRequest(BaseModel):
    tool: str
    arguments: dict[str, Any]


class MCPToolCallResponse(BaseModel):
    content: list[dict[str, Any]]
    error: Optional[dict[str, Any]] = None


class ErrorDetail(BaseModel):
    code: str
    message: str


class TaskCreateOutput(BaseModel):
    task_id: str
    created_at: str


class TaskStatusProgressTask(BaseModel):
    name: str
    pct: float


class TaskStatusProgress(BaseModel):
    overall: float
    current_task: TaskStatusProgressTask


class TaskStatusTiming(BaseModel):
    started_at: str | None
    elapsed_sec: float


class TaskStatusArtifact(BaseModel):
    path: str
    updated_at: str


class TaskStatusOutput(BaseModel):
    task_id: str | None = None
    state: Literal["stopped", "running", "completed", "failed", "stopping"] | None = None
    progress: TaskStatusProgress | None = None
    timing: TaskStatusTiming | None = None
    latest_artifacts: list[TaskStatusArtifact] | None = None
    error: ErrorDetail | None = None


class ReportRange(BaseModel):
    start: int
    length: int


class ReportResultOutput(BaseModel):
    state: Literal["running", "failed", "ready"] | None = None
    content_type: str | None = None
    sha256: str | None = None
    download_size: int | None = None
    download_url: str | None = None
    content: str | None = None
    total_size: int | None = None
    range: ReportRange | None = None
    truncated: bool | None = None
    next_range: ReportRange | None = None
    error: ErrorDetail | None = None


def extract_text_content(text_contents: Sequence[Any]) -> list[dict[str, Any]]:
    """Extract text content from MCP TextContent objects."""
    result = []
    for item in text_contents:
        if hasattr(item, 'text'):
            result.append({"text": item.text})
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append({"text": str(item)})
    return result


def _parse_error_from_text(text: Any) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    if not text or text[:1] not in ("{", "["):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and "error" in parsed:
        error = parsed["error"]
        if isinstance(error, dict):
            return error
        return {"message": str(error)}
    return None


def _normalize_tool_result(result: Any) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    if isinstance(result, CallToolResult):
        content_blocks = result.content
        content = extract_text_content(content_blocks)
        error = None
        for item in content:
            if isinstance(item, dict) and "error" in item:
                error = item["error"]
                break
            if isinstance(item, dict) and "text" in item:
                parsed_error = _parse_error_from_text(item["text"])
                if parsed_error:
                    error = parsed_error
                    break
        return content, error
    if isinstance(result, ContentBlock):
        content_blocks: Sequence[Any] = [result]
    elif isinstance(result, list):
        content_blocks = result
    elif isinstance(result, dict):
        content_blocks = [result]
    else:
        content_blocks = [TextContent(type="text", text=str(result))]

    content = extract_text_content(content_blocks)
    error = None
    for item in content:
        if isinstance(item, dict) and "error" in item:
            error = item["error"]
            break
        if isinstance(item, dict) and "text" in item:
            parsed_error = _parse_error_from_text(item["text"])
            if parsed_error:
                error = parsed_error
                break
    return content, error


async def task_create(
    idea: str,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[CallToolResult, TaskCreateOutput]:
    return await handle_task_create({"idea": idea, "config": config, "metadata": metadata})


async def task_status(task_id: str) -> Annotated[CallToolResult, TaskStatusOutput]:
    return await handle_task_status({"task_id": task_id})


async def task_stop(
    task_id: str,
    mode: str = "graceful",
) -> CallToolResult:
    return await handle_task_stop({"task_id": task_id, "mode": mode})


async def get_result(
    task_id: str,
) -> Annotated[CallToolResult, ReportResultOutput]:
    return await handle_report_read({"task_id": task_id})


def _register_tools(server: FastMCP) -> None:
    server.tool(
        name="task_create",
        description="Creates a new task and output namespace",
    )(task_create)
    server.tool(
        name="task_status",
        description="Returns run status and progress",
    )(task_status)
    server.tool(
        name="task_stop",
        description="Stops the active run",
    )(task_stop)
    server.tool(
        name="task_result",
        description="Returns download metadata for the generated report",
    )(get_result)


fastmcp_server = FastMCP(
    name="planexe-mcp-server",
    instructions="HTTP wrapper for PlanExe MCP interface",
    host=HTTP_HOST,
    port=HTTP_PORT,
    streamable_http_path="/",
    json_response=True,
    stateless_http=True,
)
_register_tools(fastmcp_server)
fastmcp_http_app = fastmcp_server.streamable_http_app()


def _get_fastmcp(request: Request) -> FastMCP:
    fastmcp_server = getattr(request.app.state, "fastmcp_server", None)
    if fastmcp_server is None:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    return fastmcp_server


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.fastmcp_server = fastmcp_server
    stop_event = asyncio.Event()
    sweeper_task = asyncio.create_task(_sweep_rate_buckets(stop_event))
    try:
        async with fastmcp_server.session_manager.run():
            yield
    finally:
        stop_event.set()
        sweeper_task.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper_task


app = FastAPI(
    title="PlanExe MCP Server (HTTP)",
    description="HTTP wrapper for PlanExe MCP interface",
    version="1.0.0",
    lifespan=_lifespan,
)

app.mount("/mcp", fastmcp_http_app)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "API_KEY"],
)


@app.middleware("http")
async def enforce_api_key(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.url.path.startswith("/mcp") or request.url.path.startswith("/download"):
        error_response = _validate_api_key(request)
        if error_response:
            return error_response

    error_response = await _enforce_body_size(request)
    if error_response:
        return error_response

    error_response = await _enforce_rate_limit(request)
    if error_response:
        return error_response

    response = await call_next(request)
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
    return response


async def call_tool_via_registry(
    server: FastMCP,
    tool_name: str,
    arguments: dict[str, Any],
) -> MCPToolCallResponse:
    """Call tools via the FastMCP registry."""
    try:
        result = await server.call_tool(tool_name, arguments)
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
        return MCPToolCallResponse(
            content=[],
            error={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        )

    content, error = _normalize_tool_result(result)
    return MCPToolCallResponse(content=content, error=error)


@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def call_tool(
    payload: MCPToolCallRequest,
    fastmcp_server: FastMCP = Depends(_get_fastmcp),
) -> MCPToolCallResponse:
    """
    Call an MCP tool by name with arguments.

    This endpoint wraps the stdio-based MCP tool handlers for HTTP access.
    """
    return await call_tool_via_registry(fastmcp_server, payload.tool, payload.arguments)


@app.get("/mcp/tools")
async def list_tools(fastmcp_server: FastMCP = Depends(_get_fastmcp)) -> dict[str, Any]:
    """List all available MCP tools."""
    tools = await fastmcp_server.list_tools()
    sanitized = []
    for tool in tools:
        tool_entry = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
        }
        if tool.title:
            tool_entry["title"] = tool.title
        if tool.outputSchema:
            tool_entry["outputSchema"] = tool.outputSchema
        if tool.annotations:
            tool_entry["annotations"] = tool.annotations
        if tool.icons:
            tool_entry["icons"] = tool.icons
        sanitized.append(tool_entry)
    return {"tools": sanitized}

@app.get("/download/{task_id}/{filename}")
async def download_report(task_id: str, filename: str) -> Response:
    """Download the generated report HTML for a task."""
    if filename != REPORT_FILENAME:
        raise HTTPException(status_code=404, detail="Report not found")
    task = resolve_task_for_task_id(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    content_bytes = await fetch_artifact_from_worker_plan(str(task.id), REPORT_FILENAME)
    if content_bytes is None:
        raise HTTPException(status_code=404, detail="Report not found")
    headers = {"Content-Disposition": f'inline; filename="{REPORT_FILENAME}"'}
    return Response(content=content_bytes, media_type=REPORT_CONTENT_TYPE, headers=headers)


@app.get("/healthcheck")
def healthcheck() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "planexe-mcp-http",
        "api_key_configured": REQUIRED_API_KEY is not None
    }


@app.get("/")
def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "PlanExe MCP Server (HTTP)",
        "version": "1.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "tools": "/mcp/tools",
            "call": "/mcp/tools/call",
            "health": "/healthcheck",
        "download": f"/download/{{task_id}}/{REPORT_FILENAME}",
        },
        "documentation": "See /docs for OpenAPI documentation",
        "authentication": "Authorization: Bearer <key>, X-API-Key, or ?api_key= (set PLANEXE_MCP_API_KEY)"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting PlanExe MCP HTTP server on {HTTP_HOST}:{HTTP_PORT}")
    if REQUIRED_API_KEY:
        logger.info("API key authentication enabled")
    else:
        logger.warning("API key authentication disabled - set PLANEXE_MCP_API_KEY")

    uvicorn.run("http_server:app", host=HTTP_HOST, port=HTTP_PORT, reload=False)

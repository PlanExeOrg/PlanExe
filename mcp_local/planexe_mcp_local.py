"""
PlanExe MCP local proxy.

Runs locally over stdio and forwards tool calls to mcp_cloud, the MCP server
running in the cloud.
Downloads artifacts to disk for task_download.
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://your-railway-app.up.railway.app/mcp"
REPORT_FILENAME = "030-report.html"
ZIP_FILENAME = "run.zip"
SpeedVsDetailInput = Literal[
    "ping",
    "fast",
    "all",
]


class TaskCreateRequest(BaseModel):
    prompt: str
    speed_vs_detail: Optional[SpeedVsDetailInput] = None


class TaskStatusRequest(BaseModel):
    task_id: str


class TaskStopRequest(BaseModel):
    task_id: str


class TaskDownloadRequest(BaseModel):
    task_id: str
    artifact: str = "report"


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else default


def _get_mcp_base_url() -> str:
    raw_url = _get_env("PLANEXE_URL", DEFAULT_MCP_URL)
    if not raw_url:
        raw_url = DEFAULT_MCP_URL
    raw_url = raw_url.strip()
    parsed = urlparse(raw_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/mcp/tools/call"):
        path = path[: -len("/tools/call")]
    elif path.endswith("/mcp/tools"):
        path = path[: -len("/tools")]
    elif path.endswith("/tools/call"):
        path = path[: -len("/tools/call")]
    elif path.endswith("/tools"):
        path = path[: -len("/tools")]
    if not path.endswith("/mcp"):
        path = f"{path}/mcp".rstrip("/")
    normalized = parsed._replace(path=path, params="", query="", fragment="").geturl()
    return normalized


def _get_download_base_url() -> str:
    base_url = _get_mcp_base_url()
    if base_url.endswith("/mcp"):
        return base_url[:-4]
    return base_url


def _build_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    api_key = _get_env("PLANEXE_MCP_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_request_with_redirects(
    method: str,
    url: str,
    body: Optional[bytes],
    headers: dict[str, str],
    max_redirects: int = 5,
    max_retries: int = 3,
    retry_delay_base: float = 1.0,
) -> bytes:
    """Make an HTTP request with redirect handling and retry logic.
    
    Retries on 5xx server errors and network timeouts with exponential backoff.
    Does not retry on 4xx client errors (except redirects).
    """
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries):
        if attempt > 0:
            delay = retry_delay_base * (2 ** (attempt - 1))
            logger.warning(
                "Retrying HTTP request to %s (attempt %d/%d) after %.1fs delay",
                url,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
        
        try:
            for redirect_count in range(max_redirects + 1):
                request = Request(url, data=body, method=method, headers=headers)
                try:
                    with urlopen(request, timeout=60) as response:
                        return response.read()
                except HTTPError as exc:
                    # Handle redirects (3xx)
                    if exc.code in (301, 302, 303, 307, 308):
                        location = exc.headers.get("Location")
                        if not location:
                            raise
                        url = urljoin(url, location)
                        if exc.code == 303:
                            method = "GET"
                            body = None
                        continue
                    # Retry on 5xx server errors
                    if 500 <= exc.code < 600:
                        last_exception = exc
                        break  # Break redirect loop, will retry outer loop
                    # Don't retry on 4xx client errors (except redirects handled above)
                    raise
                except (URLError, OSError, TimeoutError) as exc:
                    # Retry on network errors and timeouts
                    last_exception = exc
                    break  # Break redirect loop, will retry outer loop
            else:
                # Redirect loop exhausted without success
                raise HTTPError(url, 310, "Too many redirects", None, None)
        except HTTPError as exc:
            # 4xx errors: don't retry
            if 400 <= exc.code < 500:
                raise
            # If we get here with a 5xx, last_exception is set and we'll retry
            if attempt == max_retries - 1:
                raise
    
    # All retries exhausted
    if last_exception:
        raise last_exception
    raise HTTPError(url, 500, "Request failed after retries", None, None)


def _http_json_request(method: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    response_body = _http_request_with_redirects(method, url, body, _build_headers())
    decoded = response_body.decode("utf-8") if response_body else ""
    return json.loads(decoded) if decoded else {}


def _http_get_bytes(url: str) -> bytes:
    return _http_request_with_redirects("GET", url, None, _build_headers())


def _extract_payload(content: list[dict[str, Any]]) -> dict[str, Any]:
    for item in content:
        text = item.get("text") if isinstance(item, dict) else None
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    return {}


def _call_remote_tool_rpc(
    tool: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    mcp_base_url = _get_mcp_base_url()
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    response = _http_json_request("POST", mcp_base_url, payload)
    error = response.get("error")
    if error:
        return {}, error
    result = response.get("result")
    if isinstance(result, dict):
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            if result.get("isError") and "error" in structured:
                return {}, structured.get("error")
            return structured, None
        content = result.get("content", [])
        if isinstance(content, list):
            return _extract_payload(content), None
    return {}, None


def _call_remote_tool(tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    mcp_base_url = _get_mcp_base_url()
    url = f"{mcp_base_url}/tools/call"
    payload = {"tool": tool, "arguments": arguments}
    try:
        response = _http_json_request("POST", url, payload)
    except HTTPError as exc:
        if exc.code == 404:
            try:
                return _call_remote_tool_rpc(tool, arguments)
            except Exception as rpc_exc:
                logger.error("Remote MCP JSON-RPC failed: %s", rpc_exc)
                return {}, {"code": "REMOTE_ERROR", "message": str(rpc_exc)}
        logger.error("Remote MCP request failed: %s", exc)
        return {}, {"code": "REMOTE_ERROR", "message": f"{exc} ({url})"}
    except Exception as exc:
        logger.error("Remote MCP request failed: %s", exc)
        return {}, {"code": "REMOTE_ERROR", "message": str(exc)}
    error = response.get("error")
    if error:
        return {}, error
    content = response.get("content", [])
    payload = _extract_payload(content)
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return {}, payload["error"]
    return payload, None


def _hash_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _derive_download_url(task_id: str, artifact: str) -> str:
    if artifact == "zip":
        path = f"/download/{task_id}/{ZIP_FILENAME}"
    else:
        path = f"/download/{task_id}/{REPORT_FILENAME}"
    return urljoin(_get_download_base_url().rstrip("/") + "/", path.lstrip("/"))


def _ensure_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"PLANEXE_PATH is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _choose_output_path(task_id: str, download_url: str, artifact: str) -> Path:
    base_path = Path(_get_env("PLANEXE_PATH", str(Path.cwd()))).expanduser()
    _ensure_directory(base_path)

    basename = Path(urlparse(download_url).path).name
    if not basename:
        basename = REPORT_FILENAME if artifact == "report" else ZIP_FILENAME
    filename = f"{task_id}-{basename}"
    candidate = base_path / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(1, 1000):
        fallback = base_path / f"{stem}-{index}{suffix}"
        if not fallback.exists():
            return fallback
    raise ValueError(f"Unable to find available filename in {base_path}")


def _download_to_path(download_url: str, destination: Path) -> int:
    content = _http_get_bytes(download_url)
    destination.write_bytes(content)
    return len(content)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: Optional[dict[str, Any]] = None


ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["code", "message"],
}

TASK_CREATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "What the plan should cover. Good prompts are often 300–800 words. "
                "Use prompt_examples to get example prompts; use these as examples for task_create. Short prompts produce less detailed plans."
            ),
        },
        "speed_vs_detail": {
            "type": "string",
            "enum": ["ping", "fast", "all"],
            "default": "ping",
            "description": "How much work to run. 'ping': single LLM call to check the pipeline is reachable (check logs if it fails). 'fast': minimal run (approx 5-10 min) through all pipeline steps, skipping where possible—use to verify the pipeline works. 'all': full plan with full detail (approx 10-20 min).",
        },
    },
    "required": ["prompt"],
}

TASK_STATUS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "UUID of the task (returned by task_create).",
        },
    },
    "required": ["task_id"],
}

TASK_STOP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "UUID of the task to stop (returned by task_create).",
        },
    },
    "required": ["task_id"],
}

TASK_DOWNLOAD_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "UUID of the task (returned by task_create).",
        },
        "artifact": {
            "type": "string",
            "enum": ["report", "zip"],
            "default": "report",
            "description": "What to download: 'report' = HTML report, 'zip' = full output bundle.",
        },
    },
    "required": ["task_id"],
}

PROMPT_EXAMPLES_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}
PROMPT_EXAMPLES_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Example prompts to copy or adapt when calling task_create.",
        },
        "message": {"type": "string"},
    },
    "required": ["samples", "message"],
}

TASK_CREATE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "created_at": {"type": "string"},
    },
    "required": ["task_id", "created_at"],
}

TASK_STATUS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": ["string", "null"]},
        "state": {"type": ["string", "null"]},
        "progress_percentage": {"type": ["number", "null"]},
    },
}

TASK_STOP_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "error": ERROR_SCHEMA,
    },
}

TASK_DOWNLOAD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string"},
        "sha256": {"type": "string"},
        "download_size": {"type": "integer"},
        "download_url": {"type": "string"},
        "saved_path": {"type": "string"},
        "error": ERROR_SCHEMA,
    },
    "additionalProperties": False,
}

TOOL_DEFINITIONS = [
    ToolDefinition(
        name="prompt_examples",
        description=(
            "Return curated example prompts from the PlanExe prompt catalog (entries marked mcp_example). "
            "Use these to see the level of detail that produces good plans; iterate with your local LLM "
            "Use these as examples for task_create."
        ),
        input_schema=PROMPT_EXAMPLES_INPUT_SCHEMA,
        output_schema=PROMPT_EXAMPLES_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_create",
        description=(
            "Start creating a new plan. Call prompt_examples for example prompts to use with task_create. "
            "Runs in the background and usually takes 10-20 minutes to complete."
        ),
        input_schema=TASK_CREATE_INPUT_SCHEMA,
        output_schema=TASK_CREATE_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_status",
        description=(
            "Returns status and progress of the plan currently being created. "
            "Poll at reasonable intervals only (e.g. every 5 minutes): plan generation takes 15–20+ minutes "
            "and frequent polling is unnecessary."
        ),
        input_schema=TASK_STATUS_INPUT_SCHEMA,
        output_schema=TASK_STATUS_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_stop",
        description="Stop a plan that is currently being created.",
        input_schema=TASK_STOP_INPUT_SCHEMA,
        output_schema=TASK_STOP_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_download",
        description=(
            "Download the plan output and save it locally. Choose the HTML "
            "report (default) or a zip of all generated files."
        ),
        input_schema=TASK_DOWNLOAD_INPUT_SCHEMA,
        output_schema=TASK_DOWNLOAD_OUTPUT_SCHEMA,
    ),
]

# Shown in MCP initialize response (e.g. Inspector) so clients know what PlanExe is.
PLANEXE_SERVER_INSTRUCTIONS = (
    "PlanExe generates rough-draft project plans from a natural-language prompt. "
    "You describe a large goal (e.g. open a clinic, launch a product, build a moon base)—the kind of project that takes months or years. "
    "PlanExe produces a structured draft with steps and deliverables (Gantt chart, risk analysis, etc.); the plan is not executable yet, it's a draft to refine. "
    "Creating a plan is a long-running task (100+ LLM calls). Main output: large HTML file (approx 700KB) and a zip of intermediary files (md, json, csv). "
    "Call prompt_examples first, then task_create; poll task_status and use task_download when complete."
)

mcp_local = Server("planexe-mcp-local", instructions=PLANEXE_SERVER_INSTRUCTIONS)


@mcp_local.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name=definition.name,
            description=definition.description,
            inputSchema=definition.input_schema,
            outputSchema=definition.output_schema,
        )
        for definition in TOOL_DEFINITIONS
    ]


def _wrap_response(payload: dict[str, Any], is_error: Optional[bool] = None) -> CallToolResult:
    if is_error is None:
        is_error = isinstance(payload.get("error"), dict)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=is_error,
    )


@mcp_local.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        response = {"error": {"code": "INVALID_TOOL", "message": f"Unknown tool: {name}"}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )
    return await handler(arguments)


async def handle_task_create(arguments: dict[str, Any]) -> CallToolResult:
    """Create a task in mcp_cloud via the local HTTP proxy.

    Examples:
        - {"prompt": "Start a dental clinic in Copenhagen with 3 treatment rooms, targeting families and children. Budget 2.5M DKK. Open within 12 months."} → task_id + created_at
        - {"prompt": "Launch a bike repair shop in Amsterdam with retail sales, service bays, and mobile repair van. Budget 150k EUR. Profitability goal: month 18.", "speed_vs_detail": "fast"}

    Args:
        - prompt: What the plan should cover (goal, context, constraints).
        - speed_vs_detail: Optional mode ("ping" | "fast" | "all").

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: task_id/created_at payload or error.
        - isError: True when the remote tool call fails.
    """
    req = TaskCreateRequest(**arguments)
    payload, error = _call_remote_tool(
        "task_create",
        {"prompt": req.prompt, "speed_vs_detail": req.speed_vs_detail} if req.speed_vs_detail else {"prompt": req.prompt},
    )
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_prompt_examples(arguments: dict[str, Any]) -> CallToolResult:
    """Return curated prompts from mcp_cloud so LLMs can see example detail."""
    payload, error = _call_remote_tool("prompt_examples", arguments or {})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_task_status(arguments: dict[str, Any]) -> CallToolResult:
    """Fetch status/progress for a task from mcp_cloud.

    Examples:
        - {"task_id": "uuid"} → state/progress/timing

    Args:
        - task_id: Task UUID returned by task_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: status payload or error.
        - isError: True when the remote tool call fails.
    """
    req = TaskStatusRequest(**arguments)
    payload, error = _call_remote_tool("task_status", {"task_id": req.task_id})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_task_stop(arguments: dict[str, Any]) -> CallToolResult:
    """Request mcp_cloud to stop a running task.

    Examples:
        - {"task_id": "uuid"} → stop request acknowledged

    Args:
        - task_id: Task UUID returned by task_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: {"state": "stopped"} or error.
        - isError: True when the remote tool call fails.
    """
    req = TaskStopRequest(**arguments)
    payload, error = _call_remote_tool("task_stop", {"task_id": req.task_id})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_task_download(arguments: dict[str, Any]) -> CallToolResult:
    """Download report/zip for a task from mcp_cloud and save it locally.

    Examples:
        - {"task_id": "uuid"} → download report (default)
        - {"task_id": "uuid", "artifact": "zip"} → download zip

    Args:
        - task_id: Task UUID returned by task_create.
        - artifact: Optional "report" or "zip".

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: metadata + saved_path or error.
        - isError: True when download fails or remote tool errors.
    """
    req = TaskDownloadRequest(**arguments)
    artifact = (req.artifact or "report").strip().lower()
    if artifact not in ("report", "zip"):
        artifact = "report"

    payload, error = _call_remote_tool(
        "task_file_info",
        {"task_id": req.task_id, "artifact": artifact},
    )
    if error:
        return _wrap_response({"error": error}, is_error=True)
    if not payload:
        return _wrap_response(payload)

    download_url = payload.get("download_url")
    if isinstance(download_url, str) and download_url.startswith("/"):
        download_url = urljoin(_get_download_base_url().rstrip("/") + "/", download_url.lstrip("/"))
    if not download_url:
        download_url = _derive_download_url(req.task_id, artifact)

    try:
        destination = _choose_output_path(req.task_id, download_url, artifact)
        downloaded_size = _download_to_path(download_url, destination)
    except Exception as exc:
        return _wrap_response(
            {"error": {"code": "DOWNLOAD_FAILED", "message": str(exc)}},
            is_error=True,
        )

    payload["download_url"] = download_url
    payload["saved_path"] = str(destination)

    sha256 = payload.get("sha256")
    if isinstance(sha256, str):
        actual_sha = _hash_sha256(destination.read_bytes())
        if sha256 != actual_sha:
            logger.warning("SHA256 mismatch for %s (expected %s, got %s)", destination, sha256, actual_sha)

    size_value = payload.get("download_size")
    if isinstance(size_value, (int, float)) and int(size_value) != downloaded_size:
        logger.warning(
            "Download size mismatch for %s (expected %s, got %s)",
            destination,
            size_value,
            downloaded_size,
        )

    return _wrap_response(payload)


TOOL_HANDLERS = {
    "task_create": handle_task_create,
    "task_status": handle_task_status,
    "task_stop": handle_task_stop,
    "task_download": handle_task_download,
    "prompt_examples": handle_prompt_examples,
}


async def main() -> None:
    logger.info("Starting PlanExe MCP local proxy using %s", _get_mcp_base_url())
    async with stdio_server() as streams:
        await mcp_local.run(
            streams[0],
            streams[1],
            mcp_local.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from mcp.server import Server
try:
    from mcp.server import request_ctx
except Exception:  # pragma: no cover - optional in older MCP SDKs
    request_ctx = None
try:
    from mcp.server.errors import InvalidParamsError
except Exception:  # pragma: no cover - optional in older MCP SDKs
    try:
        from mcp.server.exceptions import InvalidParamsError
    except Exception:  # pragma: no cover - optional in older MCP SDKs
        InvalidParamsError = None
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
    idea: str
    speed_vs_detail: Optional[SpeedVsDetailInput] = None


class PlanGenerateRequest(BaseModel):
    idea: str
    speed_vs_detail: Optional[SpeedVsDetailInput] = None
    idempotency_key: Optional[str] = None


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
) -> bytes:
    for _ in range(max_redirects + 1):
        request = Request(url, data=body, method=method, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                if not location:
                    raise
                url = urljoin(url, location)
                if exc.code == 303:
                    method = "GET"
                    body = None
                continue
            raise
    raise HTTPError(url, 310, "Too many redirects", None, None)


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


def _get_request_context() -> Optional[Any]:
    if request_ctx is not None:
        try:
            return request_ctx.get()
        except Exception:
            try:
                return request_ctx()
            except Exception:
                return None
    return getattr(mcp_local, "request_context", None)


def _extract_task_and_meta(arguments: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    task_hint: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = None
    ctx = _get_request_context()
    request_obj = None
    if ctx is not None:
        request_obj = getattr(ctx, "request", None) or getattr(ctx, "raw_request", None)
        if request_obj is None and isinstance(ctx, dict):
            request_obj = ctx.get("request") or ctx.get("raw_request")
    params_obj = None
    if request_obj is not None:
        params_obj = getattr(request_obj, "params", None)
        if params_obj is None and isinstance(request_obj, dict):
            params_obj = request_obj.get("params", request_obj)
    if params_obj is not None:
        if isinstance(params_obj, dict):
            task_hint = params_obj.get("task")
            meta = params_obj.get("_meta") or params_obj.get("meta")
        else:
            task_hint = getattr(params_obj, "task", None)
            meta = getattr(params_obj, "_meta", None) or getattr(params_obj, "meta", None)

    if task_hint is None:
        candidate = arguments.get("task")
        if candidate is not None:
            task_hint = candidate
    if meta is None:
        candidate = arguments.get("_meta") or arguments.get("meta")
        if isinstance(candidate, dict):
            meta = candidate
    return task_hint, meta


def _raise_invalid_params(message: str) -> None:
    if InvalidParamsError is not None:
        raise InvalidParamsError(message)
    raise ValueError(message)


def _call_remote_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    mcp_base_url = _get_mcp_base_url()
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": params,
    }
    return _http_json_request("POST", mcp_base_url, payload)


def _call_remote_tool_rpc_raw(
    tool: str,
    arguments: dict[str, Any],
    task: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    params: dict[str, Any] = {"name": tool, "arguments": arguments}
    if task is not None:
        params["task"] = task
    response = _call_remote_rpc("tools/call", params)
    error = response.get("error")
    if error:
        return {}, error
    result = response.get("result")
    return result if isinstance(result, dict) else {}, None


def _call_remote_tool_rpc(
    tool: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    result, error = _call_remote_tool_rpc_raw(tool, arguments)
    if error:
        return {}, error
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


def _call_remote_tool(
    tool: str,
    arguments: dict[str, Any],
    task: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    if task is not None:
        result, error = _call_remote_tool_rpc_raw(tool, arguments, task=task)
        if error:
            return {}, error
        return result, None

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


def _call_remote_tasks_method(
    method: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    response = _call_remote_rpc(method, params)
    error = response.get("error")
    if error:
        return {}, error
    result = response.get("result")
    return result if isinstance(result, dict) else {}, None


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
    task_support: Optional[str] = None


ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["code", "message"],
}

ARTIFACT_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string"},
        "sha256": {"type": "string"},
        "download_size": {"type": "integer"},
        "download_url": {"type": "string"},
    },
    "required": ["content_type", "sha256", "download_size"],
}

PLAN_GENERATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string"},
        "speed_vs_detail": {
            "type": "string",
            "enum": ["ping", "fast", "all"],
            "default": "ping",
        },
        "idempotency_key": {"type": "string"},
    },
    "required": ["idea"],
}

TASK_CREATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string"},
        "speed_vs_detail": {
            "type": "string",
            "enum": ["ping", "fast", "all"],
            "default": "ping",
        },
    },
    "required": ["idea"],
}

TASK_STATUS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"task_id": {"type": "string"}},
    "required": ["task_id"],
}

TASK_STOP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"task_id": {"type": "string"}},
    "required": ["task_id"],
}

TASK_DOWNLOAD_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "artifact": {"type": "string", "enum": ["report", "zip"], "default": "report"},
    },
    "required": ["task_id"],
}

PLAN_GENERATE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "progress_percentage": {"type": ["number", "null"]},
        "message": {"type": ["string", "null"]},
        "report": ARTIFACT_INFO_SCHEMA,
        "zip": ARTIFACT_INFO_SCHEMA,
        "error": ERROR_SCHEMA,
    },
    "required": ["task_id", "status"],
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
        name="plan_generate",
        description=(
            "Generate a new plan. Supports MCP task augmentation for long-running runs."
        ),
        input_schema=PLAN_GENERATE_INPUT_SCHEMA,
        output_schema=PLAN_GENERATE_OUTPUT_SCHEMA,
        task_support="optional",
    ),
    ToolDefinition(
        name="task_create",
        description=(
            "Start creating a new plan. Plan creation is long-running and "
            "typically takes 10-20 minutes to finish."
        ),
        input_schema=TASK_CREATE_INPUT_SCHEMA,
        output_schema=TASK_CREATE_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
    ToolDefinition(
        name="task_status",
        description="Returns status and progress of the plan currently being created.",
        input_schema=TASK_STATUS_INPUT_SCHEMA,
        output_schema=TASK_STATUS_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
    ToolDefinition(
        name="task_stop",
        description="Stops the plan that is currently being created.",
        input_schema=TASK_STOP_INPUT_SCHEMA,
        output_schema=TASK_STOP_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
    ToolDefinition(
        name="task_download",
        description="Download report or zip for a task and save it locally.",
        input_schema=TASK_DOWNLOAD_INPUT_SCHEMA,
        output_schema=TASK_DOWNLOAD_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
]

mcp_local = Server("planexe-mcp-local")


@mcp_local.list_tools()
async def handle_list_tools() -> list[Tool]:
    tools: list[Tool] = []
    for definition in TOOL_DEFINITIONS:
        execution = None
        if definition.task_support:
            execution = {"taskSupport": definition.task_support}
        tool_kwargs = {
            "name": definition.name,
            "description": definition.description,
            "inputSchema": definition.input_schema,
            "outputSchema": definition.output_schema,
        }
        if execution:
            try:
                tools.append(Tool(**tool_kwargs, execution=execution))
                continue
            except TypeError:
                logger.debug("Tool execution field not supported by MCP SDK.")
        tools.append(Tool(**tool_kwargs))
    return tools


def _wrap_response(
    payload: dict[str, Any],
    is_error: Optional[bool] = None,
    meta: Optional[dict[str, Any]] = None,
) -> CallToolResult:
    if is_error is None:
        is_error = isinstance(payload.get("error"), dict)
    kwargs = {
        "content": [TextContent(type="text", text=json.dumps(payload))],
        "structuredContent": payload,
        "isError": is_error,
    }
    if meta:
        model_fields = getattr(CallToolResult, "model_fields", None)
        if isinstance(model_fields, dict) and "meta" in model_fields:
            kwargs["meta"] = meta
        elif isinstance(model_fields, dict) and "_meta" in model_fields:
            kwargs["_meta"] = meta
        else:
            legacy_fields = getattr(CallToolResult, "__fields__", {})
            if isinstance(legacy_fields, dict) and "meta" in legacy_fields:
                kwargs["meta"] = meta
            elif isinstance(legacy_fields, dict) and "_meta" in legacy_fields:
                kwargs["_meta"] = meta
    return CallToolResult(**kwargs)


def _extract_task_id(params: Any) -> Optional[str]:
    if isinstance(params, str):
        return params
    if isinstance(params, dict):
        return params.get("taskId") or params.get("task_id")
    task_id = getattr(params, "taskId", None) or getattr(params, "task_id", None)
    if isinstance(task_id, str):
        return task_id
    return None


def _extract_list_params(params: Any) -> tuple[Optional[str], Optional[int]]:
    if isinstance(params, dict):
        return params.get("cursor"), params.get("limit")
    cursor = getattr(params, "cursor", None)
    limit = getattr(params, "limit", None)
    return cursor, limit


@mcp_local.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult | dict[str, Any]:
    task_hint, meta = _extract_task_and_meta(arguments)
    cleaned_arguments = {
        key: value
        for key, value in (arguments or {}).items()
        if key not in ("task", "_meta", "meta")
    }
    result = await _dispatch_tool_call(name, cleaned_arguments, task_hint, meta)
    if isinstance(result, CallToolResult):
        return result
    return result


async def _dispatch_tool_call(
    name: str,
    arguments: dict[str, Any],
    task_hint: Optional[dict[str, Any]],
    meta: Optional[dict[str, Any]],
) -> CallToolResult | dict[str, Any]:
    if name == "plan_generate":
        return await handle_plan_generate(arguments, task_hint, meta)

    if task_hint is not None:
        return _wrap_response(
            {
                "error": {
                    "code": "TASKS_NOT_SUPPORTED",
                    "message": f"Tool does not support task execution: {name}",
                }
            },
            is_error=True,
        )

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return _wrap_response(
            {"error": {"code": "INVALID_TOOL", "message": f"Unknown tool: {name}"}},
            is_error=True,
        )
    return await handler(arguments)


async def handle_plan_generate(
    arguments: dict[str, Any],
    task_hint: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> CallToolResult | dict[str, Any]:
    """Generate a plan via the remote MCP server."""
    req = PlanGenerateRequest(**arguments)
    payload_args: dict[str, Any] = {
        "idea": req.idea,
        "speed_vs_detail": req.speed_vs_detail or "ping",
    }
    if req.idempotency_key:
        payload_args["idempotency_key"] = req.idempotency_key

    if task_hint is not None:
        payload, error = _call_remote_tool("plan_generate", payload_args, task=task_hint)
        if error:
            return _wrap_response({"error": error}, is_error=True)
        return payload

    payload, error = _call_remote_tool("plan_generate", payload_args)
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_tasks_get(params: Any) -> dict[str, Any]:
    task_id = _extract_task_id(params)
    if not task_id:
        _raise_invalid_params("taskId is required.")
    payload, error = _call_remote_tasks_method("tasks/get", {"taskId": task_id})
    if error:
        _raise_invalid_params(error.get("message", "Remote error"))
    return payload


async def handle_tasks_result(params: Any) -> dict[str, Any]:
    task_id = _extract_task_id(params)
    if not task_id:
        _raise_invalid_params("taskId is required.")
    payload, error = _call_remote_tasks_method("tasks/result", {"taskId": task_id})
    if error:
        _raise_invalid_params(error.get("message", "Remote error"))
    return payload


async def handle_tasks_cancel(params: Any) -> dict[str, Any]:
    task_id = _extract_task_id(params)
    if not task_id:
        _raise_invalid_params("taskId is required.")
    payload, error = _call_remote_tasks_method("tasks/cancel", {"taskId": task_id})
    if error:
        _raise_invalid_params(error.get("message", "Remote error"))
    return payload


async def handle_tasks_list(params: Any) -> dict[str, Any]:
    cursor, limit_value = _extract_list_params(params)
    request_params: dict[str, Any] = {}
    if cursor:
        request_params["cursor"] = cursor
    if isinstance(limit_value, (int, float)) and int(limit_value) > 0:
        request_params["limit"] = int(limit_value)
    payload, error = _call_remote_tasks_method("tasks/list", request_params)
    if error:
        _raise_invalid_params(error.get("message", "Remote error"))
    return payload


def _register_mcp_method(server: Server, name: str, handler: Any) -> None:
    registrar = getattr(server, "register_method", None) or getattr(server, "method", None)
    if registrar is None:
        logger.warning("MCP SDK does not support method registration; %s disabled.", name)
        return
    registrar(name)(handler)


async def handle_task_create(arguments: dict[str, Any]) -> CallToolResult:
    """Create a task in mcp_cloud via the local HTTP proxy.

    Examples:
        - {"idea": "Write a market research plan"} → task_id + created_at
        - {"idea": "Generate onboarding plan", "speed_vs_detail": "fast"}

    Args:
        - idea: Prompt/goal for the plan.
        - speed_vs_detail: Optional mode ("ping" | "fast" | "all").

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: task_id/created_at payload or error.
        - isError: True when the remote tool call fails.
    """
    req = TaskCreateRequest(**arguments)
    payload, error = _call_remote_tool(
        "task_create",
        {"idea": req.idea, "speed_vs_detail": req.speed_vs_detail} if req.speed_vs_detail else {"idea": req.idea},
    )
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
}

_register_mcp_method(mcp_local, "tasks/get", handle_tasks_get)
_register_mcp_method(mcp_local, "tasks/result", handle_tasks_result)
_register_mcp_method(mcp_local, "tasks/cancel", handle_tasks_cancel)
_register_mcp_method(mcp_local, "tasks/list", handle_tasks_list)

def _apply_tasks_capability(initialization_options: Any) -> Any:
    tasks_capability = {
        "requests": True,
        "cancel": True,
        "list": True,
    }
    try:
        capabilities = getattr(initialization_options, "capabilities", None)
        if capabilities is None and isinstance(initialization_options, dict):
            capabilities = initialization_options.get("capabilities")
        if isinstance(capabilities, dict):
            capabilities["tasks"] = tasks_capability
        elif capabilities is not None:
            try:
                setattr(capabilities, "tasks", tasks_capability)
            except Exception:
                if hasattr(capabilities, "model_copy"):
                    initialization_options.capabilities = capabilities.model_copy(
                        update={"tasks": tasks_capability}
                    )
    except Exception as exc:
        logger.warning("Unable to apply tasks capability: %s", exc)
    return initialization_options


async def main() -> None:
    logger.info("Starting PlanExe MCP local proxy using %s", _get_mcp_base_url())
    async with stdio_server() as streams:
        init_options = _apply_tasks_capability(mcp_local.create_initialization_options())
        await mcp_local.run(
            streams[0],
            streams[1],
            init_options,
        )


if __name__ == "__main__":
    asyncio.run(main())

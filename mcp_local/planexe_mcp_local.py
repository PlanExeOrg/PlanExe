"""
PlanExe MCP local proxy.

Runs locally over stdio and forwards tool calls to mcp_cloud, the MCP server
running in the cloud.
Downloads artifacts to disk for plan_download.
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
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://your-railway-app.up.railway.app/mcp"
REPORT_FILENAME = "030-report.html"
ZIP_FILENAME = "run.zip"
ModelProfileInput = Literal[
    "baseline",
    "premium",
    "frontier",
    "custom",
]


class PlanCreateRequest(BaseModel):
    prompt: str
    model_profile: Optional[ModelProfileInput] = None
    user_api_key: Optional[str] = None


class PlanStatusRequest(BaseModel):
    plan_id: str


class PlanStopRequest(BaseModel):
    plan_id: str


class PlanRetryRequest(BaseModel):
    plan_id: str
    model_profile: ModelProfileInput = "baseline"


class PlanResumeRequest(BaseModel):
    plan_id: str
    model_profile: ModelProfileInput = "baseline"


class PlanDownloadRequest(BaseModel):
    plan_id: str
    artifact: str = "report"


class PlanListRequest(BaseModel):
    user_api_key: Optional[str] = None
    limit: int = 10


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
        headers["X-API-Key"] = api_key
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


def _derive_download_url(plan_id: str, artifact: str) -> str:
    if artifact == "zip":
        path = f"/download/{plan_id}/{ZIP_FILENAME}"
    else:
        path = f"/download/{plan_id}/{REPORT_FILENAME}"
    return urljoin(_get_download_base_url().rstrip("/") + "/", path.lstrip("/"))


def _ensure_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"PLANEXE_PATH is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _choose_output_path(plan_id: str, download_url: str, artifact: str) -> Path:
    base_path = Path(_get_env("PLANEXE_PATH", str(Path.cwd()))).expanduser()
    _ensure_directory(base_path)

    basename = Path(urlparse(download_url).path).name
    if not basename:
        basename = REPORT_FILENAME if artifact == "report" else ZIP_FILENAME
    filename = f"{plan_id}-{basename}"
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
    annotations: Optional[dict[str, Any]] = None


ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "details": {"type": ["object", "null"]},
    },
    "required": ["code", "message"],
}

PLAN_CREATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "What the plan should cover. Good prompts are often 300–800 words. "
                "Use example_prompts to get example prompts; use these as examples for plan_create. "
                "Good prompt shape: objective, scope, constraints, timeline, stakeholders, "
                "budget/resources, and success criteria. "
                "Write as flowing prose, not structured markdown. Include banned approaches, "
                "governance preferences, and phasing inline. "
                "Short prompts produce less detailed plans. "
                "Do not use plan_create for tiny one-shot outputs (e.g., a 5-point checklist)."
            ),
        },
        "model_profile": {
            "type": "string",
            "enum": ["baseline", "premium", "frontier", "custom"],
            "default": "baseline",
            "description": (
                "Model profile selection: baseline (cheap/fast), premium (higher quality), "
                "frontier (most capable), custom (user-defined). Call model_profiles for runtime availability."
            ),
        },
        "user_api_key": {
            "type": ["string", "null"],
            "default": None,
            "description": "Optional user API key for credits and attribution.",
        },
    },
    "required": ["prompt"],
}

PLAN_STATUS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {
            "type": "string",
            "description": "UUID of the plan (returned by plan_create).",
        },
    },
    "required": ["plan_id"],
}

PLAN_STOP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {
            "type": "string",
            "description": "UUID of the plan to stop (returned by plan_create).",
        },
    },
    "required": ["plan_id"],
}

PLAN_RETRY_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {
            "type": "string",
            "description": "UUID of the failed plan to retry.",
        },
        "model_profile": {
            "type": "string",
            "enum": ["baseline", "premium", "frontier", "custom"],
            "default": "baseline",
            "description": "Model profile used for retry. Defaults to baseline.",
        },
    },
    "required": ["plan_id"],
}

PLAN_DOWNLOAD_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {
            "type": "string",
            "description": "UUID of the plan (returned by plan_create).",
        },
        "artifact": {
            "type": "string",
            "enum": ["report", "zip"],
            "default": "report",
            "description": "What to download: 'report' = HTML report, 'zip' = full output bundle.",
        },
    },
    "required": ["plan_id"],
}

EXAMPLE_PROMPTS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}
EXAMPLE_PROMPTS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Example prompts to copy or adapt when calling plan_create.",
        },
        "message": {"type": "string"},
    },
    "required": ["samples", "message"],
}
MODEL_PROFILES_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}
MODEL_PROFILES_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "default_profile": {
            "type": "string",
            "enum": ["baseline", "premium", "frontier", "custom"],
        },
        "profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "enum": ["baseline", "premium", "frontier", "custom"],
                    },
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "model_count": {"type": "integer"},
                    "models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "provider_class": {"type": ["string", "null"]},
                                "model": {"type": ["string", "null"]},
                                "priority": {"type": ["integer", "null"]},
                            },
                            "required": ["key"],
                        },
                    },
                },
                "required": [
                    "profile",
                    "title",
                    "summary",
                    "model_count",
                    "models",
                ],
            },
        },
        "message": {"type": "string"},
    },
    "required": [
        "default_profile",
        "profiles",
        "message",
    ],
}

EXAMPLE_PLANS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}
EXAMPLE_PLANS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title describing the example plan."},
                    "report_url": {"type": "string", "description": "URL to the static HTML report."},
                    "zip_url": {"type": "string", "description": "URL to the zip bundle."},
                },
                "required": ["title", "report_url", "zip_url"],
            },
            "description": "Curated example plans with download links for reports and zip bundles.",
        },
        "message": {"type": "string"},
    },
    "required": ["plans", "message"],
}

PLAN_CREATE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "created_at": {"type": "string"},
    },
    "required": ["plan_id", "created_at"],
}

PLAN_STATUS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {"type": ["string", "null"]},
        "state": {"type": ["string", "null"]},
        "progress_percentage": {"type": ["number", "null"]},
        "timing": {
            "type": ["object", "null"],
            "properties": {
                "started_at": {"type": ["string", "null"]},
                "elapsed_sec": {"type": "number"},
            },
        },
        "files": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "updated_at": {"type": "string"},
                },
            },
        },
    },
}

PLAN_STOP_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "stop_requested": {"type": "boolean"},
        "error": ERROR_SCHEMA,
    },
}

PLAN_RETRY_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "state": {"type": "string"},
        "model_profile": {
            "type": "string",
            "enum": ["baseline", "premium", "frontier", "custom"],
        },
        "retried_at": {"type": "string"},
        "error": ERROR_SCHEMA,
    },
}

PLAN_RESUME_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {
            "type": "string",
            "description": "UUID of the failed plan to resume.",
        },
        "model_profile": {
            "type": "string",
            "enum": ["baseline", "premium", "frontier", "custom"],
            "default": "baseline",
            "description": "Model profile used for the resumed run. Defaults to baseline.",
        },
    },
    "required": ["plan_id"],
}

PLAN_RESUME_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "state": {"type": "string"},
        "model_profile": {
            "type": "string",
            "enum": ["baseline", "premium", "frontier", "custom"],
        },
        "resume_count": {"type": "integer"},
        "resumed_at": {"type": "string"},
        "error": ERROR_SCHEMA,
    },
}

PLAN_DOWNLOAD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string", "description": "Artifact content type."},
        "sha256": {"type": "string", "description": "SHA-256 hash of artifact bytes."},
        "download_size": {"type": "integer", "description": "Artifact size in bytes."},
        "download_url": {"type": "string", "description": "Remote URL used for download."},
        "saved_path": {
            "type": "string",
            "description": "Local file path written by plan_download.",
        },
        "error": ERROR_SCHEMA,
    },
    "additionalProperties": False,
}

PLAN_LIST_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_api_key": {
            "type": ["string", "null"],
            "default": None,
            "description": "Optional user API key for credits and attribution.",
        },
        "limit": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 50,
            "description": "Maximum number of plans to return (1-50). Newest plans are returned first.",
        },
    },
    "required": [],
}
PLAN_LIST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "state": {"type": "string"},
                    "progress_percentage": {"type": "number"},
                    "created_at": {"type": "string"},
                    "prompt_excerpt": {"type": "string"},
                },
            },
            "description": "Plans for the authenticated user, newest first.",
        },
        "message": {"type": "string"},
        "error": ERROR_SCHEMA,
    },
    "additionalProperties": False,
}

TOOL_DEFINITIONS = [
    ToolDefinition(
        name="example_plans",
        description=(
            "Returns a curated list of example plans with download links for reports and zip bundles. "
            "Use this to preview what PlanExe output looks like before creating your own plan. "
            "Especially useful when the user asks what the output looks like before committing to a plan. "
            "No API key required."
        ),
        input_schema=EXAMPLE_PLANS_INPUT_SCHEMA,
        output_schema=EXAMPLE_PLANS_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="example_prompts",
        description=(
            "Call this first. Returns example prompts that define what a good prompt looks like. "
            "Do NOT call plan_create yet. Optional before plan_create: call model_profiles to choose model_profile. "
            "Next is a non-tool step: formulate a prompt (use examples as a baseline, similar structure) and get user approval. "
            "Good prompt shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. "
            "Write the prompt as flowing prose, not structured markdown with headers or bullet lists. "
            "Weave technical specs, constraints, and targets naturally into sentences. Include banned words/approaches and governance preferences inline. "
            "The examples demonstrate this prose style — match their tone and density. "
            "Then call plan_create. "
            "PlanExe is not for tiny one-shot outputs like a 5-point checklist; and it does not support selecting only some internal pipeline steps."
        ),
        input_schema=EXAMPLE_PROMPTS_INPUT_SCHEMA,
        output_schema=EXAMPLE_PROMPTS_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="model_profiles",
        description=(
            "Optional helper before plan_create. Returns model_profile options with plain-language guidance "
            "and currently available models in each profile. "
            "If no models are available, returns error code MODEL_PROFILES_UNAVAILABLE."
        ),
        input_schema=MODEL_PROFILES_INPUT_SCHEMA,
        output_schema=MODEL_PROFILES_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_create",
        description=(
            "Call only after example_prompts and after you have completed prompt drafting/approval (non-tool step). "
            "PlanExe turns the approved prompt into a strategic project-plan draft (20+ sections) in ~10-20 min. "
            "Sections include: executive summary, interactive Gantt charts, investor pitch, project plan with SMART criteria, "
            "strategic decision analysis, scenario comparison, assumptions with expert review, governance structure, "
            "SWOT analysis, team role profiles, simulated expert criticism, work breakdown structure, "
            "plan review (critical issues, KPIs, financial strategy, automation opportunities), Q&A, "
            "premortem with failure scenarios, self-audit checklist, and adversarial premise attacks that argue against the project. "
            "The adversarial sections (premortem, self-audit, premise attacks) surface risks and questions the prompter may not have considered. "
            "Returns plan_id (UUID); use it for plan_status, plan_stop, plan_retry, and plan_download. "
            "If you lose a plan_id, call plan_list to recover it. "
            "Each plan_create call creates a new plan_id (proxied to cloud; no server-side dedup). "
            "If you are unsure which model_profile to choose, call model_profiles first. "
            "If your deployment uses credits, include user_api_key to charge the correct account. "
            "Common proxied error codes: INVALID_USER_API_KEY, USER_API_KEY_REQUIRED, INSUFFICIENT_CREDITS, REMOTE_ERROR."
        ),
        input_schema=PLAN_CREATE_INPUT_SCHEMA,
        output_schema=PLAN_CREATE_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_status",
        description=(
            "Returns status and progress of the plan currently being created. "
            "Poll at reasonable intervals only (e.g. every 5 minutes): plan generation typically takes 10-20 minutes "
            "(baseline profile) and may take longer on higher-quality profiles. "
            "State contract: pending/processing => keep polling; completed => download is ready; failed => terminal error. "
            "progress_percentage is 0-100 (integer-like float); 100 when completed. "
            "files lists intermediate outputs produced so far; use their updated_at timestamps to detect stalls. "
            "Unknown plan_id returns PLAN_NOT_FOUND (or REMOTE_ERROR when transport fails). "
            "Troubleshooting: pending for >5 minutes likely means queued but not picked up by a worker. "
            "processing with no file-output changes for >20 minutes likely means failed/stalled. "
            "Report these issues to https://github.com/PlanExeOrg/PlanExe/issues ."
        ),
        input_schema=PLAN_STATUS_INPUT_SCHEMA,
        output_schema=PLAN_STATUS_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_stop",
        description=(
            "Request the plan generation to stop. Pass the plan_id (the UUID returned by plan_create). "
            "Stopping is asynchronous: the stop flag is set immediately but the plan may continue briefly before halting. "
            "A stopped plan will eventually transition to the failed state. "
            "If the plan is already completed or failed, stop_requested returns false (the plan already finished). "
            "Unknown plan_id returns PLAN_NOT_FOUND (or REMOTE_ERROR when transport fails)."
        ),
        input_schema=PLAN_STOP_INPUT_SCHEMA,
        output_schema=PLAN_STOP_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_retry",
        description=(
            "Retry a plan that is currently in failed state. "
            "Pass the failed plan_id and optionally model_profile (defaults to baseline). "
            "The same plan_id is requeued and reset to pending on the cloud service. "
            "Unknown plan_id returns PLAN_NOT_FOUND; non-failed plans return PLAN_NOT_FAILED."
        ),
        input_schema=PLAN_RETRY_INPUT_SCHEMA,
        output_schema=PLAN_RETRY_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_resume",
        description=(
            "Resume a failed plan without discarding completed intermediary files. "
            "Plan generation restarts from the first incomplete step, skipping all steps that already produced output files. "
            "Use plan_resume when plan_status shows 'failed' and plan generation was interrupted before completing all steps "
            "(network drop, timeout, plan_stop, worker crash). "
            "For a full restart or to change model_profile, use plan_retry instead. "
            "Only failed plans can be resumed. "
            "Returns PLAN_NOT_FOUND when plan_id is unknown and PLAN_NOT_RESUMABLE when the plan is not in failed state. "
            "Returns PIPELINE_VERSION_MISMATCH when the snapshot was created by a different pipeline version; use plan_retry instead."
        ),
        input_schema=PLAN_RESUME_INPUT_SCHEMA,
        output_schema=PLAN_RESUME_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_download",
        description=(
            "Download the plan output and save it locally to PLANEXE_PATH. "
            "Use artifact='report' (default) for the interactive HTML report (~700KB, self-contained with embedded JS "
            "for collapsible sections and interactive Gantt charts — open in a browser). "
            "Use artifact='zip' for the full pipeline output bundle (md, json, csv intermediary files that fed the report). "
            "If PLANEXE_PATH is unset, files are saved to the current working directory. "
            "Filename format is <plan_id>-<artifact_name> with numeric suffixes when collisions occur. "
            "Common local error codes: DOWNLOAD_FAILED, REMOTE_ERROR."
        ),
        input_schema=PLAN_DOWNLOAD_INPUT_SCHEMA,
        output_schema=PLAN_DOWNLOAD_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_list",
        description=(
            "List the most recent plans for an authenticated user. "
            "Returns up to `limit` plans (default 10, max 50) newest-first, each with plan_id, state, "
            "progress_percentage, created_at (ISO 8601), and a prompt_excerpt (first 100 chars). "
            "Use this to recover a lost plan_id or to review recent activity."
        ),
        input_schema=PLAN_LIST_INPUT_SCHEMA,
        output_schema=PLAN_LIST_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
]

# Shown in MCP initialize response (e.g. Inspector) so clients know what PlanExe is.
PLANEXE_SERVER_INSTRUCTIONS = (
    "PlanExe generates strategic project-plan drafts from a natural-language prompt. "
    "Output is a self-contained interactive HTML report (~700KB) with 20+ sections including "
    "executive summary, interactive Gantt charts, risk analysis, SWOT, governance, investor pitch, "
    "team profiles, work breakdown, scenario comparison, expert criticism, and adversarial sections "
    "(premortem, self-audit checklist, premise attacks) that stress-test whether the plan holds up. "
    "The output is a draft to refine, not final ground truth — but it surfaces hard questions the prompter may not have considered. "
    "Use PlanExe for substantial multi-phase projects with constraints, stakeholders, budgets, and timelines. "
    "Do not use PlanExe for tiny one-shot outputs (for example: 'give me a 5-point checklist'); use a normal LLM response for that. "
    "The planning pipeline is fixed end-to-end; callers cannot select individual internal pipeline steps to run. "
    "Required interaction order: call example_plans first (optional, to preview what PlanExe output looks like — curated example reports and zip bundles). "
    "Then call example_prompts. "
    "Optional before plan_create: call model_profiles to see profile guidance and available models in each profile. "
    "Then perform a non-tool step: draft a strong prompt as flowing prose (not structured markdown with headers or bullets), "
    "typically ~300-800 words, and get user approval. "
    "Good prompt shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. "
    "Write the prompt as flowing prose — weave specs, constraints, and targets naturally into sentences. "
    "Only after approval, call plan_create. "
    "Each plan_create call creates a new plan_id; the server does not enforce a global per-client concurrency limit. "
    "Then poll plan_status (about every 5 minutes); use plan_download when complete. "
    "If plan generation fails before completing all steps, call plan_resume to continue from where it left off without discarding completed work. "
    "Use plan_retry instead for a full restart. "
    "Both accept the failed plan_id and optional model_profile (defaults to baseline). "
    "To stop, call plan_stop with the plan_id from plan_create; stopping is asynchronous and the plan will eventually transition to failed. "
    "If model_profiles returns MODEL_PROFILES_UNAVAILABLE, inform the user that no models are currently configured and the server administrator needs to set up model profiles. "
    "Tool errors use {error:{code,message}}. plan_download may return REMOTE_ERROR or DOWNLOAD_FAILED. "
    "plan_download saves to PLANEXE_PATH (default: current working directory) and returns saved_path. "
    "To list recent plans for a user call plan_list; returns plan_id, state, progress_percentage, created_at, and prompt_excerpt. "
    "plan_status state contract: pending/processing => keep polling; completed => download is ready; failed => terminal error. "
    "Troubleshooting: if plan_status stays in pending for longer than 5 minutes, the plan was likely queued but not picked up by a worker (server issue). "
    "If plan_status is in processing and output files do not change for longer than 20 minutes, the run likely failed/stalled. "
    "In both cases, report the issue to PlanExe developers on GitHub: https://github.com/PlanExeOrg/PlanExe/issues . "
    "Main output: a self-contained interactive HTML report (~700KB) with collapsible sections and interactive Gantt charts — open in a browser. "
    "The zip contains the intermediary pipeline files (md, json, csv) that fed the report. "
    "New users: create an account and obtain an API key at https://home.planexe.org/ ."
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
            annotations=ToolAnnotations(**definition.annotations) if definition.annotations else None,
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


async def handle_plan_create(arguments: dict[str, Any]) -> CallToolResult:
    """Create a plan in mcp_cloud via the local HTTP proxy.

    Examples:
        - {"prompt": "Start a dental clinic in Copenhagen with 3 treatment rooms, targeting families and children. Budget 2.5M DKK. Open within 12 months."} → plan_id + created_at

    Args:
        - prompt: What the plan should cover (goal, context, constraints).
        - model_profile: Optional profile ("baseline" | "premium" | "frontier" | "custom"). Call model_profiles to inspect options.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: plan_id/created_at payload or error.
        - isError: True when the remote tool call fails.
    """
    req = PlanCreateRequest(**arguments)
    payload: dict[str, Any] = {"prompt": req.prompt}
    if req.model_profile:
        payload["model_profile"] = req.model_profile
    if req.user_api_key:
        payload["user_api_key"] = req.user_api_key

    metadata = arguments.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = metadata

    payload, error = _call_remote_tool(
        "plan_create",
        payload,
    )
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_example_prompts(arguments: dict[str, Any]) -> CallToolResult:
    """Return curated prompts from mcp_cloud so LLMs can see example detail."""
    payload, error = _call_remote_tool("example_prompts", arguments or {})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_model_profiles(arguments: dict[str, Any]) -> CallToolResult:
    """Return model_profile options and available models from mcp_cloud."""
    payload, error = _call_remote_tool("model_profiles", arguments or {})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_example_plans(arguments: dict[str, Any]) -> CallToolResult:
    """Return curated example plans with download links from mcp_cloud."""
    payload, error = _call_remote_tool("example_plans", arguments or {})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_plan_status(arguments: dict[str, Any]) -> CallToolResult:
    """Fetch status/progress for a plan from mcp_cloud.

    Examples:
        - {"plan_id": "uuid"} → state/progress/timing

    Args:
        - plan_id: Plan UUID returned by plan_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: status payload or error.
        - isError: True when the remote tool call fails.
    """
    req = PlanStatusRequest(**arguments)
    payload, error = _call_remote_tool("plan_status", {"plan_id": req.plan_id})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_plan_stop(arguments: dict[str, Any]) -> CallToolResult:
    """Request mcp_cloud to stop an active plan.

    Examples:
        - {"plan_id": "uuid"} → stop request acknowledged

    Args:
        - plan_id: Plan UUID returned by plan_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: {"state": "pending|processing|completed|failed", "stop_requested": bool} or error.
        - isError: True when the remote tool call fails.
    """
    req = PlanStopRequest(**arguments)
    payload, error = _call_remote_tool("plan_stop", {"plan_id": req.plan_id})
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_plan_retry(arguments: dict[str, Any]) -> CallToolResult:
    """Request mcp_cloud to retry a failed plan."""
    req = PlanRetryRequest(**arguments)
    payload, error = _call_remote_tool(
        "plan_retry",
        {"plan_id": req.plan_id, "model_profile": req.model_profile},
    )
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_plan_resume(arguments: dict[str, Any]) -> CallToolResult:
    """Request mcp_cloud to resume a failed plan without discarding completed outputs."""
    req = PlanResumeRequest(**arguments)
    payload, error = _call_remote_tool(
        "plan_resume",
        {"plan_id": req.plan_id, "model_profile": req.model_profile},
    )
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


async def handle_plan_download(arguments: dict[str, Any]) -> CallToolResult:
    """Download report/zip for a plan from mcp_cloud and save it locally.

    Examples:
        - {"plan_id": "uuid"} → download report (default)
        - {"plan_id": "uuid", "artifact": "zip"} → download zip

    Args:
        - plan_id: Plan UUID returned by plan_create.
        - artifact: Optional "report" or "zip".

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: metadata + saved_path or error.
        - isError: True when download fails or remote tool errors.
    """
    req = PlanDownloadRequest(**arguments)
    artifact = (req.artifact or "report").strip().lower()
    if artifact not in ("report", "zip"):
        return _wrap_response(
            {"error": {"code": "INVALID_ARGUMENT", "message": f"Invalid artifact type: {req.artifact!r}. Must be 'report' or 'zip'."}},
            is_error=True,
        )

    payload, error = _call_remote_tool(
        "plan_file_info",
        {"plan_id": req.plan_id, "artifact": artifact},
    )
    if error:
        return _wrap_response({"error": error}, is_error=True)
    if not payload:
        return _wrap_response(payload)

    download_url = payload.get("download_url")
    if isinstance(download_url, str) and download_url.startswith("/"):
        download_url = urljoin(_get_download_base_url().rstrip("/") + "/", download_url.lstrip("/"))
    if not download_url:
        download_url = _derive_download_url(req.plan_id, artifact)

    try:
        destination = _choose_output_path(req.plan_id, download_url, artifact)
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


async def handle_plan_list(arguments: dict[str, Any]) -> CallToolResult:
    """List recent plans for an authenticated user via mcp_cloud."""
    req = PlanListRequest(**arguments)
    payload_args: dict[str, Any] = {"limit": req.limit}
    if req.user_api_key:
        payload_args["user_api_key"] = req.user_api_key
    payload, error = _call_remote_tool("plan_list", payload_args)
    if error:
        return _wrap_response({"error": error}, is_error=True)
    return _wrap_response(payload)


TOOL_HANDLERS = {
    "example_plans": handle_example_plans,
    "example_prompts": handle_example_prompts,
    "model_profiles": handle_model_profiles,
    "plan_create": handle_plan_create,
    "plan_status": handle_plan_status,
    "plan_stop": handle_plan_stop,
    "plan_retry": handle_plan_retry,
    "plan_resume": handle_plan_resume,
    "plan_download": handle_plan_download,
    "plan_list": handle_plan_list,
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

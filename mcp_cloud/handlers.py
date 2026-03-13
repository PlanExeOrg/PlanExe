"""PlanExe MCP Cloud – MCP tool handlers and dispatch."""
import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from mcp.types import CallToolResult, Tool, TextContent, ToolAnnotations
from worker_plan_api.format_datetime import format_datetime_utc

from mcp_cloud.db_setup import (
    PlanState,
    REPORT_CONTENT_TYPE,
    REPORT_FILENAME,
    ZIP_CONTENT_TYPE,
    ModelProfileInput,
    PlanCreateRequest,
    PlanStatusRequest,
    PlanStopRequest,
    PlanRetryRequest,
    PlanResumeRequest,
    PlanFileInfoRequest,
    PlanListRequest,
    ModelProfilesRequest,
    mcp_cloud_server,
)
from mcp_cloud.auth import _resolve_user_from_api_key
from mcp_cloud.db_queries import (
    _create_plan_sync,
    _get_plan_status_snapshot_sync,
    _request_plan_stop_sync,
    _retry_failed_plan_sync,
    _resume_plan_sync,
    _get_plan_for_report_sync,
    _list_plans_sync,
    get_plan_state_mapping,
    _extract_plan_create_metadata_overrides,
    _merge_plan_create_config,
)
from mcp_cloud.zip_utils import (
    list_files_from_zip_snapshot,
    compute_sha256,
)
from mcp_cloud.worker_fetchers import (
    fetch_artifact_from_worker_plan,
    fetch_file_list_from_worker_plan,
    list_files_from_local_run_dir,
    fetch_user_downloadable_zip,
)
from mcp_cloud.model_profiles import _get_model_profiles_sync
from mcp_cloud.download_tokens import build_report_download_url, build_zip_download_url, _get_download_base_url, DOWNLOAD_TOKEN_TTL_SECONDS
from mcp_cloud.example_prompts import _load_mcp_example_prompts
from mcp_cloud.schemas import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


@mcp_cloud_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name=definition.name,
            description=definition.description,
            outputSchema=definition.output_schema,
            inputSchema=definition.input_schema,
            annotations=ToolAnnotations(**definition.annotations) if definition.annotations else None,
        )
        for definition in TOOL_DEFINITIONS
    ]

@mcp_cloud_server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Dispatch MCP tool calls and return structured JSON errors for unknown tools."""
    start = time.monotonic()
    try:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            logger.warning("tool_call tool=%s result=unknown_tool", name)
            response = {"error": {"code": "INVALID_TOOL", "message": f"Unknown tool: {name}"}}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(response))],
                structuredContent=response,
                isError=True,
            )
        result = await handler(arguments)
        elapsed_ms = (time.monotonic() - start) * 1000
        if result.isError:
            logger.info("tool_call tool=%s result=error duration_ms=%.0f", name, elapsed_ms)
        else:
            logger.info("tool_call tool=%s result=ok duration_ms=%.0f", name, elapsed_ms)
        return result
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error("tool_call tool=%s result=exception duration_ms=%.0f error=%s", name, elapsed_ms, e, exc_info=True)
        response = {"error": {"code": "INTERNAL_ERROR", "message": str(e)}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

async def handle_plan_create(arguments: dict[str, Any]) -> CallToolResult:
    """Create a new PlanExe task and enqueue it for processing.

    Examples:
        - {"prompt": "Start a dental clinic in Copenhagen with 3 treatment rooms, targeting families and children. Budget 2.5M DKK. Open within 12 months."} → returns plan_id (UUID) + created_at

    Args:
        - prompt: What the plan should cover (goal, context, constraints).
        - model_profile: Optional profile ("baseline" | "premium" | "frontier" | "custom"). Call model_profiles to inspect options.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: {"plan_id": "<uuid>", "created_at": ...}
        - isError: False on success.
    """
    req = PlanCreateRequest(**arguments)
    metadata_overrides = _extract_plan_create_metadata_overrides(arguments)
    metadata_model_profile = metadata_overrides.get("model_profile")
    model_profile = req.model_profile
    if model_profile is None and isinstance(metadata_model_profile, str):
        model_profile = metadata_model_profile

    merged_config = _merge_plan_create_config(None, model_profile)
    require_user_key = os.environ.get("PLANEXE_MCP_REQUIRE_USER_KEY", "false").lower() in ("1", "true", "yes", "on")
    user_context = None
    if req.user_api_key:
        user_context = _resolve_user_from_api_key(req.user_api_key.strip())
        if not user_context:
            response = {"error": {"code": "INVALID_USER_API_KEY", "message": "Invalid user_api_key. Verify your key or create an account at https://home.planexe.org/"}}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(response))],
                structuredContent=response,
                isError=True,
            )
    elif require_user_key:
        response = {"error": {"code": "USER_API_KEY_REQUIRED", "message": "user_api_key is required for plan_create. Create an account at https://home.planexe.org/"}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    if user_context and float(user_context.get("credits_balance", 0.0)) <= 0.0:
        response = {"error": {"code": "INSUFFICIENT_CREDITS", "message": "Not enough credits. Top up at https://home.planexe.org/"}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    metadata = None
    if user_context:
        metadata = {"user_id": str(user_context["user_id"])}
        if user_context.get("api_key_id"):
            metadata["api_key_id"] = user_context["api_key_id"]
    response = await asyncio.to_thread(
        _create_plan_sync,
        req.prompt,
        merged_config,
        metadata,
    )
    base_url = _get_download_base_url()
    if base_url and response.get("plan_id"):
        response["sse_url"] = f"{base_url}/sse/plan/{response['plan_id']}"
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )


async def handle_example_prompts(arguments: dict[str, Any]) -> CallToolResult:
    """Return curated prompts from the catalog (mcp_example true) so LLMs can see example detail."""
    samples = _load_mcp_example_prompts()
    payload = {
        "samples": samples,
        "message": (
            "Next: complete the non-tool step by drafting a detailed prompt (typically ~300-800 words) using these as a baseline (similar structure), then get user approval. "
            "Good prompt shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. "
            "Write the prompt as flowing prose, not structured markdown with headers or bullet lists. "
            "Weave technical specs, constraints, and targets naturally into sentences. Include banned words/approaches and governance preferences inline. "
            "The examples demonstrate this prose style — match their tone and density. "
            "Only after approval, call plan_create. "
            "Do not use PlanExe for tiny one-shot requests (e.g., rewrite this email, summarize this document). "
            "PlanExe always runs the full fixed planning pipeline; callers cannot run only selected internal steps."
        ),
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


async def handle_example_plans(arguments: dict[str, Any]) -> CallToolResult:
    """Return a curated list of example plans with download links."""
    payload = {
        "plans": [
            {
                "title": "CBC Validation",
                "report_url": "https://planexe.org/20260114_cbc_validation_report.html",
                "zip_url": "https://planexe.org/20260114_cbc_validation.zip",
            },
            {
                "title": "Minecraft Escape",
                "report_url": "https://planexe.org/20251016_minecraft_escape_report.html",
                "zip_url": "https://planexe.org/20251016_minecraft_escape.zip",
            },
        ],
        "message": (
            "These are curated example plans showing what PlanExe output looks like. "
            "Open the report URLs in a browser to see the interactive HTML reports with "
            "collapsible sections and Gantt charts. The zip bundles contain the intermediary "
            "pipeline files (md, json, csv) that fed each report."
        ),
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


async def handle_model_profiles(arguments: dict[str, Any]) -> CallToolResult:
    """Return model profile options and currently available models in each profile."""
    _ = ModelProfilesRequest(**(arguments or {}))
    payload = await asyncio.to_thread(_get_model_profiles_sync)
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) == 0:
        response = {
            "error": {
                "code": "MODEL_PROFILES_UNAVAILABLE",
                "message": (
                    "No models are currently configured. "
                    "Inform the user that the server administrator needs to set up model profiles before plans can be created."
                ),
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


async def handle_plan_status(arguments: dict[str, Any]) -> CallToolResult:
    """Fetch the current plan status, progress, and recent files for a plan.

    Examples:
        - {"plan_id": "uuid"} → state/progress/timing + recent files

    Args:
        - plan_id: Plan UUID returned by plan_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: status payload or error.
        - isError: True only when plan_id is unknown.
    """
    req = PlanStatusRequest(**arguments)
    plan_id = req.plan_id

    plan_snapshot = await asyncio.to_thread(_get_plan_status_snapshot_sync, plan_id)
    if plan_snapshot is None:
        response = {
            "error": {
                "code": "PLAN_NOT_FOUND",
                "message": f"Plan not found: {plan_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    progress_percentage = float(plan_snapshot.get("progress_percentage") or 0.0)

    plan_state = plan_snapshot["state"]
    state = get_plan_state_mapping(plan_state)
    # Normalize completed plans: state is the source of truth, so progress
    # fields must agree even if the final progress DB write was lost.
    if plan_state == PlanState.completed:
        progress_percentage = 100.0

    # Collect files — try fast local sources first, then fall back to worker HTTP.
    plan_uuid = plan_snapshot["id"]
    files = []
    if plan_uuid:
        files_list = await asyncio.to_thread(list_files_from_zip_snapshot, plan_uuid)
        if not files_list:
            files_list = await asyncio.to_thread(list_files_from_local_run_dir, plan_uuid)
        if not files_list:
            try:
                files_list = await asyncio.wait_for(
                    fetch_file_list_from_worker_plan(plan_uuid), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Worker file list fetch timed out for plan %s", plan_uuid)
                files_list = None
        if files_list:
            for file_name, updated_at in files_list:
                if file_name != "log.txt":
                    files.append({
                        "path": file_name,
                        "updated_at": updated_at,
                    })

    steps_completed = plan_snapshot.get("steps_completed")
    steps_total = plan_snapshot.get("steps_total")
    current_step = plan_snapshot.get("current_step")
    if plan_state == PlanState.completed and steps_total is not None:
        steps_completed = steps_total

    created_at = plan_snapshot["timestamp_created"]
    if created_at and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    response = {
        "plan_id": plan_uuid,
        "state": state,
        "progress_percentage": progress_percentage,
        "steps_completed": steps_completed,
        "steps_total": steps_total,
        "current_step": current_step,
        "timing": {
            "started_at": format_datetime_utc(created_at) if created_at else None,
            "elapsed_sec": (datetime.now(UTC) - created_at).total_seconds() if created_at else 0,
        },
        "files_count": len(files),
        "files": files[-10:],
    }

    if state == "failed":
        progress_msg = plan_snapshot.get("progress_message")
        response["error"] = {
            "failure_reason": plan_snapshot.get("failure_reason"),
            "failed_step": plan_snapshot.get("failed_step"),
            "message": plan_snapshot.get("error_message") or progress_msg or "Plan generation failed.",
            "recoverable": plan_snapshot.get("recoverable"),
        }

    if state not in ("completed", "failed", "stopped"):
        base_url = _get_download_base_url()
        if base_url:
            response["sse_url"] = f"{base_url}/sse/plan/{plan_uuid}"

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )

async def handle_plan_stop(arguments: dict[str, Any]) -> CallToolResult:
    """Request an active plan to stop.

    Examples:
        - {"plan_id": "uuid"} → stop request accepted

    Args:
        - plan_id: Plan UUID returned by plan_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: {"state": "pending|processing|completed|failed", "stop_requested": bool} or error payload.
        - isError: True only when plan_id is unknown.
    """
    req = PlanStopRequest(**arguments)
    plan_id = req.plan_id

    stop_result = await asyncio.to_thread(_request_plan_stop_sync, plan_id)
    if stop_result is None:
        response = {
            "error": {
                "code": "PLAN_NOT_FOUND",
                "message": f"Plan not found: {plan_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    response = stop_result

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )


async def handle_plan_retry(arguments: dict[str, Any]) -> CallToolResult:
    """Retry a failed plan by resetting it back to pending."""
    req = PlanRetryRequest(**arguments)
    plan_id = req.plan_id

    # Resolve caller identity so the retried plan is attributed to the new key.
    user_api_key = arguments.get("user_api_key")
    caller_metadata: Optional[dict[str, str]] = None
    if user_api_key:
        user_context = _resolve_user_from_api_key(user_api_key.strip())
        if user_context:
            caller_metadata = {"user_id": str(user_context["user_id"])}
            if user_context.get("api_key_id"):
                caller_metadata["api_key_id"] = user_context["api_key_id"]

    retry_result = await asyncio.to_thread(_retry_failed_plan_sync, plan_id, req.model_profile, caller_metadata)

    if retry_result is None:
        response = {
            "error": {
                "code": "PLAN_NOT_FOUND",
                "message": f"Plan not found: {plan_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    if isinstance(retry_result.get("error"), dict):
        response = retry_result
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    response = retry_result
    base_url = _get_download_base_url()
    if base_url and response.get("plan_id"):
        response["sse_url"] = f"{base_url}/sse/plan/{response['plan_id']}"
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )


async def handle_plan_resume(arguments: dict[str, Any]) -> CallToolResult:
    """Resume a failed plan without discarding completed intermediary files."""
    req = PlanResumeRequest(**arguments)
    plan_id = req.plan_id

    resume_result = await asyncio.to_thread(_resume_plan_sync, plan_id, req.model_profile)

    if resume_result is None:
        response = {
            "error": {
                "code": "PLAN_NOT_FOUND",
                "message": f"Plan not found: {plan_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    if isinstance(resume_result.get("error"), dict):
        response = resume_result
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    response = resume_result
    base_url = _get_download_base_url()
    if base_url and response.get("plan_id"):
        response["sse_url"] = f"{base_url}/sse/plan/{response['plan_id']}"
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )


async def handle_plan_file_info(arguments: dict[str, Any]) -> CallToolResult:
    """Return download metadata for a plan's report or zip artifact.

    Examples:
        - {"plan_id": "uuid"} → report metadata (default)
        - {"plan_id": "uuid", "artifact": "zip"} → zip metadata

    Args:
        - plan_id: Plan UUID returned by plan_create.
        - artifact: Optional "report" or "zip".

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: metadata (content_type, sha256, download_size,
          optional download_url) or {} if not ready, or error payload.
        - isError: True only when plan_id is unknown.
    """
    req = PlanFileInfoRequest(**arguments)
    plan_id = req.plan_id
    artifact = req.artifact.strip().lower() if isinstance(req.artifact, str) else "report"
    if artifact not in ("report", "zip"):
        response = {
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": f"Invalid artifact type: {req.artifact!r}. Must be 'report' or 'zip'.",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )
    plan_snapshot = await asyncio.to_thread(_get_plan_for_report_sync, plan_id)
    if plan_snapshot is None:
        response = {
            "error": {
                "code": "PLAN_NOT_FOUND",
                "message": f"Plan not found: {plan_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    run_id = plan_snapshot["id"]
    if artifact == "zip":
        content_bytes = await fetch_user_downloadable_zip(run_id)
        if content_bytes is None:
            plan_state = plan_snapshot["state"]
            if plan_state in (PlanState.pending, PlanState.processing) or plan_state is None:
                response = {"ready": False, "reason": "processing"}
            else:
                response = {
                    "error": {
                        "code": "content_unavailable",
                        "message": "zip content_bytes is None",
                    },
                }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(response))],
                structuredContent=response,
                isError=False,
            )

        total_size = len(content_bytes)
        content_hash = compute_sha256(content_bytes)
        response = {
            "content_type": ZIP_CONTENT_TYPE,
            "sha256": content_hash,
            "download_size": total_size,
        }
        download_url = build_zip_download_url(run_id)
        if download_url:
            response["download_url"] = download_url
            response["expires_at"] = (datetime.now(UTC) + timedelta(seconds=DOWNLOAD_TOKEN_TTL_SECONDS)).isoformat()

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )

    plan_state = plan_snapshot["state"]
    if plan_state in (PlanState.pending, PlanState.processing) or plan_state is None:
        response = {"ready": False, "reason": "processing"}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )
    if plan_state == PlanState.stopped:
        response = {"ready": False, "reason": "stopped"}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )
    if plan_state == PlanState.failed:
        message = plan_snapshot["progress_message"] or "Plan generation failed."
        response = {"ready": False, "reason": "failed", "error": {"code": "generation_failed", "message": message}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )

    content_bytes = await fetch_artifact_from_worker_plan(run_id, REPORT_FILENAME)
    if content_bytes is None:
        response = {
            "error": {
                "code": "content_unavailable",
                "message": "content_bytes is None",
            },
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )

    total_size = len(content_bytes)
    content_hash = compute_sha256(content_bytes)
    response = {
        "content_type": REPORT_CONTENT_TYPE,
        "sha256": content_hash,
        "download_size": total_size,
    }
    download_url = build_report_download_url(run_id)
    if download_url:
        response["download_url"] = download_url
        response["expires_at"] = (datetime.now(UTC) + timedelta(seconds=DOWNLOAD_TOKEN_TTL_SECONDS)).isoformat()

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )

async def handle_plan_list(arguments: dict[str, Any]) -> CallToolResult:
    """Return recent plans for an authenticated user."""
    try:
        req = PlanListRequest(**arguments)
    except Exception as exc:
        response = {"error": {"code": "INVALID_ARGUMENTS", "message": str(exc)}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )
    require_user_key = os.environ.get("PLANEXE_MCP_REQUIRE_USER_KEY", "false").lower() in ("1", "true", "yes", "on")
    user_context = None
    if req.user_api_key:
        user_context = _resolve_user_from_api_key(req.user_api_key.strip())
        if not user_context:
            response = {"error": {"code": "INVALID_USER_API_KEY", "message": "Invalid user_api_key. Verify your key or create an account at https://home.planexe.org/"}}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(response))],
                structuredContent=response,
                isError=True,
            )
    elif require_user_key:
        response = {"error": {"code": "USER_API_KEY_REQUIRED", "message": "user_api_key is required for plan_list. Create an account at https://home.planexe.org/"}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )
    user_id = str(user_context["user_id"]) if user_context else None
    limit = max(1, min(req.limit, 50))
    plans = await asyncio.to_thread(_list_plans_sync, user_id, limit)
    response = {
        "plans": plans,
        "message": f"Returned {len(plans)} plan(s).",
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )


TOOL_HANDLERS = {
    "example_plans": handle_example_plans,
    "example_prompts": handle_example_prompts,
    "model_profiles": handle_model_profiles,
    "plan_create": handle_plan_create,
    "plan_status": handle_plan_status,
    "plan_stop": handle_plan_stop,
    "plan_retry": handle_plan_retry,
    "plan_resume": handle_plan_resume,
    "plan_file_info": handle_plan_file_info,
    "plan_list": handle_plan_list,
}

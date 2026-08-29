"""
PlanExe MCP Cloud — tool HTTP bridge

Pydantic request/response models, MCP result normalization, and thin
async wrapper functions that connect MCP tool registrations to the
handler implementations in ``mcp_cloud.app``.
"""
import json
import logging
from typing import Annotated, Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ContentBlock, TextContent

from mcp_cloud.app import (
    handle_plan_create,
    handle_plan_list,
    handle_model_profiles,
    handle_plan_status,
    handle_plan_retry,
    handle_plan_resume,
    handle_plan_stop,
    handle_plan_file_info,
    handle_example_prompts,
    handle_example_plans,
    handle_send_feedback,
)
from mcp_cloud.middleware import _get_authenticated_user_api_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class MCPToolCallRequest(BaseModel):
    tool: str
    arguments: dict[str, Any]
    metadata: Optional[dict[str, Any]] = None


class MCPToolCallResponse(BaseModel):
    content: list[dict[str, Any]]
    error: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Content extraction & error parsing
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Type aliases for tool parameters
# ---------------------------------------------------------------------------
ModelProfileInput = Literal["baseline", "premium", "frontier", "custom"]
ResultArtifactInput = Literal["report", "zip"]


# ---------------------------------------------------------------------------
# MCP tool wrapper functions
# ---------------------------------------------------------------------------
async def plan_create(
    prompt: str,
    model_profile: Annotated[
        ModelProfileInput,
        Field(description="Model profile: baseline, premium, frontier, custom. Call model_profiles to inspect options."),
    ] = "baseline",
    start_date: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional plan start date in ISO 8601 format with timezone offset "
                "(e.g. '2025-06-15T09:00:00+02:00'). "
                "When omitted, the plan starts now. "
                "Use this to set a past or future start date for the plan."
            ),
        ),
    ] = None,
) -> CallToolResult:
    """Create a new PlanExe task. Use example_prompts first for example prompts."""
    authenticated_user_api_key = _get_authenticated_user_api_key()
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "model_profile": model_profile,
    }
    if start_date:
        arguments["start_date"] = start_date
    if authenticated_user_api_key:
        arguments["user_api_key"] = authenticated_user_api_key
    return await handle_plan_create(
        arguments,
    )


async def plan_status(
    plan_id: str = Field(..., description="Plan UUID returned by plan_create."),
) -> CallToolResult:
    return await handle_plan_status({"plan_id": plan_id})


async def plan_stop(
    plan_id: str = Field(..., description="Plan UUID returned by plan_create. Use it to stop the plan creation."),
) -> CallToolResult:
    return await handle_plan_stop({"plan_id": plan_id})


async def plan_retry(
    plan_id: str = Field(..., description="UUID of the failed plan to retry."),
    model_profile: Annotated[
        ModelProfileInput,
        Field(description="Model profile used for retry. Defaults to baseline."),
    ] = "baseline",
) -> CallToolResult:
    arguments: dict[str, Any] = {"plan_id": plan_id, "model_profile": model_profile}
    authenticated_user_api_key = _get_authenticated_user_api_key()
    if authenticated_user_api_key:
        arguments["user_api_key"] = authenticated_user_api_key
    return await handle_plan_retry(arguments)


async def plan_resume(
    plan_id: str = Field(..., description="UUID of the failed plan to resume."),
    model_profile: Annotated[
        ModelProfileInput,
        Field(description="Model profile used for the resumed plan. Defaults to baseline."),
    ] = "baseline",
) -> CallToolResult:
    return await handle_plan_resume({"plan_id": plan_id, "model_profile": model_profile})


async def plan_file_info(
    plan_id: str = Field(..., description="Plan UUID returned by plan_create. Use it to download the created plan."),
    artifact: Annotated[
        ResultArtifactInput,
        Field(description="Download artifact type: report or zip."),
    ] = "report",
) -> CallToolResult:
    return await handle_plan_file_info({"plan_id": plan_id, "artifact": artifact})


async def example_prompts() -> CallToolResult:
    """Return curated example prompts from the catalog (no arguments)."""
    return await handle_example_prompts({})


async def model_profiles() -> CallToolResult:
    """Return model_profile options with currently available models."""
    return await handle_model_profiles({})


async def example_plans() -> CallToolResult:
    """Return curated example plans with download links (no arguments)."""
    return await handle_example_plans({})


async def plan_list(
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of plans to return (1–50). Newest plans are returned first."),
) -> CallToolResult:
    """List the most recent plans for an authenticated user."""
    authenticated_user_api_key = _get_authenticated_user_api_key()
    arguments: dict[str, Any] = {"limit": limit}
    if authenticated_user_api_key:
        arguments["user_api_key"] = authenticated_user_api_key
    return await handle_plan_list(arguments)


async def send_feedback(
    category: str = Field(..., description="Feedback category: mcp, plan, code, docs, or other."),
    message: str = Field(..., description="Free-text feedback. Include environment context if reporting an issue."),
    plan_id: Optional[str] = Field(default=None, description="Optional plan UUID to attach this feedback to."),
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="Sentiment: 1=strong negative, 2=weak negative, 3=neutral, 4=weak positive, 5=strong positive."),
) -> CallToolResult:
    """Submit your feedback about PlanExe."""
    authenticated_user_api_key = _get_authenticated_user_api_key()
    arguments: dict[str, Any] = {"category": category, "message": message}
    if plan_id is not None:
        arguments["plan_id"] = plan_id
    if rating is not None:
        arguments["rating"] = rating
    if authenticated_user_api_key:
        arguments["user_api_key"] = authenticated_user_api_key
    return await handle_send_feedback(arguments)


# ---------------------------------------------------------------------------
# Registry-based tool call (used by REST endpoint)
# ---------------------------------------------------------------------------
async def call_tool_via_registry(
    server: MCPServer,
    tool_name: str,
    arguments: dict[str, Any],
) -> MCPToolCallResponse:
    """Call tools via the MCP tool registry."""
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

"""
PlanExe MCP Cloud — route registration

MCP tool registration, MCP prompts, and all FastAPI route handlers.
Called by ``server_boot.py`` during application assembly.
"""
import asyncio
import logging
import os
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from mcp_cloud.app import (
    REPORT_CONTENT_TYPE,
    REPORT_FILENAME,
    TOOL_DEFINITIONS,
    ZIP_CONTENT_TYPE,
    ZIP_FILENAME,
    fetch_artifact_from_worker_plan,
    fetch_user_downloadable_zip,
    handle_plan_create,
    resolve_plan_by_id,
    validate_download_token,
)
from mcp_cloud.tool_http_bridge import (
    MCPToolCallRequest,
    MCPToolCallResponse,
    _normalize_tool_result,
    call_tool_via_registry,
    example_plans,
    example_prompts,
    model_profiles,
    plan_create,
    plan_file_info,
    plan_list,
    plan_resume,
    plan_retry,
    plan_status,
    plan_stop,
    send_feedback,
)
from mcp_cloud.middleware import _get_authenticated_user_api_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------
def _register_tools(server: MCPServer) -> None:
    handler_map = {
        "example_plans": example_plans,
        "example_prompts": example_prompts,
        "model_profiles": model_profiles,
        "plan_create": plan_create,
        "plan_status": plan_status,
        "plan_stop": plan_stop,
        "plan_retry": plan_retry,
        "plan_resume": plan_resume,
        "plan_file_info": plan_file_info,
        "plan_list": plan_list,
        "send_feedback": send_feedback,
    }
    for tool_def in TOOL_DEFINITIONS:
        handler = handler_map.get(tool_def.name)
        if handler is None:
            logger.warning("No HTTP handler registered for tool %s", tool_def.name)
            continue
        server.tool(
            name=tool_def.name,
            description=tool_def.description,
            annotations=ToolAnnotations(**tool_def.annotations) if tool_def.annotations else None,
        )(handler)

    # Inject the canonical outputSchema from TOOL_DEFINITIONS into each
    # registered tool so that list_tools advertises the schema we control.
    #
    # We set the schema as an instance attribute on the Tool, which shadows
    # the cached_property (Tool.output_schema reads fn_metadata.output_schema).
    # This way list_tools() sees the canonical schema, but fn_metadata stays
    # untouched so convert_result() does not try to validate against a missing
    # output_model.
    #
    # Skip oneOf schemas: MCP clients (e.g. Inspector) require outputSchema to
    # be {"type": "object", ...}. Tools with multiple response shapes
    # (plan_status, plan_file_info) use oneOf which clients reject. These tools
    # work correctly without an advertised outputSchema.
    for tool_def in TOOL_DEFINITIONS:
        schema = tool_def.output_schema
        if schema is None:
            continue
        if "oneOf" in schema:
            continue
        registered_tool = server._tool_manager.get_tool(tool_def.name)
        if registered_tool is None:
            continue
        registered_tool.__dict__["output_schema"] = schema


# ---------------------------------------------------------------------------
# MCP Prompts
# ---------------------------------------------------------------------------
def _register_prompts(server: MCPServer) -> None:
    @server.prompt()
    def getting_started() -> str:
        """Quick-start guide for using PlanExe to create a project plan."""
        return (
            "You have access to PlanExe, an MCP server that generates strategic "
            "project-plan drafts from a natural-language prompt.\n\n"
            "To get started:\n"
            "1. Call the example_prompts tool to see what good prompts look like.\n"
            "2. Optionally call model_profiles to see available quality tiers.\n"
            "3. Draft a detailed prompt (300-800 words) covering: objective, scope, "
            "constraints, timeline, stakeholders, budget, and success criteria. "
            "Write it as flowing prose, not bullet lists.\n"
            "4. Show the draft to the user for approval. Iterate — the user may "
            "want to refine scope, add constraints, or adjust priorities. "
            "A few rounds of feedback typically produce the best plans.\n"
            "5. Once the user is satisfied, call plan_create with the approved prompt.\n"
            "6. Poll plan_status every 5 minutes until state is completed.\n"
            "7. Call plan_file_info to get the download URL for the HTML report.\n\n"
            "The report contains 20+ sections including executive summary, Gantt charts, "
            "risk analysis, SWOT, governance, investor pitch, and adversarial stress-tests."
        )

    @server.prompt()
    def plan_a_project(topic: str, location: str = "") -> str:
        """Draft a project plan for a given topic. Guides the agent through the full PlanExe workflow."""
        location_line = f"\nLocation/region: {location}\n" if location else ""
        return (
            f"The user wants to create a project plan about: {topic}\n"
            f"{location_line}\n"
            "Follow these steps:\n"
            "1. Call example_prompts to see baseline prompt quality and structure.\n"
            "2. Using those examples as inspiration, expand the user's topic into a "
            "detailed prompt of 300-800 words. Include: objective, scope, constraints, "
            "timeline, stakeholders, budget/resources, and success criteria. Write as "
            "flowing prose — weave specs and targets naturally into sentences.\n"
            "3. Present the draft prompt to the user and ask for approval or changes.\n"
            "4. Once approved, call plan_create with the prompt.\n"
            "5. Poll plan_status every 5 minutes (plan generation takes 10-20 minutes).\n"
            "6. When state is completed, call plan_file_info with artifact='report' "
            "and share the download URL with the user."
        )


def register_tools_and_prompts(server: MCPServer) -> None:
    """Register all MCP tools and prompts on the MCP server."""
    _register_tools(server)
    _register_prompts(server)


# ---------------------------------------------------------------------------
# Module-level route handler functions
#
# These are defined at module level so they can be accessed by tests via
# ``http_server.options_mcp`` etc. They are registered on the app inside
# ``register_routes()``.
# ---------------------------------------------------------------------------
async def options_mcp() -> Response:
    """Handle CORS preflight for /mcp so browser-based tools (e.g. MCP Inspector) succeed."""
    return Response(status_code=200)


async def head_mcp_trailing_slash() -> Response:
    """Handle HEAD /mcp/ for health-check probes (e.g. Smithery scanner).

    The mounted Streamable HTTP app does not support HEAD and returns
    405.  This explicit route intercepts the request so scanners get a clean
    200 instead of bouncing off the sub-app.
    """
    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


# ---------------------------------------------------------------------------
# FastAPI route handlers
# ---------------------------------------------------------------------------
def register_routes(
    app: FastAPI,
    mcp_server: MCPServer,
    get_mcp_server: Callable[..., MCPServer],
) -> None:
    """Register all HTTP route handlers on the FastAPI app.

    Must be called before ``app.mount("/mcp", ...)`` so that explicit routes
    like ``/mcp/tools`` and ``/mcp/tools/call`` take priority over the mounted
    MCP sub-app.
    """
    from mcp_cloud.server_boot import (
        AUTH_REQUIRED,
        GLAMA_MAINTAINER_EMAIL,
        SERVER_VERSION,
    )

    # -- CORS / HEAD handlers for /mcp ------------------------------------

    app.options("/mcp")(options_mcp)
    app.options("/mcp/")(options_mcp)

    app.head("/mcp/")(head_mcp_trailing_slash)

    # -- REST tool call endpoint -------------------------------------------

    @app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
    async def call_tool(
        payload: MCPToolCallRequest,
        mcp_server: MCPServer = Depends(get_mcp_server),
    ) -> MCPToolCallResponse:
        """
        Call an MCP tool by name with arguments.

        This endpoint wraps the stdio-based MCP tool handlers for HTTP access.
        """
        arguments = dict(payload.arguments or {})
        if payload.tool in ("plan_create", "plan_list"):
            authenticated_user_api_key = _get_authenticated_user_api_key()
            if authenticated_user_api_key and not arguments.get("user_api_key"):
                arguments["user_api_key"] = authenticated_user_api_key

        if payload.tool == "plan_create":
            if isinstance(payload.metadata, dict):
                arguments["metadata"] = dict(payload.metadata)

            result = await handle_plan_create(arguments)
            content, error = _normalize_tool_result(result)
            return MCPToolCallResponse(content=content, error=error)

        return await call_tool_via_registry(mcp_server, payload.tool, arguments)

    # -- Tools list endpoint -----------------------------------------------

    @app.get("/mcp/tools")
    async def list_tools(mcp_server: MCPServer = Depends(get_mcp_server)) -> dict[str, Any]:
        """List all available MCP tools."""
        tools = await mcp_server.list_tools()
        sanitized = []
        for tool in tools:
            tool_entry = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            if tool.title:
                tool_entry["title"] = tool.title
            if tool.output_schema:
                tool_entry["outputSchema"] = tool.output_schema
            if tool.annotations:
                tool_entry["annotations"] = tool.annotations
            if tool.icons:
                tool_entry["icons"] = tool.icons
            sanitized.append(tool_entry)
        return {"tools": sanitized}

    # -- Download endpoint -------------------------------------------------

    @app.get("/download/{plan_id}/{filename}")
    async def download_report(
        plan_id: str,
        filename: str,
        token: Optional[str] = None,
    ) -> Response:
        """Download the generated report HTML or zip for a plan.

        Authentication: either a valid ``?token=...`` query parameter (signed,
        expiring) or a valid API key in the request headers (existing behaviour).
        The middleware enforces one of these; the token is re-validated here for
        defence-in-depth.
        """
        if filename not in (REPORT_FILENAME, ZIP_FILENAME):
            raise HTTPException(status_code=404, detail="Report not found")
        # Defence-in-depth: if a token was supplied, it must be valid for this artifact.
        if token is not None and not validate_download_token(token, plan_id, filename):
            raise HTTPException(status_code=401, detail="Invalid or expired download token")
        plan = await asyncio.to_thread(resolve_plan_by_id, plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="Plan not found")
        if filename == ZIP_FILENAME:
            content_bytes = await fetch_user_downloadable_zip(str(plan.id))
            if content_bytes is None:
                raise HTTPException(status_code=404, detail="Report not found")
            headers = {"Content-Disposition": f'attachment; filename="{plan_id}.zip"'}
            return Response(content=content_bytes, media_type=ZIP_CONTENT_TYPE, headers=headers)

        content_bytes = await fetch_artifact_from_worker_plan(str(plan.id), REPORT_FILENAME)
        if content_bytes is None:
            raise HTTPException(status_code=404, detail="Report not found")
        headers = {"Content-Disposition": f'inline; filename="{REPORT_FILENAME}"'}
        return Response(content=content_bytes, media_type=REPORT_CONTENT_TYPE, headers=headers)

    # -- SSE endpoint ------------------------------------------------------

    @app.get("/sse/plan/{plan_id}")
    async def sse_plan_progress(plan_id: str, request: Request) -> Response:
        """SSE endpoint that streams real-time plan progress updates."""
        from mcp_cloud.sse import (
            SSEConnectionLimitError,
            _track_sse_connection,
            plan_progress_stream,
        )

        client_ip = request.client.host if request.client else "unknown"

        try:
            async def event_generator():
                async with _track_sse_connection(client_ip):
                    disconnect_event = asyncio.Event()
                    async for event in plan_progress_stream(plan_id, disconnect_event):
                        if await request.is_disconnected():
                            disconnect_event.set()
                            break
                        yield event

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except SSEConnectionLimitError as exc:
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "SSE_CONNECTION_LIMIT", "message": str(exc)}},
            )

    # -- Health & metadata endpoints ---------------------------------------

    @app.get("/healthcheck")
    def healthcheck() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "planexe-mcp-cloud",
            "authentication_required": AUTH_REQUIRED,
        }

    @app.get("/")
    def root() -> dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "service": "PlanExe – AI Project Planning",
            "description": "MCP server that generates rough-draft project plans from a natural-language prompt",
            "version": SERVER_VERSION,
            "endpoints": {
                "mcp": "/mcp",
                "tools": "/mcp/tools",
                "call": "/mcp/tools/call",
                "health": "/healthcheck",
                "mcp_server_card": "/.well-known/mcp/server-card.json",
                "glama_connector": "/.well-known/glama.json",
                "download": f"/download/{{plan_id}}/{REPORT_FILENAME}",
                "sse": "/sse/plan/{plan_id}",
                "llms_txt": "/llms.txt",
            },
            "documentation": "See /docs for OpenAPI documentation",
            "authentication": (
                "Required: X-API-Key (Obtain an api key at home.planexe.org)"
                if AUTH_REQUIRED
                else "Disabled (PLANEXE_MCP_REQUIRE_AUTH=false)"
            ),
        }

    def _llms_txt_path() -> str:
        """
        Resolve canonical llms metadata file path.
        """
        repo_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(repo_root, "public", "llms.txt")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="llms.txt not found")
        return path

    @app.get("/.well-known/mcp/server-card.json")
    def mcp_server_card() -> dict[str, Any]:
        """Serve MCP Server Card for discovery (SEP-1649).
        https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/schema/2025-06-18

        This allows registries like Smithery to discover the server's capabilities
        without performing a full MCP handshake.
        """
        return {
            "version": "1.0",
            "protocolVersion": "2025-06-18",
            "serverInfo": {
                "name": "planexe-mcp-server",
                "title": "PlanExe - AI Project Planning",
                "version": SERVER_VERSION,
            },
            "description": (
                "MCP server that generates strategic project-plan drafts from a "
                "natural-language prompt. Output is a self-contained interactive "
                "HTML report with 20+ sections including executive summary, "
                "interactive Gantt charts, risk analysis, SWOT, governance, "
                "investor pitch, and adversarial stress-test sections."
            ),
            "documentationUrl": "https://docs.planexe.org/",
            "transport": {
                "type": "streamable-http",
                "endpoint": "/mcp/",
            },
            "capabilities": {
                "tools": {},
            },
            "authentication": {
                "required": AUTH_REQUIRED,
                "schemes": ["api-key"],
            },
            "tools": ["dynamic"],
            "prompts": ["dynamic"],
            "resources": ["dynamic"],
        }

    @app.get("/.well-known/glama.json")
    def glama_connector_metadata() -> dict[str, Any]:
        """Serve Glama connector ownership metadata."""
        return {
            "$schema": "https://glama.ai/mcp/schemas/connector.json",
            "maintainers": [{"email": GLAMA_MAINTAINER_EMAIL}],
        }

    @app.get("/llms.txt")
    def llms_txt():
        """
        Serve llms.txt for AI agent discoverability.

        This endpoint provides information about PlanExe for autonomous AI agents
        looking for project planning and execution tools. Designed for agent-first
        organizations and AI workforce deployments.
        """
        return FileResponse(_llms_txt_path(), media_type="text/plain; charset=utf-8")

    @app.get("/llm.txt")
    def llm_txt() -> RedirectResponse:
        """Legacy alias that redirects to canonical /llms.txt."""
        return RedirectResponse(url="/llms.txt", status_code=308)

    @app.get("/robots.txt")
    def robots_txt() -> Response:
        """Allow crawler discovery of public MCP metadata endpoints."""
        content = (
            "User-agent: *\n"
            "Allow: /\n"
            "Sitemap: https://mcp.planexe.org/llms.txt\n"
        )
        return Response(content=content, media_type="text/plain; charset=utf-8")

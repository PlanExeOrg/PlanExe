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
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env file early
from dotenv import load_dotenv
_module_dir = Path(__file__).parent
_dotenv_loaded = load_dotenv(_module_dir / ".env")
if not _dotenv_loaded:
    load_dotenv(_module_dir.parent / ".env")

# Import MCP tool handlers from app.py
from mcp_server.app import (
    handle_session_create,
    handle_session_start,
    handle_session_status,
    handle_session_stop,
    handle_session_resume,
    handle_artifact_list,
    handle_artifact_read,
    handle_artifact_write,
    handle_session_events,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlanExe MCP Server (HTTP)",
    description="HTTP wrapper for PlanExe MCP interface",
    version="1.0.0"
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key validation
REQUIRED_API_KEY = os.environ.get("PLANEXE_MCP_API_KEY")
if not REQUIRED_API_KEY:
    logger.warning("PLANEXE_MCP_API_KEY not set. API key authentication disabled (not recommended for production)")

def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Header(None, alias="API_KEY")
):
    """Verify API key from header. Supports both X-API-Key and API_KEY headers."""
    if not REQUIRED_API_KEY:
        # No API key configured, allow all (development mode)
        return True
    
    # Support both header formats
    provided_key = x_api_key or api_key
    
    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key or API_KEY header."
        )
    
    if provided_key != REQUIRED_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

# Request/Response models
class MCPToolCallRequest(BaseModel):
    tool: str
    arguments: dict[str, Any]

class MCPToolCallResponse(BaseModel):
    content: list[dict[str, Any]]
    error: Optional[dict[str, Any]] = None

def extract_text_content(text_contents: list) -> list[dict[str, Any]]:
    """Extract text content from MCP TextContent objects."""
    result = []
    for item in text_contents:
        if hasattr(item, 'text'):
            # Try to parse as JSON, fallback to plain text
            try:
                parsed = json.loads(item.text)
                result.append(parsed)
            except json.JSONDecodeError:
                result.append({"text": item.text})
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append({"text": str(item)})
    return result

@app.post("/mcp", response_model=MCPToolCallResponse)
async def mcp_endpoint(
    request: MCPToolCallRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Main MCP endpoint for tool calls.
    
    Compatible with MCP clients that expect a single endpoint.
    """
    return await call_tool_internal(request.tool, request.arguments)

@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def call_tool(
    request: MCPToolCallRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Call an MCP tool by name with arguments.
    
    This endpoint wraps the stdio-based MCP tool handlers for HTTP access.
    """
    return await call_tool_internal(request.tool, request.arguments)

async def call_tool_internal(tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResponse:
    """Internal tool call handler."""
    try:
        # Route to appropriate handler
        if tool_name == "planexe.session.create":
            result = await handle_session_create(arguments)
        elif tool_name == "planexe.session.start":
            result = await handle_session_start(arguments)
        elif tool_name == "planexe.session.status":
            result = await handle_session_status(arguments)
        elif tool_name == "planexe.session.stop":
            result = await handle_session_stop(arguments)
        elif tool_name == "planexe.session.resume":
            result = await handle_session_resume(arguments)
        elif tool_name == "planexe.artifact.list":
            result = await handle_artifact_list(arguments)
        elif tool_name == "planexe.artifact.read":
            result = await handle_artifact_read(arguments)
        elif tool_name == "planexe.artifact.write":
            result = await handle_artifact_write(arguments)
        elif tool_name == "planexe.session.events":
            result = await handle_session_events(arguments)
        else:
            return MCPToolCallResponse(
                content=[],
                error={
                    "code": "INVALID_TOOL",
                    "message": f"Unknown tool: {tool_name}"
                }
            )
        
        # Extract text content from MCP response format
        content = extract_text_content(result)
        
        # Check if any content contains an error
        error = None
        for item in content:
            if isinstance(item, dict) and "error" in item:
                error = item["error"]
                break
        
        return MCPToolCallResponse(content=content, error=error)
        
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
        return MCPToolCallResponse(
            content=[],
            error={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        )

@app.get("/mcp/tools")
async def list_tools(_: bool = Depends(verify_api_key)):
    """
    List all available MCP tools.
    """
    tools = [
        {
            "name": "planexe.session.create",
            "description": "Creates a new session and output namespace"
        },
        {
            "name": "planexe.session.start",
            "description": "Starts execution for a target DAG output"
        },
        {
            "name": "planexe.session.status",
            "description": "Returns run status and progress"
        },
        {
            "name": "planexe.session.stop",
            "description": "Stops the active run"
        },
        {
            "name": "planexe.session.resume",
            "description": "Resumes execution, reusing cached Luigi outputs"
        },
        {
            "name": "planexe.artifact.list",
            "description": "Lists artifacts under output namespace"
        },
        {
            "name": "planexe.artifact.read",
            "description": "Reads an artifact"
        },
        {
            "name": "planexe.artifact.write",
            "description": "Writes an artifact (enables Stop → Edit → Resume)"
        },
        {
            "name": "planexe.session.events",
            "description": "Provides incremental events for a session since a cursor"
        },
    ]
    return {"tools": tools}

@app.get("/healthcheck")
def healthcheck():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "planexe-mcp-http",
        "api_key_configured": REQUIRED_API_KEY is not None
    }

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "PlanExe MCP Server (HTTP)",
        "version": "1.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "tools": "/mcp/tools",
            "call": "/mcp/tools/call",
            "health": "/healthcheck"
        },
        "documentation": "See /docs for OpenAPI documentation",
        "authentication": "X-API-Key header required (set PLANEXE_MCP_API_KEY)"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("PLANEXE_MCP_HTTP_HOST", "0.0.0.0")
    # Railway provides PORT env var, otherwise use PLANEXE_MCP_HTTP_PORT or default
    port = int(os.environ.get("PORT") or os.environ.get("PLANEXE_MCP_HTTP_PORT", "8001"))
    
    logger.info(f"Starting PlanExe MCP HTTP server on {host}:{port}")
    if REQUIRED_API_KEY:
        logger.info("API key authentication enabled")
    else:
        logger.warning("API key authentication disabled - set PLANEXE_MCP_API_KEY")
    
    uvicorn.run("http_server:app", host=host, port=port, reload=False)

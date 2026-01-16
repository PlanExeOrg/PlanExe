"""
PlanExe MCP Server

Implements the Model Context Protocol interface for PlanExe as specified in
extra/mcp-spec1.md. Communicates with worker_plan_database via the shared
database_api models.
"""
import asyncio
import hashlib
import io
import json
import logging
import os
import sys
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus
from io import BytesIO
import httpx
from sqlalchemy import cast, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel

# Load .env file early
from mcp_server.dotenv_utils import load_planexe_dotenv
_dotenv_loaded, _dotenv_paths = load_planexe_dotenv(Path(__file__).parent)

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

# Database imports
from database_api.planexe_db_singleton import db
from database_api.model_taskitem import TaskItem, TaskState
from database_api.model_event import EventItem, EventType
from flask import Flask, has_app_context

# Initialize Flask app for database access
app = Flask(__name__)
app.config.from_pyfile('config.py')

def build_postgres_uri_from_env(env: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Construct a SQLAlchemy URI for Postgres using environment variables."""
    host = env.get("PLANEXE_POSTGRES_HOST") or "database_postgres"
    port = str(env.get("PLANEXE_POSTGRES_PORT") or "5432")
    dbname = env.get("PLANEXE_POSTGRES_DB") or "planexe"
    user = env.get("PLANEXE_POSTGRES_USER") or "planexe"
    password = env.get("PLANEXE_POSTGRES_PASSWORD") or "planexe"
    uri = f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    safe_config = {"host": host, "port": port, "dbname": dbname, "user": user}
    return uri, safe_config

# Load database configuration
sqlalchemy_database_uri = os.environ.get("SQLALCHEMY_DATABASE_URI")
if sqlalchemy_database_uri is None:
    sqlalchemy_database_uri, db_settings = build_postgres_uri_from_env(os.environ)
    logger.info(f"SQLALCHEMY_DATABASE_URI not set. Using Postgres defaults: {db_settings}")
else:
    logger.info("Using SQLALCHEMY_DATABASE_URI from environment.")

app.config['SQLALCHEMY_DATABASE_URI'] = sqlalchemy_database_uri
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle': 280, 'pool_pre_ping': True}
db.init_app(app)

def ensure_taskitem_stop_columns() -> None:
    insp = inspect(db.engine)
    columns = {col["name"] for col in insp.get_columns("task_item")}
    with db.engine.begin() as conn:
        if "stop_requested" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested BOOLEAN"))
        if "stop_requested_timestamp" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested_timestamp TIMESTAMP"))

with app.app_context():
    ensure_taskitem_stop_columns()

# MCP Server setup
mcp_server = Server("planexe-mcp-server")

# Base directory for run artifacts (not used directly, fetched via worker_plan HTTP API)
BASE_DIR_RUN = Path(os.environ.get("PLANEXE_RUN_DIR", Path(__file__).parent.parent / "run")).resolve()

# Worker plan HTTP API URL
WORKER_PLAN_URL = os.environ.get("PLANEXE_WORKER_PLAN_URL", "http://worker_plan:8000")

# Pydantic models for request/response validation
class SessionCreateRequest(BaseModel):
    idea: str
    config: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None

class SessionStartRequest(BaseModel):
    session_id: str
    target: str = "build_plan_and_validate"
    inputs: Optional[dict[str, Any]] = None

class SessionStatusRequest(BaseModel):
    session_id: str

class SessionStopRequest(BaseModel):
    session_id: str
    run_id: Optional[str] = None
    mode: str = "graceful"

class SessionResumeRequest(BaseModel):
    session_id: str
    target: str = "build_plan_and_validate"
    resume_policy: str = "luigi_up_to_date"
    invalidate: Optional[dict[str, Any]] = None

class ArtifactListRequest(BaseModel):
    session_id: str
    path: str = ""
    include_metadata: bool = True

class ArtifactReadRequest(BaseModel):
    artifact_uri: str
    range: Optional[dict[str, int]] = None

class ArtifactWriteRequest(BaseModel):
    artifact_uri: str
    content: str
    edit_reason: Optional[str] = None
    lock: Optional[dict[str, str]] = None

class SessionEventsRequest(BaseModel):
    session_id: str
    since: Optional[str] = None

# Helper functions
def find_task_by_session_id(session_id: str) -> Optional[TaskItem]:
    """Find TaskItem by session_id stored in parameters."""
    def _query() -> Optional[TaskItem]:
        query = db.session.query(TaskItem)
        if db.engine.dialect.name == "postgresql":
            tasks = query.filter(
                cast(TaskItem.parameters, JSONB).contains({"_mcp_session_id": session_id})
            ).all()
        else:
            tasks = query.filter(
                TaskItem.parameters.contains({"_mcp_session_id": session_id})
            ).all()
        if tasks:
            return tasks[0]
        return None

    if has_app_context():
        return _query()
    with app.app_context():
        return _query()

def get_task_id_for_session(session_id: str) -> Optional[str]:
    """Get the task_id (run_id) for a session."""
    task = find_task_by_session_id(session_id)
    if task:
        return str(task.id)
    # Fallback: try parsing session_id format
    if "__" in session_id:
        # Extract UUID portion from pxe_YYYY_MM_DD__{uuid}
        uuid_part = session_id.split("__", 1)[1]
        # Try to find task by UUID
        try:
            task_id = uuid.UUID(uuid_part)
            return str(task_id)
        except ValueError:
            pass
    return None

async def fetch_artifact_from_worker_plan(run_id: str, file_path: str) -> Optional[bytes]:
    """Fetch an artifact file from worker_plan via HTTP."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # For report.html, use the dedicated report endpoint (most efficient)
            if file_path == "report.html" or file_path.endswith("/report.html"):
                report_response = await client.get(f"{WORKER_PLAN_URL}/runs/{run_id}/report")
                if report_response.status_code == 200:
                    return report_response.content
                logger.warning(f"Worker plan returned {report_response.status_code} for report: {run_id}")
                return None
            
            # For other files, fetch the zip and extract the file
            # This is less efficient but works without a file serving endpoint
            zip_response = await client.get(f"{WORKER_PLAN_URL}/runs/{run_id}/zip")
            if zip_response.status_code != 200:
                logger.warning(f"Worker plan returned {zip_response.status_code} for zip: {run_id}")
                return None
            
            # Extract the file from the zip in memory
            zip_bytes = zip_response.content
            with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_file:
                try:
                    # Normalize path (remove leading slash if present)
                    file_path_normalized = file_path.lstrip('/')
                    file_data = zip_file.read(file_path_normalized)
                    return file_data
                except KeyError:
                    logger.warning(f"File {file_path} not found in zip for run {run_id}")
                    return None
            
    except Exception as e:
        logger.error(f"Error fetching artifact from worker_plan: {e}", exc_info=True)
        return None

async def fetch_file_list_from_worker_plan(run_id: str) -> Optional[list[str]]:
    """Fetch the list of files from worker_plan via HTTP."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{WORKER_PLAN_URL}/runs/{run_id}/files")
            if response.status_code == 200:
                data = response.json()
                return data.get("files", [])
            logger.warning(f"Worker plan returned {response.status_code} for files list: {run_id}")
            return None
    except Exception as e:
        logger.error(f"Error fetching file list from worker_plan: {e}", exc_info=True)
        return None

def compute_sha256(content: str | bytes) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def get_task_state_mapping(task_state: TaskState) -> str:
    """Map TaskState to MCP run state."""
    mapping = {
        TaskState.pending: "stopped",
        TaskState.processing: "running",
        TaskState.completed: "completed",
        TaskState.failed: "failed",
    }
    return mapping.get(task_state, "stopped")

# MCP Tool implementations
@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name="planexe.session.create",
            description="Creates a new session and output namespace",
            inputSchema={
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "The idea/prompt for the plan"},
                    "config": {"type": "object", "description": "Optional configuration"},
                    "metadata": {"type": "object", "description": "Optional metadata including user_id"},
                },
                "required": ["idea"],
            },
        ),
        Tool(
            name="planexe.session.start",
            description="Starts execution for a target DAG output",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "target": {"type": "string", "default": "build_plan_and_validate"},
                    "inputs": {"type": "object"},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="planexe.session.status",
            description="Returns run status and progress",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="planexe.session.stop",
            description="Stops the active run",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "mode": {"type": "string", "default": "graceful"},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="planexe.session.resume",
            description="Resumes execution, reusing cached Luigi outputs",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "target": {"type": "string", "default": "build_plan_and_validate"},
                    "resume_policy": {"type": "string", "default": "luigi_up_to_date"},
                    "invalidate": {"type": "object"},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="planexe.artifact.list",
            description="Lists artifacts under output namespace",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "path": {"type": "string", "default": ""},
                    "include_metadata": {"type": "boolean", "default": True},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="planexe.artifact.read",
            description="Reads an artifact",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_uri": {"type": "string"},
                    "range": {"type": "object"},
                },
                "required": ["artifact_uri"],
            },
        ),
        Tool(
            name="planexe.artifact.write",
            description="Writes an artifact (enables Stop → Edit → Resume)",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_uri": {"type": "string"},
                    "content": {"type": "string"},
                    "edit_reason": {"type": "string"},
                    "lock": {"type": "object"},
                },
                "required": ["artifact_uri", "content"],
            },
        ),
        Tool(
            name="planexe.session.events",
            description="Provides incremental events for a session since a cursor",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "since": {"type": "string"},
                },
                "required": ["session_id"],
            },
        ),
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "planexe.session.create":
            return await handle_session_create(arguments)
        elif name == "planexe.session.start":
            return await handle_session_start(arguments)
        elif name == "planexe.session.status":
            return await handle_session_status(arguments)
        elif name == "planexe.session.stop":
            return await handle_session_stop(arguments)
        elif name == "planexe.session.resume":
            return await handle_session_resume(arguments)
        elif name == "planexe.artifact.list":
            return await handle_artifact_list(arguments)
        elif name == "planexe.artifact.read":
            return await handle_artifact_read(arguments)
        elif name == "planexe.artifact.write":
            return await handle_artifact_write(arguments)
        elif name == "planexe.session.events":
            return await handle_session_events(arguments)
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "INVALID_TOOL", "message": f"Unknown tool: {name}"}})
            )]
    except Exception as e:
        logger.error(f"Error handling tool {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INTERNAL_ERROR", "message": str(e)}})
        )]

async def handle_session_create(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.session.create"""
    req = SessionCreateRequest(**arguments)
    
    with app.app_context():
        # Create a new TaskItem (which represents a session in our model)
        task = TaskItem(
            prompt=req.idea,
            state=TaskState.pending,
            user_id=req.metadata.get("user_id", "mcp_user") if req.metadata else "mcp_user",
            parameters=req.config or {},
        )
        db.session.add(task)
        db.session.commit()
        
        # Generate session_id in format: pxe_{date}__{short_uuid}
        date_str = datetime.now(UTC).strftime("%Y_%m_%d")
        short_uuid = str(task.id).replace("-", "")[:8]
        session_id = f"pxe_{date_str}__{short_uuid}"
        output_dir_uri = f"planexe://sessions/{session_id}/out"
        
        # Store session_id mapping in task parameters for later lookup
        parameters = dict(task.parameters or {})
        parameters["_mcp_session_id"] = session_id
        task.parameters = parameters
        event_context = {
            "task_id": str(task.id),
            "session_id": session_id,
            "prompt": task.prompt,
            "user_id": task.user_id,
            "config": req.config,
            "metadata": req.metadata,
            "parameters": task.parameters,
        }
        event = EventItem(
            event_type=EventType.TASK_PENDING,
            message="Enqueued task via MCP",
            context=event_context
        )
        db.session.add(event)
        db.session.commit()
        
        response = {
            "session_id": session_id,
            "output_dir_uri": output_dir_uri,
            "created_at": task.timestamp_created.isoformat() + "Z",
        }
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_session_start(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.session.start"""
    req = SessionStartRequest(**arguments)
    session_id = req.session_id
    
    with app.app_context():
        task = find_task_by_session_id(session_id)
        if task is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        # Check if already running
        if task.state == TaskState.processing:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "RUN_ALREADY_ACTIVE", "message": "A run is currently active for this session."}})
            )]
        
        # Update task to pending (worker_plan_database will pick it up and change to processing)
        task.state = TaskState.pending
        task.progress_percentage = 0.0
        task.progress_message = "Starting..."
        task.last_seen_timestamp = datetime.now(UTC)
        task.stop_requested = False
        task.stop_requested_timestamp = None

        db.session.commit()
        
        run_id = f"run_{str(task.id).replace('-', '_')}"
        response = {
            "run_id": run_id,
            "state": "running",
        }
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_session_status(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.session.status"""
    req = SessionStatusRequest(**arguments)
    session_id = req.session_id
    
    with app.app_context():
        task = find_task_by_session_id(session_id)
        if task is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        # Determine phase based on progress
        progress_pct = float(task.progress_percentage) if task.progress_percentage else 0.0
        if progress_pct == 0.0:
            phase = "initializing"
        elif progress_pct < 50.0:
            phase = "generating_plan"
        elif progress_pct < 90.0:
            phase = "validating"
        elif progress_pct < 100.0:
            phase = "exporting"
        else:
            phase = "finalizing"
        
        run_id = f"run_{str(task.id).replace('-', '_')}"
        state = get_task_state_mapping(task.state)
        if task.state == TaskState.processing and task.stop_requested:
            state = "stopping"
        
        # Collect artifacts from worker_plan
        run_id = get_task_id_for_session(session_id)
        latest_artifacts = []
        if run_id:
            files_list = await fetch_file_list_from_worker_plan(run_id)
            if files_list:
                for file_name in files_list[:10]:  # Limit to 10 most recent
                    if file_name != "log.txt":
                        artifact_uri = f"planexe://sessions/{session_id}/out/{file_name}"
                        latest_artifacts.append({
                            "path": file_name,
                            "artifact_uri": artifact_uri,
                            "kind": "intermediate",
                            "updated_at": datetime.now(UTC).isoformat() + "Z",  # Approximate
                        })
        
        created_at = task.timestamp_created
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        response = {
            "session_id": session_id,
            "run_id": run_id,
            "state": state,
            "phase": phase,
            "progress": {
                "overall": progress_pct / 100.0,
                "current_task": {
                    "name": task.progress_message or "Unknown",
                    "pct": progress_pct / 100.0,
                },
            },
            "timing": {
                "started_at": created_at.isoformat().replace("+00:00", "Z") if created_at else None,
                "elapsed_sec": (datetime.now(UTC) - created_at).total_seconds() if created_at else 0,
            },
            "latest_artifacts": latest_artifacts[:10],  # Limit to 10 most recent
            "warnings": [],
        }
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_session_stop(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.session.stop"""
    req = SessionStopRequest(**arguments)
    session_id = req.session_id
    
    with app.app_context():
        task = find_task_by_session_id(session_id)
        if task is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        if task.state in (TaskState.pending, TaskState.processing):
            task.stop_requested = True
            task.stop_requested_timestamp = datetime.now(UTC)
            task.progress_message = "Stop requested by user."
            db.session.commit()
            logger.info("Stop requested for session %s; stop flag set on task %s.", session_id, task.id)
        
        response = {
            "state": "stopped",
        }
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_session_resume(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.session.resume"""
    req = SessionResumeRequest(**arguments)
    session_id = req.session_id
    
    # Resume is similar to start - we create a new run (TaskItem state -> pending)
    # For now, we'll reuse the same task
    with app.app_context():
        task = find_task_by_session_id(session_id)
        if task is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        if task.state == TaskState.processing:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "RUN_ALREADY_ACTIVE", "message": "A run is currently active for this session."}})
            )]
        
        # Reset to pending to trigger a new run
        task.state = TaskState.pending
        task.progress_percentage = 0.0
        task.progress_message = "Resuming..."
        task.last_seen_timestamp = datetime.now(UTC)
        task.stop_requested = False
        task.stop_requested_timestamp = None
        db.session.commit()
        
        run_id = f"run_{str(task.id).replace('-', '_')}"
        response = {
            "run_id": run_id,
            "state": "running",
        }
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_artifact_list(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.artifact.list"""
    req = ArtifactListRequest(**arguments)
    session_id = req.session_id
    
    run_id = get_task_id_for_session(session_id)
    if not run_id:
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
        )]
    
    entries = []
    
    # Fetch file list from worker_plan
    files_list = await fetch_file_list_from_worker_plan(run_id)
    if files_list is None:
        # Return empty list if run doesn't exist yet
        response = {"entries": []}
        return [TextContent(type="text", text=json.dumps(response))]
    
    # Filter by path if specified
    if req.path:
        files_list = [f for f in files_list if f.startswith(req.path)]
    
    for file_name in files_list:
        # Skip log.txt as per spec
        if file_name == "log.txt":
            continue
        
        artifact_uri = f"planexe://sessions/{session_id}/out/{file_name}"
        
        # Guess content type from extension
        content_type = "application/octet-stream"
        if file_name.endswith(".md"):
            content_type = "text/markdown"
        elif file_name.endswith(".html"):
            content_type = "text/html"
        elif file_name.endswith(".json"):
            content_type = "application/json"
        elif file_name.endswith(".txt"):
            content_type = "text/plain"
        
        # For metadata, we'd need to fetch the file, but that's expensive
        # For now, return basic info without size/hash if metadata not required
        entry = {
            "type": "file",
            "path": file_name,
            "artifact_uri": artifact_uri,
            "content_type": content_type,
            "kind": "intermediate",
        }
        
        if req.include_metadata:
            # Fetch file to compute hash (expensive, but required by spec)
            content_bytes = await fetch_artifact_from_worker_plan(run_id, file_name)
            if content_bytes:
                entry["size"] = len(content_bytes)
                entry["sha256"] = compute_sha256(content_bytes)
                entry["updated_at"] = datetime.now(UTC).isoformat() + "Z"  # Approximate
            else:
                entry["size"] = 0
                entry["sha256"] = ""
                entry["updated_at"] = datetime.now(UTC).isoformat() + "Z"
        
        entries.append(entry)
    
    response = {"entries": entries}
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_artifact_read(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.artifact.read"""
    req = ArtifactReadRequest(**arguments)
    artifact_uri = req.artifact_uri
    
    # Parse artifact_uri: planexe://sessions/{session_id}/out/{path}
    if not artifact_uri.startswith("planexe://sessions/"):
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INVALID_ARTIFACT_URI", "message": f"Invalid artifact URI format: {artifact_uri}"}})
        )]
    
    parts = artifact_uri.replace("planexe://sessions/", "").split("/out/", 1)
    if len(parts) != 2:
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INVALID_ARTIFACT_URI", "message": f"Invalid artifact URI format: {artifact_uri}"}})
        )]
    
    session_id = parts[0]
    artifact_path = parts[1]
    
    # Security: ensure path doesn't contain path traversal
    if ".." in artifact_path or "/" in artifact_path and artifact_path.startswith("/"):
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "PERMISSION_DENIED", "message": "Path traversal detected"}})
        )]
    
    run_id = get_task_id_for_session(session_id)
    if not run_id:
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
        )]
    
    # Fetch artifact from worker_plan
    content_bytes = await fetch_artifact_from_worker_plan(run_id, artifact_path)
    if content_bytes is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INVALID_ARTIFACT_URI", "message": f"Artifact not found: {artifact_uri}"}})
        )]
    
    try:
        content_hash = compute_sha256(content_bytes)
        
        # Apply range if specified
        if req.range:
            start = req.range.get("start", 0)
            length = req.range.get("length", len(content_bytes))
            content_bytes = content_bytes[start:start+length]
        
        # Decode as text (try UTF-8, fallback to errors='replace')
        try:
            content = content_bytes.decode('utf-8')
            content_type = "text/plain"
            if artifact_path.endswith(".md"):
                content_type = "text/markdown"
            elif artifact_path.endswith(".html"):
                content_type = "text/html"
            elif artifact_path.endswith(".json"):
                content_type = "application/json"
        except UnicodeDecodeError:
            # Binary file, return base64 or indicate binary
            content = f"[Binary file, {len(content_bytes)} bytes]"
            content_type = "application/octet-stream"
        
        response = {
            "artifact_uri": artifact_uri,
            "content_type": content_type,
            "sha256": content_hash,
            "content": content,
        }
    except Exception as e:
        logger.error(f"Error processing artifact {artifact_uri}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INTERNAL_ERROR", "message": str(e)}})
        )]
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_artifact_write(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.artifact.write"""
    req = ArtifactWriteRequest(**arguments)
    artifact_uri = req.artifact_uri
    
    # Parse artifact_uri
    if not artifact_uri.startswith("planexe://sessions/"):
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INVALID_ARTIFACT_URI", "message": f"Invalid artifact URI format: {artifact_uri}"}})
        )]
    
    parts = artifact_uri.replace("planexe://sessions/", "").split("/out/", 1)
    if len(parts) != 2:
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INVALID_ARTIFACT_URI", "message": f"Invalid artifact URI format: {artifact_uri}"}})
        )]
    
    session_id = parts[0]
    artifact_path = parts[1]
    
    with app.app_context():
        # Check if session exists and is not running
        task = find_task_by_session_id(session_id)
        if task is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        # Check if run is active (strict policy: reject writes while running)
        if task.state == TaskState.processing:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "RUNNING_READONLY", "message": "Cannot write artifacts while run is active. Stop the run first."}})
            )]
        
        run_id = get_task_id_for_session(session_id)
        if not run_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        # Security check: ensure path doesn't contain path traversal
        if ".." in artifact_path or "/" in artifact_path and artifact_path.startswith("/"):
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "PERMISSION_DENIED", "message": "Path traversal detected"}})
            )]
        
        # Optimistic locking check - fetch current file first
        if req.lock and req.lock.get("expected_sha256"):
            current_bytes = await fetch_artifact_from_worker_plan(run_id, artifact_path)
            if current_bytes:
                current_hash = compute_sha256(current_bytes)
                if current_hash != req.lock["expected_sha256"]:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": {"code": "CONFLICT", "message": "SHA256 mismatch. Artifact was modified."}})
                    )]
        
        # Write file - for now, this is not supported via worker_plan HTTP API
        # We'd need to add a write endpoint to worker_plan, or use a different approach
        # For now, return an error indicating write is not yet supported via HTTP
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "NOT_IMPLEMENTED", "message": "Artifact write not yet supported via HTTP. Use database-backed approach or file system mount."}})
        )]
    
    return [TextContent(type="text", text=json.dumps(response))]

async def handle_session_events(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle planexe.session.events"""
    req = SessionEventsRequest(**arguments)
    session_id = req.session_id
    
    with app.app_context():
        task = find_task_by_session_id(session_id)
        if task is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"}})
            )]
        
        # Query events for this task
        since_id = int(req.since.replace("cursor_", "")) if req.since and req.since.startswith("cursor_") else 0
        
        events_query = db.session.query(EventItem).filter(
            EventItem.context.contains({"task_id": str(task.id)})
        ).filter(EventItem.id > since_id).order_by(EventItem.id.asc())
        
        events = events_query.all()
        
        event_list = []
        for event in events:
            event_type_map = {
                EventType.TASK_PROCESSING: "run_started",
                EventType.TASK_COMPLETED: "run_completed",
                EventType.TASK_FAILED: "run_failed",
                EventType.TASK_PENDING: "run_started",
            }
            
            mcp_event_type = event_type_map.get(event.event_type, "log")
            event_list.append({
                "ts": event.timestamp.isoformat() + "Z",
                "type": mcp_event_type,
                "data": event.context or {},
            })
        
        cursor = f"cursor_{events[-1].id}" if events else (req.since or "cursor_0")
        
        response = {
            "cursor": cursor,
            "events": event_list,
        }
    
    return [TextContent(type="text", text=json.dumps(response))]

async def main():
    """Main entry point for MCP server."""
    logger.info("Starting PlanExe MCP Server...")
    
    # Initialize database
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")
    
    # Run MCP server over stdio
    async with stdio_server() as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

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
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus
from io import BytesIO
import httpx
from sqlalchemy import cast, text
from sqlalchemy.dialects.postgresql import JSONB
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, Tool, TextContent
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
from mcp_server.tool_models import (
    ErrorDetail,
    ReportReadyOutput,
    TaskCreateOutput,
    TaskStatusSuccess,
    TaskStopOutput,
)

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
    statements = (
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested BOOLEAN",
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested_timestamp TIMESTAMP",
    )
    with db.engine.begin() as conn:
        for statement in statements:
            try:
                conn.execute(text(statement))
            except Exception as exc:
                logger.warning("Schema update failed for %s: %s", statement, exc, exc_info=True)

with app.app_context():
    ensure_taskitem_stop_columns()

# MCP Server setup
mcp_server = Server("planexe-mcp-server")

# Base directory for run artifacts (not used directly, fetched via worker_plan HTTP API)
BASE_DIR_RUN = Path(os.environ.get("PLANEXE_RUN_DIR", Path(__file__).parent.parent / "run")).resolve()

# Worker plan HTTP API URL
WORKER_PLAN_URL = os.environ.get("PLANEXE_WORKER_PLAN_URL", "http://worker_plan:8000")

REPORT_FILENAME = "030-report.html"
REPORT_CONTENT_TYPE = "text/html; charset=utf-8"
ZIP_FILENAME = "run.zip"
ZIP_CONTENT_TYPE = "application/zip"
ZIP_SNAPSHOT_MAX_BYTES = 100_000_000

SPEED_VS_DETAIL_DEFAULT = "ping_llm"
SPEED_VS_DETAIL_DEFAULT_ALIAS = "ping"
SPEED_VS_DETAIL_VALUES = (
    "ping_llm",
    "fast_but_skip_details",
    "all_details_but_slow",
)
SPEED_VS_DETAIL_INPUT_VALUES = (
    "ping",
    "fast",
    "all",
)
SPEED_VS_DETAIL_ALIASES = {
    "ping": "ping_llm",
    "fast": "fast_but_skip_details",
    "all": "all_details_but_slow",
}

# Pydantic models for request/response validation
class TaskCreateRequest(BaseModel):
    idea: str
    speed_vs_detail: Optional[str] = None

class TaskStatusRequest(BaseModel):
    task_id: str

class TaskStopRequest(BaseModel):
    task_id: str

class ReportReadRequest(BaseModel):
    task_id: str
    artifact: Optional[str] = None

# Helper functions
def find_task_by_task_id(task_id: str) -> Optional[TaskItem]:
    """Find TaskItem by MCP task_id (UUID), with legacy fallback."""
    task = get_task_by_id(task_id)
    if task is not None:
        return task

    def _query_legacy() -> Optional[TaskItem]:
        query = db.session.query(TaskItem)
        if db.engine.dialect.name == "postgresql":
            tasks = query.filter(
                cast(TaskItem.parameters, JSONB).contains({"_mcp_task_id": task_id})
            ).all()
        else:
            tasks = query.filter(
                TaskItem.parameters.contains({"_mcp_task_id": task_id})
            ).all()
        if tasks:
            return tasks[0]
        return None

    if has_app_context():
        legacy_task = _query_legacy()
    else:
        with app.app_context():
            legacy_task = _query_legacy()
    if legacy_task is not None:
        logger.debug("Resolved legacy MCP task id %s to task %s", task_id, legacy_task.id)
    return legacy_task

def get_task_by_id(task_id: str) -> Optional[TaskItem]:
    """Fetch a TaskItem by its UUID string."""
    def _query() -> Optional[TaskItem]:
        try:
            task_uuid = uuid.UUID(task_id)
        except ValueError:
            return None
        return db.session.get(TaskItem, task_uuid)

    if has_app_context():
        return _query()
    with app.app_context():
        return _query()

def resolve_task_for_task_id(task_id: str) -> Optional[TaskItem]:
    """Resolve a TaskItem from a task_id (UUID), with legacy fallback."""
    return find_task_by_task_id(task_id)

def _create_task_sync(
    idea: str,
    config: Optional[dict[str, Any]],
    metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    with app.app_context():
        parameters = dict(config or {})
        parameters["speed_vs_detail"] = resolve_speed_vs_detail(parameters)

        task = TaskItem(
            prompt=idea,
            state=TaskState.pending,
            user_id=metadata.get("user_id", "mcp_user") if metadata else "mcp_user",
            parameters=parameters,
        )
        db.session.add(task)
        db.session.commit()

        task_id = str(task.id)
        event_context = {
            "task_id": task_id,
            "task_handle": task_id,
            "prompt": task.prompt,
            "user_id": task.user_id,
            "config": config,
            "metadata": metadata,
            "parameters": task.parameters,
        }
        event = EventItem(
            event_type=EventType.TASK_PENDING,
            message="Enqueued task via MCP",
            context=event_context,
        )
        db.session.add(event)
        db.session.commit()

        created_at = task.timestamp_created
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return {
            "task_id": task_id,
            "created_at": created_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }

def _get_task_status_snapshot_sync(task_id: str) -> Optional[dict[str, Any]]:
    with app.app_context():
        task = find_task_by_task_id(task_id)
        if task is None:
            return None
        return {
            "id": str(task.id),
            "state": task.state,
            "stop_requested": bool(task.stop_requested),
            "progress_percentage": task.progress_percentage,
            "timestamp_created": task.timestamp_created,
        }

def _request_task_stop_sync(task_id: str) -> bool:
    with app.app_context():
        task = find_task_by_task_id(task_id)
        if task is None:
            return False
        if task.state in (TaskState.pending, TaskState.processing):
            task.stop_requested = True
            task.stop_requested_timestamp = datetime.now(UTC)
            task.progress_message = "Stop requested by user."
            db.session.commit()
            logger.info("Stop requested for task %s; stop flag set on task %s.", task_id, task.id)
        return True

def _get_task_for_report_sync(task_id: str) -> Optional[dict[str, Any]]:
    with app.app_context():
        task = resolve_task_for_task_id(task_id)
        if task is None:
            return None
        return {
            "id": str(task.id),
            "state": task.state,
            "progress_message": task.progress_message,
        }

def list_files_from_zip_bytes(zip_bytes: bytes) -> list[str]:
    """List file entries from an in-memory zip archive."""
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_file:
            files = [name for name in zip_file.namelist() if not name.endswith("/")]
            return sorted(files)
    except Exception as exc:
        logger.warning("Unable to list files from zip snapshot: %s", exc)
        return []

def extract_file_from_zip_bytes(zip_bytes: bytes, file_path: str) -> Optional[bytes]:
    """Extract a file from an in-memory zip archive."""
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zip_file:
            file_path_normalized = file_path.lstrip('/')
            try:
                return zip_file.read(file_path_normalized)
            except KeyError:
                return None
    except Exception as exc:
        logger.warning("Unable to read %s from zip snapshot: %s", file_path, exc)
        return None

def extract_file_from_zip_file(file_handle: io.BufferedIOBase, file_path: str) -> Optional[bytes]:
    """Extract a file from a seekable zip file handle."""
    try:
        with zipfile.ZipFile(file_handle, 'r') as zip_file:
            file_path_normalized = file_path.lstrip('/')
            try:
                return zip_file.read(file_path_normalized)
            except KeyError:
                return None
    except Exception as exc:
        logger.warning("Unable to read %s from zip stream: %s", file_path, exc)
        return None

def fetch_report_from_db(task_id: str) -> Optional[bytes]:
    """Fetch the report HTML stored in the TaskItem."""
    task = get_task_by_id(task_id)
    if task and task.generated_report_html is not None:
        return task.generated_report_html.encode("utf-8")
    return None

def fetch_zip_snapshot(task_id: str) -> Optional[bytes]:
    """Fetch the zip snapshot stored in the TaskItem."""
    task = get_task_by_id(task_id)
    if task and task.run_zip_snapshot is not None:
        return task.run_zip_snapshot
    return None

def fetch_file_from_zip_snapshot(task_id: str, file_path: str) -> Optional[bytes]:
    """Fetch a file from the TaskItem zip snapshot."""
    task = get_task_by_id(task_id)
    if task and task.run_zip_snapshot is not None:
        return extract_file_from_zip_bytes(task.run_zip_snapshot, file_path)
    return None

def list_files_from_zip_snapshot(task_id: str) -> Optional[list[str]]:
    """List files from the TaskItem zip snapshot."""
    task = get_task_by_id(task_id)
    if task and task.run_zip_snapshot is not None:
        return list_files_from_zip_bytes(task.run_zip_snapshot)
    return None

async def fetch_artifact_from_worker_plan(run_id: str, file_path: str) -> Optional[bytes]:
    """Fetch an artifact file from worker_plan via HTTP."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # For report.html, use the dedicated report endpoint (most efficient)
            if (
                file_path == "report.html"
                or file_path.endswith("/report.html")
                or file_path == REPORT_FILENAME
                or file_path.endswith(f"/{REPORT_FILENAME}")
            ):
                report_response = await client.get(f"{WORKER_PLAN_URL}/runs/{run_id}/report")
                if report_response.status_code == 200:
                    return report_response.content
                logger.warning(f"Worker plan returned {report_response.status_code} for report: {run_id}")
                report_from_db = await asyncio.to_thread(fetch_report_from_db, run_id)
                if report_from_db is not None:
                    return report_from_db
                report_from_zip = await asyncio.to_thread(
                    fetch_file_from_zip_snapshot, run_id, REPORT_FILENAME
                )
                if report_from_zip is not None:
                    return report_from_zip
                return None
            
            # For other files, fetch the zip and extract the file
            # This is less efficient but works without a file serving endpoint
            async with client.stream("GET", f"{WORKER_PLAN_URL}/runs/{run_id}/zip") as zip_response:
                if zip_response.status_code != 200:
                    logger.warning(f"Worker plan returned {zip_response.status_code} for zip: {run_id}")
                else:
                    zip_too_large = False
                    content_length = zip_response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > ZIP_SNAPSHOT_MAX_BYTES:
                                logger.warning(
                                    "Zip snapshot too large (%s bytes) for run %s; skipping.",
                                    content_length,
                                    run_id,
                                )
                                zip_too_large = True
                        except ValueError:
                            logger.warning(
                                "Invalid Content-Length for zip snapshot: %s", content_length
                            )
                    if not zip_too_large:
                        with tempfile.TemporaryFile() as tmp_file:
                            size = 0
                            async for chunk in zip_response.aiter_bytes():
                                size += len(chunk)
                                if size > ZIP_SNAPSHOT_MAX_BYTES:
                                    logger.warning(
                                        "Zip snapshot exceeded max size (%s bytes) for run %s; skipping.",
                                        ZIP_SNAPSHOT_MAX_BYTES,
                                        run_id,
                                    )
                                    zip_too_large = True
                                    break
                                tmp_file.write(chunk)
                            if not zip_too_large:
                                tmp_file.seek(0)
                                file_data = extract_file_from_zip_file(tmp_file, file_path)
                                if file_data is not None:
                                    return file_data

            snapshot_file = await asyncio.to_thread(fetch_file_from_zip_snapshot, run_id, file_path)
            if snapshot_file is not None:
                return snapshot_file
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
            fallback_files = await asyncio.to_thread(list_files_from_zip_snapshot, run_id)
            if fallback_files is not None:
                return fallback_files
            return None
    except Exception as e:
        logger.error(f"Error fetching file list from worker_plan: {e}", exc_info=True)
        return None

async def fetch_zip_from_worker_plan(run_id: str) -> Optional[bytes]:
    """Fetch the zip snapshot from worker_plan via HTTP."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", f"{WORKER_PLAN_URL}/runs/{run_id}/zip") as response:
                if response.status_code != 200:
                    logger.warning("Worker plan returned %s for zip: %s", response.status_code, run_id)
                else:
                    zip_too_large = False
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > ZIP_SNAPSHOT_MAX_BYTES:
                                logger.warning(
                                    "Zip snapshot too large (%s bytes) for run %s; skipping.",
                                    content_length,
                                    run_id,
                                )
                                zip_too_large = True
                        except ValueError:
                            logger.warning(
                                "Invalid Content-Length for zip snapshot: %s", content_length
                            )
                    if not zip_too_large:
                        buffer = BytesIO()
                        size = 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > ZIP_SNAPSHOT_MAX_BYTES:
                                logger.warning(
                                    "Zip snapshot exceeded max size (%s bytes) for run %s; skipping.",
                                    ZIP_SNAPSHOT_MAX_BYTES,
                                    run_id,
                                )
                                zip_too_large = True
                                break
                            buffer.write(chunk)
                        if not zip_too_large:
                            return buffer.getvalue()

            snapshot_bytes = await asyncio.to_thread(fetch_zip_snapshot, run_id)
            if snapshot_bytes is not None:
                return snapshot_bytes
            return None
    except Exception as e:
        logger.error(f"Error fetching zip from worker_plan: {e}", exc_info=True)
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

def resolve_speed_vs_detail(config: Optional[dict[str, Any]]) -> str:
    value: Optional[str] = None
    if isinstance(config, dict):
        raw_value = config.get("speed_vs_detail") or config.get("speed")
        if isinstance(raw_value, str):
            value = raw_value.strip().lower()
    if value in SPEED_VS_DETAIL_ALIASES:
        return SPEED_VS_DETAIL_ALIASES[value]
    if value in SPEED_VS_DETAIL_VALUES:
        return value
    return SPEED_VS_DETAIL_DEFAULT

def _merge_task_create_config(
    config: Optional[dict[str, Any]],
    speed_vs_detail: Optional[str],
) -> Optional[dict[str, Any]]:
    merged = dict(config or {})
    if isinstance(speed_vs_detail, str):
        candidate = speed_vs_detail.strip()
        if candidate and "speed_vs_detail" not in merged and "speed" not in merged:
            merged["speed_vs_detail"] = candidate
    return merged or None

def build_report_download_path(task_id: str) -> str:
    return f"/download/{task_id}/{REPORT_FILENAME}"

def build_report_download_url(task_id: str) -> Optional[str]:
    base_url = os.environ.get("PLANEXE_MCP_PUBLIC_BASE_URL")
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}{build_report_download_path(task_id)}"

def build_zip_download_path(task_id: str) -> str:
    return f"/download/{task_id}/{ZIP_FILENAME}"

def build_zip_download_url(task_id: str) -> Optional[str]:
    base_url = os.environ.get("PLANEXE_MCP_PUBLIC_BASE_URL")
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}{build_zip_download_path(task_id)}"

# Output schemas for MCP tools.
ERROR_SCHEMA = ErrorDetail.model_json_schema()
TASK_CREATE_OUTPUT_SCHEMA = TaskCreateOutput.model_json_schema()
TASK_STATUS_SUCCESS_SCHEMA = TaskStatusSuccess.model_json_schema()
TASK_STATUS_OUTPUT_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"error": ERROR_SCHEMA},
            "required": ["error"],
        },
        TASK_STATUS_SUCCESS_SCHEMA,
    ]
}
TASK_STOP_OUTPUT_SCHEMA = TaskStopOutput.model_json_schema()
REPORT_READY_OUTPUT_SCHEMA = ReportReadyOutput.model_json_schema()
TASK_RESULT_OUTPUT_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"error": ERROR_SCHEMA},
            "required": ["error"],
        },
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        REPORT_READY_OUTPUT_SCHEMA,
    ]
}

TASK_CREATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string", "description": "The idea/prompt for the plan"},
        "speed_vs_detail": {
            "type": "string",
            "enum": list(SPEED_VS_DETAIL_INPUT_VALUES),
            "default": SPEED_VS_DETAIL_DEFAULT_ALIAS,
            "description": (
                "Defaults to ping (alias for ping_llm). Options: ping, fast, all."
            ),
        },
    },
    "required": ["idea"],
}
TASK_STATUS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
    },
    "required": ["task_id"],
}
TASK_STOP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
    },
    "required": ["task_id"],
}
TASK_RESULT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "artifact": {
            "type": "string",
            "enum": ["report", "zip"],
            "default": "report",
            "description": "Download artifact type: report or zip.",
        },
    },
    "required": ["task_id"],
}

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: Optional[dict[str, Any]] = None

TOOL_DEFINITIONS = [
    ToolDefinition(
        name="task_create",
        description=(
            "Start creating a new plan. speed_vs_detail modes: "
            "'all' runs the full pipeline with all details (slower, higher token usage/cost). "
            "'fast' runs the full pipeline with minimal work per step (faster, fewer details), "
            "useful to verify the pipeline is working. "
            "'ping' runs the pipeline entrypoint and makes a single LLM call to verify the "
            "worker_plan_database is processing tasks and can reach the LLM."
        ),
        input_schema=TASK_CREATE_INPUT_SCHEMA,
        output_schema=TASK_CREATE_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_status",
        description="Returns status and progress of the plan currently being created.",
        input_schema=TASK_STATUS_INPUT_SCHEMA,
        output_schema=TASK_STATUS_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_stop",
        description="Stops the plan that is currently being created.",
        input_schema=TASK_STOP_INPUT_SCHEMA,
        output_schema=TASK_STOP_OUTPUT_SCHEMA,
    ),
    ToolDefinition(
        name="task_result",
        description="Returns download metadata for the report or zip snapshot.",
        input_schema=TASK_RESULT_INPUT_SCHEMA,
        output_schema=TASK_RESULT_OUTPUT_SCHEMA,
    ),
]

# MCP Tool implementations
@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name=definition.name,
            description=definition.description,
            outputSchema=definition.output_schema,
            inputSchema=definition.input_schema,
        )
        for definition in TOOL_DEFINITIONS
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "INVALID_TOOL", "message": f"Unknown tool: {name}"}})
            )]
        return await handler(arguments)
    except Exception as e:
        logger.error(f"Error handling tool {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "INTERNAL_ERROR", "message": str(e)}})
        )]

async def handle_task_create(arguments: dict[str, Any]) -> CallToolResult:
    """Handle task_create"""
    req = TaskCreateRequest(**arguments)

    merged_config = _merge_task_create_config(None, req.speed_vs_detail)
    response = await asyncio.to_thread(
        _create_task_sync,
        req.idea,
        merged_config,
        None,
    )
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )

async def handle_task_status(arguments: dict[str, Any]) -> CallToolResult:
    """Handle task_status"""
    req = TaskStatusRequest(**arguments)
    task_id = req.task_id

    task_snapshot = await asyncio.to_thread(_get_task_status_snapshot_sync, task_id)
    if task_snapshot is None:
        response = {
            "error": {
                "code": "TASK_NOT_FOUND",
                "message": f"Task not found: {task_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    progress_percent = (
        int(round(float(task_snapshot["progress_percentage"])))
        if task_snapshot["progress_percentage"]
        else 0
    )

    task_state = task_snapshot["state"]
    state = get_task_state_mapping(task_state)
    if task_state == TaskState.processing and task_snapshot["stop_requested"]:
        state = "stopping"
    if task_state == TaskState.completed:
        progress_percent = 100

    # Collect files from worker_plan
    task_uuid = task_snapshot["id"]
    files = []
    if task_uuid:
        files_list = await fetch_file_list_from_worker_plan(task_uuid)
        if files_list:
            for file_name in files_list[:10]:  # Limit to 10 most recent
                if file_name != "log.txt":
                    updated_at = datetime.now(UTC).replace(microsecond=0)
                    files.append({
                        "path": file_name,
                        "updated_at": updated_at.isoformat().replace("+00:00", "Z"),  # Approximate
                    })

    created_at = task_snapshot["timestamp_created"]
    if created_at and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    response = {
        "task_id": task_uuid,
        "state": state,
        "progress_percent": progress_percent,
        "timing": {
            "started_at": (
                created_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if created_at
                else None
            ),
            "elapsed_sec": (datetime.now(UTC) - created_at).total_seconds() if created_at else 0,
        },
        "files": files[:10],  # Limit to 10 most recent
    }

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )

async def handle_task_stop(arguments: dict[str, Any]) -> CallToolResult:
    """Handle task_stop"""
    req = TaskStopRequest(**arguments)
    task_id = req.task_id

    found = await asyncio.to_thread(_request_task_stop_sync, task_id)
    if not found:
        response = {
            "error": {
                "code": "TASK_NOT_FOUND",
                "message": f"Task not found: {task_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    response = {
        "state": "stopped",
    }

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )

async def handle_report_read(arguments: dict[str, Any]) -> CallToolResult:
    """Handle task_result."""
    req = ReportReadRequest(**arguments)
    task_id = req.task_id
    artifact = req.artifact.strip().lower() if isinstance(req.artifact, str) else "report"
    if artifact not in ("report", "zip"):
        artifact = "report"
    task_snapshot = await asyncio.to_thread(_get_task_for_report_sync, task_id)
    if task_snapshot is None:
        response = {
            "error": {
                "code": "TASK_NOT_FOUND",
                "message": f"Task not found: {task_id}",
            }
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

    task_state = task_snapshot["state"]
    if task_state in (TaskState.pending, TaskState.processing) or task_state is None:
        response = {}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )
    if task_state == TaskState.failed:
        message = task_snapshot["progress_message"] or "Plan generation failed."
        response = {"error": {"code": "generation_failed", "message": message}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=False,
        )

    run_id = task_snapshot["id"]
    if artifact == "zip":
        content_bytes = await fetch_zip_from_worker_plan(run_id)
        if content_bytes is None:
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

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response))],
        structuredContent=response,
        isError=False,
    )

TOOL_HANDLERS = {
    "task_create": handle_task_create,
    "task_status": handle_task_status,
    "task_stop": handle_task_stop,
    "task_result": handle_report_read,
}

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

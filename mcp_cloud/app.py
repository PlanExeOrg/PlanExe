"""
PlanExe MCP Cloud

Implements the Model Context Protocol interface for PlanExe as specified in
 extra/planexe_mcp_interface.md. Communicates with worker_plan_database via the shared
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote_plus
from io import BytesIO
import httpx
from sqlalchemy import cast, text, or_
from sqlalchemy.dialects.postgresql import JSONB
from mcp.server import Server
try:
    from mcp.server import request_ctx
except Exception:  # pragma: no cover - optional in older MCP SDKs
    request_ctx = None
try:
    from mcp.server.errors import InvalidParamsError, MethodNotFoundError
except Exception:  # pragma: no cover - optional in older MCP SDKs
    try:
        from mcp.server.exceptions import InvalidParamsError, MethodNotFoundError
    except Exception:  # pragma: no cover - optional in older MCP SDKs
        InvalidParamsError = None
        MethodNotFoundError = None
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, Tool, TextContent
from pydantic import BaseModel

from mcp_cloud.dotenv_utils import load_planexe_dotenv
_dotenv_loaded, _dotenv_paths = load_planexe_dotenv(Path(__file__).parent)

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

from database_api.planexe_db_singleton import db
from database_api.model_taskitem import TaskItem, TaskState
from database_api.model_event import EventItem, EventType
from flask import Flask, has_app_context
from mcp_cloud.tool_models import (
    ErrorDetail,
    PlanGenerateOutput,
    TaskFileInfoReadyOutput,
    TaskCreateOutput,
    TaskStatusSuccess,
    TaskStopOutput,
)

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
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS timestamp_updated TIMESTAMP",
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS task_ttl_ms INTEGER",
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS task_expires_at TIMESTAMP",
    )
    with db.engine.begin() as conn:
        for statement in statements:
            try:
                conn.execute(text(statement))
            except Exception as exc:
                logger.warning("Schema update failed for %s: %s", statement, exc, exc_info=True)

with app.app_context():
    ensure_taskitem_stop_columns()

mcp_cloud = Server("planexe-mcp-cloud")

# Base directory for run artifacts (not used directly, fetched via worker_plan HTTP API)
BASE_DIR_RUN = Path(os.environ.get("PLANEXE_RUN_DIR", Path(__file__).parent.parent / "run")).resolve()

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
SpeedVsDetailInput = Literal[
    "ping",
    "fast",
    "all",
]
SPEED_VS_DETAIL_ALIASES = {
    "ping": "ping_llm",
    "fast": "fast_but_skip_details",
    "all": "all_details_but_slow",
}

TASK_TTL_MS_DEFAULT = 12 * 60 * 60 * 1000
TASK_TTL_MS_MIN = 60 * 60 * 1000
TASK_TTL_MS_MAX = 24 * 60 * 60 * 1000
TASK_POLL_INTERVAL_MS = 1000
TASK_LIST_DEFAULT_LIMIT = 50
TASK_CURSOR_PREFIX = "cursor_"

class TaskNotFoundError(Exception):
    pass

def _raise_invalid_params(message: str) -> None:
    if InvalidParamsError is not None:
        raise InvalidParamsError(message)
    raise TaskNotFoundError(message)

def _raise_method_not_found(message: str) -> None:
    if MethodNotFoundError is not None:
        raise MethodNotFoundError(message)
    raise ValueError(message)

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

class TaskFileInfoRequest(BaseModel):
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

def _normalize_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value

def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    normalized = _normalize_datetime(value)
    if normalized is None:
        return None
    return normalized.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _get_request_context() -> Optional[Any]:
    if request_ctx is not None:
        try:
            return request_ctx.get()
        except Exception:
            try:
                return request_ctx()
            except Exception:
                return None
    return getattr(mcp_cloud, "request_context", None)

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

def _resolve_user_id(meta: Optional[dict[str, Any]]) -> str:
    if isinstance(meta, dict):
        for key in ("user_id", "userId", "user", "username"):
            value = meta.get(key)
            if value:
                return str(value)
    return "mcp_user"

def _normalize_task_ttl_ms(task_hint: Optional[dict[str, Any]]) -> int:
    ttl_value = None
    if isinstance(task_hint, dict):
        ttl_value = task_hint.get("ttl")
    if ttl_value is None:
        return TASK_TTL_MS_DEFAULT
    try:
        ttl_ms = int(ttl_value)
    except (TypeError, ValueError):
        return TASK_TTL_MS_DEFAULT
    if ttl_ms <= 0:
        return TASK_TTL_MS_DEFAULT
    if ttl_ms < TASK_TTL_MS_MIN:
        return TASK_TTL_MS_MIN
    if ttl_ms > TASK_TTL_MS_MAX:
        return TASK_TTL_MS_MAX
    return ttl_ms

def _compute_task_expires_at(created_at: datetime, ttl_ms: int) -> datetime:
    created_at_normalized = _normalize_datetime(created_at) or datetime.now(UTC)
    return created_at_normalized + timedelta(milliseconds=ttl_ms)

def _is_task_expired(task: TaskItem, now: Optional[datetime] = None) -> bool:
    expires_at = _normalize_datetime(task.task_expires_at)
    if expires_at is None:
        return False
    now_value = _normalize_datetime(now) or datetime.now(UTC)
    return expires_at <= now_value

def _sweep_expired_tasks_sync(now: Optional[datetime] = None) -> int:
    now_value = _normalize_datetime(now) or datetime.now(UTC)
    def _query() -> int:
        query = (
            db.session.query(TaskItem)
            .filter(TaskItem.task_expires_at.isnot(None))
            .filter(TaskItem.task_expires_at < now_value)
        )
        query = query.filter(
            or_(
                TaskItem.state.in_([TaskState.completed, TaskState.failed]),
                TaskItem.stop_requested.is_(True),
            )
        )
        tasks = query.all()
        for task in tasks:
            db.session.delete(task)
        if tasks:
            db.session.commit()
        return len(tasks)

    if has_app_context():
        return _query()
    with app.app_context():
        return _query()

def _find_task_by_idempotency_sync(user_id: str, idempotency_key: str) -> Optional[TaskItem]:
    def _query() -> Optional[TaskItem]:
        query = db.session.query(TaskItem).filter(TaskItem.user_id == user_id)
        if db.engine.dialect.name == "postgresql":
            query = query.filter(
                cast(TaskItem.parameters, JSONB).contains({"idempotency_key": idempotency_key})
            )
        else:
            query = query.filter(
                TaskItem.parameters.contains({"idempotency_key": idempotency_key})
            )
        return query.order_by(TaskItem.timestamp_created.desc()).first()

    if has_app_context():
        return _query()
    with app.app_context():
        return _query()

def _create_or_get_task_sync(
    idea: str,
    config: Optional[dict[str, Any]],
    user_id: str,
    idempotency_key: Optional[str],
    task_ttl_ms: int,
) -> tuple[TaskItem, bool]:
    with app.app_context():
        now_value = datetime.now(UTC)
        _sweep_expired_tasks_sync()
        if idempotency_key:
            existing = _find_task_by_idempotency_sync(user_id, idempotency_key)
            if existing is not None and not _is_task_expired(existing):
                if existing.task_expires_at is None:
                    existing.task_ttl_ms = task_ttl_ms
                    existing.task_expires_at = _compute_task_expires_at(
                        existing.timestamp_created or now_value, task_ttl_ms
                    )
                    existing.timestamp_updated = now_value
                    db.session.commit()
                return existing, False

        parameters = dict(config or {})
        parameters["speed_vs_detail"] = resolve_speed_vs_detail(parameters)
        if idempotency_key:
            parameters["idempotency_key"] = idempotency_key

        task = TaskItem(
            prompt=idea,
            state=TaskState.pending,
            user_id=user_id,
            parameters=parameters,
        )
        task.timestamp_updated = now_value
        task.task_ttl_ms = task_ttl_ms
        task.task_expires_at = _compute_task_expires_at(task.timestamp_created or now_value, task_ttl_ms)
        db.session.add(task)
        db.session.commit()

        task_id = str(task.id)
        event_context = {
            "task_id": task_id,
            "task_handle": task_id,
            "prompt": task.prompt,
            "user_id": task.user_id,
            "config": config,
            "parameters": task.parameters,
        }
        event = EventItem(
            event_type=EventType.TASK_PENDING,
            message="Enqueued task via MCP",
            context=event_context,
        )
        db.session.add(event)
        db.session.commit()
        return task, True

def _create_task_sync(
    idea: str,
    config: Optional[dict[str, Any]],
    metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    user_id = metadata.get("user_id", "mcp_user") if isinstance(metadata, dict) else "mcp_user"
    task, _created = _create_or_get_task_sync(
        idea=idea,
        config=config,
        user_id=user_id,
        idempotency_key=None,
        task_ttl_ms=TASK_TTL_MS_DEFAULT,
    )
    created_at = _format_datetime(task.timestamp_created)
    return {
        "task_id": str(task.id),
        "created_at": created_at,
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
            "timestamp_updated": task.timestamp_updated,
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
            task.timestamp_updated = datetime.now(UTC)
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

def _get_mcp_task_status(task: TaskItem) -> str:
    if task.stop_requested:
        return "cancelled"
    if task.state in (TaskState.pending, TaskState.processing):
        return "working"
    if task.state == TaskState.completed:
        return "completed"
    if task.state == TaskState.failed:
        return "failed"
    return "working"

def _get_task_status_message(task: TaskItem) -> Optional[str]:
    message = task.progress_message if isinstance(task.progress_message, str) else None
    if message:
        return message
    if task.stop_requested:
        return "Cancellation requested."
    return None

def _build_task_payload(task: TaskItem, poll_interval_ms: int) -> dict[str, Any]:
    created_at = _normalize_datetime(task.timestamp_created) or datetime.now(UTC)
    last_updated = _normalize_datetime(task.timestamp_updated) or _normalize_datetime(
        task.stop_requested_timestamp
    ) or created_at
    payload = {
        "taskId": str(task.id),
        "status": _get_mcp_task_status(task),
        "createdAt": _format_datetime(created_at),
        "lastUpdatedAt": _format_datetime(last_updated),
        "ttl": int(task.task_ttl_ms or TASK_TTL_MS_DEFAULT),
        "pollInterval": int(poll_interval_ms),
    }
    status_message = _get_task_status_message(task)
    if status_message:
        payload["statusMessage"] = status_message
    return payload

def _build_related_task_meta(task_id: str, immediate_response: Optional[str] = None) -> dict[str, Any]:
    meta = {"io.modelcontextprotocol/related-task": {"taskId": task_id}}
    if immediate_response:
        meta["io.modelcontextprotocol/model-immediate-response"] = immediate_response
    return meta

def _build_create_task_result(task: TaskItem, poll_interval_ms: int) -> dict[str, Any]:
    return {
        "task": _build_task_payload(task, poll_interval_ms),
        "_meta": _build_related_task_meta(
            str(task.id),
            immediate_response="Task accepted. Use tasks/get or tasks/result to continue.",
        ),
    }

def _is_terminal_status(status: str) -> bool:
    return status in ("completed", "failed", "cancelled", "input_required")

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

async def _build_artifact_info(task_id: str, artifact: str, task_state: TaskState) -> Optional[dict[str, Any]]:
    if artifact == "report":
        if task_state != TaskState.completed:
            return None
        content_bytes = await fetch_artifact_from_worker_plan(task_id, REPORT_FILENAME)
        if content_bytes is None:
            return None
        response = {
            "content_type": REPORT_CONTENT_TYPE,
            "sha256": compute_sha256(content_bytes),
            "download_size": len(content_bytes),
        }
        download_url = build_report_download_url(task_id)
        if download_url:
            response["download_url"] = download_url
        return response

    if task_state not in (TaskState.completed, TaskState.failed):
        return None
    content_bytes = await fetch_zip_from_worker_plan(task_id)
    if content_bytes is None:
        return None
    response = {
        "content_type": ZIP_CONTENT_TYPE,
        "sha256": compute_sha256(content_bytes),
        "download_size": len(content_bytes),
    }
    download_url = build_zip_download_url(task_id)
    if download_url:
        response["download_url"] = download_url
    return response

async def _build_plan_generate_output(task: TaskItem) -> dict[str, Any]:
    status = _get_mcp_task_status(task)
    progress_value = float(task.progress_percentage or 0.0)
    if status == "completed":
        progress_value = 100.0

    payload: dict[str, Any] = {
        "task_id": str(task.id),
        "status": status,
        "progress_percentage": progress_value,
    }

    message = _get_task_status_message(task)
    if message:
        payload["message"] = message

    if status in ("completed", "failed", "cancelled"):
        report_info = await _build_artifact_info(str(task.id), "report", task.state)
        if report_info:
            payload["report"] = report_info
        zip_info = await _build_artifact_info(str(task.id), "zip", task.state)
        if zip_info:
            payload["zip"] = zip_info

    if status == "failed":
        payload["error"] = {
            "code": "TASK_FAILED",
            "message": task.progress_message or "Plan generation failed.",
        }
    elif status == "cancelled":
        payload["error"] = {
            "code": "TASK_CANCELLED",
            "message": task.progress_message or "Task cancelled.",
        }
    return payload

async def _wait_for_task_terminal(task_id: str, poll_interval_ms: int) -> TaskItem:
    while True:
        task = await asyncio.to_thread(resolve_task_for_task_id, task_id)
        if task is None:
            raise TaskNotFoundError(f"Task not found: {task_id}")
        status = _get_mcp_task_status(task)
        if _is_terminal_status(status):
            return task
        await asyncio.sleep(poll_interval_ms / 1000.0)

def _make_task_cursor(task_id: str) -> str:
    return f"{TASK_CURSOR_PREFIX}{task_id}"

def _parse_task_cursor(cursor: Optional[str]) -> Optional[str]:
    if not cursor or not cursor.startswith(TASK_CURSOR_PREFIX):
        return None
    return cursor[len(TASK_CURSOR_PREFIX):]

def _list_tasks_sync(
    user_id: Optional[str],
    cursor: Optional[str],
    limit: int,
) -> tuple[list[TaskItem], Optional[str]]:
    with app.app_context():
        query = db.session.query(TaskItem)
        if user_id:
            query = query.filter(TaskItem.user_id == user_id)

        cursor_task_id = _parse_task_cursor(cursor)
        if cursor_task_id:
            cursor_task = get_task_by_id(cursor_task_id)
            if cursor_task and cursor_task.timestamp_created:
                query = query.filter(TaskItem.timestamp_created < cursor_task.timestamp_created)
        query = query.order_by(TaskItem.timestamp_created.desc())
        tasks = query.limit(limit + 1).all()
        next_cursor = None
        if len(tasks) > limit:
            next_cursor = _make_task_cursor(str(tasks[limit - 1].id))
            tasks = tasks[:limit]
        return tasks, next_cursor

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

def _extract_request_meta_only() -> Optional[dict[str, Any]]:
    ctx = _get_request_context()
    request_obj = None
    if ctx is not None:
        request_obj = getattr(ctx, "request", None) or getattr(ctx, "raw_request", None)
        if request_obj is None and isinstance(ctx, dict):
            request_obj = ctx.get("request") or ctx.get("raw_request")
    if request_obj is None:
        return None
    if isinstance(request_obj, dict):
        meta = request_obj.get("_meta") or request_obj.get("meta")
        return meta if isinstance(meta, dict) else None
    meta_value = getattr(request_obj, "_meta", None) or getattr(request_obj, "meta", None)
    return meta_value if isinstance(meta_value, dict) else None

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

ERROR_SCHEMA = ErrorDetail.model_json_schema()
PLAN_GENERATE_OUTPUT_SCHEMA = PlanGenerateOutput.model_json_schema()
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
TASK_FILE_INFO_READY_OUTPUT_SCHEMA = TaskFileInfoReadyOutput.model_json_schema()
TASK_FILE_INFO_OUTPUT_SCHEMA = {
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
        TASK_FILE_INFO_READY_OUTPUT_SCHEMA,
    ]
}

PLAN_GENERATE_INPUT_SCHEMA = {
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
        "idempotency_key": {
            "type": "string",
            "description": (
                "Optional key to dedupe repeated requests for the same user."
            ),
        },
    },
    "required": ["idea"],
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
TASK_FILE_INFO_INPUT_SCHEMA = {
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
    task_support: Optional[str] = None

TOOL_DEFINITIONS = [
    ToolDefinition(
        name="plan_generate",
        description=(
            "Generate a new plan. Supports MCP task augmentation for long-running runs. "
            "speed_vs_detail modes: 'all' runs the full pipeline with all details (slower, higher token usage/cost). "
            "'fast' runs the full pipeline with minimal work per step (faster, fewer details), "
            "useful to verify the pipeline is working. "
            "'ping' runs the pipeline entrypoint and makes a single LLM call to verify the "
            "worker_plan_database is processing tasks and can reach the LLM."
        ),
        input_schema=PLAN_GENERATE_INPUT_SCHEMA,
        output_schema=PLAN_GENERATE_OUTPUT_SCHEMA,
        task_support="optional",
    ),
    ToolDefinition(
        name="task_create",
        description=(
            "Legacy wrapper: start creating a new plan. speed_vs_detail modes: "
            "'all' runs the full pipeline with all details (slower, higher token usage/cost). "
            "'fast' runs the full pipeline with minimal work per step (faster, fewer details), "
            "useful to verify the pipeline is working. "
            "'ping' runs the pipeline entrypoint and makes a single LLM call to verify the "
            "worker_plan_database is processing tasks and can reach the LLM."
        ),
        input_schema=TASK_CREATE_INPUT_SCHEMA,
        output_schema=TASK_CREATE_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
    ToolDefinition(
        name="task_status",
        description="Legacy wrapper: returns status and progress of the plan currently being created.",
        input_schema=TASK_STATUS_INPUT_SCHEMA,
        output_schema=TASK_STATUS_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
    ToolDefinition(
        name="task_stop",
        description="Legacy wrapper: stops the plan that is currently being created.",
        input_schema=TASK_STOP_INPUT_SCHEMA,
        output_schema=TASK_STOP_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
    ToolDefinition(
        name="task_file_info",
        description="Legacy wrapper: returns file metadata for the report or zip snapshot.",
        input_schema=TASK_FILE_INFO_INPUT_SCHEMA,
        output_schema=TASK_FILE_INFO_OUTPUT_SCHEMA,
        task_support="forbidden",
    ),
]

@mcp_cloud.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available MCP tools."""
    tools: list[Tool] = []
    for definition in TOOL_DEFINITIONS:
        execution = None
        if definition.task_support:
            execution = {"taskSupport": definition.task_support}
        tool_kwargs = {
            "name": definition.name,
            "description": definition.description,
            "outputSchema": definition.output_schema,
            "inputSchema": definition.input_schema,
        }
        if execution:
            try:
                tools.append(Tool(**tool_kwargs, execution=execution))
                continue
            except TypeError:
                logger.debug("Tool execution field not supported by MCP SDK.")
        tools.append(Tool(**tool_kwargs))
    return tools

def _build_call_tool_result(
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

async def _dispatch_tool_call(
    name: str,
    arguments: dict[str, Any],
    task_hint: Optional[dict[str, Any]],
    meta: Optional[dict[str, Any]],
) -> CallToolResult | dict[str, Any]:
    if name == "plan_generate":
        return await handle_plan_generate(arguments, task_hint, meta)

    if task_hint is not None:
        response = {
            "error": {
                "code": "TASKS_NOT_SUPPORTED",
                "message": f"Tool does not support task execution: {name}",
            }
        }
        return _build_call_tool_result(response, is_error=True)

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        response = {"error": {"code": "INVALID_TOOL", "message": f"Unknown tool: {name}"}}
        return _build_call_tool_result(response, is_error=True)
    return await handler(arguments)

@mcp_cloud.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult | dict[str, Any]:
    """Dispatch MCP tool calls and return structured JSON errors for unknown tools."""
    try:
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
    except Exception as e:
        logger.error(f"Error handling tool {name}: {e}", exc_info=True)
        response = {"error": {"code": "INTERNAL_ERROR", "message": str(e)}}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response))],
            structuredContent=response,
            isError=True,
        )

async def handle_plan_generate(
    arguments: dict[str, Any],
    task_hint: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> CallToolResult | dict[str, Any]:
    """Generate a plan, optionally as an MCP task."""
    req = PlanGenerateRequest(**arguments)
    user_id = _resolve_user_id(meta)
    merged_config = _merge_task_create_config(None, req.speed_vs_detail)
    task_ttl_ms = _normalize_task_ttl_ms(task_hint)
    task, _created = await asyncio.to_thread(
        _create_or_get_task_sync,
        req.idea,
        merged_config,
        user_id,
        req.idempotency_key,
        task_ttl_ms,
    )

    if task_hint is not None:
        return _build_create_task_result(task, TASK_POLL_INTERVAL_MS)

    task_terminal = await _wait_for_task_terminal(str(task.id), TASK_POLL_INTERVAL_MS)
    payload = await _build_plan_generate_output(task_terminal)
    return _build_call_tool_result(payload)

async def handle_task_create(arguments: dict[str, Any]) -> CallToolResult:
    """Create a new PlanExe task and enqueue it for processing.

    Examples:
        - {"idea": "Draft a 3-day Tokyo itinerary"} → returns task_id + created_at
        - {"idea": "Generate onboarding plan", "speed_vs_detail": "fast"} → faster run

    Args:
        - idea: Prompt/goal for the plan.
        - speed_vs_detail: Optional mode ("ping" | "fast" | "all").

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: {"task_id": ..., "created_at": ...}
        - isError: False on success.
    """
    req = TaskCreateRequest(**arguments)

    merged_config = _merge_task_create_config(None, req.speed_vs_detail)
    response = await asyncio.to_thread(
        _create_task_sync,
        req.idea,
        merged_config,
        None,
    )
    return _build_call_tool_result(response)

async def handle_task_status(arguments: dict[str, Any]) -> CallToolResult:
    """Fetch the current run status, progress, and recent files for a task.

    Examples:
        - {"task_id": "uuid"} → state/progress/timing + recent files

    Args:
        - task_id: Task UUID returned by task_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: status payload or error.
        - isError: True only when task_id is unknown.
    """
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
        return _build_call_tool_result(response, is_error=True)

    progress_percentage = float(task_snapshot.get("progress_percentage") or 0.0)

    task_state = task_snapshot["state"]
    state = get_task_state_mapping(task_state)
    if task_state == TaskState.processing and task_snapshot["stop_requested"]:
        state = "stopping"
    if task_state == TaskState.completed:
        progress_percentage = 100.0

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
        "progress_percentage": progress_percentage,
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

    return _build_call_tool_result(response)

async def handle_task_stop(arguments: dict[str, Any]) -> CallToolResult:
    """Request the active run for a task to stop.

    Examples:
        - {"task_id": "uuid"} → stop request accepted

    Args:
        - task_id: Task UUID returned by task_create.

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: {"state": "stopped"} or error payload.
        - isError: True only when task_id is unknown.
    """
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
        return _build_call_tool_result(response, is_error=True)

    response = {
        "state": "stopped",
    }

    return _build_call_tool_result(response)

async def handle_task_file_info(arguments: dict[str, Any]) -> CallToolResult:
    """Return download metadata for a task's report or zip artifact.

    Examples:
        - {"task_id": "uuid"} → report metadata (default)
        - {"task_id": "uuid", "artifact": "zip"} → zip metadata

    Args:
        - task_id: Task UUID returned by task_create.
        - artifact: Optional "report" or "zip".

    Returns:
        - content: JSON string matching structuredContent.
        - structuredContent: metadata (content_type, sha256, download_size,
          optional download_url) or {} if not ready, or error payload.
        - isError: True only when task_id is unknown.
    """
    req = TaskFileInfoRequest(**arguments)
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
        return _build_call_tool_result(response, is_error=True)

    run_id = task_snapshot["id"]
    if artifact == "zip":
        content_bytes = await fetch_zip_from_worker_plan(run_id)
        if content_bytes is None:
            task_state = task_snapshot["state"]
            if task_state in (TaskState.pending, TaskState.processing) or task_state is None:
                response = {}
            else:
                response = {
                    "error": {
                        "code": "content_unavailable",
                        "message": "zip content_bytes is None",
                    },
                }
            return _build_call_tool_result(response)

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

        return _build_call_tool_result(response)

    task_state = task_snapshot["state"]
    if task_state in (TaskState.pending, TaskState.processing) or task_state is None:
        response = {}
        return _build_call_tool_result(response)
    if task_state == TaskState.failed:
        message = task_snapshot["progress_message"] or "Plan generation failed."
        response = {"error": {"code": "generation_failed", "message": message}}
        return _build_call_tool_result(response)

    content_bytes = await fetch_artifact_from_worker_plan(run_id, REPORT_FILENAME)
    if content_bytes is None:
        response = {
            "error": {
                "code": "content_unavailable",
                "message": "content_bytes is None",
            },
        }
        return _build_call_tool_result(response)

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

    return _build_call_tool_result(response)

async def handle_tasks_get(params: Any) -> dict[str, Any]:
    await asyncio.to_thread(_sweep_expired_tasks_sync)
    task_id = _extract_task_id(params)
    if not task_id:
        _raise_invalid_params("taskId is required.")
    task = await asyncio.to_thread(resolve_task_for_task_id, task_id)
    if task is None:
        _raise_invalid_params(f"Task not found: {task_id}")
    return _build_task_payload(task, TASK_POLL_INTERVAL_MS)

async def handle_tasks_result(params: Any) -> CallToolResult:
    await asyncio.to_thread(_sweep_expired_tasks_sync)
    task_id = _extract_task_id(params)
    if not task_id:
        _raise_invalid_params("taskId is required.")
    task_terminal = await _wait_for_task_terminal(task_id, TASK_POLL_INTERVAL_MS)
    payload = await _build_plan_generate_output(task_terminal)
    meta = _build_related_task_meta(task_id)
    return _build_call_tool_result(payload, meta=meta)

async def handle_tasks_cancel(params: Any) -> dict[str, Any]:
    await asyncio.to_thread(_sweep_expired_tasks_sync)
    task_id = _extract_task_id(params)
    if not task_id:
        _raise_invalid_params("taskId is required.")
    task = await asyncio.to_thread(resolve_task_for_task_id, task_id)
    if task is None:
        _raise_invalid_params(f"Task not found: {task_id}")
    status = _get_mcp_task_status(task)
    if _is_terminal_status(status):
        _raise_invalid_params(f"Task already terminal: {task_id}")
    await asyncio.to_thread(_request_task_stop_sync, task_id)
    updated_task = await asyncio.to_thread(resolve_task_for_task_id, task_id)
    if updated_task is None:
        _raise_invalid_params(f"Task not found: {task_id}")
    return _build_task_payload(updated_task, TASK_POLL_INTERVAL_MS)

async def handle_tasks_list(params: Any) -> dict[str, Any]:
    await asyncio.to_thread(_sweep_expired_tasks_sync)
    cursor, limit_value = _extract_list_params(params)
    limit = TASK_LIST_DEFAULT_LIMIT
    if isinstance(limit_value, (int, float)) and int(limit_value) > 0:
        limit = int(limit_value)
    meta = _extract_request_meta_only()
    user_id = _resolve_user_id(meta) if meta else None
    tasks, next_cursor = await asyncio.to_thread(_list_tasks_sync, user_id, cursor, limit)
    payload = {"tasks": [_build_task_payload(task, TASK_POLL_INTERVAL_MS) for task in tasks]}
    if next_cursor:
        payload["nextCursor"] = next_cursor
    return payload

def _register_mcp_method(server: Server, name: str, handler: Any) -> None:
    registrar = getattr(server, "register_method", None) or getattr(server, "method", None)
    if registrar is None:
        logger.warning("MCP SDK does not support method registration; %s disabled.", name)
        return
    registrar(name)(handler)

TOOL_HANDLERS = {
    "task_create": handle_task_create,
    "task_status": handle_task_status,
    "task_stop": handle_task_stop,
    "task_file_info": handle_task_file_info,
}

_register_mcp_method(mcp_cloud, "tasks/get", handle_tasks_get)
_register_mcp_method(mcp_cloud, "tasks/result", handle_tasks_result)
_register_mcp_method(mcp_cloud, "tasks/cancel", handle_tasks_cancel)
_register_mcp_method(mcp_cloud, "tasks/list", handle_tasks_list)

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

async def main():
    """Main entry point for MCP server."""
    logger.info("Starting PlanExe MCP Cloud...")
    
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")
    
    async with stdio_server() as streams:
        init_options = _apply_tasks_capability(mcp_cloud.create_initialization_options())
        await mcp_cloud.run(
            streams[0],
            streams[1],
            init_options
        )

if __name__ == "__main__":
    asyncio.run(main())

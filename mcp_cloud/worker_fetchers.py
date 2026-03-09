"""PlanExe MCP Cloud – HTTP fetchers for worker_plan artifacts."""
import asyncio
import logging
import tempfile
import uuid as _uuid
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, Optional

import httpx
from flask import has_app_context
from worker_plan_api.format_datetime import format_datetime_utc

from mcp_cloud.db_setup import (
    BASE_DIR_RUN,
    REPORT_FILENAME,
    WORKER_PLAN_URL,
    ZIP_SNAPSHOT_MAX_BYTES,
)
from mcp_cloud.zip_utils import (
    _sanitize_legacy_zip_snapshot,
    extract_file_from_zip_file,
    fetch_file_from_zip_snapshot,
    fetch_report_from_db,
    fetch_zip_snapshot,
    list_files_from_zip_snapshot,
)

logger = logging.getLogger(__name__)


async def fetch_artifact_from_worker_plan(run_id: str, file_path: str) -> Optional[bytes]:
    """Fetch an artifact file from worker_plan via HTTP.

    For report artifacts, three fallback steps are tried independently:
    1. DB lookup (generated_report_html column)
    2. Zip snapshot extraction
    3. HTTP request to worker service (last resort)

    Each step has its own try/except so a failure in one does not skip
    the remaining fallbacks.  Fast local sources are tried first so that
    a slow/unreachable worker does not block the response.
    """
    is_report = (
        file_path == "report.html"
        or file_path.endswith("/report.html")
        or file_path == REPORT_FILENAME
        or file_path.endswith(f"/{REPORT_FILENAME}")
    )

    if is_report:
        return await _fetch_report_with_fallbacks(run_id)

    # For other files, fetch the zip and extract the file.
    # This is less efficient but works without a file serving endpoint.
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
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


async def _fetch_report_with_fallbacks(run_id: str) -> Optional[bytes]:
    """Fetch report HTML using three isolated fallback steps.

    Order: DB first, then zip snapshot, then HTTP (last resort).
    Fast local sources are tried before the network call so that a
    slow/unreachable worker does not block the response.

    Each step is wrapped in its own try/except so that a failure in one
    does not prevent the next fallback from being attempted.
    """
    # Step 1: Try DB lookup (generated_report_html column) — fast
    try:
        report_from_db = await asyncio.to_thread(fetch_report_from_db, run_id)
        if report_from_db is not None:
            return report_from_db
    except Exception as e:
        logger.warning(f"DB fetch failed for report {run_id}: {e}")

    # Step 2: Try zip snapshot extraction — fast
    try:
        report_from_zip = await asyncio.to_thread(
            fetch_file_from_zip_snapshot, run_id, REPORT_FILENAME
        )
        if report_from_zip is not None:
            return report_from_zip
    except Exception as e:
        logger.warning(f"Zip snapshot fetch failed for report {run_id}: {e}")

    # Step 3: Try HTTP request to worker service — last resort, short timeout
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            report_response = await client.get(f"{WORKER_PLAN_URL}/runs/{run_id}/report")
            if report_response.status_code == 200:
                return report_response.content
            logger.warning(f"Worker plan returned {report_response.status_code} for report: {run_id}")
    except Exception as e:
        logger.warning(f"HTTP fetch failed for report {run_id}: {e}")

    return None

async def fetch_file_list_from_worker_plan(run_id: str) -> Optional[list[tuple[str, str]]]:
    """Fetch the list of files from worker_plan via HTTP.

    Returns list of (filename, ISO-8601 UTC timestamp) tuples, or None.
    """

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            response = await client.get(f"{WORKER_PLAN_URL}/runs/{run_id}/files")
            if response.status_code == 200:
                data = response.json()
                # Prefer files_with_timestamps when available (new worker)
                timestamped = data.get("files_with_timestamps")
                if timestamped:
                    return [(e["name"], e["updated_at"]) for e in timestamped]
                # Fall back to plain filenames with current time
                files = data.get("files", [])
                if files:
                    now = format_datetime_utc(datetime.now(UTC))
                    return [(f, now) for f in files]
                fallback_files = await asyncio.to_thread(list_files_from_zip_snapshot, run_id)
                if fallback_files:
                    return fallback_files
                return []
            logger.warning(f"Worker plan returned {response.status_code} for files list: {run_id}")
            fallback_files = await asyncio.to_thread(list_files_from_zip_snapshot, run_id)
            if fallback_files is not None:
                return fallback_files
            return None
    except Exception as e:
        logger.error(f"Error fetching file list from worker_plan: {e}", exc_info=True)
        return None


def list_files_from_local_run_dir(run_id: str) -> Optional[list[tuple[str, str]]]:
    """
    List files from local run directory when this service shares PLANEXE_RUN_DIR
    with the worker (e.g., Docker compose).

    Returns list of (filename, ISO-8601 UTC timestamp) tuples sorted by name,
    or None if the directory does not exist.
    """

    run_dir = (BASE_DIR_RUN / run_id).resolve()
    try:
        if not run_dir.is_relative_to(BASE_DIR_RUN):
            return None
    except ValueError:
        return None
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    try:
        results = []
        for path in run_dir.iterdir():
            if path.is_file():
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                mtime_str = format_datetime_utc(mtime)
                results.append((path.name, mtime_str))
        results.sort(key=lambda t: t[0])
        return results
    except Exception as exc:
        logger.warning("Unable to list local run dir files for %s: %s", run_id, exc)
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


def _load_zip_snapshot_sync(plan_id: str) -> dict[str, Any]:
    """Load zip snapshot and layout version inside a single app context.

    Returns dict with 'id', 'run_zip_snapshot', 'run_artifact_layout_version',
    or empty dict if plan not found.
    """
    from mcp_cloud.db_setup import app, db
    from database_api.model_planitem import PlanItem

    def _query():
        try:
            plan_uuid = _uuid.UUID(plan_id)
        except ValueError:
            return {}
        plan = db.session.get(PlanItem, plan_uuid)
        if plan is None:
            return {}
        return {
            "id": str(plan.id),
            "run_zip_snapshot": plan.run_zip_snapshot,
            "run_artifact_layout_version": plan.run_artifact_layout_version or 0,
        }

    if has_app_context():
        return _query()
    with app.app_context():
        return _query()


async def fetch_user_downloadable_zip(plan_id: str) -> Optional[bytes]:
    """
    Fetch a user-downloadable zip for a plan.
    New layout snapshots are served directly from PlanItem.run_zip_snapshot.
    Legacy fallbacks are sanitized to remove track_activity.jsonl.
    """
    snapshot_info = await asyncio.to_thread(_load_zip_snapshot_sync, plan_id)
    if not snapshot_info:
        return None

    snapshot_bytes = snapshot_info["run_zip_snapshot"]
    layout_version = snapshot_info["run_artifact_layout_version"]
    if snapshot_bytes is not None:
        if layout_version >= 2:
            return snapshot_bytes
        return _sanitize_legacy_zip_snapshot(snapshot_bytes)

    worker_plan_zip = await fetch_zip_from_worker_plan(snapshot_info["id"])
    if worker_plan_zip is None:
        return None
    return _sanitize_legacy_zip_snapshot(worker_plan_zip)

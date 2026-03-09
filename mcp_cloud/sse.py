"""PlanExe MCP Cloud – Server-Sent Events for real-time plan progress monitoring."""
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import AsyncGenerator, Optional

from mcp_cloud.db_queries import _get_plan_status_snapshot_sync, get_plan_state_mapping
from worker_plan_api.format_datetime import format_datetime_utc
from database_api.model_planitem import PlanState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
SSE_POLL_INTERVAL: float = float(os.environ.get("PLANEXE_SSE_POLL_INTERVAL", "3.0"))
SSE_HEARTBEAT_INTERVAL: float = float(os.environ.get("PLANEXE_SSE_HEARTBEAT_INTERVAL", "20.0"))
SSE_MAX_DURATION: int = int(os.environ.get("PLANEXE_SSE_MAX_DURATION", "3600"))
SSE_MAX_CONNECTIONS_PER_CLIENT: int = int(os.environ.get("PLANEXE_SSE_MAX_CONNECTIONS", "5"))
SSE_MAX_TOTAL_CONNECTIONS: int = int(os.environ.get("PLANEXE_SSE_MAX_TOTAL_CONNECTIONS", "200"))

TERMINAL_STATES = {"completed", "failed"}


class SSEConnectionLimitError(Exception):
    """Raised when SSE connection limits are exceeded."""
    pass


# ---------------------------------------------------------------------------
# Connection tracking
# ---------------------------------------------------------------------------
_connections_per_client: dict[str, int] = defaultdict(int)
_total_connections: int = 0
_connections_lock = asyncio.Lock()


@asynccontextmanager
async def _track_sse_connection(client_id: str):
    """Track SSE connections with per-client and server-wide limits."""
    global _total_connections
    async with _connections_lock:
        if _total_connections >= SSE_MAX_TOTAL_CONNECTIONS:
            raise SSEConnectionLimitError(
                f"Server-wide SSE connection limit ({SSE_MAX_TOTAL_CONNECTIONS}) reached"
            )
        if _connections_per_client[client_id] >= SSE_MAX_CONNECTIONS_PER_CLIENT:
            raise SSEConnectionLimitError(
                f"Per-client SSE connection limit ({SSE_MAX_CONNECTIONS_PER_CLIENT}) reached"
            )
        _connections_per_client[client_id] += 1
        _total_connections += 1
    try:
        yield
    finally:
        async with _connections_lock:
            _connections_per_client[client_id] -= 1
            if _connections_per_client[client_id] <= 0:
                del _connections_per_client[client_id]
            _total_connections -= 1


# ---------------------------------------------------------------------------
# SSE event formatting
# ---------------------------------------------------------------------------

def _format_sse_event(event: str, data: dict) -> str:
    """Format a single SSE event according to the SSE spec."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# SSE stream generator
# ---------------------------------------------------------------------------

async def plan_progress_stream(
    plan_id: str,
    disconnect_event: asyncio.Event,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events for plan progress.

    - Verifies plan exists (sends error event + closes if not found)
    - Polls DB every ~SSE_POLL_INTERVAL seconds
    - Sends 'status' event only when state or progress_percentage changes
    - Sends 'heartbeat' event every ~SSE_HEARTBEAT_INTERVAL seconds of silence
    - Sends 'complete' event on terminal state then closes
    - Absolute timeout at SSE_MAX_DURATION seconds
    """
    # Send retry hint for client reconnection
    yield f"retry: 5000\n\n"

    # Verify plan exists
    snapshot = await asyncio.to_thread(_get_plan_status_snapshot_sync, plan_id)
    if snapshot is None:
        yield _format_sse_event("error", {
            "code": "PLAN_NOT_FOUND",
            "message": f"Plan not found: {plan_id}",
        })
        return

    start_time = time.monotonic()
    last_state: Optional[str] = None
    last_progress: Optional[float] = None
    last_event_time = time.monotonic()

    while True:
        # Check absolute timeout
        elapsed = time.monotonic() - start_time
        if elapsed >= SSE_MAX_DURATION:
            yield _format_sse_event("error", {
                "code": "SSE_TIMEOUT",
                "message": f"SSE stream timed out after {SSE_MAX_DURATION} seconds",
            })
            return

        # Poll DB
        snapshot = await asyncio.to_thread(_get_plan_status_snapshot_sync, plan_id)
        if snapshot is None:
            yield _format_sse_event("error", {
                "code": "PLAN_NOT_FOUND",
                "message": f"Plan not found: {plan_id}",
            })
            return

        plan_state = snapshot["state"]
        state = get_plan_state_mapping(plan_state)
        progress = float(snapshot.get("progress_percentage") or 0.0)
        if plan_state == PlanState.completed:
            progress = 100.0

        created_at = snapshot.get("timestamp_created")
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        elapsed_sec = (datetime.now(UTC) - created_at).total_seconds() if created_at else 0.0

        # Check if state is terminal
        if state in TERMINAL_STATES:
            payload = {
                "plan_id": plan_id,
                "state": state,
                "progress_percentage": progress,
                "elapsed_sec": round(elapsed_sec, 1),
            }
            if state == "failed":
                message = snapshot.get("progress_message") or "Plan generation failed."
                payload["error"] = {"code": "generation_failed", "message": message}
            yield _format_sse_event("complete", payload)
            return

        # Dedup: only send status when something changed
        if state != last_state or progress != last_progress:
            yield _format_sse_event("status", {
                "plan_id": plan_id,
                "state": state,
                "progress_percentage": progress,
                "elapsed_sec": round(elapsed_sec, 1),
            })
            last_state = state
            last_progress = progress
            last_event_time = time.monotonic()
        else:
            # Send heartbeat if enough silence has passed
            if time.monotonic() - last_event_time >= SSE_HEARTBEAT_INTERVAL:
                yield _format_sse_event("heartbeat", {
                    "timestamp": format_datetime_utc(datetime.now(UTC)),
                })
                last_event_time = time.monotonic()

        # Wait for poll interval or disconnect
        try:
            await asyncio.wait_for(disconnect_event.wait(), timeout=SSE_POLL_INTERVAL)
            # disconnect_event was set — client disconnected
            return
        except asyncio.TimeoutError:
            # Normal: poll interval elapsed, loop again
            pass

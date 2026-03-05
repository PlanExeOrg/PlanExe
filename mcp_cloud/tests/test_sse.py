import asyncio
import json
import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

from database_api.model_planitem import PlanState


def _collect_events(gen) -> list[str]:
    """Run an async generator to completion and collect all yielded strings."""
    results = []
    async def _run():
        async for item in gen:
            results.append(item)
    asyncio.run(_run())
    return results


def _parse_sse_events(raw_chunks: list[str]) -> list[dict]:
    """Parse raw SSE text chunks into structured events.

    Returns list of dicts with 'event' and 'data' keys.
    Skips retry hints and empty chunks.
    """
    events = []
    for chunk in raw_chunks:
        lines = chunk.strip().split("\n")
        event_type = None
        data_line = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data_line = line[len("data: "):]
            elif line.startswith("retry:"):
                # retry hint, not a real event
                pass
        if event_type and data_line:
            events.append({"event": event_type, "data": json.loads(data_line)})
    return events


def _make_snapshot(plan_id, state=PlanState.processing, progress=25.0):
    return {
        "id": plan_id,
        "state": state,
        "stop_requested": False,
        "progress_percentage": progress,
        "timestamp_created": datetime.now(UTC),
    }


class TestPlanProgressStream(unittest.TestCase):
    def test_plan_not_found(self):
        """When plan doesn't exist, an error event is emitted and the stream closes."""
        from mcp_cloud.sse import plan_progress_stream

        plan_id = str(uuid.uuid4())
        disconnect = asyncio.Event()

        with patch("mcp_cloud.sse._get_plan_status_snapshot_sync", return_value=None):
            chunks = _collect_events(plan_progress_stream(plan_id, disconnect))

        events = _parse_sse_events(chunks)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertEqual(events[0]["data"]["code"], "PLAN_NOT_FOUND")

    def test_emits_status_on_progress_change(self):
        """Status events are emitted when progress changes between polls."""
        from mcp_cloud.sse import plan_progress_stream

        plan_id = str(uuid.uuid4())
        disconnect = asyncio.Event()

        snapshots = [
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.processing, 50.0),
            _make_snapshot(plan_id, PlanState.completed, 100.0),
        ]
        call_count = 0

        def mock_snapshot(pid):
            nonlocal call_count
            idx = min(call_count, len(snapshots) - 1)
            call_count += 1
            return snapshots[idx]

        with patch("mcp_cloud.sse._get_plan_status_snapshot_sync", side_effect=mock_snapshot):
            chunks = _collect_events(plan_progress_stream(plan_id, disconnect))

        events = _parse_sse_events(chunks)
        status_events = [e for e in events if e["event"] == "status"]
        complete_events = [e for e in events if e["event"] == "complete"]

        self.assertEqual(len(status_events), 2)
        self.assertEqual(status_events[0]["data"]["progress_percentage"], 25.0)
        self.assertEqual(status_events[1]["data"]["progress_percentage"], 50.0)
        self.assertEqual(len(complete_events), 1)
        self.assertEqual(complete_events[0]["data"]["state"], "completed")

    def test_emits_complete_on_terminal(self):
        """A completed plan emits a complete event and the generator stops."""
        from mcp_cloud.sse import plan_progress_stream

        plan_id = str(uuid.uuid4())
        disconnect = asyncio.Event()

        snapshot = _make_snapshot(plan_id, PlanState.completed, 100.0)

        with patch("mcp_cloud.sse._get_plan_status_snapshot_sync", return_value=snapshot):
            chunks = _collect_events(plan_progress_stream(plan_id, disconnect))

        events = _parse_sse_events(chunks)
        complete_events = [e for e in events if e["event"] == "complete"]
        self.assertEqual(len(complete_events), 1)
        self.assertEqual(complete_events[0]["data"]["state"], "completed")
        self.assertEqual(complete_events[0]["data"]["progress_percentage"], 100.0)

    def test_deduplication(self):
        """Same state/progress should not emit duplicate status events."""
        from mcp_cloud.sse import plan_progress_stream

        plan_id = str(uuid.uuid4())
        disconnect = asyncio.Event()

        # Return same state 3 times, then change, then terminal
        snapshots = [
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.processing, 50.0),
            _make_snapshot(plan_id, PlanState.completed, 100.0),
        ]
        call_count = 0

        def mock_snapshot(pid):
            nonlocal call_count
            idx = min(call_count, len(snapshots) - 1)
            call_count += 1
            return snapshots[idx]

        with patch("mcp_cloud.sse._get_plan_status_snapshot_sync", side_effect=mock_snapshot), \
             patch("mcp_cloud.sse.SSE_HEARTBEAT_INTERVAL", 9999):  # Suppress heartbeats
            chunks = _collect_events(plan_progress_stream(plan_id, disconnect))

        events = _parse_sse_events(chunks)
        status_events = [e for e in events if e["event"] == "status"]
        # Only 2 status events: 25% and 50% (dedup removed the repeats)
        self.assertEqual(len(status_events), 2)

    def test_heartbeat(self):
        """Heartbeat events are emitted when no state changes occur."""
        from mcp_cloud.sse import plan_progress_stream

        plan_id = str(uuid.uuid4())
        disconnect = asyncio.Event()

        # Same state 3 times then terminal
        snapshots = [
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.processing, 25.0),
            _make_snapshot(plan_id, PlanState.completed, 100.0),
        ]
        call_count = 0

        def mock_snapshot(pid):
            nonlocal call_count
            idx = min(call_count, len(snapshots) - 1)
            call_count += 1
            return snapshots[idx]

        with patch("mcp_cloud.sse._get_plan_status_snapshot_sync", side_effect=mock_snapshot), \
             patch("mcp_cloud.sse.SSE_HEARTBEAT_INTERVAL", 0), \
             patch("mcp_cloud.sse.SSE_POLL_INTERVAL", 0.01):
            chunks = _collect_events(plan_progress_stream(plan_id, disconnect))

        events = _parse_sse_events(chunks)
        heartbeat_events = [e for e in events if e["event"] == "heartbeat"]
        self.assertGreaterEqual(len(heartbeat_events), 1)

    def test_already_terminal_on_connect(self):
        """A plan that is already failed emits a single complete event."""
        from mcp_cloud.sse import plan_progress_stream

        plan_id = str(uuid.uuid4())
        disconnect = asyncio.Event()

        snapshot = _make_snapshot(plan_id, PlanState.failed, 30.0)

        with patch("mcp_cloud.sse._get_plan_status_snapshot_sync", return_value=snapshot):
            chunks = _collect_events(plan_progress_stream(plan_id, disconnect))

        events = _parse_sse_events(chunks)
        complete_events = [e for e in events if e["event"] == "complete"]
        self.assertEqual(len(complete_events), 1)
        self.assertEqual(complete_events[0]["data"]["state"], "failed")


class TestSSEConnectionTracking(unittest.TestCase):
    def test_connection_limit(self):
        """Exceeding per-client limit raises SSEConnectionLimitError."""
        from mcp_cloud.sse import SSEConnectionLimitError, _track_sse_connection

        async def _run():
            # Open connections up to the limit, then try one more
            contexts = []
            client_id = "test-client"

            with patch("mcp_cloud.sse.SSE_MAX_CONNECTIONS_PER_CLIENT", 2), \
                 patch("mcp_cloud.sse.SSE_MAX_TOTAL_CONNECTIONS", 200):
                # Reset global state for test isolation
                from mcp_cloud.sse import _connections_per_client
                _connections_per_client.clear()
                import mcp_cloud.sse as sse_mod
                sse_mod._total_connections = 0

                ctx1 = _track_sse_connection(client_id)
                cm1 = await ctx1.__aenter__()
                contexts.append((ctx1, cm1))

                ctx2 = _track_sse_connection(client_id)
                cm2 = await ctx2.__aenter__()
                contexts.append((ctx2, cm2))

                with self.assertRaises(SSEConnectionLimitError):
                    ctx3 = _track_sse_connection(client_id)
                    await ctx3.__aenter__()

                # Clean up
                for ctx, _ in contexts:
                    await ctx.__aexit__(None, None, None)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()

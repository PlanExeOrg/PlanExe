import asyncio
import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult
from database_api.model_taskitem import TaskState
from mcp_server.app import handle_session_status


class TestSessionStatusTool(unittest.TestCase):
    def test_session_status_returns_structured_content(self):
        session_id = "pxe_2026_01_18__b1530380"
        task = SimpleNamespace(
            id="b1530380-7942-4e84-827c-6a699b1c1e92",
            state=TaskState.completed,
            stop_requested=False,
            progress_percentage=0.0,
            progress_message="Picked up by server",
            timestamp_created=datetime.now(UTC),
        )
        with patch("mcp_server.app.app.app_context", return_value=nullcontext()), patch(
            "mcp_server.app.find_task_by_session_id", return_value=task
        ), patch(
            "mcp_server.app.get_task_id_for_session", return_value=str(task.id)
        ), patch(
            "mcp_server.app.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_session_status({"session_id": session_id}))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertEqual(result.structuredContent["session_id"], session_id)
        self.assertIn("state", result.structuredContent)
        self.assertIn("progress", result.structuredContent)


if __name__ == "__main__":
    unittest.main()

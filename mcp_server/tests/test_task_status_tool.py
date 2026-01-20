import asyncio
import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult
from database_api.model_taskitem import TaskState
from mcp_server.app import handle_task_status


class TestTaskStatusTool(unittest.TestCase):
    def test_task_status_returns_structured_content(self):
        task_id = "pxe_2026_01_18__b1530380"
        task = SimpleNamespace(
            id="b1530380-7942-4e84-827c-6a699b1c1e92",
            state=TaskState.completed,
            stop_requested=False,
            progress_percentage=0.0,
            progress_message="Picked up by server",
            timestamp_created=datetime.now(UTC),
        )
        with patch("mcp_server.app.app.app_context", return_value=nullcontext()), patch(
            "mcp_server.app.find_task_by_task_id", return_value=task
        ), patch(
            "mcp_server.app.get_task_uuid_for_task_id", return_value=str(task.id)
        ), patch(
            "mcp_server.app.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_task_status({"task_id": task_id}))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertEqual(result.structuredContent["task_id"], task_id)
        self.assertIn("state", result.structuredContent)
        self.assertIn("progress_percent", result.structuredContent)
        self.assertIsInstance(result.structuredContent["progress_percent"], int)
        self.assertEqual(result.structuredContent["progress_percent"], 100)


if __name__ == "__main__":
    unittest.main()

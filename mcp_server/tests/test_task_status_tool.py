import asyncio
import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult
from database_api.model_taskitem import TaskState
from mcp_server.app import handle_task_status


class TestTaskStatusTool(unittest.TestCase):
    def test_task_status_returns_structured_content(self):
        task_id = str(uuid.uuid4())
        task_snapshot = {
            "id": task_id,
            "state": TaskState.completed,
            "stop_requested": False,
            "progress_percentage": 0.0,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_server.app._get_task_status_snapshot_sync",
            return_value=task_snapshot,
        ), patch(
            "mcp_server.app.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_task_status({"task_id": task_id}))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertEqual(result.structuredContent["task_id"], task_id)
        self.assertIn("state", result.structuredContent)
        self.assertIn("progress_percentage", result.structuredContent)
        self.assertIsInstance(result.structuredContent["progress_percentage"], float)
        self.assertEqual(result.structuredContent["progress_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()

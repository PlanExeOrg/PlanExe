import asyncio
import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult
from database_api.model_taskitem import TaskState
from mcp_cloud.app import handle_plan_generate
from typing import Optional


class StubTaskItem:
    def __init__(self, task_id: Optional[str] = None):
        self.id = uuid.UUID(task_id) if task_id else uuid.uuid4()
        self.state = TaskState.pending
        self.timestamp_created = datetime.now(UTC)
        self.timestamp_updated = self.timestamp_created
        self.stop_requested = False
        self.task_ttl_ms = None
        self.progress_message = None


class TestPlanGenerateTool(unittest.TestCase):
    def test_plan_generate_task_hint_returns_task(self):
        stub_task = StubTaskItem()
        with patch(
            "mcp_cloud.app._create_or_get_task_sync",
            return_value=(stub_task, True),
        ):
            result = asyncio.run(
                handle_plan_generate({"idea": "demo"}, task_hint={"ttl": 3600_000})
            )

        self.assertIsInstance(result, dict)
        self.assertIn("task", result)
        self.assertIn("_meta", result)
        self.assertEqual(result["task"]["taskId"], str(stub_task.id))

    def test_plan_generate_sync_returns_call_tool_result(self):
        stub_task = StubTaskItem()
        with patch(
            "mcp_cloud.app._create_or_get_task_sync",
            return_value=(stub_task, True),
        ), patch(
            "mcp_cloud.app._wait_for_task_terminal",
            new=AsyncMock(return_value=stub_task),
        ), patch(
            "mcp_cloud.app._build_plan_generate_output",
            new=AsyncMock(return_value={"task_id": str(stub_task.id), "status": "completed"}),
        ):
            result = asyncio.run(handle_plan_generate({"idea": "demo"}))

        self.assertIsInstance(result, CallToolResult)
        self.assertEqual(result.structuredContent["task_id"], str(stub_task.id))


if __name__ == "__main__":
    unittest.main()

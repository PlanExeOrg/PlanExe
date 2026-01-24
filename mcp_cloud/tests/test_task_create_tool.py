import asyncio
import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

from mcp.types import CallToolResult
from database_api.model_taskitem import TaskState
from mcp_cloud.app import handle_task_create


class TestTaskCreateTool(unittest.TestCase):
    def test_task_create_returns_structured_content(self):
        arguments = {"idea": "xcv"}
        class StubTaskItem:
            def __init__(self):
                self.id = uuid.uuid4()
                self.state = TaskState.pending
                self.timestamp_created = datetime.now(UTC)
                self.timestamp_updated = self.timestamp_created
                self.stop_requested = False
                self.task_ttl_ms = None
                self.progress_message = None

        with patch(
            "mcp_cloud.app._create_or_get_task_sync",
            return_value=(StubTaskItem(), True),
        ):
            result = asyncio.run(handle_task_create(arguments))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertIn("task_id", result.structuredContent)
        self.assertIn("created_at", result.structuredContent)
        self.assertIsInstance(uuid.UUID(result.structuredContent["task_id"]), uuid.UUID)


if __name__ == "__main__":
    unittest.main()

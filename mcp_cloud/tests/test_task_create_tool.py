import asyncio
import unittest
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from mcp.types import CallToolResult
from mcp_cloud.app import handle_task_create


class TestTaskCreateTool(unittest.TestCase):
    def test_task_create_returns_structured_content(self):
        arguments = {"idea": "xcv", "config": None, "metadata": None}
        fake_session = MagicMock()
        class StubTaskItem:
            def __init__(self, prompt: str, state, user_id: str, parameters):
                self.id = uuid.uuid4()
                self.prompt = prompt
                self.state = state
                self.user_id = user_id
                self.parameters = parameters
                self.timestamp_created = datetime.now(UTC)

        with patch("mcp_cloud.app.app.app_context", return_value=nullcontext()), patch(
            "mcp_cloud.app.db.session", fake_session
        ), patch(
            "mcp_cloud.app.TaskItem", StubTaskItem
        ):
            result = asyncio.run(handle_task_create(arguments))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertIn("task_id", result.structuredContent)
        self.assertIn("created_at", result.structuredContent)
        self.assertIsInstance(uuid.UUID(result.structuredContent["task_id"]), uuid.UUID)


if __name__ == "__main__":
    unittest.main()

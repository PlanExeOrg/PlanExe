import asyncio
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from mcp.types import CallToolResult
from mcp_server.app import handle_session_create


class TestSessionCreateTool(unittest.TestCase):
    def test_session_create_returns_structured_content(self):
        arguments = {"idea": "xcv", "config": None, "metadata": None}
        fake_session = MagicMock()
        with patch("mcp_server.app.app.app_context", return_value=nullcontext()), patch(
            "mcp_server.app.db.session", fake_session
        ):
            result = asyncio.run(handle_session_create(arguments))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertIn("session_id", result.structuredContent)
        self.assertIn("output_dir_uri", result.structuredContent)
        self.assertIn("created_at", result.structuredContent)


if __name__ == "__main__":
    unittest.main()

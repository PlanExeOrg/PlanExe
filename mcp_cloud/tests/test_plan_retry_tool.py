import asyncio
import unittest
import uuid
from unittest.mock import patch

from mcp.types import CallToolResult
from mcp_cloud.app import handle_list_tools, handle_plan_retry


class TestPlanRetryTool(unittest.TestCase):
    def test_plan_retry_tool_listed(self):
        tools = asyncio.run(handle_list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertIn("plan_retry", tool_names)

    def test_plan_retry_returns_structured_content(self):
        plan_id = str(uuid.uuid4())
        payload = {
            "plan_id": plan_id,
            "state": "pending",
            "model_profile": "baseline",
            "retried_at": "2026-01-01T00:00:00Z",
        }
        with patch("mcp_cloud.handlers._retry_failed_plan_sync", return_value=payload):
            result = asyncio.run(handle_plan_retry({"plan_id": plan_id}))

        self.assertIsInstance(result, CallToolResult)
        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["plan_id"], plan_id)
        self.assertEqual(result.structuredContent["state"], "pending")
        self.assertEqual(result.structuredContent["model_profile"], "baseline")

    def test_plan_retry_includes_sse_url(self):
        plan_id = str(uuid.uuid4())
        payload = {
            "plan_id": plan_id,
            "state": "pending",
            "model_profile": "baseline",
            "retried_at": "2026-01-01T00:00:00Z",
        }
        base_url = "https://example.com"
        with (
            patch("mcp_cloud.handlers._retry_failed_plan_sync", return_value=payload),
            patch("mcp_cloud.handlers._get_download_base_url", return_value=base_url),
        ):
            result = asyncio.run(handle_plan_retry({"plan_id": plan_id}))

        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["sse_url"], f"{base_url}/sse/plan/{plan_id}")

    def test_plan_retry_returns_plan_not_found(self):
        plan_id = str(uuid.uuid4())
        with patch("mcp_cloud.handlers._retry_failed_plan_sync", return_value=None):
            result = asyncio.run(handle_plan_retry({"plan_id": plan_id}))

        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["error"]["code"], "PLAN_NOT_FOUND")

    def test_plan_retry_returns_plan_not_failed(self):
        plan_id = str(uuid.uuid4())
        payload = {"error": {"code": "PLAN_NOT_FAILED", "message": "Plan is not failed."}}
        with patch("mcp_cloud.handlers._retry_failed_plan_sync", return_value=payload):
            result = asyncio.run(handle_plan_retry({"plan_id": plan_id}))

        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["error"]["code"], "PLAN_NOT_FAILED")


if __name__ == "__main__":
    unittest.main()

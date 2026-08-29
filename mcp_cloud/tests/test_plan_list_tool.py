import asyncio
import unittest
from unittest.mock import patch

from mcp.types import CallToolResult
from mcp_cloud.app import handle_list_tools, handle_plan_list


class TestPlanListTool(unittest.TestCase):
    def test_plan_list_tool_listed(self):
        tools = asyncio.run(handle_list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertIn("plan_list", tool_names)

    def test_plan_list_returns_plans(self):
        fake_plans = [
            {
                "plan_id": "aaa-111",
                "state": "completed",
                "progress_percentage": 100.0,
                "created_at": "2026-01-01T00:00:00Z",
                "prompt_excerpt": "Build a rocket",
            },
            {
                "plan_id": "bbb-222",
                "state": "processing",
                "progress_percentage": 42.0,
                "created_at": "2026-01-02T00:00:00Z",
                "prompt_excerpt": "Open a bakery",
            },
        ]
        user_context = {"user_id": "user-1", "credits_balance": 10.0}
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=user_context), \
             patch("mcp_cloud.handlers._list_plans_sync", return_value=fake_plans):
            result = asyncio.run(handle_plan_list({"user_api_key": "pex_test", "limit": 10}))

        self.assertIsInstance(result, CallToolResult)
        self.assertFalse(result.is_error)
        self.assertEqual(len(result.structured_content["plans"]), 2)
        self.assertIn("Returned 2 plan(s)", result.structured_content["message"])

    def test_plan_list_empty_result(self):
        user_context = {"user_id": "user-1", "credits_balance": 10.0}
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=user_context), \
             patch("mcp_cloud.handlers._list_plans_sync", return_value=[]):
            result = asyncio.run(handle_plan_list({"user_api_key": "pex_test"}))

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["plans"], [])
        self.assertIn("Returned 0 plan(s)", result.structured_content["message"])

    def test_plan_list_clamps_limit(self):
        """Limit is clamped to [1, 50]."""
        user_context = {"user_id": "user-1", "credits_balance": 10.0}
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=user_context), \
             patch("mcp_cloud.handlers._list_plans_sync", return_value=[]) as mock_list:
            asyncio.run(handle_plan_list({"user_api_key": "pex_test", "limit": 999}))
            _, call_args = mock_list.call_args[0][0], mock_list.call_args[0][1]
            self.assertEqual(call_args, 50)

            asyncio.run(handle_plan_list({"user_api_key": "pex_test", "limit": -5}))
            _, call_args = mock_list.call_args[0][0], mock_list.call_args[0][1]
            self.assertEqual(call_args, 1)

    def test_plan_list_invalid_user_api_key(self):
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=None):
            result = asyncio.run(handle_plan_list({"user_api_key": "pex_bad"}))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "INVALID_USER_API_KEY")

    def test_plan_list_requires_key_when_env_set(self):
        with patch.dict("os.environ", {"PLANEXE_MCP_REQUIRE_USER_KEY": "true"}):
            result = asyncio.run(handle_plan_list({"limit": 5}))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "USER_API_KEY_REQUIRED")

    def test_plan_list_no_key_when_not_required(self):
        """When key is not required and not provided, returns all tasks (user_id=None)."""
        with patch.dict("os.environ", {"PLANEXE_MCP_REQUIRE_USER_KEY": "false"}), \
             patch("mcp_cloud.handlers._list_plans_sync", return_value=[]) as mock_list:
            result = asyncio.run(handle_plan_list({"limit": 5}))

        self.assertFalse(result.is_error)
        # user_id should be None
        self.assertIsNone(mock_list.call_args[0][0])

    def test_plan_list_uses_default_limit(self):
        user_context = {"user_id": "user-1", "credits_balance": 10.0}
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=user_context), \
             patch("mcp_cloud.handlers._list_plans_sync", return_value=[]) as mock_list:
            asyncio.run(handle_plan_list({"user_api_key": "pex_test"}))
            _, call_args = mock_list.call_args[0][0], mock_list.call_args[0][1]
            self.assertEqual(call_args, 10)


if __name__ == "__main__":
    unittest.main()

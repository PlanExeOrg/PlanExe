import asyncio
import unittest
from unittest.mock import patch

from mcp.types import CallToolResult
from mcp_cloud.app import handle_list_tools, handle_send_feedback


class TestSendFeedbackTool(unittest.TestCase):
    def test_send_feedback_tool_listed(self):
        tools = asyncio.run(handle_list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertIn("send_feedback", tool_names)

    def test_feedback_success_minimal(self):
        """Minimal feedback with only required fields succeeds."""
        with patch("mcp_cloud.handlers._create_feedback_sync"):
            result = asyncio.run(handle_send_feedback({
                "category": "mcp",
                "message": "Add dark mode support",
            }))

        self.assertIsInstance(result, CallToolResult)
        self.assertFalse(result.is_error)
        self.assertIn("feedback_id", result.structured_content)
        self.assertIn("received_at", result.structured_content)
        self.assertEqual(result.structured_content["message"], "Feedback received. Thank you.")

    def test_feedback_success_all_fields(self):
        """Feedback with all optional fields succeeds."""
        plan_snapshot = {
            "plan_id": "aaa-111",
            "state": "processing",
            "progress_percentage": 42.0,
            "current_step": "SWOT Analysis",
        }
        with patch("mcp_cloud.handlers._get_plan_snapshot_for_feedback_sync", return_value=plan_snapshot), \
             patch("mcp_cloud.handlers._create_feedback_sync"):
            result = asyncio.run(handle_send_feedback({
                "category": "plan",
                "message": "The SWOT section is too generic",
                "plan_id": "aaa-111",
                "rating": 3,
            }))

        self.assertFalse(result.is_error)
        self.assertIn("feedback_id", result.structured_content)

    def test_feedback_invalid_category(self):
        """Invalid category returns INVALID_FEEDBACK error."""
        result = asyncio.run(handle_send_feedback({
            "category": "nonexistent_category",
            "message": "test",
        }))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "INVALID_FEEDBACK")

    def test_feedback_missing_message(self):
        """Missing required message field returns INVALID_FEEDBACK error."""
        result = asyncio.run(handle_send_feedback({
            "category": "mcp",
        }))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "INVALID_FEEDBACK")

    def test_feedback_plan_not_found(self):
        """plan_id that doesn't exist returns PLAN_NOT_FOUND error."""
        with patch("mcp_cloud.handlers._get_plan_snapshot_for_feedback_sync", return_value=None):
            result = asyncio.run(handle_send_feedback({
                "category": "plan",
                "message": "test feedback",
                "plan_id": "nonexistent-uuid",
            }))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "PLAN_NOT_FOUND")

    def test_feedback_rating_out_of_range(self):
        """Rating outside 1-5 returns INVALID_FEEDBACK error."""
        result = asyncio.run(handle_send_feedback({
            "category": "other",
            "message": "Great tool!",
            "rating": 10,
        }))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "INVALID_FEEDBACK")

    def test_feedback_db_failure_returns_success(self):
        """DB write failure is logged but success is returned (fire-and-forget)."""
        with patch("mcp_cloud.handlers._create_feedback_sync", side_effect=RuntimeError("DB is down")):
            result = asyncio.run(handle_send_feedback({
                "category": "other",
                "message": "test feedback",
            }))

        self.assertFalse(result.is_error)
        self.assertIn("feedback_id", result.structured_content)
        self.assertEqual(result.structured_content["message"], "Feedback received. Thank you.")

    def test_feedback_all_categories_accepted(self):
        """All 4 defined categories are accepted."""
        categories = ["mcp", "plan", "code", "docs", "other"]
        for category in categories:
            with patch("mcp_cloud.handlers._create_feedback_sync"):
                result = asyncio.run(handle_send_feedback({
                    "category": category,
                    "message": f"Test {category}",
                }))
            self.assertFalse(result.is_error, f"Category {category} should be accepted")

    def test_feedback_invalid_user_api_key(self):
        """Invalid user_api_key returns INVALID_USER_API_KEY error."""
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=None):
            result = asyncio.run(handle_send_feedback({
                "category": "mcp",
                "message": "test",
                "user_api_key": "pex_bad_key",
            }))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "INVALID_USER_API_KEY")

    def test_feedback_requires_key_when_env_set(self):
        """When PLANEXE_MCP_REQUIRE_USER_KEY is true, missing key returns error."""
        with patch.dict("os.environ", {"PLANEXE_MCP_REQUIRE_USER_KEY": "true"}):
            result = asyncio.run(handle_send_feedback({
                "category": "mcp",
                "message": "test",
            }))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "USER_API_KEY_REQUIRED")

    def test_feedback_no_key_when_not_required(self):
        """When key is not required and not provided, feedback succeeds."""
        with patch.dict("os.environ", {"PLANEXE_MCP_REQUIRE_USER_KEY": "false"}), \
             patch("mcp_cloud.handlers._create_feedback_sync"):
            result = asyncio.run(handle_send_feedback({
                "category": "mcp",
                "message": "test",
            }))

        self.assertFalse(result.is_error)

    def test_feedback_passes_user_id_from_api_key(self):
        """Valid user_api_key resolves user_id and passes it to _create_feedback_sync."""
        user_context = {"user_id": "user-42", "credits_balance": 10.0}
        with patch("mcp_cloud.handlers._resolve_user_from_api_key", return_value=user_context), \
             patch("mcp_cloud.handlers._create_feedback_sync") as mock_create:
            asyncio.run(handle_send_feedback({
                "category": "other",
                "message": "Great tool!",
                "user_api_key": "pex_valid",
            }))

        call_kwargs = mock_create.call_args
        self.assertEqual(call_kwargs.kwargs["user_id"], "user-42")

    def test_feedback_captures_plan_snapshot(self):
        """When plan_id is provided, plan snapshot is passed to _create_feedback_sync."""
        plan_snapshot = {
            "plan_id": "test-uuid",
            "state": "completed",
            "progress_percentage": 100.0,
            "current_step": "Report Generation",
        }
        with patch("mcp_cloud.handlers._get_plan_snapshot_for_feedback_sync", return_value=plan_snapshot) as mock_get, \
             patch("mcp_cloud.handlers._create_feedback_sync") as mock_create:
            asyncio.run(handle_send_feedback({
                "category": "plan",
                "message": "Report looks great",
                "plan_id": "test-uuid",
            }))

        mock_get.assert_called_once_with("test-uuid")
        # Verify plan_snapshot was passed through
        call_kwargs = mock_create.call_args
        self.assertEqual(call_kwargs.kwargs["plan_snapshot"], plan_snapshot)


if __name__ == "__main__":
    unittest.main()

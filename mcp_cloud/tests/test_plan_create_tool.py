import asyncio
import unittest
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from mcp.types import CallToolResult
from mcp_cloud.app import handle_list_tools, handle_plan_create


class StubPlanItem:
    def __init__(self, prompt: str, state, user_id: str, parameters, api_key_id=None):
        self.id = uuid.uuid4()
        self.prompt = prompt
        self.state = state
        self.user_id = user_id
        self.parameters = parameters
        self.api_key_id = api_key_id
        self.timestamp_created = datetime.now(UTC)


class TestPlanCreateTool(unittest.TestCase):
    def test_plan_create_visible_schema_exposes_prompt_and_model_profile(self):
        tools = asyncio.run(handle_list_tools())
        plan_create_tool = next(tool for tool in tools if tool.name == "plan_create")
        properties = plan_create_tool.input_schema.get("properties", {})
        self.assertIn("prompt", properties)
        self.assertIn("model_profile", properties)

    def test_plan_create_returns_structured_content(self):
        arguments = {"prompt": "xcv", "config": None, "metadata": None}
        fake_session = MagicMock()

        with patch("mcp_cloud.db_queries.app.app_context", return_value=nullcontext()), patch(
            "mcp_cloud.db_queries.db.session", fake_session
        ), patch(
            "mcp_cloud.db_queries.PlanItem", StubPlanItem
        ), patch(
            "mcp_cloud.db_queries._find_recent_duplicate_plan", return_value=None
        ):
            result = asyncio.run(handle_plan_create(arguments))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structured_content, dict)
        self.assertIn("plan_id", result.structured_content)
        self.assertIn("created_at", result.structured_content)
        self.assertIsInstance(uuid.UUID(result.structured_content["plan_id"]), uuid.UUID)
        # New plan should not have deduplicated key
        self.assertNotIn("deduplicated", result.structured_content)

    def test_plan_create_dedup_returns_existing_plan(self):
        """When _find_recent_duplicate_plan returns a plan, plan_create returns it with deduplicated=True."""
        arguments = {"prompt": "build a spaceship", "config": None, "metadata": None}
        fake_session = MagicMock()

        existing_id = uuid.uuid4()
        existing_dict = {
            "id": existing_id,
            "timestamp_created": datetime.now(UTC),
        }

        with patch("mcp_cloud.db_queries.app.app_context", return_value=nullcontext()), patch(
            "mcp_cloud.db_queries.db.session", fake_session
        ), patch(
            "mcp_cloud.db_queries.PlanItem", StubPlanItem
        ), patch(
            "mcp_cloud.db_queries._find_recent_duplicate_plan", return_value=existing_dict
        ):
            result = asyncio.run(handle_plan_create(arguments))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structured_content, dict)
        self.assertEqual(result.structured_content["plan_id"], str(existing_id))
        self.assertTrue(result.structured_content["deduplicated"])

    def test_find_recent_duplicate_plan_returns_none_when_window_zero(self):
        """Opt-out: window_seconds=0 always returns None."""
        from mcp_cloud.db_queries import _find_recent_duplicate_plan

        result = _find_recent_duplicate_plan(
            user_id="u1",
            prompt="anything",
            model_profile="baseline",
            window_seconds=0,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

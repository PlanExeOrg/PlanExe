import asyncio
import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult
from database_api.model_planitem import PlanState
from mcp_cloud.app import handle_plan_status


class TestPlanStatusTool(unittest.TestCase):
    def test_plan_status_returns_structured_content(self):
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.completed,
            "stop_requested": False,
            "progress_percentage": 0.0,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertIsInstance(result, CallToolResult)
        self.assertIsInstance(result.structuredContent, dict)
        self.assertEqual(result.structuredContent["plan_id"], plan_id)
        self.assertIn("state", result.structuredContent)
        self.assertIn("progress_percentage", result.structuredContent)
        self.assertIsInstance(result.structuredContent["progress_percentage"], float)
        self.assertEqual(result.structuredContent["progress_percentage"], 100.0)

    def test_plan_status_falls_back_to_zip_snapshot_files_when_primary_source_empty(self):
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": False,
            "progress_percentage": 34.23,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan",
            new=AsyncMock(return_value=[]),
        ), patch(
            "mcp_cloud.handlers.list_files_from_zip_snapshot",
            return_value=[("001-2-plan.txt", "2026-03-08T23:49:53Z"), ("log.txt", "2026-03-08T23:50:00Z")],
        ), patch(
            "mcp_cloud.handlers.list_files_from_local_run_dir",
            return_value=None,
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        files = result.structuredContent["files"]
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], "001-2-plan.txt")
        self.assertEqual(files[0]["updated_at"], "2026-03-08T23:49:53Z")

    def test_plan_status_uses_processing_state_name(self):
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": True,
            "progress_percentage": 10.0,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan",
            new=AsyncMock(return_value=[]),
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertEqual(result.structuredContent["state"], "processing")

    def test_plan_status_returns_plan_not_found_error(self):
        plan_id = str(uuid.uuid4())
        with patch("mcp_cloud.handlers._get_plan_status_snapshot_sync", return_value=None):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["error"]["code"], "PLAN_NOT_FOUND")


    def test_plan_status_completed_normalizes_steps(self):
        """Completed plans must show steps_completed == steps_total even if DB is stale."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.completed,
            "stop_requested": False,
            "progress_percentage": 96.0,
            "steps_completed": 29,
            "steps_total": 30,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structuredContent
        self.assertEqual(sc["progress_percentage"], 100.0)
        self.assertEqual(sc["steps_completed"], 30)
        self.assertEqual(sc["steps_total"], 30)

    def test_plan_status_includes_file_counts_from_db(self):
        """steps_completed and steps_total are read from DB columns."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": False,
            "progress_percentage": 76.67,
            "progress_message": "23 of 30",
            "steps_completed": 23,
            "steps_total": 30,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structuredContent
        self.assertEqual(sc["steps_completed"], 23)
        self.assertEqual(sc["steps_total"], 30)

    def test_plan_status_file_counts_with_extra_files(self):
        """steps_completed counts only expected files, not extra ones."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": False,
            "progress_percentage": 50.0,
            "progress_message": "15 of 30. Extra files: 3",
            "steps_completed": 15,
            "steps_total": 30,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structuredContent
        self.assertEqual(sc["steps_completed"], 15)
        self.assertEqual(sc["steps_total"], 30)

    def test_plan_status_file_counts_null_when_pending(self):
        """steps_completed and steps_total are null before the worker starts."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.pending,
            "stop_requested": False,
            "progress_percentage": 0.0,
            "steps_completed": None,
            "steps_total": None,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structuredContent
        self.assertIsNone(sc["steps_completed"])
        self.assertIsNone(sc["steps_total"])
        self.assertIsNone(sc["current_step"])

    def test_plan_status_includes_current_step(self):
        """current_step is the human-readable label from the DB."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": False,
            "progress_percentage": 50.0,
            "progress_message": "15 of 30",
            "steps_completed": 15,
            "steps_total": 30,
            "current_step": "SWOT Analysis",
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertEqual(result.structuredContent["current_step"], "SWOT Analysis")


if __name__ == "__main__":
    unittest.main()

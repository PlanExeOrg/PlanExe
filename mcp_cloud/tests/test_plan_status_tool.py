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
        self.assertIsInstance(result.structured_content, dict)
        self.assertEqual(result.structured_content["plan_id"], plan_id)
        self.assertIn("state", result.structured_content)
        self.assertIn("progress_percentage", result.structured_content)
        self.assertIsInstance(result.structured_content["progress_percentage"], float)
        self.assertEqual(result.structured_content["progress_percentage"], 100.0)

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
            return_value=[("plan.txt", "2026-03-08T23:49:53Z"), ("log.txt", "2026-03-08T23:50:00Z")],
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        files = result.structured_content["files"]
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], "plan.txt")
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

        self.assertEqual(result.structured_content["state"], "processing")

    def test_plan_status_returns_plan_not_found_error(self):
        plan_id = str(uuid.uuid4())
        with patch("mcp_cloud.handlers._get_plan_status_snapshot_sync", return_value=None):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["error"]["code"], "PLAN_NOT_FOUND")


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

        sc = result.structured_content
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

        sc = result.structured_content
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

        sc = result.structured_content
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

        sc = result.structured_content
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

        self.assertEqual(result.structured_content["current_step"], "SWOT Analysis")

    def test_plan_status_stopped_returns_stopped_state(self):
        """User-stopped plan has state='stopped' and no stop_reason field."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.stopped,
            "stop_requested": True,
            "progress_percentage": 42.0,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertEqual(result.structured_content["state"], "stopped")
        self.assertNotIn("stop_reason", result.structured_content)

    def test_plan_status_actual_failure_has_no_stop_reason(self):
        """Failed plan response does not contain stop_reason field."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.failed,
            "stop_requested": False,
            "progress_percentage": 5.5,
            "timestamp_created": datetime.now(UTC),
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        self.assertEqual(result.structured_content["state"], "failed")
        self.assertNotIn("stop_reason", result.structured_content)
        self.assertIn("error", result.structured_content)

    def test_plan_status_failed_includes_failure_diagnostics(self):
        """Failed plan with all four diagnostic fields populated surfaces them in response."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.failed,
            "stop_requested": False,
            "progress_percentage": 53.0,
            "progress_message": "Generation error",
            "steps_completed": 16,
            "steps_total": 30,
            "current_step": "016-expert_criticism",
            "timestamp_created": datetime.now(UTC),
            "failure_reason": "generation_error",
            "failed_step": "016-expert_criticism",
            "error_message": "LLM provider returned 503",
            "recoverable": True,
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structured_content
        self.assertEqual(sc["state"], "failed")
        self.assertIn("error", sc)
        err = sc["error"]
        self.assertEqual(err["failure_reason"], "generation_error")
        self.assertEqual(err["failed_step"], "016-expert_criticism")
        self.assertEqual(err["message"], "LLM provider returned 503")
        self.assertTrue(err["recoverable"])

    def test_plan_status_failed_diagnostics_null_when_not_set(self):
        """Legacy failed rows with no diagnostics return None for all four fields."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.failed,
            "stop_requested": False,
            "progress_percentage": 0.0,
            "progress_message": "Plan generation failed.",
            "timestamp_created": datetime.now(UTC),
            "failure_reason": None,
            "failed_step": None,
            "error_message": None,
            "recoverable": None,
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structured_content
        self.assertEqual(sc["state"], "failed")
        self.assertIn("error", sc)
        err = sc["error"]
        self.assertIsNone(err["failure_reason"])
        self.assertIsNone(err["failed_step"])
        # message falls back to progress_message when error_message is None
        self.assertEqual(err["message"], "Plan generation failed.")
        self.assertIsNone(err["recoverable"])

    def test_plan_status_includes_last_progress_at(self):
        """Processing plan with last_progress_at set returns ISO 8601 string in timing."""
        plan_id = str(uuid.uuid4())
        progress_ts = datetime(2026, 3, 12, 14, 30, 0, tzinfo=UTC)
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": False,
            "progress_percentage": 50.0,
            "steps_completed": 15,
            "steps_total": 30,
            "current_step": "SWOT Analysis",
            "timestamp_created": datetime.now(UTC),
            "failure_reason": None,
            "failed_step": None,
            "error_message": None,
            "recoverable": None,
            "last_progress_at": progress_ts,
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        timing = result.structured_content["timing"]
        self.assertIn("last_progress_at", timing)
        self.assertIsInstance(timing["last_progress_at"], str)
        self.assertIn("2026-03-12", timing["last_progress_at"])

    def test_plan_status_last_progress_at_null_when_pending(self):
        """Pending plan with no progress has last_progress_at as None in timing."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.pending,
            "stop_requested": False,
            "progress_percentage": 0.0,
            "steps_completed": None,
            "steps_total": None,
            "current_step": None,
            "timestamp_created": datetime.now(UTC),
            "failure_reason": None,
            "failed_step": None,
            "error_message": None,
            "recoverable": None,
            "last_progress_at": None,
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        timing = result.structured_content["timing"]
        self.assertIn("last_progress_at", timing)
        self.assertIsNone(timing["last_progress_at"])

    def test_plan_status_non_failed_omits_diagnostics(self):
        """Processing/completed plans have no failure_reason etc. in response."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": plan_id,
            "state": PlanState.processing,
            "stop_requested": False,
            "progress_percentage": 50.0,
            "steps_completed": 15,
            "steps_total": 30,
            "current_step": "SWOT Analysis",
            "timestamp_created": datetime.now(UTC),
            "failure_reason": None,
            "failed_step": None,
            "error_message": None,
            "recoverable": None,
        }
        with patch(
            "mcp_cloud.handlers._get_plan_status_snapshot_sync",
            return_value=plan_snapshot,
        ), patch(
            "mcp_cloud.handlers.fetch_file_list_from_worker_plan", new=AsyncMock(return_value=[])
        ):
            result = asyncio.run(handle_plan_status({"plan_id": plan_id}))

        sc = result.structured_content
        self.assertEqual(sc["state"], "processing")
        self.assertNotIn("error", sc)


if __name__ == "__main__":
    unittest.main()

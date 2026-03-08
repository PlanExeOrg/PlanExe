import asyncio
import unittest
import uuid
import zipfile
from io import BytesIO
from unittest.mock import AsyncMock, patch

from database_api.model_planitem import PlanState
from mcp_cloud.app import (
    REPORT_FILENAME,
    ZIP_CONTENT_TYPE,
    _sanitize_legacy_zip_snapshot,
    extract_file_from_zip_bytes,
    handle_plan_file_info,
    handle_list_tools,
    list_files_from_zip_bytes,
)


class TestPlanFileInfoTool(unittest.TestCase):
    def test_plan_file_info_tool_listed(self):
        tools = asyncio.run(handle_list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertIn("plan_file_info", tool_names)

    def test_zip_helpers(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(REPORT_FILENAME, "<html>ok</html>")
            zip_file.writestr("001-2-plan.txt", "Plan prompt")
        zip_bytes = buffer.getvalue()

        files = list_files_from_zip_bytes(zip_bytes)
        file_names = [name for name, _ in files]
        self.assertIn(REPORT_FILENAME, file_names)
        self.assertIn("001-2-plan.txt", file_names)

        report_bytes = extract_file_from_zip_bytes(zip_bytes, REPORT_FILENAME)
        self.assertEqual(report_bytes, b"<html>ok</html>")

    def test_report_read_defaults_to_metadata(self):
        plan_id = str(uuid.uuid4())
        content_bytes = b"a" * 10
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.completed,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_artifact_from_worker_plan",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_plan_file_info({"plan_id": plan_id}))

        payload = result.structuredContent
        self.assertEqual(payload["download_size"], len(content_bytes))
        self.assertEqual(payload["content_type"], "text/html; charset=utf-8")
        self.assertNotIn("download_path", payload)
        self.assertNotIn("content", payload)
        self.assertNotIn("state", payload)

    def test_report_read_zip(self):
        plan_id = str(uuid.uuid4())
        content_bytes = b"zipdata"
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.completed,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_user_downloadable_zip",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "zip"}))

        payload = result.structuredContent
        self.assertEqual(payload["download_size"], len(content_bytes))
        self.assertEqual(payload["content_type"], ZIP_CONTENT_TYPE)

    def test_report_read_zip_for_failed_task(self):
        plan_id = str(uuid.uuid4())
        content_bytes = b"zipdata"
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.failed,
            "progress_message": "Stopped",
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_user_downloadable_zip",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "zip"}))

        payload = result.structuredContent
        self.assertEqual(payload["download_size"], len(content_bytes))
        self.assertEqual(payload["content_type"], ZIP_CONTENT_TYPE)

    def test_plan_file_info_returns_empty_object_when_pending(self):
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.pending,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            result = asyncio.run(handle_plan_file_info({"plan_id": plan_id}))

        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent, {"ready": False, "reason": "processing"})

    def test_plan_file_info_not_ready_preserves_structured_content(self):
        """Not-ready responses must carry structuredContent so MCP clients see the payload."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.processing,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "report"}))

        self.assertFalse(result.isError)
        sc = result.structuredContent
        self.assertIsNotNone(sc, "structuredContent must be present for not-ready responses")
        self.assertFalse(sc["ready"])
        self.assertEqual(sc["reason"], "processing")

    def test_plan_file_info_pending_zip_preserves_structured_content(self):
        """Not-ready zip responses must also carry structuredContent."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.pending,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_user_downloadable_zip",
                new=AsyncMock(return_value=None),
            ):
                result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "zip"}))

        self.assertFalse(result.isError)
        sc = result.structuredContent
        self.assertIsNotNone(sc)
        self.assertFalse(sc["ready"])
        self.assertEqual(sc["reason"], "processing")

    def test_plan_file_info_not_found_has_structured_content(self):
        """PLAN_NOT_FOUND error must include structuredContent with error detail."""
        plan_id = str(uuid.uuid4())
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=None):
            result = asyncio.run(handle_plan_file_info({"plan_id": plan_id}))

        self.assertTrue(result.isError)
        sc = result.structuredContent
        self.assertIsNotNone(sc, "Error responses must include structuredContent")
        self.assertEqual(sc["error"]["code"], "PLAN_NOT_FOUND")

    def test_plan_file_info_content_mirrors_structured_content(self):
        """content text and structuredContent must carry the same data."""
        import json
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.pending,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            result = asyncio.run(handle_plan_file_info({"plan_id": plan_id}))

        sc = result.structuredContent
        content_text = result.content[0].text
        self.assertEqual(json.loads(content_text), sc)

    def test_plan_file_info_ready_report_has_structured_content(self):
        """Successful report response must include structuredContent with download metadata."""
        plan_id = str(uuid.uuid4())
        content_bytes = b"<html>report</html>"
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.completed,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_artifact_from_worker_plan",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "report"}))

        self.assertFalse(result.isError)
        sc = result.structuredContent
        self.assertIsNotNone(sc)
        self.assertIn("content_type", sc)
        self.assertIn("sha256", sc)
        self.assertIn("download_size", sc)
        self.assertEqual(sc["download_size"], len(content_bytes))

    def test_plan_file_info_generation_failed_has_structured_content_with_error(self):
        """generation_failed must include structuredContent with error code and message."""
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.failed,
            "progress_message": "Out of memory",
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "report"}))

        self.assertFalse(result.isError)
        sc = result.structuredContent
        self.assertIsNotNone(sc)
        self.assertEqual(sc["error"]["code"], "generation_failed")
        self.assertIn("Out of memory", sc["error"]["message"])
        self.assertFalse(sc["ready"])

    def test_plan_file_info_returns_generation_failed_payload(self):
        plan_id = str(uuid.uuid4())
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.failed,
            "progress_message": "Pipeline failed",
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "report"}))

        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["error"]["code"], "generation_failed")

    def test_report_expires_at_present_when_download_url_set(self):
        """expires_at must be an ISO 8601 UTC timestamp when download_url is present."""
        import mcp_cloud.download_tokens as _dt_mod
        from datetime import datetime, timezone

        plan_id = str(uuid.uuid4())
        content_bytes = b"<html>report</html>"
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.completed,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_artifact_from_worker_plan",
                new=AsyncMock(return_value=content_bytes),
            ):
                with patch.object(_dt_mod, "_get_download_base_url", return_value="https://example.com"):
                    result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "report"}))

        sc = result.structuredContent
        self.assertIn("download_url", sc)
        self.assertIn("expires_at", sc)
        expires = datetime.fromisoformat(sc["expires_at"])
        self.assertGreater(expires, datetime.now(timezone.utc))

    def test_zip_expires_at_present_when_download_url_set(self):
        """expires_at must be present for zip artifacts when download_url is set."""
        import mcp_cloud.download_tokens as _dt_mod
        from datetime import datetime, timezone

        plan_id = str(uuid.uuid4())
        content_bytes = b"zipdata"
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.completed,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_user_downloadable_zip",
                new=AsyncMock(return_value=content_bytes),
            ):
                with patch.object(_dt_mod, "_get_download_base_url", return_value="https://example.com"):
                    result = asyncio.run(handle_plan_file_info({"plan_id": plan_id, "artifact": "zip"}))

        sc = result.structuredContent
        self.assertIn("download_url", sc)
        self.assertIn("expires_at", sc)
        expires = datetime.fromisoformat(sc["expires_at"])
        self.assertGreater(expires, datetime.now(timezone.utc))

    def test_expires_at_absent_when_no_download_url(self):
        """expires_at must NOT be present when download_url is absent."""
        plan_id = str(uuid.uuid4())
        content_bytes = b"<html>report</html>"
        plan_snapshot = {
            "id": "plan-id",
            "state": PlanState.completed,
            "progress_message": None,
        }
        with patch("mcp_cloud.handlers._get_plan_for_report_sync", return_value=plan_snapshot):
            with patch(
                "mcp_cloud.handlers.fetch_artifact_from_worker_plan",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_plan_file_info({"plan_id": plan_id}))

        sc = result.structuredContent
        self.assertNotIn("download_url", sc)
        self.assertNotIn("expires_at", sc)

    def test_sanitize_legacy_zip_snapshot_removes_track_activity_jsonl(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(REPORT_FILENAME, "<html>ok</html>")
            zip_file.writestr("nested/track_activity.jsonl", "{\"event\":\"secret\"}\n")
        sanitized = _sanitize_legacy_zip_snapshot(buffer.getvalue())
        self.assertIsNotNone(sanitized)
        assert sanitized is not None
        files = list_files_from_zip_bytes(sanitized)
        file_names = [name for name, _ in files]
        self.assertIn(REPORT_FILENAME, file_names)
        self.assertNotIn("nested/track_activity.jsonl", file_names)


if __name__ == "__main__":
    unittest.main()

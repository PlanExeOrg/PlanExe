import asyncio
import unittest
import uuid
import zipfile
from io import BytesIO
from unittest.mock import AsyncMock, patch

from database_api.model_taskitem import TaskState
from mcp_server.app import (
    REPORT_FILENAME,
    ZIP_CONTENT_TYPE,
    extract_file_from_zip_bytes,
    handle_report_read,
    handle_list_tools,
    list_files_from_zip_bytes,
)


class TestReportTool(unittest.TestCase):
    def test_report_tool_listed(self):
        tools = asyncio.run(handle_list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertIn("task_result", tool_names)

    def test_zip_helpers(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(REPORT_FILENAME, "<html>ok</html>")
            zip_file.writestr("001-2-plan.txt", "Plan prompt")
        zip_bytes = buffer.getvalue()

        files = list_files_from_zip_bytes(zip_bytes)
        self.assertIn(REPORT_FILENAME, files)
        self.assertIn("001-2-plan.txt", files)

        report_bytes = extract_file_from_zip_bytes(zip_bytes, REPORT_FILENAME)
        self.assertEqual(report_bytes, b"<html>ok</html>")

    def test_report_read_defaults_to_metadata(self):
        task_id = str(uuid.uuid4())
        content_bytes = b"a" * 10
        task_snapshot = {
            "id": "task-id",
            "state": TaskState.completed,
            "progress_message": None,
        }
        with patch("mcp_server.app._get_task_for_report_sync", return_value=task_snapshot):
            with patch(
                "mcp_server.app.fetch_artifact_from_worker_plan",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_report_read({"task_id": task_id}))

        payload = result.structuredContent
        self.assertEqual(payload["download_size"], len(content_bytes))
        self.assertEqual(payload["content_type"], "text/html; charset=utf-8")
        self.assertNotIn("download_path", payload)
        self.assertNotIn("content", payload)
        self.assertNotIn("state", payload)

    def test_report_read_zip(self):
        task_id = str(uuid.uuid4())
        content_bytes = b"zipdata"
        task_snapshot = {
            "id": "task-id",
            "state": TaskState.completed,
            "progress_message": None,
        }
        with patch("mcp_server.app._get_task_for_report_sync", return_value=task_snapshot):
            with patch(
                "mcp_server.app.fetch_zip_from_worker_plan",
                new=AsyncMock(return_value=content_bytes),
            ):
                result = asyncio.run(handle_report_read({"task_id": task_id, "artifact": "zip"}))

        payload = result.structuredContent
        self.assertEqual(payload["download_size"], len(content_bytes))
        self.assertEqual(payload["content_type"], ZIP_CONTENT_TYPE)


if __name__ == "__main__":
    unittest.main()

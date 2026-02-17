import unittest
import uuid
from unittest.mock import MagicMock, patch

from mcp_cloud.app import find_task_by_task_id, normalize_task_id_input


class TestTaskIdResolution(unittest.TestCase):
    def test_normalize_task_id_input_canonicalizes_uuid(self):
        raw = str(uuid.uuid4()).upper()
        normalized, is_uuid = normalize_task_id_input(raw)
        self.assertTrue(is_uuid)
        self.assertEqual(normalized, raw.lower())

    def test_normalize_task_id_input_preserves_legacy_identifier(self):
        legacy = "PlanExe_19841231_195936"
        normalized, is_uuid = normalize_task_id_input(legacy)
        self.assertFalse(is_uuid)
        self.assertEqual(normalized, legacy)

    def test_find_task_by_task_id_uses_uuid_lookup_first(self):
        task_id = str(uuid.uuid4())
        found = object()
        with patch("mcp_cloud.app.get_task_by_id", return_value=found) as mock_get:
            result = find_task_by_task_id(task_id.upper())
        self.assertIs(result, found)
        mock_get.assert_called_once_with(task_id)

    def test_find_task_by_task_id_falls_back_to_legacy_query(self):
        legacy = "PlanExe_19841231_195936"
        legacy_task = object()

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [legacy_task]

        with patch("mcp_cloud.app.get_task_by_id") as mock_get, patch(
            "mcp_cloud.app.has_app_context", return_value=True
        ), patch("mcp_cloud.app.db.session.query", return_value=mock_query):
            result = find_task_by_task_id(legacy)

        self.assertIs(result, legacy_task)
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()

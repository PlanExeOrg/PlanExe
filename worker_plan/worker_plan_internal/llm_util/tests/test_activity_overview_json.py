import json
import tempfile
from pathlib import Path
import unittest

from worker_plan_internal.llm_util.track_activity import TrackActivity
from worker_plan_api.filenames import ExtraFilenameEnum


class TestActivityOverviewJson(unittest.TestCase):
    def test_updates_overview_with_cost_and_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "track_activity.jsonl"
            jsonl_path.touch()
            tracker = TrackActivity(jsonl_file_path=jsonl_path, write_to_logger=False)

            event_data = {
                "response": {
                    "raw": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "cost": 0.25,
                        },
                        "model": "test-model",
                        "provider": "TestProvider",
                    }
                }
            }

            tracker._update_activity_overview(event_data)

            overview_path = jsonl_path.parent / ExtraFilenameEnum.ACTIVITY_OVERVIEW_JSON.value
            self.assertTrue(overview_path.exists())

            with open(overview_path, "r", encoding="utf-8") as f:
                overview = json.load(f)

            self.assertAlmostEqual(overview["total_cost"], 0.25)
            self.assertEqual(overview["total_input_tokens"], 10)
            self.assertEqual(overview["total_output_tokens"], 5)
            self.assertEqual(overview["total_tokens"], 15)
            self.assertIn("TestProvider:test-model", overview["models"])

            model_stats = overview["models"]["TestProvider:test-model"]
            self.assertAlmostEqual(model_stats["total_cost"], 0.25)
            self.assertEqual(model_stats["input_tokens"], 10)
            self.assertEqual(model_stats["output_tokens"], 5)
            self.assertEqual(model_stats["total_tokens"], 15)
            self.assertEqual(model_stats["calls"], 1)


if __name__ == '__main__':
    unittest.main()

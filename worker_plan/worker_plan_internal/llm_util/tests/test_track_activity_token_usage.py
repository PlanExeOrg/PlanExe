import tempfile
from pathlib import Path
import unittest

from worker_plan_internal.llm_util.track_activity import TrackActivity


class TestTrackActivityTokenUsage(unittest.TestCase):
    def _make_tracker(self) -> TrackActivity:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        tmp.close()
        return TrackActivity(jsonl_file_path=Path(tmp.name), write_to_logger=False)

    def test_extracts_usage_from_response_usage(self):
        tracker = self._make_tracker()
        event_data = {
            "response": {
                "usage": {"prompt_tokens": 12, "completion_tokens": 7}
            }
        }
        usage = tracker._extract_token_usage(event_data)
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 19)

    def test_extracts_usage_from_response_raw_usage(self):
        tracker = self._make_tracker()
        event_data = {
            "response": {
                "raw": {
                    "usage": {"input_tokens": 5, "output_tokens": 9}
                }
            }
        }
        usage = tracker._extract_token_usage(event_data)
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 5)
        self.assertEqual(usage["output_tokens"], 9)
        self.assertEqual(usage["total_tokens"], 14)


if __name__ == '__main__':
    unittest.main()

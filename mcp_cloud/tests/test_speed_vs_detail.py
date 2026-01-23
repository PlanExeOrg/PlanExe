import unittest

from pydantic import ValidationError

from mcp_cloud.app import (
    SPEED_VS_DETAIL_DEFAULT,
    TaskCreateRequest,
    _merge_task_create_config,
    resolve_speed_vs_detail,
)


class TestResolveSpeedVsDetail(unittest.TestCase):
    def test_default(self):
        self.assertEqual(resolve_speed_vs_detail(None), SPEED_VS_DETAIL_DEFAULT)

    def test_fast_alias(self):
        self.assertEqual(resolve_speed_vs_detail({"speed_vs_detail": "fast"}), "fast_but_skip_details")

    def test_all_alias(self):
        self.assertEqual(resolve_speed_vs_detail({"speed": "all"}), "all_details_but_slow")

    def test_ping_alias(self):
        self.assertEqual(resolve_speed_vs_detail({"speed_vs_detail": "ping"}), "ping_llm")

    def test_passthrough(self):
        self.assertEqual(resolve_speed_vs_detail({"speed_vs_detail": "ping_llm"}), "ping_llm")

    def test_merge_task_create_config_injects_speed(self):
        merged = _merge_task_create_config(None, "fast")
        self.assertEqual(merged, {"speed_vs_detail": "fast"})

    def test_merge_task_create_config_preserves_existing(self):
        merged = _merge_task_create_config({"speed_vs_detail": "all_details_but_slow"}, "fast")
        self.assertEqual(merged, {"speed_vs_detail": "all_details_but_slow"})

    def test_merge_task_create_config_ignores_blank(self):
        merged = _merge_task_create_config({}, "   ")
        self.assertIsNone(merged)


class TestTaskCreateRequest(unittest.TestCase):
    def test_speed_vs_detail_accepts_enum(self):
        for value in ("ping", "fast", "all"):
            req = TaskCreateRequest(idea="demo", speed_vs_detail=value)
            self.assertEqual(req.speed_vs_detail, value)

    def test_speed_vs_detail_rejects_invalid(self):
        with self.assertRaises(ValidationError):
            TaskCreateRequest(idea="demo", speed_vs_detail="slow")


if __name__ == "__main__":
    unittest.main()

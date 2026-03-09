import unittest
from datetime import UTC, datetime, timezone, timedelta

from worker_plan_api.format_datetime import format_datetime_utc


class TestFormatDatetimeUtc(unittest.TestCase):
    def test_basic(self):
        dt = datetime(2026, 3, 8, 23, 49, 53, tzinfo=UTC)
        self.assertEqual(format_datetime_utc(dt), "2026-03-08T23:49:53Z")

    def test_truncates_microseconds(self):
        dt = datetime(2026, 1, 15, 12, 0, 0, 999999, tzinfo=UTC)
        self.assertEqual(format_datetime_utc(dt), "2026-01-15T12:00:00Z")

    def test_midnight(self):
        dt = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
        self.assertEqual(format_datetime_utc(dt), "2026-06-01T00:00:00Z")

    def test_non_utc_timezone(self):
        tz_plus2 = timezone(timedelta(hours=2))
        dt = datetime(2026, 3, 8, 23, 0, 0, tzinfo=tz_plus2)
        result = format_datetime_utc(dt)
        self.assertEqual(result, "2026-03-08T23:00:00+02:00")


if __name__ == "__main__":
    unittest.main()

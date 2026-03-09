"""UTC datetime formatting for ISO 8601 with Z suffix."""
from datetime import datetime


def format_datetime_utc(dt: datetime) -> str:
    """Format a datetime as ISO 8601 with whole seconds and Z suffix.

    Example: datetime(2026, 3, 8, 23, 49, 53, 123456, tzinfo=UTC) -> "2026-03-08T23:49:53Z"
    """
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

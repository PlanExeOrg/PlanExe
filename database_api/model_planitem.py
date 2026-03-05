import enum
import uuid
from datetime import datetime, UTC
from typing import TYPE_CHECKING
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType
from sqlalchemy import JSON
from sqlalchemy.orm import column_property, deferred
from sqlalchemy import event


def _sanitize_utf8_text(value):
    """Normalize values into valid UTF-8-safe text for persistence."""
    if value is None:
        return None

    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        text = bytes(value).decode("utf-8", errors="replace")
    else:
        text = str(value)

    # Postgres text does not support embedded NULL bytes.
    if "\x00" in text:
        text = text.replace("\x00", "")

    # Replace unpaired surrogates or other non-encodable code points.
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="replace").decode("utf-8")
    return text

class PlanState(enum.Enum):
    pending = 1
    processing = 2
    completed = 3
    failed = 4


class PlanItem(db.Model):
    __tablename__ = "task_item"

    # A unique identifier for the task.
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)

    # Time was the /run endpoint called.
    timestamp_created = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    # A new task is created with state=pending.
    # When the task is picked up from the queue, the state is set to processing.
    # When the plan has been generated successfully the state is set to completed.
    # If anything fails or the task is aborted, the state is set to failed.
    state = db.Column(db.Enum(PlanState), nullable=True)

    # The prompt that was submitted to the /run endpoint, that PlanExe will attempt to generate a plan for.
    # The limit is 4GB of text.
    prompt = db.Column(db.Text)

    # Progress percentage, from 0.0 to 100.0.
    progress_percentage = db.Column(db.Numeric(5, 2), default=0.0)

    # Example: "Awaiting server to start…" or "42 of 89. Extra files: 8"
    progress_message = db.Column(db.String(128))

    # When was the last time the browser fetched the /progress endpoint.
    # This is used to determine if the task is still active.
    # If the task is not active, it will be stopped.
    last_seen_timestamp = db.Column(db.DateTime, nullable=True, default=lambda: datetime.now(UTC))

    # Stop requests from external callers (e.g., MCP or admin tools).
    stop_requested = db.Column(db.Boolean, nullable=True, default=False)

    # When a stop was requested (UTC).
    stop_requested_timestamp = db.Column(db.DateTime, nullable=True, default=None)

    # Identifies who invoked the /run endpoint, that is charged credits for generating the plan.
    user_id = db.Column(db.String(256))

    # Which API key created this plan (NULL for legacy/frontend plans).
    api_key_id = db.Column(db.String(36), nullable=True, index=True)

    # Extra parameters provided to the /run endpoint, that may control speedvsdetail, loglevel, and other developer settings.
    parameters = db.Column(JSON, nullable=True, default=None)

    # The generated report HTML (stored when the pipeline succeeds).
    # HEAVY (~800KB) — deferred so it is only loaded when explicitly accessed.
    # Use has_generated_report_html to check presence without loading the data.
    generated_report_html = deferred(db.Column(db.Text, nullable=True))

    # A zip archive of the run directory for this plan (stored for both success and failure).
    # HEAVY (~5MB) — deferred so it is only loaded when explicitly accessed.
    # Use has_run_zip_snapshot to check presence without loading the data.
    run_zip_snapshot = deferred(db.Column(db.LargeBinary, nullable=True))

    # Internal-only raw activity log (contains sensitive provider payloads).
    # HEAVY (~20MB) — deferred so it is only loaded when explicitly accessed.
    # Use has_run_track_activity_jsonl to check presence without loading the data.
    run_track_activity_jsonl = deferred(db.Column(db.Text, nullable=True))

    # Original byte size for run_track_activity_jsonl (for observability/migration checks).
    run_track_activity_bytes = db.Column(db.Integer, nullable=True)

    # User-facing usage/cost summary from activity_overview.json.
    run_activity_overview_json = db.Column(JSON, nullable=True, default=None)

    # Artifact schema/version marker (legacy snapshots are NULL/1, split-storage snapshots are 2+).
    run_artifact_layout_version = db.Column(db.Integer, nullable=True, default=None)

    # Lightweight IS NOT NULL checks (actual column_property assigned after class body).
    # TYPE_CHECKING-only annotations so pyright knows the types without confusing
    # SQLAlchemy's declarative metaclass at runtime.
    if TYPE_CHECKING:
        has_generated_report_html: bool
        has_run_zip_snapshot: bool
        has_run_track_activity_jsonl: bool

    def __repr__(self):
        return f"{self.id}: {self.timestamp_created}, {self.state}, {self.prompt!r}, parameters: {self.parameters!r}"

    def has_parameter_key(self, key: str) -> bool:
        if not isinstance(self.parameters, dict):
            return False
        return key in self.parameters

    @classmethod
    def demo_items(cls) -> list['PlanItem']:
        task1 = PlanItem(
            state=PlanState.failed,
            prompt="Eurovision 2026 in Austria, following the country's 2025 victory. Budget of €30-40 million, funded by the European Broadcasting Union (EBU), Austrian broadcaster ORF, and host city contributions. Host city likely to be Vienna. Venue capable of accommodating 10,000-15,000 spectators.",
            progress_percentage=0.0,
            progress_message="Awaiting server to start…",
            user_id="demo_user_1"
        )
        task2 = PlanItem(
            state=PlanState.completed,
            prompt="It's 2025 and humanoid robots are entering mainstream society, with China already showcasing robotic athletes in sports events. Plan a 2026 Robot Olympics, outline innovative events, rules, and challenges to test the humanoid robots.",
            progress_percentage=100.0,
            progress_message="Completed",
            user_id="demo_user_1"
        )
        task3 = PlanItem(
            state=PlanState.completed,
            prompt="It's 2025 and humanoid robots are entering mainstream society, with China already showcasing robotic athletes in sports events. Plan a 2026 Robot Olympics, outline innovative events, rules, and challenges to test the humanoid robots.",
            progress_percentage=100.0,
            progress_message="Completed",
            user_id="demo_user_1",
            parameters={
                "budget": 100000000,
                "location": "Tokyo",
                "date": "1984-12-31"
            }
        )
        return [task1, task2, task3]


# Lightweight IS NOT NULL checks — defined outside the class body because
# deferred() descriptors do not support .isnot() at class-definition time
# (produces NotImplemented).  Using __table__.c gives us the raw Column
# objects which support SQL operators correctly.
_t = PlanItem.__table__.c
PlanItem.has_generated_report_html = column_property(_t.generated_report_html.isnot(None))
PlanItem.has_run_zip_snapshot = column_property(_t.run_zip_snapshot.isnot(None))
PlanItem.has_run_track_activity_jsonl = column_property(_t.run_track_activity_jsonl.isnot(None))
del _t


@event.listens_for(PlanItem, "before_insert")
@event.listens_for(PlanItem, "before_update")
def _sanitize_planitem_fields(_mapper, _connection, target):
    # Enforce valid UTF-8-safe prompt text regardless of writer path.
    target.prompt = _sanitize_utf8_text(target.prompt)

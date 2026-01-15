import enum
import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType
from sqlalchemy import JSON

class TaskState(enum.Enum):
    pending = 1
    processing = 2
    completed = 3
    failed = 4


class TaskItem(db.Model):
    # A unique identifier for the task.
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)

    # Time was the /run endpoint called.
    timestamp_created = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    # A new task is created with state=pending.
    # When the task is picked up from the queue, the state is set to processing.
    # When the plan has been generated successfully the state is set to completed.
    # If anything fails or the task is aborted, the state is set to failed.
    state = db.Column(db.Enum(TaskState), nullable=True)

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

    # Extra parameters provided to the /run endpoint, that may control speedvsdetail, loglevel, and other developer settings.
    parameters = db.Column(JSON, nullable=True, default=None)

    # The generated report HTML (stored when the pipeline succeeds).
    generated_report_html = db.Column(db.Text, nullable=True)

    # A zip archive of the run directory for this task (stored for both success and failure).
    run_zip_snapshot = db.Column(db.LargeBinary, nullable=True)

    def __repr__(self):
        return f"{self.id}: {self.timestamp_created}, {self.state}, {self.prompt!r}, parameters: {self.parameters!r}"

    def has_parameter_key(self, key: str) -> bool:
        if not isinstance(self.parameters, dict):
            return False
        return key in self.parameters
    
    @classmethod
    def demo_items(cls) -> list['TaskItem']:
        task1 = TaskItem(
            state=TaskState.failed,
            prompt="Eurovision 2026 in Austria, following the country’s 2025 victory. Budget of €30-40 million, funded by the European Broadcasting Union (EBU), Austrian broadcaster ORF, and host city contributions. Host city likely to be Vienna. Venue capable of accommodating 10,000-15,000 spectators.",
            progress_percentage=0.0,
            progress_message="Awaiting server to start…",
            user_id="demo_user_1"
        )
        task2 = TaskItem(
            state=TaskState.completed,
            prompt="It's 2025 and humanoid robots are entering mainstream society, with China already showcasing robotic athletes in sports events. Plan a 2026 Robot Olympics, outline innovative events, rules, and challenges to test the humanoid robots.",
            progress_percentage=100.0,
            progress_message="Completed",
            user_id="demo_user_1"
        )
        task3 = TaskItem(
            state=TaskState.completed,
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

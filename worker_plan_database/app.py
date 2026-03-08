"""
This project monitors the database for pending PlanItems and automatically changes their status to processing
when found. It then executes the pipeline for each task.

PROMPT> PLANEXE_WORKER_ID=1 python -m app.py
"""
from datetime import UTC, datetime
from decimal import Decimal
import json
import os
import shutil
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Any, cast
import uuid
from worker_plan_api.model_profile import ModelProfileEnum
from urllib.parse import quote_plus
import io
import zipfile
import requests
from sqlalchemy import func, inspect, text, or_
from worker_plan_database.worker_identity import resolve_and_set_worker_id


def _new_model(model_cls: Any, **kwargs: Any) -> Any:
    """Instantiate ORM models through Any to accommodate dynamic Flask-SQLAlchemy typing."""
    return cast(Any, model_cls)(**kwargs)

# Load .env file early, before any imports that require environment variables (e.g., machai.py).
# This allows configuration via .env file instead of shell exports.
# Search in worker_plan_database/ first, then fall back to the project root.
from dotenv import load_dotenv
_module_dir = Path(__file__).parent
_dotenv_loaded = load_dotenv(_module_dir / ".env")
if not _dotenv_loaded:
    load_dotenv(_module_dir.parent / ".env")

try:
    WORKER_ID = resolve_and_set_worker_id(os.environ)
except ValueError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger(__name__).critical(str(exc))
    sys.exit(1)

# Attempt to configure Luigi VERY EARLY to prevent its default logging setup.
try:
    import luigi
    import luigi.configuration
    luigi_config = luigi.configuration.get_config()
    luigi_config.set('core', 'no_configure_logging', 'true')
except ImportError:
    pass # Luigi might be imported later by worker_plan_internal

# --- Global Paths ---
BASE_DIR = Path(__file__).parent.parent.absolute()
# Default to shared PLANEXE_RUN_DIR (mounted volume) so worker_plan can read outputs.
BASE_DIR_RUN = Path(os.environ.get("PLANEXE_RUN_DIR", BASE_DIR / "run")).resolve()
BASE_DIR_RUN.mkdir(exist_ok=True)

PLANEXE_CONFIG_PATH_VAR = BASE_DIR

# Since 2021, Chrome penalizes tabs that are not in focus, disallowing faster than 60 Hz updates.
# So considering 60 seconds of inactivity, and a few seconds of processing time, 60 + some buffer, I end up with 80 seconds.
# https://developer.chrome.com/blog/timer-throttling-in-chrome-88/
BROWSER_INACTIVE_AFTER_N_SECONDS = 80
CONTINUE_GENERATING_PLAN_DESPITE_BROWSER_INACTIVE = True
HEARTBEAT_INTERVAL_IN_SECONDS = 60
CREDIT_SCALE = Decimal("0.000000001")

# --- Configure Logging Section ---
replica_id_for_logging = os.environ.get("RAILWAY_REPLICA_ID", "local")
log_format_str = (
    f"%(asctime)s - %(name)s - %(levelname)s - "
    f"[replica={replica_id_for_logging} worker_id={WORKER_ID}] - %(message)s"
)
log_formatter = logging.Formatter(log_format_str)
log_level_name = os.environ.get("PLANEXE_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, None)
invalid_log_level = not isinstance(log_level, int)
if invalid_log_level:
    log_level = logging.INFO
    log_level_name = "INFO"

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=log_level,
    handlers=[stream_handler],
    force=True
)

# Capture standard warnings and route them through the logging system.
logging.captureWarnings(True) # 'py.warnings' logger will propagate to root.

# Get the logger for the current module (__main__) and log script start.
logger = logging.getLogger(__name__) # Gets __main__ logger
if invalid_log_level:
    logger.warning("Invalid PLANEXE_LOG_LEVEL provided; defaulting to INFO.")
root_handlers = [type(h).__name__ for h in logging.getLogger().handlers]
logger.info(
    "Logging configured for worker_plan_database (level=%s, handlers=%s, stream=%s)",
    logging.getLevelName(log_level),
    root_handlers,
    getattr(stream_handler.stream, "name", "stdout")
)
logger.info(f"----- PlanExe-server: {Path(__file__).name} SCRIPT IS BEING ACCESSED (WORKER_ID: {WORKER_ID}) -----")

# Configure specific loggers to send their output to stdout via the root logger.
loggers_to_redirect_via_root = {
    'luigi': logging.DEBUG,
    'luigi-interface': logging.DEBUG,
    'luigi.worker': logging.DEBUG,
    'luigi.scheduler': logging.DEBUG,
    'luigi.task': logging.DEBUG,
    'transformers': logging.INFO,
    'httpx': logging.WARNING,
}

for name, level in loggers_to_redirect_via_root.items():
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.handlers = []
    lg.propagate = True

logger.debug("Logging fully configured. All configured loggers now write to stdout via root.")

# --- Environment Setup ---
os.environ["PLANEXE_CONFIG_PATH"] = str(PLANEXE_CONFIG_PATH_VAR)
logger.debug(f"PLANEXE_CONFIG_PATH set to: {PLANEXE_CONFIG_PATH_VAR}")

# --- Imports (after logging setup) ---
try:
    logger.debug("Importing required modules... LlamaIndex.")
    from llama_index.core.instrumentation import get_dispatcher
    logger.debug("Importing required modules... PlanExe.")
    from worker_plan_internal.plan.run_plan_pipeline import ExecutePipeline, HandleTaskCompletionParameters
    from worker_plan_internal.plan.pipeline_config import PIPELINE_CONFIG
    from worker_plan_internal.plan.speedvsdetail import SpeedVsDetailEnum
    from worker_plan_api.start_time import StartTime
    from worker_plan_api.plan_file import PlanFile
    from worker_plan_internal.plan.filenames import FilenameEnum
    from worker_plan_api.planexe_dotenv import PlanExeDotEnv
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName, PipelineStopRequested
    from worker_plan_internal.llm_util.token_instrumentation import set_current_task_id, set_current_user_id, set_current_api_key_id
    from worker_plan_internal.llm_util.track_activity import TrackActivity
    from worker_plan_internal.plan.filenames import ExtraFilenameEnum
    from worker_plan_internal.plan.ping_llm import run_ping_llm_report
    logger.debug("Importing required modules... PlanExe-server.")
    from database_api.planexe_db_singleton import db
    from database_api.model_planitem import PlanItem, PlanState
    from database_api.model_event import EventType, EventItem
    from database_api.model_worker import WorkerItem
    from database_api.model_user_account import UserAccount
    from database_api.model_user_api_key import UserApiKey
    from database_api.model_credit_history import CreditHistory
    from database_api.model_token_metrics import TokenMetrics
    from worker_plan_database.speedvsdetail import resolve_speedvsdetail
    from worker_plan_database.model_profile import resolve_model_profile
    from worker_plan_database.machai import MachAI
    from flask import Flask
    logger.debug("All modules imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import required components. Error: {e}", exc_info=True)
    sys.exit(1)

planexe_dotenv = PlanExeDotEnv.load()
logger.info(f"{Path(__file__).name}. planexe_dotenv: {planexe_dotenv!r}")

def build_postgres_uri_from_env(env: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Construct a SQLAlchemy URI for Postgres using environment variables."""
    host = env.get("PLANEXE_POSTGRES_HOST") or "database_postgres"
    port = str(env.get("PLANEXE_POSTGRES_PORT") or "5432")
    dbname = env.get("PLANEXE_POSTGRES_DB") or "planexe"
    user = env.get("PLANEXE_POSTGRES_USER") or "planexe"
    password = env.get("PLANEXE_POSTGRES_PASSWORD") or "planexe"
    uri = f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    safe_config = {"host": host, "port": port, "dbname": dbname, "user": user}
    return uri, safe_config

PIPELINE_CONFIG.enable_csv_export = True
logger.info(f"PIPELINE_CONFIG: {PIPELINE_CONFIG!r}")

# Initialize Flask app for database access
app = Flask(__name__)
app.config.from_pyfile('config.py')
sqlalchemy_database_uri = planexe_dotenv.get("SQLALCHEMY_DATABASE_URI")
if sqlalchemy_database_uri is None:
    sqlalchemy_database_uri, db_settings = build_postgres_uri_from_env(planexe_dotenv.dotenv_dict)
    logger.info(f"SQLALCHEMY_DATABASE_URI not set. Using Postgres defaults from environment: {db_settings}")
else:
    logger.info("Using SQLALCHEMY_DATABASE_URI from environment or .env file.")
app.config['SQLALCHEMY_DATABASE_URI'] = sqlalchemy_database_uri
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle' : 280, 'pool_pre_ping': True}
db.init_app(app)

def ensure_planitem_artifact_columns() -> None:
    insp = inspect(db.engine)
    columns = {col["name"] for col in insp.get_columns("task_item")}
    with db.engine.begin() as conn:
        if "generated_report_html" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS generated_report_html TEXT"))
        if "run_zip_snapshot" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_zip_snapshot BYTEA"))
        if "run_track_activity_jsonl" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_track_activity_jsonl TEXT"))
        if "run_track_activity_bytes" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_track_activity_bytes INTEGER"))
        if "run_activity_overview_json" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_activity_overview_json JSON"))
        if "run_artifact_layout_version" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_artifact_layout_version INTEGER"))
        if "stop_requested" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested BOOLEAN"))
        if "stop_requested_timestamp" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested_timestamp TIMESTAMP"))


def ensure_token_metrics_columns() -> None:
    insp = inspect(db.engine)
    if "token_metrics" not in insp.get_table_names():
        return
    columns = {col["name"] for col in insp.get_columns("token_metrics")}
    with db.engine.begin() as conn:
        # Remove legacy identifiers. Token metrics should reference tasks only.
        if "run_id" in columns:
            conn.execute(text("ALTER TABLE token_metrics DROP COLUMN IF EXISTS run_id"))
        if "task_name" in columns:
            conn.execute(text("ALTER TABLE token_metrics DROP COLUMN IF EXISTS task_name"))
        if "task_id" not in columns:
            conn.execute(text("ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS task_id VARCHAR(255)"))
        if "user_id" not in columns:
            conn.execute(text("ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)"))
        if "upstream_provider" not in columns:
            conn.execute(text("ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS upstream_provider VARCHAR(255)"))
        if "upstream_model" not in columns:
            conn.execute(text("ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS upstream_model VARCHAR(255)"))
        if "cost_usd" not in columns:
            conn.execute(text("ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION"))


def ensure_fractional_credit_columns() -> None:
    if db.engine.dialect.name != "postgresql":
        return
    insp = inspect(db.engine)
    table_names = set(insp.get_table_names())
    with db.engine.begin() as conn:
        if "user_account" in table_names:
            conn.execute(text(
                "ALTER TABLE user_account "
                "ALTER COLUMN credits_balance TYPE NUMERIC(18,9) "
                "USING credits_balance::NUMERIC(18,9)"
            ))
            conn.execute(text("ALTER TABLE user_account ALTER COLUMN credits_balance SET DEFAULT 0"))
        if "credit_history" in table_names:
            conn.execute(text(
                "ALTER TABLE credit_history "
                "ALTER COLUMN delta TYPE NUMERIC(18,9) "
                "USING delta::NUMERIC(18,9)"
            ))
        if "payment_record" in table_names:
            conn.execute(text(
                "ALTER TABLE payment_record "
                "ALTER COLUMN credits TYPE NUMERIC(18,9) "
                "USING credits::NUMERIC(18,9)"
            ))


def ensure_step_count_columns() -> None:
    """Add steps_completed, steps_total, and current_step columns to task_item (idempotent)."""
    insp = inspect(db.engine)
    columns = {col["name"] for col in insp.get_columns("task_item")}
    with db.engine.begin() as conn:
        if "steps_completed" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS steps_completed INTEGER"))
        if "steps_total" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS steps_total INTEGER"))
        if "current_step" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS current_step VARCHAR(128)"))


def ensure_multi_api_key_columns() -> None:
    """Add columns for multi-API-key support (idempotent)."""
    statements = (
        "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS label VARCHAR(128)",
        "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS key_plaintext VARCHAR(64)",
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
        "ALTER TABLE credit_history ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
        "ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
    )
    with db.engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)

def worker_process_started() -> None:
    planexe_worker_id = os.environ.get("PLANEXE_WORKER_ID")
    event_context = {
        "pid": str(os.getpid()),
        "WORKER_ID": str(WORKER_ID),
        "environment variable PLANEXE_WORKER_ID": str(planexe_worker_id)
    }
    if planexe_worker_id != WORKER_ID:
        event_context["issue with worker_id"] = "ERROR: PLANEXE_WORKER_ID != WORKER_ID. This is an inconsistency. The process may have been started without a PLANEXE_WORKER_ID environment variable."

    with app.app_context():
        event = _new_model(
            EventItem,
            event_type=EventType.GENERIC_EVENT,
            message="Worker started",
            context=event_context
        )
        db.session.add(event)
        db.session.commit()

worker_process_started()

def update_task_state_with_retry(task_id: str, new_state: PlanState, max_retries: int = 3, retry_delay: int = 5) -> bool:
    """Helper function to update task state with retry logic for database operations."""
    for attempt in range(max_retries):
        try:
            task = db.session.get(PlanItem, task_id)
            if task is None:
                logger.error(f"Task with ID {task_id!r} not found in database. Cannot update task state.")
                return False
            if task.state == new_state:
                logger.info(f"Task {task_id!r} already in state {new_state}. No update needed.")
                return True            
            task.state = new_state
            db.session.commit()
            logger.info(f"Updated task {task_id!r} state to {new_state}")
            return True
        except Exception as e:
            logger.error(f"Database error updating task state (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
            db.session.rollback()
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached for task state update")
                return False
    return False

def update_task_progress_with_retry(
    task_id: str,
    progress_percentage: float,
    progress_message: str,
    steps_completed: Optional[int] = None,
    steps_total: Optional[int] = None,
    current_step: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> bool:
    """Update task progress in the database, retrying on transient failures.

    Args:
        task_id: PlanItem primary key (UUID as string).
        progress_percentage: Completion progress from 0.0 to 100.0.
        progress_message: Human-readable status, e.g. ``"23 of 30"``.
        steps_completed: Number of plan generation steps completed so far.
        steps_total: Total number of plan generation steps expected.
        current_step: Human-readable label of the most recently completed step.
        max_retries: Number of attempts before giving up.
        retry_delay: Seconds to wait between retries.

    Returns:
        True if the update was committed, False on persistent failure or missing task.
    """
    for attempt in range(max_retries):
        try:
            task = db.session.get(PlanItem, task_id)
            if task is None:
                logger.error(f"Task with ID {task_id!r} not found in database. Cannot update task progress.")
                return False

            task.progress_percentage = progress_percentage
            task.progress_message = progress_message
            task.steps_completed = steps_completed
            task.steps_total = steps_total
            task.current_step = current_step
            db.session.commit()
            logger.debug(f"Updated task {task_id!r} progress to {progress_percentage}%: {progress_message}")
            return True
        except Exception as e:
            logger.error(f"Database error updating task progress (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
            db.session.rollback()
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")    
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached for task progress update")
                return False
    return False


class ServerExecutePipeline(ExecutePipeline):
    def __init__(
        self,
        task_id: str,
        run_id_dir: Path,
        speedvsdetail: SpeedVsDetailEnum,
        llm_models: list[str],
        model_profile: ModelProfileEnum,
    ):
        super().__init__(
            run_id_dir=run_id_dir,
            speedvsdetail=speedvsdetail,
            llm_models=llm_models,
            model_profile=model_profile,
        )
        self.task_id = task_id

    def _handle_task_completion(self, parameters: HandleTaskCompletionParameters) -> None:
        logger.debug(f"ServerExecutePipeline._handle_task_completion")

        with app.app_context():
            WorkerItem.upsert_heartbeat(worker_id=WORKER_ID, current_task_id=self.task_id)

        # Lookup the taskitem in the database by self.task_id
        with app.app_context():
            task = db.session.get(PlanItem, self.task_id)
            if task is None:
                logger.error(f"Task with ID {self.task_id!r} not found in database, while running the pipeline. This is an inconsistency.")
                raise Exception(f"Task with ID {self.task_id!r} not found in database, while running the pipeline. This is an inconsistency.")
            stop_requested = bool(task.stop_requested)

        if task.last_seen_timestamp is None:
            # A new PlanItem is supposed to have a last_seen_timestamp.
            # If it doesn't have a last_seen_timestamp, it's an inconsistency that should be fixed.
            logger.error(f"Task with ID {self.task_id!r} has no last_seen_timestamp. This is an inconsistency.")
            raise Exception(f"Task with ID {self.task_id!r} has no last_seen_timestamp. This is an inconsistency.")

        if stop_requested:
            logger.info("Stopping task %s because a stop was requested.", self.task_id)
            with app.app_context():
                update_task_progress_with_retry(
                    task_id=self.task_id,
                    progress_percentage=parameters.progress.progress_percentage,
                    progress_message="Stop requested by user.",
                    steps_completed=parameters.progress.steps_completed,
                    steps_total=parameters.progress.steps_total,
                    current_step=parameters.progress.current_step,
                )
            raise PipelineStopRequested(f"Stopping task {self.task_id!r} because a stop was requested.")

        # Detect if the browser has been inactive for N seconds.
        # Make last_seen_timestamp timezone-aware if it isn't already
        last_seen_aware = task.last_seen_timestamp
        if last_seen_aware.tzinfo is None:
            last_seen_aware = last_seen_aware.replace(tzinfo=UTC)
        
        limit = BROWSER_INACTIVE_AFTER_N_SECONDS
        time_since_last_seen = (datetime.now(UTC) - last_seen_aware).total_seconds()
        if time_since_last_seen > limit:
            # The browser has been inactive for more than N seconds. 
            # The user appears to have navigated away from the progress bar page, or closed the browser.
            if CONTINUE_GENERATING_PLAN_DESPITE_BROWSER_INACTIVE:
                logger.debug(f"Task {self.task_id!r} has been inactive for {time_since_last_seen} seconds. Continuing to generate the plan.")
            else:
                # Optimization: Stop generating the plan and save resources. So other users can use the server.
                logger.info(f"Stopping task {self.task_id!r} because it the browser has not been active for {limit} seconds")
                raise PipelineStopRequested(f"Stopping task {self.task_id!r} because it the browser has not been active for {limit} seconds")
        
        # The browser is still open and the progress bar is visible. 
        # The user is still interested in continuing generating the plan.
        logger.info(f"Task {self.task_id!r} is still active. The user is still interested in continuing generating the plan.")

        with app.app_context():
            update_task_progress_with_retry(
                task_id=self.task_id,
                progress_percentage=parameters.progress.progress_percentage,
                progress_message=parameters.progress.progress_message,
                steps_completed=parameters.progress.steps_completed,
                steps_total=parameters.progress.steps_total,
                current_step=parameters.progress.current_step,
            )

        # Charge credits incrementally so usage is visible in real time.
        try:
            _charge_incremental_usage(self.task_id, self.run_id_dir)
        except Exception as exc:
            logger.warning("Incremental billing failed for task %s: %s", self.task_id, exc)

# Every time a LLM/reasoning model is used, it gets registered in the "track_activity" file.
# The llm_executor_uuid is written to stdout, and referenced in the "track_activity" file, so it's possible to
# cross-reference what happened when there is a problem with the LLM/reasoning model.
# Storing the track_activity file in the BASE_DIR_RUN is a fallback, so it's possible to see the track_activity file even if the run_id_dir is not available.
# The "track_activity" file is supposed to be stored in the run_id_dir.
# If there exist a "track_activity_X_fallback.jsonl" file, then the LLM/reasoning models have been used outside the worker_plan pipeline, which is unusual.
sanitized_worker_id = WORKER_ID.replace(':', '_').replace('/', '_')
track_activity_file_name_fallback = f"track_activity_{sanitized_worker_id}_fallback.jsonl"
track_activity_fallback_path = BASE_DIR_RUN / track_activity_file_name_fallback
track_activity = TrackActivity(jsonl_file_path=track_activity_fallback_path, write_to_logger=False)
get_dispatcher().add_event_handler(track_activity)

def create_zip_bytes(run_dir: Path) -> bytes:
    """
    Create an in-memory zip of a run directory and return the bytes.
    The downloadable snapshot excludes internal-only activity logs.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                if file == "log.txt":
                    continue
                if file == ExtraFilenameEnum.TRACK_ACTIVITY_JSONL.value:
                    continue
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(run_dir))
    buffer.seek(0)
    return buffer.read()


def read_activity_artifacts(run_id_dir: Path) -> tuple[Optional[str], Optional[int], Optional[dict[str, object]]]:
    """Read track/activity artifacts from a run directory for PlanItem persistence."""
    track_activity_path = run_id_dir / ExtraFilenameEnum.TRACK_ACTIVITY_JSONL.value
    activity_overview_path = run_id_dir / ExtraFilenameEnum.ACTIVITY_OVERVIEW_JSON.value

    run_track_activity_jsonl: Optional[str] = None
    run_track_activity_bytes: Optional[int] = None
    if track_activity_path.exists():
        try:
            raw_track_activity = track_activity_path.read_bytes()
            run_track_activity_bytes = len(raw_track_activity)
            run_track_activity_jsonl = raw_track_activity.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Unable to read %s for run %s: %s", track_activity_path, run_id_dir, exc)

    run_activity_overview_json: Optional[dict[str, object]] = None
    if activity_overview_path.exists():
        try:
            overview_payload = json.loads(activity_overview_path.read_text(encoding="utf-8"))
            if isinstance(overview_payload, dict):
                run_activity_overview_json = overview_payload
        except Exception as exc:
            logger.warning("Unable to parse %s for run %s: %s", activity_overview_path, run_id_dir, exc)

    return run_track_activity_jsonl, run_track_activity_bytes, run_activity_overview_json


def _read_inference_cost_usd_from_run_dir(run_id_dir: Path) -> float:
    """Extract total inference cost from activity_overview.json for a run."""
    activity_overview_path = run_id_dir / ExtraFilenameEnum.ACTIVITY_OVERVIEW_JSON.value
    if not activity_overview_path.exists():
        return 0.0
    try:
        payload = json.loads(activity_overview_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unable to parse %s: %s", activity_overview_path, exc)
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    try:
        return max(0.0, float(payload.get("total_cost", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _credits_for_usd(usd_amount: float) -> Decimal:
    """Convert USD amount to credits with fractional precision."""
    if usd_amount <= 0:
        return Decimal("0").quantize(CREDIT_SCALE)
    credit_price_cents = max(1, int(os.environ.get("PLANEXE_CREDIT_PRICE_CENTS", "100")))
    credit_price_usd = Decimal(str(credit_price_cents)) / Decimal("100")
    usd_decimal = Decimal(str(usd_amount))
    return (usd_decimal / credit_price_usd).quantize(CREDIT_SCALE)


def _resolve_user_for_billing(task_user_id: str) -> Optional[UserAccount]:
    """Resolve the UserAccount for billing from a PlanItem.user_id.

    Handles both UUID-based user_ids (OAuth users, new admin plans) and
    legacy string user_ids (e.g. ``"admin"``) by falling back to the first
    admin UserAccount row.
    """
    try:
        user_uuid = uuid.UUID(str(task_user_id))
        user = db.session.get(UserAccount, user_uuid)
        if user is not None:
            return user
    except (ValueError, AttributeError):
        pass
    # Fallback: legacy admin string → find any admin UserAccount.
    return UserAccount.query.filter_by(is_admin=True).first()


def _sum_already_charged_credits(task_id: str) -> Decimal:
    """Sum all incremental credit charges already recorded for a task."""
    rows = (
        CreditHistory.query
        .filter_by(source="usage_billing_progress", external_id=str(task_id))
        .with_entities(func.sum(CreditHistory.delta))
        .first()
    )
    total = rows[0] if rows and rows[0] is not None else Decimal("0")
    # delta is negative (deduction), return absolute value
    return abs(Decimal(str(total))).quantize(CREDIT_SCALE)


def _charge_incremental_usage(task_id: str, run_id_dir: Path) -> None:
    """Charge the user for inference cost accumulated so far.

    Called periodically during plan execution so credits are deducted in
    real time rather than only at completion.  Each call reads the current
    ``activity_overview.json``, compares against what has already been
    charged (``source='usage_billing_progress'`` ledger entries), and
    creates a new entry for the delta.
    """
    current_cost_usd = _read_inference_cost_usd_from_run_dir(run_id_dir)
    if current_cost_usd <= 0:
        return

    with app.app_context():
        task = db.session.get(PlanItem, task_id)
        if task is None:
            return

        if isinstance(task.parameters, dict) and bool(task.parameters.get("billing_skip_usage_charge")):
            return

        user = _resolve_user_for_billing(task.user_id)
        if user is None:
            return

        # Resolve api_key_id: use task's value, or fall back to user's first active key.
        api_key_id = getattr(task, "api_key_id", None)
        if not api_key_id:
            first_key = (
                UserApiKey.query
                .filter_by(user_id=user.id, revoked_at=None)
                .order_by(UserApiKey.created_at.asc())
                .first()
            )
            if first_key:
                api_key_id = str(first_key.id)

        already_charged = _sum_already_charged_credits(task_id)
        total_credits = _credits_for_usd(current_cost_usd)
        delta = (total_credits - already_charged).quantize(CREDIT_SCALE)
        if delta <= 0:
            return

        current_balance = Decimal(str(user.credits_balance or 0)).quantize(CREDIT_SCALE)
        user.credits_balance = (current_balance - delta).quantize(CREDIT_SCALE)
        ledger = _new_model(
            CreditHistory,
            user_id=user.id,
            delta=-delta,
            reason="plan_usage_in_progress",
            source="usage_billing_progress",
            external_id=str(task_id),
            api_key_id=api_key_id,
        )
        db.session.add(ledger)
        db.session.commit()
        logger.debug(
            "Incremental billing for task %s: charged %s credits (total so far: %s)",
            task_id, delta, total_credits,
        )


def _charge_usage_credits_once(task_id: str, run_id_dir: Path, success: bool) -> dict[str, float | Decimal | bool]:
    """Final billing for a completed/failed task.

    Reads the final ``activity_overview.json``, subtracts any incremental
    charges already recorded (``source='usage_billing_progress'``), and
    creates one ``source='usage_billing'`` ledger entry for the remainder
    plus the success fee (if applicable).
    """
    usage_cost_usd = _read_inference_cost_usd_from_run_dir(run_id_dir)
    success_fee_usd = 0.0
    should_charge = True

    with app.app_context():
        task = db.session.get(PlanItem, task_id)
        if task is None:
            logger.warning("Unable to bill task %s: task row not found.", task_id)
            return {
                "usage_cost_usd": usage_cost_usd,
                "success_fee_usd": success_fee_usd,
                "total_charge_usd": usage_cost_usd,
                "charged_credits": Decimal("0"),
                "charged": False,
            }

        if isinstance(task.parameters, dict) and bool(task.parameters.get("billing_skip_usage_charge")):
            should_charge = False

        user = _resolve_user_for_billing(task.user_id)
        if user is None:
            should_charge = False

        # Resolve api_key_id: use task's value, or fall back to user's first active key.
        api_key_id = getattr(task, "api_key_id", None)
        if not api_key_id and user is not None:
            first_key = (
                UserApiKey.query
                .filter_by(user_id=user.id, revoked_at=None)
                .order_by(UserApiKey.created_at.asc())
                .first()
            )
            if first_key:
                api_key_id = str(first_key.id)

        speed_mode = resolve_speedvsdetail(task.parameters if isinstance(task.parameters, dict) else None)
        is_ping_task = speed_mode == SpeedVsDetailEnum.PING_LLM

        if success and should_charge and not is_ping_task:
            success_fee_usd = float(os.environ.get("PLANEXE_SUCCESS_PLAN_FEE_USD", "1.0"))

        total_charge_usd = usage_cost_usd + success_fee_usd
        total_credits = _credits_for_usd(total_charge_usd) if should_charge else Decimal("0")

        existing = CreditHistory.query.filter_by(
            source="usage_billing",
            external_id=str(task_id),
        ).first()
        if existing is not None:
            return {
                "usage_cost_usd": usage_cost_usd,
                "success_fee_usd": success_fee_usd,
                "total_charge_usd": total_charge_usd,
                "charged_credits": Decimal("0"),
                "charged": False,
            }

        # Subtract credits already charged incrementally during execution.
        already_charged = _sum_already_charged_credits(task_id) if should_charge else Decimal("0")
        remaining_credits = (total_credits - already_charged).quantize(CREDIT_SCALE)

        if user is not None and remaining_credits > 0:
            current_balance = Decimal(str(user.credits_balance or 0)).quantize(CREDIT_SCALE)
            user.credits_balance = (current_balance - remaining_credits).quantize(CREDIT_SCALE)
            ledger = _new_model(
                CreditHistory,
                user_id=user.id,
                delta=-remaining_credits,
                reason="plan_created_with_usage_cost" if success else "plan_failed_usage_cost",
                source="usage_billing",
                external_id=str(task_id),
                api_key_id=api_key_id,
            )
            db.session.add(ledger)
            db.session.commit()
            return {
                "usage_cost_usd": usage_cost_usd,
                "success_fee_usd": success_fee_usd,
                "total_charge_usd": total_charge_usd,
                "charged_credits": total_credits,
                "charged": True,
            }

        # If remaining_credits <= 0, everything was already charged incrementally.
        # Still record a zero final entry so the idempotency guard works.
        if user is not None and should_charge and already_charged > 0:
            ledger = _new_model(
                CreditHistory,
                user_id=user.id,
                delta=Decimal("0"),
                reason="plan_created_with_usage_cost" if success else "plan_failed_usage_cost",
                source="usage_billing",
                external_id=str(task_id),
                api_key_id=api_key_id,
            )
            db.session.add(ledger)
            db.session.commit()
            return {
                "usage_cost_usd": usage_cost_usd,
                "success_fee_usd": success_fee_usd,
                "total_charge_usd": total_charge_usd,
                "charged_credits": total_credits,
                "charged": True,
            }

        return {
            "usage_cost_usd": usage_cost_usd,
            "success_fee_usd": success_fee_usd,
            "total_charge_usd": total_charge_usd,
            "charged_credits": Decimal("0"),
            "charged": False,
        }

def upload_report_to_worker_plan(run_id: str, report_path: Path) -> None:
    """
    Best-effort upload of the generated report to the worker_plan service so the frontend can fetch it
    even when worker_plan and worker_plan_database do not share a filesystem (e.g., Railway).
    """
    worker_plan_url = os.environ.get("PLANEXE_WORKER_PLAN_URL")
    if not worker_plan_url:
        return

    if not report_path.exists():
        logger.warning("Report path not found for run %s at %s; skipping upload to worker_plan.", run_id, report_path)
        return

    worker_plan_url = worker_plan_url.rstrip("/")
    url = f"{worker_plan_url}/runs/{run_id}/report"

    try:
        report_html = report_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Unable to read report for run %s: %s", run_id, exc)
        return

    try:
        response = requests.post(url, json={"report_html": report_html}, timeout=15)
    except Exception as exc:
        logger.warning("Error uploading report for run %s to worker_plan: %s", run_id, exc)
        return

    if response.status_code >= 300:
        logger.warning(
            "worker_plan returned %s when uploading report for run %s: %s",
            response.status_code,
            run_id,
            response.text[:500],
        )

def execute_pipeline_for_job(
    task_id: str,
    user_id: str,
    run_id_dir: Path,
    speedvsdetail: SpeedVsDetailEnum,
    model_profile: ModelProfileEnum,
    use_machai_developer_endpoint: bool,
    api_key_id: Optional[str] = None,
):
    start_time = time.time()
    logger.info(
        f"Executing pipeline for task_id: {task_id!r}, run_id_dir: {run_id_dir!r}, "
        f"speedvsdetail: {speedvsdetail!r}, model_profile: {model_profile.value!r}, "
        f"use_machai_developer_endpoint: {use_machai_developer_endpoint!r}..."
    )

    llm_models = ExecutePipeline.resolve_llm_models(None, model_profile=model_profile)
    pipeline_instance = ServerExecutePipeline(
        task_id=task_id,
        run_id_dir=run_id_dir,
        speedvsdetail=speedvsdetail,
        llm_models=llm_models,
        model_profile=model_profile,
    )
    # Keep a Flask app context active while running pipeline tasks so db-backed
    # instrumentation (for example token metrics) can access db.session safely.
    with app.app_context():
        set_current_task_id(task_id)
        set_current_user_id(user_id)
        set_current_api_key_id(api_key_id)
        previous_track_activity_path = track_activity.jsonl_file_path
        previous_model_profile = os.environ.get("PLANEXE_MODEL_PROFILE")
        try:
            os.environ["PLANEXE_MODEL_PROFILE"] = model_profile.value
            # Always keep activity tracking in the task run directory, including PING_LLM mode.
            track_activity.jsonl_file_path = run_id_dir / ExtraFilenameEnum.TRACK_ACTIVITY_JSONL.value

            if speedvsdetail == SpeedVsDetailEnum.PING_LLM:
                logger.info("PING_LLM mode requested; running a single LLM ping.")
                run_ping_llm_report(
                    run_id_dir=run_id_dir,
                    llm_models=LLMModelFromName.from_names(llm_models),
                )
            else:
                pipeline_instance.setup()
                logger.info(f"ExecutePipeline instance: {pipeline_instance!r}")

                pipeline_instance.run()
        finally:
            if previous_model_profile is None:
                os.environ.pop("PLANEXE_MODEL_PROFILE", None)
            else:
                os.environ["PLANEXE_MODEL_PROFILE"] = previous_model_profile
            track_activity.jsonl_file_path = previous_track_activity_path
            set_current_api_key_id(None)
            set_current_user_id(None)
            set_current_task_id(None)

    end_time = time.time()
    duration_in_seconds = end_time - start_time
    logger.info(f"Pipeline for {run_id_dir!r} executed in {duration_in_seconds:.2f} seconds")

    # Collect artifacts for storage.
    report_path = run_id_dir / FilenameEnum.REPORT.value
    report_html: Optional[str] = None
    if pipeline_instance.has_report_file and report_path.exists():
        try:
            report_html = report_path.read_text(encoding='utf-8')
        except Exception as exc:
            logger.warning("Unable to read report for task %s: %s", task_id, exc)

    run_zip_bytes: Optional[bytes] = None
    try:
        run_zip_bytes = create_zip_bytes(run_id_dir)
    except Exception as exc:
        logger.warning("Unable to create zip snapshot for task %s: %s", task_id, exc)
    run_track_activity_jsonl, run_track_activity_bytes, run_activity_overview_json = read_activity_artifacts(run_id_dir)

    # count number of files in the run_id_dir
    number_of_files_in_run_id_dir: int = len([f for f in run_id_dir.iterdir() if f.is_file()])

    event_context = {
        "task_id": str(task_id), 
        "user_id": str(user_id), 
        "run_id_dir": str(run_id_dir), 
        "speedvsdetail": str(speedvsdetail),
        "model_profile": model_profile.value,
        "duration_between_processing_and_completion": str(duration_in_seconds),
        "has_report_file": str(pipeline_instance.has_report_file),
        "has_stop_flag_file": str(pipeline_instance.has_stop_flag_file),
        "has_pipeline_complete_file": str(pipeline_instance.has_pipeline_complete_file),
        "luigi_build_return_value": str(pipeline_instance.luigi_build_return_value),
        "number_of_files_in_run_id_dir": str(number_of_files_in_run_id_dir),
        "WORKER_ID": str(WORKER_ID)
    }

    # Persist artifacts to the PlanItem record.
    stop_requested = False
    with app.app_context():
        task = db.session.get(PlanItem, task_id)
        if task is None:
            logger.error("Task %s not found while attempting to store report/zip.", task_id)
        else:
            stop_requested = bool(task.stop_requested)
            task.generated_report_html = report_html if pipeline_instance.has_report_file else None
            task.run_zip_snapshot = run_zip_bytes
            task.run_track_activity_jsonl = run_track_activity_jsonl
            task.run_track_activity_bytes = run_track_activity_bytes
            task.run_activity_overview_json = run_activity_overview_json
            task.run_artifact_layout_version = 2
            try:
                db.session.commit()
            except Exception as exc:
                logger.error("Failed to store report/zip for task %s: %s", task_id, exc, exc_info=True)
                db.session.rollback()

    event_context["stop_requested"] = str(stop_requested)

    if pipeline_instance.has_report_file:
        machai_error_message = ""
    elif pipeline_instance.has_stop_flag_file:
        if stop_requested:
            machai_error_message = 'Stopped by user.'
        else:
            machai_error_message = 'Inactive for too long, navigated away from the progress bar page, or closed the browser.'
    elif pipeline_instance.has_pipeline_complete_file:
        machai_error_message = 'Internal error. The pipeline complete file was found, but no report file was found.'
    else:
        machai_error_message = 'Error. Unable to generate the report. Likely reasons: censorship, restricted content.'

    # Update the PlanItem state to completed or failed
    final_progress = pipeline_instance.get_progress_percentage()
    with app.app_context():
        if pipeline_instance.has_report_file:
            update_task_state_with_retry(task_id, PlanState.completed)
            update_task_progress_with_retry(
                task_id, 100.0, "Completed",
                steps_completed=final_progress.steps_total,
                steps_total=final_progress.steps_total,
                current_step="Completed",
            )
            billing_result = _charge_usage_credits_once(task_id=task_id, run_id_dir=run_id_dir, success=True)
            event_context.update({
                "billing_usage_cost_usd": str(billing_result["usage_cost_usd"]),
                "billing_success_fee_usd": str(billing_result["success_fee_usd"]),
                "billing_total_charge_usd": str(billing_result["total_charge_usd"]),
                "billing_charged_credits": str(billing_result["charged_credits"]),
                "billing_charge_applied": str(billing_result["charged"]),
            })
            event = _new_model(
                EventItem,
                event_type=EventType.TASK_COMPLETED,
                message=f"Processing -> Completed",
                context=event_context
            )
            db.session.add(event)
            db.session.commit()
        else:
            update_task_state_with_retry(task_id, PlanState.failed)
            billing_result = _charge_usage_credits_once(task_id=task_id, run_id_dir=run_id_dir, success=False)
            event_context["machai_error_message"] = machai_error_message or ""
            event_context.update({
                "billing_usage_cost_usd": str(billing_result["usage_cost_usd"]),
                "billing_success_fee_usd": str(billing_result["success_fee_usd"]),
                "billing_total_charge_usd": str(billing_result["total_charge_usd"]),
                "billing_charged_credits": str(billing_result["charged_credits"]),
                "billing_charge_applied": str(billing_result["charged"]),
            })
            event = _new_model(
                EventItem,
                event_type=EventType.TASK_FAILED,
                message=f"Processing -> Failed",
                context=event_context
            )
            db.session.add(event)
            db.session.commit()

    # Post confirmation to MachAI
    machai_instance: MachAI = MachAI.create(use_machai_developer_endpoint=use_machai_developer_endpoint)
    if pipeline_instance.has_report_file:
        plan_name = 'Unnamed Plan'
        title_path = run_id_dir / FilenameEnum.WBS_LEVEL1_PROJECT_TITLE.value
        if title_path.is_file():
            plan_name = title_path.read_text(encoding='utf-8').strip()
            logger.debug(f"WBS_LEVEL1_PROJECT_TITLE file found at {title_path!r}. Using the plan_name: {plan_name!r}.")
        else:
            logger.warning(f"WBS_LEVEL1_PROJECT_TITLE file not found at {title_path!r}. Using the default plan_name: {plan_name!r}.")
        machai_instance.post_confirmation_ok_with_file(session_id=user_id, path=run_id_dir / FilenameEnum.REPORT.value, plan_name=plan_name)
        upload_report_to_worker_plan(run_id=str(task_id), report_path=run_id_dir / FilenameEnum.REPORT.value)
    else:
        machai_instance.post_confirmation_error(session_id=user_id, message=str(machai_error_message))

def process_pending_tasks() -> bool:
    """
    Attempts to claim and process one pending task.

    Pick up the oldest pending task from the FIFO queue and process it.
    """
    task_id: Optional[str] = None
    prompt: Optional[str] = None
    parameters = None
    use_machai_developer_endpoint: bool = False
    user_id: Optional[str] = None
    timestamp_created: Optional[datetime] = None
    speedvsdetail: SpeedVsDetailEnum = SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW
    model_profile: ModelProfileEnum = ModelProfileEnum.BASELINE

    with app.app_context():
        try:
            # Use a nested transaction for the claiming part.
            # This ensures that if the claim fails (e.g. row lock), we can rollback just the claim part.
            with db.session.begin_nested():
                # Atomically find and claim a task
                # Filter for pending tasks not yet assigned a worker_id
                # Order by creation time to ensure FIFO processing
                # `with_for_update(skip_locked=True)` is crucial for multi-worker
                # It tells the DB to lock the selected row and if it's already locked by another transaction,
                # skip it and try the next one, instead of waiting.
                task_to_claim = db.session.query(PlanItem)\
                    .filter(PlanItem.state == PlanState.pending)\
                    .filter(or_(PlanItem.stop_requested.is_(False), PlanItem.stop_requested.is_(None)))\
                    .order_by(PlanItem.timestamp_created.asc())\
                    .with_for_update(skip_locked=True)\
                    .first()

                if task_to_claim is None:
                    # No task available or all available tasks were locked by other workers
                    db.session.rollback() # Rollback (no changes made if no task found)
                    # logger.debug(f"No claimable pending tasks found.")
                    return False # No task claimed, sleep for a long time to avoid busy-waiting.

                # Extract all necessary data from task_to_claim BEFORE it's modified and transaction is committed
                task_id = str(task_to_claim.id)
                prompt = str(task_to_claim.prompt)
                parameters = task_to_claim.parameters if isinstance(task_to_claim.parameters, dict) else None
                speedvsdetail = resolve_speedvsdetail(parameters)
                model_profile = resolve_model_profile(parameters)
                use_machai_developer_endpoint = bool(task_to_claim.has_parameter_key('developer'))
                user_id = str(task_to_claim.user_id)
                api_key_id = getattr(task_to_claim, "api_key_id", None)
                timestamp_created = task_to_claim.timestamp_created
        
                # Now, modify the task state
                task_to_claim.state = PlanState.processing
                task_to_claim.progress_message = "Picked up by server"
                task_to_claim.progress_percentage = 0.0

                # Important: commit this nested transaction immediately to release the lock
                # and make the claim permanent.
                db.session.commit() 

        except Exception as e:
            db.session.rollback() # Rollback any potential changes from a failed claim attempt
            logger.error(f"DB error during task claiming: {e}", exc_info=True)
            return False # Error, sleep longer


    logger.info(
        f"Successfully claimed task: {task_id!r}, user_id: {user_id!r}, "
        f"timestamp_created: {timestamp_created!r}, model_profile: {model_profile.value!r}, "
        f"use_machai_developer_endpoint: {use_machai_developer_endpoint!r}"
    )

    with app.app_context():
        WorkerItem.upsert_heartbeat(worker_id=WORKER_ID, current_task_id=task_id)
        
    # Measure how long it took to pick up the task
    if timestamp_created is None:
        logger.warning("Task %s has no timestamp_created; using current time for duration estimate.", task_id)
        timestamp = datetime.now(UTC)
    else:
        timestamp = timestamp_created
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    duration_between_pending_and_processing = (datetime.now(UTC) - timestamp).total_seconds()
    logger.debug(f"Duration between pending and processing: {duration_between_pending_and_processing} seconds")

    # Create a run_id_dir for the task
    run_id_dir = BASE_DIR_RUN / task_id
    logger.debug(f"creating run_id_dir: {run_id_dir!r}")
    run_id_dir.mkdir(parents=True, exist_ok=True)

    # write the start time to the run_id_dir
    start_time: datetime = datetime.now().astimezone()
    start_time_file = StartTime.create(local_time=start_time)
    start_time_file.save(str(run_id_dir / FilenameEnum.START_TIME.value))

    # write the task prompt to the run_id_dir
    plan_file = PlanFile.create(vague_plan_description=prompt, start_time=start_time)
    plan_file.save(str(run_id_dir / FilenameEnum.INITIAL_PLAN.value))

    with app.app_context():
        event_context = {
            "task_id": str(task_id), 
            "user_id": str(user_id), 
            "run_id_dir": str(run_id_dir), 
            "speedvsdetail": str(speedvsdetail),
            "model_profile": model_profile.value,
            "duration_between_pending_and_processing": str(duration_between_pending_and_processing),
            "WORKER_ID": str(WORKER_ID)
        }
        event = _new_model(
            EventItem,
            event_type=EventType.TASK_PROCESSING,
            message=f"Pending -> Processing",
            context=event_context
        )
        db.session.add(event)
        db.session.commit()

    try:
        # Create run directory and execute pipeline
        execute_pipeline_for_job(
            task_id=task_id,
            user_id=user_id,
            run_id_dir=run_id_dir,
            speedvsdetail=speedvsdetail,
            model_profile=model_profile,
            use_machai_developer_endpoint=use_machai_developer_endpoint,
            api_key_id=api_key_id,
        )
        with app.app_context():
            WorkerItem.upsert_heartbeat(worker_id=WORKER_ID)
        return True # We just processed a task. There may be more pending tasks, don't sleep that long, so we can process the next task.
        
    except Exception as e:
        logger.error(f"Error processing task {task_id!r}: {e}", exc_info=True)
        # Update task state to failed
        with app.app_context():
            update_task_state_with_retry(task_id, PlanState.failed)
        billing_result = _charge_usage_credits_once(task_id=task_id, run_id_dir=run_id_dir, success=False)
        machai_error_message = 'Unknown error happened while processing.'
        machai_instance: MachAI = MachAI.create(use_machai_developer_endpoint=use_machai_developer_endpoint)
        machai_instance.post_confirmation_error(session_id=user_id, message=machai_error_message)
        with app.app_context():
            event_context = {
                "task_id": str(task_id), 
                "user_id": str(user_id), 
                "run_id_dir": str(run_id_dir), 
                "speedvsdetail": str(speedvsdetail),
                "model_profile": model_profile.value,
                "duration_between_pending_and_processing": str(duration_between_pending_and_processing),
                "WORKER_ID": str(WORKER_ID),
                "machai_error_message": str(machai_error_message),
                "billing_usage_cost_usd": str(billing_result["usage_cost_usd"]),
                "billing_success_fee_usd": str(billing_result["success_fee_usd"]),
                "billing_total_charge_usd": str(billing_result["total_charge_usd"]),
                "billing_charged_credits": str(billing_result["charged_credits"]),
                "billing_charge_applied": str(billing_result["charged"]),
            }
            event = _new_model(
                EventItem,
                event_type=EventType.TASK_FAILED,
                message=f"Processing -> Failed",
                context=event_context
            )
            db.session.add(event)
            db.session.commit()
        with app.app_context():
            WorkerItem.upsert_heartbeat(worker_id=WORKER_ID)
        return False # We didn't process a task. Sleep for a long time to avoid busy-waiting.
    finally:
        # Clean up the run_id_dir after the pipeline has completed and data is stored in the database.
        # This prevents the "run" directory from accumulating old session data.
        if run_id_dir.exists():
            try:
                shutil.rmtree(run_id_dir)
                logger.info(f"Cleaned up run_id_dir: {run_id_dir!r}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up run_id_dir {run_id_dir!r}: {cleanup_error}")

def startup_worker():
    with app.app_context():
        try:
            db.create_all()
            ensure_planitem_artifact_columns()
            ensure_step_count_columns()
            ensure_token_metrics_columns()
            ensure_fractional_credit_columns()
            ensure_multi_api_key_columns()
            logger.debug(f"Ensured database tables exist.")
            WorkerItem.upsert_heartbeat(worker_id=WORKER_ID)
        except Exception as e:    
            logger.critical(f"Error during startup: {e}", exc_info=True)
            raise e

def start_task_monitor():
    """Start monitoring the database for pending tasks."""
    logger.info("Started monitoring database for pending tasks.")
    try:
        last_heartbeat_time = time.time()
        while True:
            processed_something = process_pending_tasks()
            time.sleep(1 if processed_something else 5)
            
            # Wait N seconds between heartbeats, so the database doesn't get hammered with heartbeat updates. 
            new_heatbeat_time = time.time()
            if processed_something:
                # no need to update the last_heartbeat_time if we just processed a task
                last_heartbeat_time = new_heatbeat_time
            if new_heatbeat_time - last_heartbeat_time > HEARTBEAT_INTERVAL_IN_SECONDS:
                last_heartbeat_time = new_heatbeat_time
                with app.app_context():
                    WorkerItem.upsert_heartbeat(worker_id=WORKER_ID)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping task monitor...")
    except Exception as e:
        logger.critical(f"Unhandled exception in task monitor: {e}", exc_info=True)
    finally:
        logger.info("Task monitor shut down.")
        logging.shutdown()

if __name__ == "__main__":
    startup_worker()
    start_task_monitor()

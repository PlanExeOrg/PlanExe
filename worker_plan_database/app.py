"""
This project monitors the database for pending TaskItems and automatically changes their status to processing
when found. It then executes the pipeline for each task.

PROMPT> PLANEXE_WORKER_ID=1 python -m app.py
"""
from datetime import UTC, datetime
import os
import shutil
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
import uuid
import io
import zipfile
import requests
from sqlalchemy import inspect, text

WORKER_ID = os.environ.get("PLANEXE_WORKER_ID") or str(uuid.uuid4())

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

# --- Configure Logging Section ---
log_format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    from worker_plan_internal.utils.planexe_dotenv import PlanExeDotEnv
    from worker_plan_internal.llm_util.llm_executor import PipelineStopRequested
    from worker_plan_internal.llm_util.track_activity import TrackActivity
    from worker_plan_internal.plan.filenames import ExtraFilenameEnum
    logger.debug("Importing required modules... PlanExe-server.")
    from database_api.planexe_db_singleton import db
    from database_api.model_taskitem import TaskItem, TaskState
    from database_api.model_event import EventType, EventItem
    from database_api.model_worker import WorkerItem
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

def ensure_taskitem_artifact_columns() -> None:
    insp = inspect(db.engine)
    columns = {col["name"] for col in insp.get_columns("task_item")}
    with db.engine.begin() as conn:
        if "generated_report_html" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS generated_report_html TEXT"))
        if "run_zip_snapshot" not in columns:
            conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_zip_snapshot BYTEA"))

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
        event = EventItem(
            event_type=EventType.GENERIC_EVENT,
            message="Worker started",
            context=event_context
        )
        db.session.add(event)
        db.session.commit()

worker_process_started()

def update_task_state_with_retry(task_id: str, new_state: TaskState, max_retries: int = 3, retry_delay: int = 5) -> bool:
    """Helper function to update task state with retry logic for database operations."""
    for attempt in range(max_retries):
        try:
            task = db.session.get(TaskItem, task_id)
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

def update_task_progress_with_retry(task_id: str, progress_percentage: float, progress_message: str, max_retries: int = 3, retry_delay: int = 5) -> bool:
    """Helper function to update task progress with retry logic for database operations."""
    for attempt in range(max_retries):
        try:
            task = db.session.get(TaskItem, task_id)
            if task is None:
                logger.error(f"Task with ID {task_id!r} not found in database. Cannot update task progress.")
                return False
            
            task.progress_percentage = progress_percentage
            task.progress_message = progress_message
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


class ServerExecutePipeline(ExecutePipeline):
    def __init__(self, task_id: str, run_id_dir: Path, speedvsdetail: SpeedVsDetailEnum, llm_models: list[str]):
        super().__init__(run_id_dir=run_id_dir, speedvsdetail=speedvsdetail, llm_models=llm_models)
        self.task_id = task_id

    def _handle_task_completion(self, parameters: HandleTaskCompletionParameters) -> None:
        logger.debug(f"ServerExecutePipeline._handle_task_completion")

        with app.app_context():
            WorkerItem.upsert_heartbeat(worker_id=WORKER_ID, current_task_id=self.task_id)

        # Lookup the taskitem in the database by self.task_id
        with app.app_context():
            task = db.session.get(TaskItem, self.task_id)
            if task is None:
                logger.error(f"Task with ID {self.task_id!r} not found in database, while running the pipeline. This is an inconsistency.")
                raise Exception(f"Task with ID {self.task_id!r} not found in database, while running the pipeline. This is an inconsistency.")

        if task.last_seen_timestamp is None:
            # A new TaskItem is supposed to have a last_seen_timestamp.
            # If it doesn't have a last_seen_timestamp, it's an inconsistency that should be fixed.
            logger.error(f"Task with ID {self.task_id!r} has no last_seen_timestamp. This is an inconsistency.")
            raise Exception(f"Task with ID {self.task_id!r} has no last_seen_timestamp. This is an inconsistency.")

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
                progress_message=parameters.progress.progress_message
            )

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
    Create an in-memory zip of a run directory (skipping log.txt) and return the bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                if file == "log.txt":
                    continue
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(run_dir))
    buffer.seek(0)
    return buffer.read()

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

def execute_pipeline_for_job(task_id: str, user_id: str, run_id_dir: Path, speedvsdetail: SpeedVsDetailEnum, use_machai_developer_endpoint: bool):
    start_time = time.time()
    logger.info(f"Executing pipeline for task_id: {task_id!r}, run_id_dir: {run_id_dir!r}, speedvsdetail: {speedvsdetail!r}, use_machai_developer_endpoint: {use_machai_developer_endpoint!r}...")

    llm_models = ExecutePipeline.resolve_llm_models(None)
    pipeline_instance = ServerExecutePipeline(task_id=task_id, run_id_dir=run_id_dir, speedvsdetail=speedvsdetail, llm_models=llm_models)
    pipeline_instance.setup()
    logger.info(f"ExecutePipeline instance: {pipeline_instance!r}")

    # LLM/reasoning models often fail, due to censorship, invalid json, timeouts.
    # Thus I track whenever a LLM/reasoning model is used, by appended the payload+backtrace to the "track_activity" file.
    # so the developer can troubleshoot problems with the LLM/reasoning model.
    track_activity.jsonl_file_path = run_id_dir / ExtraFilenameEnum.TRACK_ACTIVITY_JSONL.value
    
    pipeline_instance.run()

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

    # count number of files in the run_id_dir
    number_of_files_in_run_id_dir: int = len([f for f in run_id_dir.iterdir() if f.is_file()])

    event_context = {
        "task_id": str(task_id), 
        "user_id": str(user_id), 
        "run_id_dir": str(run_id_dir), 
        "speedvsdetail": str(speedvsdetail), 
        "duration_between_processing_and_completion": str(duration_in_seconds),
        "has_report_file": str(pipeline_instance.has_report_file),
        "has_stop_flag_file": str(pipeline_instance.has_stop_flag_file),
        "has_pipeline_complete_file": str(pipeline_instance.has_pipeline_complete_file),
        "luigi_build_return_value": str(pipeline_instance.luigi_build_return_value),
        "number_of_files_in_run_id_dir": str(number_of_files_in_run_id_dir),
        "WORKER_ID": str(WORKER_ID)
    }

    if pipeline_instance.has_report_file:
        machai_error_message = None
    elif pipeline_instance.has_stop_flag_file:
        machai_error_message = 'Inactive for too long, navigated away from the progress bar page, or closed the browser.'
    elif pipeline_instance.has_pipeline_complete_file:
        machai_error_message = 'Internal error. The pipeline complete file was found, but no report file was found.'
    else:
        machai_error_message = 'Error. Unable to generate the report. Likely reasons: censorship, restricted content.'

    # Persist artifacts to the TaskItem record.
    with app.app_context():
        task = db.session.get(TaskItem, task_id)
        if task is None:
            logger.error("Task %s not found while attempting to store report/zip.", task_id)
        else:
            task.generated_report_html = report_html if pipeline_instance.has_report_file else None
            task.run_zip_snapshot = run_zip_bytes
            try:
                db.session.commit()
            except Exception as exc:
                logger.error("Failed to store report/zip for task %s: %s", task_id, exc, exc_info=True)
                db.session.rollback()

    # Update the TaskItem state to completed or failed
    with app.app_context():
        if pipeline_instance.has_report_file:
            update_task_state_with_retry(task_id, TaskState.completed)
            event = EventItem(
                event_type=EventType.TASK_COMPLETED,
                message=f"Processing -> Completed",
                context=event_context
            )
            db.session.add(event)
            db.session.commit()
        else:
            update_task_state_with_retry(task_id, TaskState.failed)
            event_context["machai_error_message"] = machai_error_message
            event = EventItem(
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
    fast: bool = True
    use_machai_developer_endpoint: bool = False
    user_id: Optional[str] = None
    timestamp_created: Optional[datetime] = None

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
                task_to_claim = db.session.query(TaskItem)\
                    .filter(TaskItem.state == TaskState.pending)\
                    .order_by(TaskItem.timestamp_created.asc())\
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
                fast = bool(task_to_claim.has_parameter_key('fast'))
                use_machai_developer_endpoint = bool(task_to_claim.has_parameter_key('developer'))
                user_id = str(task_to_claim.user_id)
                timestamp_created = task_to_claim.timestamp_created
        
                # Now, modify the task state
                task_to_claim.state = TaskState.processing
                task_to_claim.progress_message = "Picked up by server"
                task_to_claim.progress_percentage = 0.0

                # Important: commit this nested transaction immediately to release the lock
                # and make the claim permanent.
                db.session.commit() 

        except Exception as e:
            db.session.rollback() # Rollback any potential changes from a failed claim attempt
            logger.error(f"DB error during task claiming: {e}", exc_info=True)
            return False # Error, sleep longer


    logger.info(f"Successfully claimed task: {task_id!r}, user_id: {user_id!r}, timestamp_created: {timestamp_created!r}, use_machai_developer_endpoint: {use_machai_developer_endpoint!r}")

    with app.app_context():
        WorkerItem.upsert_heartbeat(worker_id=WORKER_ID, current_task_id=task_id)
        
    # Measure how long it took to pick up the task
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

    # Determine the speedvsdetail level
    if fast:
        speedvsdetail = SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS
    else:
        speedvsdetail = SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW

    with app.app_context():
        event_context = {
            "task_id": str(task_id), 
            "user_id": str(user_id), 
            "run_id_dir": str(run_id_dir), 
            "speedvsdetail": str(speedvsdetail), 
            "duration_between_pending_and_processing": str(duration_between_pending_and_processing),
            "WORKER_ID": str(WORKER_ID)
        }
        event = EventItem(
            event_type=EventType.TASK_PROCESSING,
            message=f"Pending -> Processing",
            context=event_context
        )
        db.session.add(event)
        db.session.commit()

    try:
        # Create run directory and execute pipeline
        execute_pipeline_for_job(task_id=task_id, user_id=user_id, run_id_dir=run_id_dir, speedvsdetail=speedvsdetail, use_machai_developer_endpoint=use_machai_developer_endpoint)
        with app.app_context():
            WorkerItem.upsert_heartbeat(worker_id=WORKER_ID)
        return True # We just processed a task. There may be more pending tasks, don't sleep that long, so we can process the next task.
        
    except Exception as e:
        logger.error(f"Error processing task {task_id!r}: {e}", exc_info=True)
        # Update task state to failed
        with app.app_context():
            update_task_state_with_retry(task_id, TaskState.failed)
        machai_error_message = 'Unknown error happened while processing.'
        machai_instance: MachAI = MachAI.create(use_machai_developer_endpoint=use_machai_developer_endpoint)
        machai_instance.post_confirmation_error(session_id=user_id, message=machai_error_message)
        with app.app_context():
            event_context = {
                "task_id": str(task_id), 
                "user_id": str(user_id), 
                "run_id_dir": str(run_id_dir), 
                "speedvsdetail": str(speedvsdetail), 
                "duration_between_pending_and_processing": str(duration_between_pending_and_processing),
                "WORKER_ID": str(WORKER_ID),
                "machai_error_message": str(machai_error_message)
            }
            event = EventItem(
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
            ensure_taskitem_artifact_columns()
            db.create_all()
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

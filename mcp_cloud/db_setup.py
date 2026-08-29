"""PlanExe MCP Cloud – database setup, Flask app, constants, and request classes."""
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote_plus

_DEBUG_STARTUP = os.environ.get("PLANEXE_DEBUG_STARTUP", "").strip().lower() in ("1", "true", "yes")

def _startup_log(msg: str) -> None:
    """Emit a startup breadcrumb to stderr when PLANEXE_DEBUG_STARTUP is set."""
    if _DEBUG_STARTUP:
        print(f"[startup] {msg}", file=sys.stderr, flush=True)

_startup_log("db_setup.py: begin imports")

from flask import Flask
from pydantic import BaseModel
from sqlalchemy import inspect, text
from worker_plan_api.model_profile import ModelProfileEnum

_startup_log("db_setup.py: imports done")

from mcp_cloud.dotenv_utils import load_planexe_dotenv

_dotenv_loaded, _dotenv_paths = load_planexe_dotenv(Path(__file__).parent)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
if not _dotenv_loaded:
    logger.warning(
        "No .env file found; searched: %s",
        ", ".join(str(path) for path in _dotenv_paths),
    )

from database_api.planexe_db_singleton import db
from database_api.model_planitem import PlanItem, PlanState
from database_api.model_event import EventItem, EventType
from database_api.model_user_account import UserAccount
from database_api.model_user_api_key import UserApiKey
from database_api.model_credit_history import CreditHistory  # noqa: F401
from database_api.model_token_metrics import TokenMetrics  # noqa: F401
from database_api.model_feedback import FeedbackItem  # noqa: F401

_startup_log("db_setup.py: creating Flask app")
app = Flask(__name__)
app.config.from_pyfile('config.py')
_startup_log("db_setup.py: Flask app created, config loaded")

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

sqlalchemy_database_uri = os.environ.get("SQLALCHEMY_DATABASE_URI")
if sqlalchemy_database_uri is None:
    sqlalchemy_database_uri, db_settings = build_postgres_uri_from_env(os.environ)
    _startup_log(f"db_setup.py: SQLALCHEMY_DATABASE_URI not set. Using Postgres defaults: {db_settings}")
    logger.info(f"SQLALCHEMY_DATABASE_URI not set. Using Postgres defaults: {db_settings}")
else:
    _startup_log("db_setup.py: Using SQLALCHEMY_DATABASE_URI from environment")
    logger.info("Using SQLALCHEMY_DATABASE_URI from environment.")

app.config['SQLALCHEMY_DATABASE_URI'] = sqlalchemy_database_uri
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 280,
    'pool_pre_ping': True,
    'connect_args': {'connect_timeout': 10},
}
_startup_log("db_setup.py: calling db.init_app(app)")
db.init_app(app)
_startup_log("db_setup.py: db.init_app(app) done")

def ensure_plans_table_name() -> None:
    """Rename legacy 'task_item' table (and its indexes) to 'plans' (idempotent).

    Wrapped in a transaction-level advisory lock so concurrent replicas do not
    race — whoever gets the lock first does the rename; the others no-op.
    Must run before db.create_all() so SQLAlchemy does not create a second
    empty 'plans' table alongside the existing 'task_item'.
    """
    sql = """
    DO $$
    BEGIN
        PERFORM pg_advisory_xact_lock(8462357421);
        IF EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = current_schema() AND table_name = 'task_item')
           AND NOT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_schema = current_schema() AND table_name = 'plans') THEN
            ALTER TABLE task_item RENAME TO plans;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_indexes
                   WHERE schemaname = current_schema() AND indexname = 'idx_task_item_user_id_timestamp_created') THEN
            ALTER INDEX idx_task_item_user_id_timestamp_created RENAME TO idx_plans_user_id_timestamp_created;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_indexes
                   WHERE schemaname = current_schema() AND indexname = 'idx_task_item_api_key_id') THEN
            ALTER INDEX idx_task_item_api_key_id RENAME TO idx_plans_api_key_id;
        END IF;
    END$$;
    """
    try:
        with db.engine.begin() as conn:
            conn.execute(text(sql))
    except Exception as exc:
        logger.warning("Rename task_item -> plans failed: %s", exc, exc_info=True)


def ensure_planitem_stop_columns() -> None:
    statements = (
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS run_track_activity_jsonl TEXT",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS run_track_activity_bytes INTEGER",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS run_activity_overview_json JSON",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS run_artifact_layout_version INTEGER",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS stop_requested BOOLEAN",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS stop_requested_timestamp TIMESTAMP",
    )
    for statement in statements:
        try:
            with db.engine.begin() as conn:
                conn.execute(text(statement))
        except Exception as exc:
            logger.warning("Schema update failed for %s: %s", statement, exc, exc_info=True)

def ensure_multi_api_key_columns() -> None:
    """Add columns for multi-API-key support (idempotent)."""
    statements = (
        "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS label VARCHAR(128)",
        "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS key_plaintext VARCHAR(64)",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
        "ALTER TABLE credit_history ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
        "ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
    )
    for stmt in statements:
        try:
            with db.engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)

def ensure_step_count_columns() -> None:
    """Add steps_completed, steps_total, and current_step columns to plans (idempotent)."""
    statements = (
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS steps_completed INTEGER",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS steps_total INTEGER",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS current_step VARCHAR(128)",
    )
    for stmt in statements:
        try:
            with db.engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)

def ensure_failure_diagnostics_columns() -> None:
    """Add failure diagnostic columns to plans (idempotent)."""
    statements = (
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS failure_reason VARCHAR(64)",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS failed_step VARCHAR(128)",
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS recoverable BOOLEAN",
    )
    for stmt in statements:
        try:
            with db.engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)
    # Rename last_error → error_message (existing DBs); add column for fresh DBs.
    # Check column existence first to avoid noisy PostgreSQL ERROR logs on every restart.
    columns = {col["name"] for col in inspect(db.engine).get_columns("plans")}
    if "error_message" not in columns:
        if "last_error" in columns:
            try:
                with db.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE plans RENAME COLUMN last_error TO error_message"))
            except Exception:
                with db.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE plans ADD COLUMN IF NOT EXISTS error_message VARCHAR(256)"))
        else:
            with db.engine.begin() as conn:
                conn.execute(text("ALTER TABLE plans ADD COLUMN IF NOT EXISTS error_message VARCHAR(256)"))

def ensure_stopped_state() -> None:
    """Add 'stopped' value to the planstate/taskstate enum type (idempotent).

    The PostgreSQL enum type is named ``taskstate`` in databases created before
    the TaskState → PlanState Python rename (proposal 74).  Fresh databases
    created after that rename will have ``planstate``.  We try both names.
    """
    for type_name in ("taskstate", "planstate"):
        try:
            with db.engine.begin() as conn:
                conn.execute(text(f"ALTER TYPE {type_name} ADD VALUE IF NOT EXISTS 'stopped'"))
        except Exception as exc:
            logger.debug("ALTER TYPE %s: %s", type_name, exc)

def ensure_last_progress_at_column() -> None:
    """Add last_progress_at column to plans (idempotent)."""
    statements = (
        "ALTER TABLE plans ADD COLUMN IF NOT EXISTS last_progress_at TIMESTAMP",
    )
    for stmt in statements:
        try:
            with db.engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)

_startup_log("db_setup.py: running schema migrations...")
with app.app_context():
    _startup_log("db_setup.py: ensure_plans_table_name")
    ensure_plans_table_name()
    _startup_log("db_setup.py: db.create_all()")
    db.create_all()
    _startup_log("db_setup.py: db.create_all() done")
    _startup_log("db_setup.py: ensure_planitem_stop_columns")
    ensure_planitem_stop_columns()
    _startup_log("db_setup.py: ensure_multi_api_key_columns")
    ensure_multi_api_key_columns()
    _startup_log("db_setup.py: ensure_step_count_columns")
    ensure_step_count_columns()
    _startup_log("db_setup.py: ensure_failure_diagnostics_columns")
    ensure_failure_diagnostics_columns()
    _startup_log("db_setup.py: ensure_stopped_state")
    ensure_stopped_state()
    _startup_log("db_setup.py: ensure_last_progress_at_column")
    ensure_last_progress_at_column()
_startup_log("db_setup.py: schema migrations complete")

# Shown in MCP initialize (e.g. Inspector) so clients know what PlanExe does.
PLANEXE_SERVER_INSTRUCTIONS = (
    "PlanExe generates strategic project-plan drafts from a natural-language prompt. "
    "Output is a self-contained interactive HTML report (~700KB) with 20+ sections including "
    "executive summary, interactive Gantt charts, risk analysis, SWOT, governance, investor pitch, "
    "team profiles, work breakdown, scenario comparison, expert criticism, and adversarial sections "
    "(premortem, self-audit checklist, premise attacks) that stress-test whether the plan holds up. "
    "The output is a draft to refine, not final ground truth — but it surfaces hard questions the prompter may not have considered. "
    "Use PlanExe for substantial multi-phase projects with constraints, stakeholders, budgets, and timelines. "
    "Do not use PlanExe for tiny one-shot outputs (for example: 'give me a 5-point checklist'); use a normal LLM response for that. "
    "The planning pipeline is fixed end-to-end; callers cannot select individual internal pipeline steps to run. "
    "Required interaction order: call example_plans first (optional, to preview what PlanExe output looks like — curated example reports and zip bundles). "
    "Then call example_prompts. "
    "Optional before plan_create: call model_profiles to see profile guidance and available models in each profile. "
    "Then perform a non-tool step: draft a strong prompt as flowing prose (not structured markdown with headers or bullets), "
    "typically ~300-800 words, and get user approval. "
    "Good prompt shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. "
    "Write the prompt as flowing prose — weave specs, constraints, and targets naturally into sentences. "
    "Only after approval, call plan_create. "
    "Each plan_create call creates a new plan_id; the server does not enforce a global per-client concurrency limit. "
    "Then poll plan_status (about every 5 minutes) to check progress; use plan_file_info when complete. "
    "plan_status is the primary way to track progress — it returns structured JSON with progress_percentage, "
    "steps_completed/steps_total, and current_step. "
    "Optionally, run `curl -N <sse_url>` in a background shell as a completion detector — "
    "the stream auto-closes when the plan reaches a terminal state (completed/failed/stopped). "
    "If plan generation fails before completing all steps, call plan_resume to continue from where it left off without discarding completed work. "
    "Use plan_retry instead for a full restart (plan must be in failed state). "
    "Both accept a failed plan_id and optional model_profile (defaults to baseline). "
    "To stop, call plan_stop with the plan_id from plan_create; stopping is asynchronous and the plan will transition to the stopped state. "
    "If model_profiles returns MODEL_PROFILES_UNAVAILABLE, inform the user that no models are currently configured and the server administrator needs to set up model profiles. "
    "Tool errors use {error:{code,message}}. plan_file_info returns {ready:false,reason:...} while the artifact is not yet ready; check readiness by testing whether download_url is present in the response. "
    "plan_file_info download_url is the absolute URL where the requested artifact can be downloaded. "
    "To list recent plans for a user call plan_list; returns plan_id, state, progress_percentage, created_at, and prompt_excerpt for each plan. "
    "plan_status state contract: pending/processing => keep polling; completed => download is ready; failed => terminal error; stopped => user called plan_stop (consider plan_resume). "
    "Troubleshooting: if plan_status stays in pending for longer than 5 minutes, the plan was likely queued but not picked up by a worker (server issue). "
    "If plan_status is in processing and output files do not change for longer than 20 minutes, the plan_create likely failed/stalled. "
    "In both cases, report the issue to PlanExe developers on GitHub: https://github.com/PlanExeOrg/PlanExe/issues . "
    "Main output: a self-contained interactive HTML report (~700KB) with collapsible sections and interactive Gantt charts — open in a browser. "
    "The zip contains the intermediary pipeline files (md, json, csv) that fed the report. "
    "Use send_feedback to report issues or share observations about plan quality, workflow friction, or the MCP interface. "
    "Feedback is fire-and-forget and never blocks the workflow. "
    "New users: create an account and obtain an API key at https://home.planexe.org/ ."
)

WORKER_PLAN_URL = os.environ.get("PLANEXE_WORKER_PLAN_URL", "http://worker_plan:8000")

REPORT_FILENAME = "report.html"
REPORT_CONTENT_TYPE = "text/html; charset=utf-8"
ZIP_FILENAME = "run.zip"
ZIP_CONTENT_TYPE = "application/zip"
ZIP_SNAPSHOT_MAX_BYTES = 100_000_000

ModelProfileInput = Literal[
    "baseline",
    "premium",
    "frontier",
    "custom",
]
MODEL_PROFILE_TITLES = {
    ModelProfileEnum.BASELINE.value: "Baseline",
    ModelProfileEnum.PREMIUM.value: "Premium",
    ModelProfileEnum.FRONTIER.value: "Frontier",
    ModelProfileEnum.CUSTOM.value: "Custom",
}
MODEL_PROFILE_SUMMARIES = {
    ModelProfileEnum.BASELINE.value: "Cheap and fast; recommended default when creating a plan.",
    ModelProfileEnum.PREMIUM.value: "Higher-cost profile tuned for stronger output quality.",
    ModelProfileEnum.FRONTIER.value: "Most capable models first; usually slowest/most expensive.",
    ModelProfileEnum.CUSTOM.value: "User-managed profile file for custom model ordering.",
}

class PlanCreateRequest(BaseModel):
    prompt: str
    model_profile: Optional[ModelProfileInput] = None
    user_api_key: Optional[str] = None
    start_date: Optional[str] = None

class PlanStatusRequest(BaseModel):
    plan_id: str

class PlanStopRequest(BaseModel):
    plan_id: str

class PlanRetryRequest(BaseModel):
    plan_id: str
    model_profile: ModelProfileInput = "baseline"

class PlanResumeRequest(BaseModel):
    plan_id: str
    model_profile: ModelProfileInput = "baseline"

class PlanFileInfoRequest(BaseModel):
    plan_id: str
    artifact: Optional[str] = None

class PlanListRequest(BaseModel):
    user_api_key: Optional[str] = None
    limit: int = 10

class ModelProfilesRequest(BaseModel):
    """No input parameters."""
    pass

class SendFeedbackRequest(BaseModel):
    category: str
    message: str
    plan_id: Optional[str] = None
    rating: Optional[int] = None
    user_api_key: Optional[str] = None

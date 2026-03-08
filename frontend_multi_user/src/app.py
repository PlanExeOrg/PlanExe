"""
Flask UI for PlanExe-server.

PROMPT> python3 -m src.app
"""
from datetime import datetime, UTC
from decimal import Decimal, ROUND_CEILING
import logging
import os
import re
import sys
import time
import json
import uuid
import io
import zipfile
import secrets
import hashlib
import tempfile
from urllib.parse import quote_plus, urlparse
from typing import ClassVar, Dict, Optional, Tuple, Any, cast
from types import SimpleNamespace
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_file, redirect, url_for, session, abort
from flask_admin import Admin, AdminIndexView, expose
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from authlib.integrations.flask_client import OAuth
from flask_wtf.csrf import CSRFProtect
from functools import wraps
import urllib.request
from urllib.error import URLError
from flask import make_response
import requests
import stripe
from worker_plan_api.start_time import StartTime
from worker_plan_api.plan_file import PlanFile
from worker_plan_api.filenames import FilenameEnum, ExtraFilenameEnum
from worker_plan_api.prompt_catalog import PromptCatalog
from sqlalchemy import text, inspect, func
from sqlalchemy.exc import OperationalError, DataError
from database_api.model_planitem import PlanItem, PlanState
from database_api.model_event import EventType, EventItem
from database_api.model_worker import WorkerItem
from database_api.model_nonce import NonceItem
from database_api.model_user_account import UserAccount
from database_api.model_user_provider import UserProvider
from database_api.model_user_api_key import UserApiKey
from database_api.model_credit_history import CreditHistory
from database_api.model_payment_record import PaymentRecord
from database_api.model_token_metrics import TokenMetrics, TokenMetricsSummary
from planexe_modelviews import WorkerItemView, PlanItemView, NonceItemView, TokenMetricsView, AdminOnlyModelView, UserAccountView
logger = logging.getLogger(__name__)

from worker_plan_api.planexe_dotenv import DotEnvKeyEnum, PlanExeDotEnv
from worker_plan_api.planexe_config import PlanExeConfig
from worker_plan_api.model_profile import ModelProfileEnum, normalize_model_profile
from worker_plan_api.llm_class_filter import (
    ENV_PLANEXE_LLM_CONFIG_WHITELISTED_CLASSES,
    is_llm_class_allowed,
    parse_llm_class_whitelist,
)

RUN_DIR = "run"

SHOW_DEMO_PLAN = False

CREDIT_SCALE = Decimal("0.000000001")

DEMO_FORM_RUN_PROMPT_UUIDS = [
    "ab700769-c3ba-4f8a-913d-8589fea4624e",
    "00e1c738-a663-476a-b950-62785922f6f0",
    "e6ddd953-939f-4d15-89ec-fd3988f79123"
]

AUTH_PROVIDER_LABELS = {
    "google": "Google",
    "github": "GitHub",
    "discord": "Discord",
    "password": "Password",
    "telegram": "Telegram",
    "open_access": "Open access",
}


def _new_model(model_cls: Any, **kwargs: Any) -> Any:
    """Instantiate ORM models through Any to accommodate dynamic Flask-SQLAlchemy typing."""
    return cast(Any, model_cls)(**kwargs)

def build_postgres_uri_from_env(env: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """Construct a SQLAlchemy URI for Postgres using environment variables."""
    host = env.get("PLANEXE_FRONTEND_MULTIUSER_DB_HOST") or env.get("PLANEXE_POSTGRES_HOST") or "database_postgres"
    port = str(env.get("PLANEXE_FRONTEND_MULTIUSER_DB_PORT") or env.get("PLANEXE_POSTGRES_PORT") or "5432")
    dbname = env.get("PLANEXE_FRONTEND_MULTIUSER_DB_NAME") or env.get("PLANEXE_POSTGRES_DB") or "planexe"
    user = env.get("PLANEXE_FRONTEND_MULTIUSER_DB_USER") or env.get("PLANEXE_POSTGRES_USER") or "planexe"
    password = env.get("PLANEXE_FRONTEND_MULTIUSER_DB_PASSWORD") or env.get("PLANEXE_POSTGRES_PASSWORD") or "planexe"
    uri = f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    safe_config = {"host": host, "port": port, "dbname": dbname, "user": user}
    return uri, safe_config

class User(UserMixin):
    def __init__(self, user_id: str, is_admin: bool = False):
        self.id = str(user_id)
        self.is_admin = is_admin

class MyAdminIndexView(AdminIndexView):
    @expose('/')
    def index(self) -> Any:
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if not current_user.is_admin:
            abort(403)
        return super(MyAdminIndexView, self).index()

    def is_accessible(self):
        return current_user.is_authenticated and getattr(current_user, "is_admin", False)

    def inaccessible_callback(self, name, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login"))
        abort(403)

def nocache(view):
    """Decorator to add 'no-cache' headers to a response."""
    @wraps(view)
    def no_cache_view(*args, **kwargs):
        # Call the original view function
        response = make_response(view(*args, **kwargs))
        # Modify headers
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1' # Or any date in the past, or 0
        return response
    return no_cache_view

def admin_required(view):
    """Decorator that requires an authenticated admin user."""
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def _profile_model_rows_map() -> Dict[str, list[dict[str, str]]]:
    whitelist = parse_llm_class_whitelist(os.environ.get(ENV_PLANEXE_LLM_CONFIG_WHITELISTED_CLASSES))
    profile_to_models: Dict[str, list[dict[str, str]]] = {}
    for profile in ModelProfileEnum:
        config = PlanExeConfig.load(model_profile_override=profile)
        config_path = config.llm_config_json_path
        if config_path is None:
            profile_to_models[profile.value] = []
            continue
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                model_map = json.load(fh)
        except Exception:
            profile_to_models[profile.value] = []
            continue
        if not isinstance(model_map, dict):
            profile_to_models[profile.value] = []
            continue

        def sort_key(item: tuple[str, dict]) -> tuple[int, str]:
            data = item[1] if isinstance(item[1], dict) else {}
            priority = data.get("priority")
            if not isinstance(priority, int):
                priority = 999999
            return priority, item[0]

        rows: list[dict[str, str]] = []
        for model_key, model_data in sorted(model_map.items(), key=sort_key):
            class_name = model_data.get("class") if isinstance(model_data, dict) else None
            if not is_llm_class_allowed(class_name, whitelist):
                continue
            model_name = ""
            comment = ""
            prio = ""
            model_info_url = ""
            pricing_kind = ""
            if isinstance(model_data, dict):
                args = model_data.get("arguments")
                if isinstance(args, dict) and isinstance(args.get("model"), str):
                    model_name = args["model"]
                if isinstance(model_data.get("comment"), str):
                    comment = model_data["comment"]
                if isinstance(model_data.get("prio"), int):
                    prio = str(model_data["prio"])
                elif isinstance(model_data.get("priority"), int):
                    prio = str(model_data["priority"])
                if isinstance(model_data.get("model_info_url"), str):
                    model_info_url = model_data["model_info_url"]
                if isinstance(model_data.get("pricing_kind"), str):
                    pricing_kind = model_data["pricing_kind"]
            rows.append(
                {
                    "key": model_key,
                    "prio": prio,
                    "pricing_kind": pricing_kind,
                    "model": model_name,
                    "comment": comment,
                    "model_info_url": model_info_url,
                }
            )
        profile_to_models[profile.value] = rows
    return profile_to_models


def _model_profile_options() -> list[dict[str, str]]:
    return [
        {
            "value": ModelProfileEnum.BASELINE.value,
            "title": "Baseline",
            "subtitle": "Cheap and fast.",
        },
        {
            "value": ModelProfileEnum.PREMIUM.value,
            "title": "Premium",
            "subtitle": "Expensive, slow, high quality.",
        },
        {
            "value": ModelProfileEnum.FRONTIER.value,
            "title": "Frontier",
            "subtitle": "Most capable models first.",
        },
        {
            "value": ModelProfileEnum.CUSTOM.value,
            "title": "Custom",
            "subtitle": "Use your own config file.",
        },
    ]

class MyFlaskApp:
    def __init__(self):
        logger.info(f"MyFlaskApp.__init__. Starting...")

        self.planexe_config = PlanExeConfig.load()
        logger.info(f"MyFlaskApp.__init__. planexe_config: {self.planexe_config!r}")

        self.planexe_dotenv = PlanExeDotEnv.load()
        logger.info(f"MyFlaskApp.__init__. planexe_dotenv: {self.planexe_dotenv!r}")

        # This is a workaround to fix the inconsistency.
        # Workaround-problem: When the Flask app launches in debug mode it runs __init__ twice, so that the app can hot reload.
        # However there is this inconsistency.
        # 1st time, the os.environ is the original environment of the shell.
        # 2nd time, the os.environ is the original environment of the shell + the .env content.
        # If it was the same in both cases, it would be easier to reason about the environment variables.
        # On following hot reloads, the os.environ continues to be the original environment of the shell + the .env content.
        # Workaround-solution: Every time update the os.environ with the .env content, so that the os.environ is always the 
        # original environment of the shell + the .env content.
        # I prefer NEVER to modify the os.environ for the current process, and instead spawn a child process with the modified os.environ.
        self.planexe_dotenv.update_os_environ()

        self.admin_username = (self.planexe_dotenv.get("PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME") or "").strip()
        self.admin_password = (self.planexe_dotenv.get("PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD") or "").strip()
        if not self.admin_username or not self.admin_password:
            raise ValueError("Admin credentials must be set via PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME and PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD.")
        if self.admin_username == "admin" and self.admin_password == "admin":
            logger.warning("Admin credentials are set to the default admin/admin; set PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME/PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD to unique values.")
        else:
            logger.info("Admin credentials loaded from PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME/PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD.")

        override_path_to_python = self.planexe_dotenv.get_absolute_path_to_file(DotEnvKeyEnum.PATH_TO_PYTHON.value)
        if isinstance(override_path_to_python, Path):
            debug_path_to_python = 'override'
            self.path_to_python = override_path_to_python
        else:
            debug_path_to_python = 'default'
            self.path_to_python = Path(sys.executable)
        logger.info(f"MyFlaskApp.__init__. path_to_python ({debug_path_to_python}): {self.path_to_python!r}")
        
        self.planexe_project_root = Path(__file__).parent.parent.parent.absolute()
        logger.info(f"MyFlaskApp.__init__. planexe_project_root: {self.planexe_project_root!r}")

        override_planexe_run_dir = self.planexe_dotenv.get_absolute_path_to_dir(DotEnvKeyEnum.PLANEXE_RUN_DIR.value)
        if isinstance(override_planexe_run_dir, Path):
            debug_planexe_run_dir = 'override'
            self.planexe_run_dir = override_planexe_run_dir
        else:
            debug_planexe_run_dir = 'default'
            self.planexe_run_dir = self.planexe_project_root / RUN_DIR
        logger.info(f"MyFlaskApp.__init__. planexe_run_dir ({debug_planexe_run_dir}): {self.planexe_run_dir!r}")

        self.worker_plan_url = (os.environ.get("PLANEXE_WORKER_PLAN_URL") or "http://worker_plan:8000").rstrip("/")
        logger.info(f"MyFlaskApp.__init__. worker_plan_url: {self.worker_plan_url}")

        self._start_check()

        # Load prompt catalog and examples.
        self.prompt_catalog = PromptCatalog()
        self.prompt_catalog.load_simple_plan_prompts()

        # Lightweight in-process cache for terminal task telemetry snapshots.
        # This avoids repeatedly re-querying/re-parsing run artifacts when the
        # plan iframe polls /plan/meta after a task is already finished.
        self._plan_telemetry_cache: dict[tuple[str, str, bool], dict[str, Any]] = {}

        # Point to the "templates" dir.
        # Prefer top-level templates dir (frontend_multi_user/templates) when running from Docker image.
        default_template_folder = Path(__file__).parent / "templates"
        alt_template_folder = Path(__file__).parent.parent / "templates"
        template_folder = default_template_folder if default_template_folder.exists() else alt_template_folder
        logger.info(f"MyFlaskApp.__init__. template_folder: {template_folder!r}")
        self.app = Flask(__name__, template_folder=str(template_folder))
        
        # Load configuration from config.py when present; otherwise use safe defaults.
        config_path = Path(__file__).with_name("config.py")
        if config_path.exists():
            logger.info("Loading Flask config from %s", config_path)
            self.app.config.from_pyfile(str(config_path))
        else:
            logger.warning("Config file not found at %s; using fallback settings.", config_path)
            self.app.config.from_mapping(
                SECRET_KEY=os.environ.get("PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY", "dev-secret-key"),
                SQLALCHEMY_TRACK_MODIFICATIONS=False,
            )

        # Env overrides: production sets PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY; honor it over config.py
        env_secret = os.environ.get("PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY")
        if env_secret:
            self.app.config["SECRET_KEY"] = env_secret

        _public_url = os.environ.get("PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL", "").strip()
        if not _public_url:
            _public_url = "http://localhost:5001"
            logger.info("PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL not set; defaulting to %s", _public_url)
        self.public_base_url = _public_url.rstrip("/")

        # Validate SECRET_KEY - check for both default values
        secret_key = self.app.config.get("SECRET_KEY")
        is_default_key = secret_key in ("dev-secret-key", "your-secret-key", None)
        is_production = os.environ.get("FLASK_ENV") == "production" or self._looks_like_production_url(self.public_base_url)

        if is_default_key:
            if is_production:
                raise ValueError(
                    "Cannot use default SECRET_KEY in production. "
                    "Set PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY environment variable. "
                    "Generate with: python -c 'import secrets; print(secrets.token_hex(32))'"
                )
            logger.warning(
                "Using default Flask SECRET_KEY (%s). "
                "Set PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY for production.",
                secret_key
            )

        # Session cookie security settings
        self.app.config.setdefault('SESSION_COOKIE_SECURE', is_production)
        self.app.config.setdefault('SESSION_COOKIE_HTTPONLY', True)
        self.app.config.setdefault('SESSION_COOKIE_SAMESITE', 'Lax')
        logger.info(f"Session cookie security: secure={is_production}, httponly=True, samesite=Lax")

        if self.public_base_url.lower().startswith("https://"):
            self.app.config["SESSION_COOKIE_SECURE"] = True
            self.app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

        # Enable CSRF protection
        self.csrf = CSRFProtect(self.app)
        logger.info("CSRF protection enabled")

        self.oauth = OAuth(self.app)
        self._register_oauth_providers()

        # Determine open access mode (no login required).
        # When no OAuth providers are configured and PLANEXE_AUTH_REQUIRED is not
        # explicitly true, the app auto-logins every request as admin so Docker
        # localhost users can create plans without setting up Google Console, etc.
        self.open_access = self._determine_open_access()
        self._api_key_show_once = os.environ.get("PLANEXE_API_KEY_SHOW_ONCE", "").strip().lower() in ("1", "true", "yes")

        db_settings: Dict[str, str] = {}
        sqlalchemy_database_uri = self.planexe_dotenv.get("SQLALCHEMY_DATABASE_URI")
        if sqlalchemy_database_uri is None:
            sqlalchemy_database_uri, db_settings = build_postgres_uri_from_env(self.planexe_dotenv.dotenv_dict)
            logger.info(
                "Using Postgres defaults for SQLAlchemy: %(host)s:%(port)s/%(dbname)s user=%(user)s",
                db_settings
            )
        else:
            logger.info("Using SQLALCHEMY_DATABASE_URI from environment or .env file.")

        self.app.config['SQLALCHEMY_DATABASE_URI'] = sqlalchemy_database_uri
        self.app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle': 280, 'pool_pre_ping': True}
        self.database_settings = db_settings if db_settings else {"uri_source": "SQLALCHEMY_DATABASE_URI"}
        
        # Initialize database
        from database_api.planexe_db_singleton import db
        self.db = db
        self.db.init_app(self.app)
        
        def _ensure_planitem_artifact_columns() -> None:
            insp = inspect(self.db.engine)
            columns = {col["name"] for col in insp.get_columns("task_item")}
            with self.db.engine.begin() as conn:
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

        def _ensure_token_metrics_columns() -> None:
            insp = inspect(self.db.engine)
            if "token_metrics" not in insp.get_table_names():
                return
            columns = {col["name"] for col in insp.get_columns("token_metrics")}
            with self.db.engine.begin() as conn:
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

        def _ensure_fractional_credit_columns() -> None:
            if self.db.engine.dialect.name != "postgresql":
                return
            insp = inspect(self.db.engine)
            table_names = set(insp.get_table_names())
            with self.db.engine.begin() as conn:
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

        def _ensure_user_account_columns() -> None:
            insp = inspect(self.db.engine)
            if "user_account" not in set(insp.get_table_names()):
                return
            columns = {col["name"] for col in insp.get_columns("user_account")}
            with self.db.engine.begin() as conn:
                if "frontend_multi_user_config" not in columns:
                    conn.execute(text("ALTER TABLE user_account ADD COLUMN IF NOT EXISTS frontend_multi_user_config JSON"))

        def _ensure_multi_api_key_columns() -> None:
            """Add columns for multi-API-key support (idempotent)."""
            statements = (
                "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS label VARCHAR(128)",
                "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS key_plaintext VARCHAR(64)",
                "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
                "ALTER TABLE credit_history ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
                "ALTER TABLE token_metrics ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
            )
            with self.db.engine.begin() as conn:
                for stmt in statements:
                    try:
                        conn.execute(text(stmt))
                    except Exception as exc:
                        logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)

        def _ensure_step_count_columns() -> None:
            insp = inspect(self.db.engine)
            columns = {col["name"] for col in insp.get_columns("task_item")}
            with self.db.engine.begin() as conn:
                if "steps_completed" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS steps_completed INTEGER"))
                if "steps_total" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS steps_total INTEGER"))
                if "current_step" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS current_step VARCHAR(128)"))

        def _ensure_planitem_indexes() -> None:
            insp = inspect(self.db.engine)
            table_names = set(insp.get_table_names())
            statements: list[str] = []
            if "task_item" in table_names:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_task_item_user_id_timestamp_created "
                    "ON task_item (user_id, timestamp_created)"
                )
                statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_task_item_api_key_id "
                    "ON task_item (api_key_id)"
                )
            if "credit_history" in table_names:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_credit_history_api_key_id "
                    "ON credit_history (api_key_id)"
                )
            for stmt in statements:
                try:
                    with self.db.engine.begin() as conn:
                        conn.execute(text(stmt))
                except Exception as exc:
                    logger.warning("Index creation skipped for %s: %s", stmt, exc)

        def _seed_initial_records() -> None:
            # Add initial records if the table is empty
            if PlanItem.query.count() == 0:
                tasks = PlanItem.demo_items()
                for task in tasks:
                    self.db.session.add(task)
                self.db.session.commit()

            if EventItem.query.count() == 0:
                events = EventItem.demo_items()
                for event in events:
                    self.db.session.add(event)
                self.db.session.commit()

            if NonceItem.query.count() == 0:
                nonce_items = NonceItem.demo_items()
                for nonce_item in nonce_items:
                    self.db.session.add(nonce_item)
                self.db.session.commit()

        def _create_tables_with_retry(attempts: int = 5, delay_seconds: float = 2.0) -> None:
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    with self.app.app_context():
                        self.db.create_all()
                        _ensure_planitem_artifact_columns()
                        _ensure_token_metrics_columns()
                        _ensure_fractional_credit_columns()
                        _ensure_user_account_columns()
                        _ensure_multi_api_key_columns()
                        _ensure_step_count_columns()
                        _ensure_planitem_indexes()
                        _seed_initial_records()
                    return
                except OperationalError as exc:
                    last_exc = exc
                    logger.warning(
                        "Database init attempt %s/%s failed: %s. Retrying in %.1fs",
                        attempt,
                        attempts,
                        exc,
                        delay_seconds,
                    )
                    time.sleep(delay_seconds)
                except Exception as exc:  # pragma: no cover - startup guardrail
                    last_exc = exc
                    logger.error(
                        "Unexpected error during database init attempt %s/%s: %s",
                        attempt,
                        attempts,
                        exc,
                        exc_info=True,
                    )
                    time.sleep(delay_seconds)
            if last_exc:
                raise last_exc

        _create_tables_with_retry()
        
        # Setup Flask-Login
        self.login_manager = LoginManager()
        self.login_manager.init_app(self.app)
        self.login_manager.login_view = cast(Any, 'login')
        
        @self.login_manager.user_loader
        def load_user(user_id):
            if user_id == self.admin_username:
                return User(user_id, is_admin=True)
            try:
                user_uuid = uuid.UUID(str(user_id))
            except ValueError:
                return None
            user = self.db.session.get(UserAccount, user_uuid)
            if not user:
                return None
            return User(user.id, is_admin=user.is_admin)
        
        # Setup Flask-Admin
        # Flask-Admin versions bundled in the image don't accept template_mode; stick with defaults.
        self.admin = Admin(self.app, name='PlanExe Admin', index_view=MyAdminIndexView())
        
        # Add database tables to admin panel
        self.admin.add_view(PlanItemView(model=PlanItem, session=self.db.session, name="Task"))
        self.admin.add_view(AdminOnlyModelView(model=EventItem, session=self.db.session, name="Event"))
        self.admin.add_view(WorkerItemView(model=WorkerItem, session=self.db.session, name="Worker"))
        self.admin.add_view(NonceItemView(model=NonceItem, session=self.db.session, name="Nonce"))
        self.admin.add_view(UserAccountView(model=UserAccount, session=self.db.session, name="User"))
        self.admin.add_view(AdminOnlyModelView(model=UserProvider, session=self.db.session, name="User Provider"))
        self.admin.add_view(AdminOnlyModelView(model=UserApiKey, session=self.db.session, name="User API Key"))
        self.admin.add_view(AdminOnlyModelView(model=CreditHistory, session=self.db.session, name="Credit History"))
        self.admin.add_view(AdminOnlyModelView(model=PaymentRecord, session=self.db.session, name="Payments"))
        self.admin.add_view(TokenMetricsView(model=TokenMetrics, session=self.db.session, name="Token Metrics"))

        self._setup_routes()

        self._track_flask_app_started()

    def _track_flask_app_started(self):
        logger.info(f"MyFlaskApp._track_flask_app_started. Starting...")
        
        # Determine if this is the main process or reloader process
        is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        is_debug_mode = self.app.debug if hasattr(self, 'app') else True
        
        event_context = {
            "pid": str(os.getpid()),
            "parent_pid": str(os.getppid()),
            "is_reloader_process": is_reloader,
            "is_debug_mode": is_debug_mode,
            "WERKZEUG_RUN_MAIN": os.environ.get('WERKZEUG_RUN_MAIN', 'not_set'),
            "python_executable": sys.executable,
            "command_line": ' '.join(sys.argv),
            "FLASK_ENV": os.environ.get('FLASK_ENV', 'not_set'),
            "FLASK_DEBUG": os.environ.get('FLASK_DEBUG', 'not_set')
        }
            
        with self.app.app_context():
            event = _new_model(
                EventItem,
                event_type=EventType.GENERIC_EVENT,
                message="Flask app started",
                context=event_context,
            )
            self.db.session.add(event)
            self.db.session.commit()
            
        logger.info(f"MyFlaskApp._track_flask_app_started. Logged {event_context!r}")

    def _start_check(self):
        # When the Flask app launches in debug mode it runs __init__ twice, so that the app can hot reload.
        # However there is this inconsistency.
        # 1st time, the os.environ is the original environment of the shell.
        # 2nd time, the os.environ is the original environment of the shell + the .env content.
        # If it was the same in both cases, it would be easier to reason about the environment variables.
        # On following hot reloads, the os.environ continues to be the original environment of the shell + the .env content.
        logger.info(f"MyFlaskApp._start_check. environment variables: {os.environ}")

        issue_count = 0
        if not self.path_to_python.exists():
            logger.error(f"The python executable does not exist at this point. However the python executable should exist: {self.path_to_python!r}")
            issue_count += 1
        if not self.planexe_project_root.exists():
            logger.error(f"The planexe_project_root does not exist at this point. However the planexe_project_root should exist: {self.planexe_project_root!r}")
            issue_count += 1
        if issue_count > 0:
            raise Exception(f"There are {issue_count} issues with the python executable and project root directory")

    def _fetch_worker_plan_llm_info(self) -> Tuple[Optional[dict], Optional[str]]:
        """
        Fetch LLM configuration info from the worker_plan service.
        Returns a tuple of (payload, error_message).
        """
        url = f"{self.worker_plan_url}/llm-info"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload, None
        except URLError as exc:
            return None, f"Failed to reach worker_plan at {url}: {exc.reason}"
        except Exception as exc:
            return None, f"Error fetching worker_plan llm-info: {exc}"

    @staticmethod
    def _looks_like_production_url(url: str) -> bool:
        """Return True when *url* looks like a real production deployment.

        Plain ``http://localhost`` / ``http://127.0.0.1`` URLs are treated as
        development so that local Docker users don't need to set a dedicated
        SECRET_KEY or deal with ``SESSION_COOKIE_SECURE`` over plain HTTP.
        """
        if not url:
            return False
        parsed = urlparse(url.lower())
        if parsed.scheme == "https":
            return True
        # http:// to localhost / loopback is clearly dev
        if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return False
        # Any other host over http is still likely a real deployment
        return True

    def _register_oauth_providers(self) -> None:
        providers = {
            "google": {
                "client_id": os.environ.get("PLANEXE_OAUTH_GOOGLE_CLIENT_ID"),
                "client_secret": os.environ.get("PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET"),
                "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
                "client_kwargs": {"scope": "openid email profile"},
            },
            "github": {
                "client_id": os.environ.get("PLANEXE_OAUTH_GITHUB_CLIENT_ID"),
                "client_secret": os.environ.get("PLANEXE_OAUTH_GITHUB_CLIENT_SECRET"),
                "authorize_url": "https://github.com/login/oauth/authorize",
                "access_token_url": "https://github.com/login/oauth/access_token",
                "api_base_url": "https://api.github.com/",
                "client_kwargs": {"scope": "read:user user:email"},
            },
            "discord": {
                "client_id": os.environ.get("PLANEXE_OAUTH_DISCORD_CLIENT_ID"),
                "client_secret": os.environ.get("PLANEXE_OAUTH_DISCORD_CLIENT_SECRET"),
                "authorize_url": "https://discord.com/oauth2/authorize",
                "access_token_url": "https://discord.com/api/oauth2/token",
                "api_base_url": "https://discord.com/api/",
                "client_kwargs": {"scope": "identify email"},
            },
        }

        self.oauth_providers: list[str] = []
        for name, config in providers.items():
            if not config["client_id"] or not config["client_secret"]:
                continue
            reg_config = dict(config)
            if name == "google":
                reg_config["redirect_uri"] = self._oauth_redirect_url("google")
            self.oauth.register(name=name, **reg_config)
            self.oauth_providers.append(name)

        if not self.oauth_providers:
            logger.warning("No OAuth providers configured. Set PLANEXE_OAUTH_* env vars to enable OAuth login.")

    def _determine_open_access(self) -> bool:
        """Check if the app should run in open access mode (no login required).

        Open access is enabled when no OAuth providers are configured and
        ``PLANEXE_AUTH_REQUIRED`` is not explicitly set to ``true``.
        This lets Docker-on-localhost users create plans immediately after
        ``docker compose up`` without setting up Google Console or any other
        OAuth provider.

        To force login even without OAuth (e.g. admin-only access via
        username/password), set ``PLANEXE_AUTH_REQUIRED=true``.
        """
        env_val = os.environ.get("PLANEXE_AUTH_REQUIRED", "").strip().lower()
        if env_val in ("1", "true", "yes"):
            if not self.oauth_providers:
                raise ValueError(
                    "PLANEXE_AUTH_REQUIRED=true but no OAuth providers are configured. "
                    "Either set PLANEXE_OAUTH_* env vars (Google / GitHub / Discord) "
                    "or remove PLANEXE_AUTH_REQUIRED to use open access mode."
                )
            logger.info(
                "Authentication required (PLANEXE_AUTH_REQUIRED=true). "
                "%d OAuth provider(s) configured.",
                len(self.oauth_providers),
            )
            return False
        if env_val in ("0", "false", "no"):
            logger.info("Open access mode forced via PLANEXE_AUTH_REQUIRED=false.")
            return True
        # Auto-detect: open access when no OAuth providers are configured.
        if not self.oauth_providers:
            logger.info(
                "Open access mode: no OAuth providers configured, login not required. "
                "Set PLANEXE_AUTH_REQUIRED=true to enforce login."
            )
            return True
        logger.info(
            "Authentication required. %d OAuth provider(s) configured.",
            len(self.oauth_providers),
        )
        return False

    def _oauth_redirect_url(self, provider: str) -> str:
        return f"{self.public_base_url}/auth/{provider}/callback"

    def _auth_provider_label(self, provider: Optional[str]) -> str:
        if not provider:
            return "Unknown"
        provider_key = str(provider).strip().lower()
        if not provider_key:
            return "Unknown"
        return AUTH_PROVIDER_LABELS.get(provider_key, provider_key.replace("_", " ").title())

    def _get_user_from_provider(self, provider: str, token: dict[str, Any]) -> dict[str, Any]:
        if provider == "google":
            client = self.oauth.create_client(provider)
            userinfo = client.parse_id_token(token)
            if userinfo:
                return userinfo
            return client.get("userinfo").json()
        if provider == "github":
            client = self.oauth.create_client(provider)
            profile = client.get("user").json()
            emails = client.get("user/emails").json()
            primary_email = None
            for item in emails:
                if item.get("primary"):
                    primary_email = item.get("email")
                    break
            if primary_email and not profile.get("email"):
                profile["email"] = primary_email
            return profile
        if provider == "discord":
            client = self.oauth.create_client(provider)
            return client.get("users/@me").json()
        raise ValueError(f"Unsupported OAuth provider: {provider}")

    def _avatar_url_from_profile(self, provider: str, profile: dict[str, Any]) -> Optional[str]:
        """Return a normalized avatar URL for OAuth profiles.

        Discord workaround: the OAuth `/users/@me` response returns `avatar` as
        a hash (not a full URL). Convert that hash to a CDN URL so templates can
        render a real image instead of a broken `<img src=\"<hash>\">`.
        """
        picture = profile.get("picture")
        if isinstance(picture, str) and picture:
            return picture

        avatar_url = profile.get("avatar_url")
        if isinstance(avatar_url, str) and avatar_url:
            return avatar_url

        avatar = profile.get("avatar")
        if provider == "discord":
            user_id = profile.get("id")
            if isinstance(avatar, str) and avatar and isinstance(user_id, str) and user_id:
                extension = "gif" if avatar.startswith("a_") else "png"
                return f"https://cdn.discordapp.com/avatars/{user_id}/{avatar}.{extension}"
            return None

        if isinstance(avatar, str) and avatar:
            return avatar
        return None

    def _upsert_user_from_oauth(self, provider: str, profile: dict[str, Any]) -> UserAccount:
        # Validate required fields
        provider_user_id = str(profile.get("sub") or profile.get("id") or "")
        if not provider_user_id:
            raise ValueError(f"OAuth profile from {provider} missing user identifier (sub/id).")

        # Email is optional for some providers - log warning if missing
        email = profile.get("email")
        if not email:
            logger.warning(f"OAuth profile from {provider} missing email for user {provider_user_id}")

        existing_provider = UserProvider.query.filter_by(
            provider=provider,
            provider_user_id=provider_user_id,
        ).first()
        now = datetime.now(UTC)

        if existing_provider:
            user = self.db.session.get(UserAccount, existing_provider.user_id)
            existing_provider.raw_profile = profile
            existing_provider.email = profile.get("email")
            existing_provider.last_login_at = now
            if user:
                user.last_login_at = now
                self._update_user_from_profile(user, provider, profile)
                self.db.session.commit()
                return user

        user = _new_model(
            UserAccount,
            email=profile.get("email"),
            name=profile.get("name") or profile.get("username") or profile.get("login"),
            given_name=profile.get("given_name"),
            family_name=profile.get("family_name"),
            locale=profile.get("locale"),
            avatar_url=self._avatar_url_from_profile(provider, profile),
            last_login_at=now,
        )
        self.db.session.add(user)
        self.db.session.commit()

        provider_row = _new_model(
            UserProvider,
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            email=profile.get("email"),
            raw_profile=profile,
            last_login_at=now,
        )
        self.db.session.add(provider_row)
        self.db.session.commit()
        return user

    def _update_user_from_profile(self, user: UserAccount, provider: str, profile: dict[str, Any]) -> None:
        user.email = profile.get("email") or user.email
        user.name = profile.get("name") or profile.get("username") or profile.get("login") or user.name
        user.given_name = profile.get("given_name") or user.given_name
        user.family_name = profile.get("family_name") or user.family_name
        user.locale = profile.get("locale") or user.locale
        user.avatar_url = self._avatar_url_from_profile(provider, profile) or user.avatar_url

    def _get_or_create_api_key(self, user: UserAccount, name: Optional[str] = None) -> str:
        api_key_secret = os.environ.get("PLANEXE_API_KEY_SECRET", "dev-api-key-secret")
        if api_key_secret == "dev-api-key-secret":
            logger.warning("PLANEXE_API_KEY_SECRET not set. Using dev secret for API key hashing.")

        active_count = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).count()
        if active_count >= 10:
            return ""

        raw_key = f"pex_{secrets.token_urlsafe(24)}"
        key_hash = hashlib.sha256(f"{api_key_secret}:{raw_key}".encode("utf-8")).hexdigest()
        key_prefix = raw_key[:10]
        sanitized_name = (name or "").strip()[:128] or None
        api_key = _new_model(
            UserApiKey,
            user_id=user.id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=sanitized_name,
            key_plaintext=raw_key if not self._api_key_show_once else None,
        )
        self.db.session.add(api_key)
        self.db.session.commit()
        return raw_key

    def _apply_credit_delta(self, user: UserAccount, delta: Decimal, reason: str, source: str, external_id: Optional[str] = None) -> None:
        current_balance = self._to_credit_decimal(user.credits_balance)
        next_balance = current_balance + self._to_credit_decimal(delta)
        user.credits_balance = max(Decimal("0"), next_balance).quantize(CREDIT_SCALE)
        ledger = _new_model(
            CreditHistory,
            user_id=user.id,
            delta=self._to_credit_decimal(delta),
            reason=reason,
            source=source,
            external_id=external_id,
        )
        self.db.session.add(ledger)
        self.db.session.commit()

    def _apply_payment_credits(
        self,
        user_id: str,
        provider: str,
        provider_payment_id: str,
        credits: Decimal,
        amount: int,
        currency: str,
        raw_payload: dict[str, Any],
    ) -> str:
        try:
            user_uuid = uuid.UUID(str(user_id))
        except ValueError:
            logger.error("Invalid user_id in payment payload: %s", user_id)
            return "invalid_user_id"
        with self.app.app_context():
            user = self.db.session.get(UserAccount, user_uuid)
            if not user:
                logger.error("Payment user not found: %s", user_id)
                return "user_not_found"
            existing = PaymentRecord.query.filter_by(
                provider=provider,
                provider_payment_id=provider_payment_id,
            ).first()
            if existing:
                return "duplicate_payment"
            credit_amount = self._to_credit_decimal(credits)
            payment = _new_model(
                PaymentRecord,
                user_id=user.id,
                provider=provider,
                provider_payment_id=provider_payment_id,
                credits=credit_amount,
                amount=amount,
                currency=currency,
                status="completed",
                raw_payload=raw_payload,
            )
            self.db.session.add(payment)
            self.db.session.commit()
            self._apply_credit_delta(
                user,
                delta=credit_amount,
                reason="credits_purchased",
                source=provider,
                external_id=provider_payment_id,
            )
            return "applied"

    def _record_event(self, event_type: EventType, message: str, context: Optional[dict[str, Any]] = None) -> None:
        """Best-effort event logging for operational visibility."""
        try:
            event = _new_model(
                EventItem,
                event_type=event_type,
                message=message,
                context=context,
            )
            self.db.session.add(event)
            self.db.session.commit()
        except Exception as exc:
            logger.error("Failed to persist event item. message=%s error=%s", message, exc, exc_info=True)
            self.db.session.rollback()

    def _finalize_stripe_checkout_session(self, user: UserAccount, checkout_session_id: str) -> str:
        """Best-effort Stripe session reconciliation from success redirect."""
        stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
        if not stripe_secret:
            self._record_event(
                EventType.GENERIC_ERROR,
                "Stripe success return ignored (Stripe not configured)",
                context={"user_id": str(user.id), "checkout_session_id": checkout_session_id},
            )
            return "stripe_not_configured"

        stripe.api_key = stripe_secret
        try:
            session_obj = stripe.checkout.Session.retrieve(checkout_session_id)
        except Exception as exc:
            self._record_event(
                EventType.GENERIC_ERROR,
                "Stripe session retrieval failed",
                context={"user_id": str(user.id), "checkout_session_id": checkout_session_id, "error": str(exc)},
            )
            return "session_retrieve_failed"

        metadata = session_obj.get("metadata") or {}
        metadata_user_id = str(metadata.get("user_id") or "")
        metadata_credits = self._to_credit_decimal(metadata.get("credits", "0") or 0)
        payment_status = session_obj.get("payment_status") or ""

        if metadata_user_id != str(user.id):
            self._record_event(
                EventType.GENERIC_ERROR,
                "Stripe session user mismatch",
                context={
                    "user_id": str(user.id),
                    "metadata_user_id": metadata_user_id,
                    "checkout_session_id": checkout_session_id,
                },
            )
            return "user_mismatch"

        if payment_status != "paid":
            self._record_event(
                EventType.GENERIC_ERROR,
                "Stripe session not paid",
                context={
                    "user_id": str(user.id),
                    "checkout_session_id": checkout_session_id,
                    "payment_status": payment_status,
                },
            )
            return "not_paid"

        if metadata_credits <= 0:
            self._record_event(
                EventType.GENERIC_ERROR,
                "Stripe session missing credits metadata",
                context={
                    "user_id": str(user.id),
                    "checkout_session_id": checkout_session_id,
                    "metadata": metadata,
                },
            )
            return "missing_credits"

        status = self._apply_payment_credits(
            user_id=str(user.id),
            provider="stripe",
            provider_payment_id=session_obj.get("id", ""),
            credits=metadata_credits,
            amount=session_obj.get("amount_total") or 0,
            currency=session_obj.get("currency") or "usd",
            raw_payload=session_obj,
        )
        self._record_event(
            EventType.GENERIC_EVENT if status in ("applied", "duplicate_payment") else EventType.GENERIC_ERROR,
            "Stripe success return processed",
            context={
                "user_id": str(user.id),
                "checkout_session_id": checkout_session_id,
                "payment_status": payment_status,
                "credits": str(metadata_credits),
                "amount_minor": session_obj.get("amount_total") or 0,
                "amount_major": (session_obj.get("amount_total") or 0) / 100.0,
                "currency": session_obj.get("currency") or "usd",
                "status": status,
            },
        )
        return status

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _extract_exception_type(message: Any) -> Optional[str]:
        text = MyFlaskApp._clean_text(message)
        if text is None:
            return None
        match = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*Error)\b", text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_nested_value(payload: Any, key_names: set[str]) -> Any:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if str(key).strip().lower() in key_names and value not in (None, "", [], {}):
                    return value
            for value in payload.values():
                found = MyFlaskApp._extract_nested_value(value, key_names)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = MyFlaskApp._extract_nested_value(item, key_names)
                if found not in (None, "", [], {}):
                    return found
        return None

    @staticmethod
    def _extract_provider_model_from_activity_key(model_key: Any) -> tuple[Optional[str], Optional[str]]:
        key = MyFlaskApp._clean_text(model_key)
        if key is None:
            return None, None
        if ":" not in key:
            return None, key
        provider, model = key.split(":", 1)
        provider_clean = provider.strip() or None
        model_clean = model.strip() or None
        return provider_clean, model_clean

    @staticmethod
    def _to_credit_decimal(value: Any) -> Decimal:
        try:
            return Decimal(str(value or 0)).quantize(CREDIT_SCALE)
        except Exception:
            return Decimal("0").quantize(CREDIT_SCALE)

    @staticmethod
    def _format_credit_display(value: Any) -> str:
        amount = MyFlaskApp._to_credit_decimal(value)
        # Round up to 3 decimal places so small charges are visible.
        quantized = amount.quantize(Decimal("0.001"), rounding=ROUND_CEILING)
        return format(quantized, ".3f")

    @staticmethod
    def _format_relative_time(value: Any) -> str:
        if not isinstance(value, datetime):
            return "-"
        now = datetime.now(UTC)
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
        seconds = max(0, int((now - dt).total_seconds()))
        if seconds >= 365 * 24 * 3600:
            n = seconds // (365 * 24 * 3600)
            return f"{n} year" if n == 1 else f"{n} years"
        if seconds >= 30 * 24 * 3600:
            n = seconds // (30 * 24 * 3600)
            return f"{n} month" if n == 1 else f"{n} months"
        if seconds >= 24 * 3600:
            n = seconds // (24 * 3600)
            return f"{n} day" if n == 1 else f"{n} days"
        if seconds >= 3600:
            n = seconds // 3600
            return f"{n} hour" if n == 1 else f"{n} hours"
        if seconds >= 60:
            n = seconds // 60
            return f"{n} min" if n == 1 else f"{n} mins"
        n = seconds
        return f"{n} sec" if n == 1 else f"{n} secs"

    def _load_prompt_preview_safe(self, task_id: Any, max_chars: int = 240) -> str:
        """Load a prompt preview for one task, tolerating corrupted UTF-8 rows."""
        try:
            preview = (
                self.db.session.query(func.substr(PlanItem.prompt, 1, max_chars))
                .filter(PlanItem.id == task_id)
                .scalar()
            )
            text = (preview or "").strip()
            if text:
                return text
        except DataError:
            self.db.session.rollback()
            logger.warning(
                "Detected invalid UTF-8 in task_item.prompt for task_id=%s; using placeholder preview.",
                task_id,
                exc_info=True,
            )
            return "[Prompt unavailable due to encoding issue]"
        except Exception:
            self.db.session.rollback()
            logger.debug("Unable to load prompt preview for task_id=%s", task_id, exc_info=True)

        return "[Prompt unavailable]"

    def _admin_user_ids(self) -> list[str]:
        """Return all possible user_id values for the admin user.

        Old plans used the admin username (e.g. ``"admin"``); new plans use
        the deterministic UUID.  Returning both lets queries find all admin
        plans regardless of when they were created.
        """
        ids = [self.admin_username]
        admin_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"planexe-admin-pref:{self.admin_username}"))
        if admin_uuid not in ids:
            ids.append(admin_uuid)
        return ids

    def _get_current_user_account(self) -> Optional[UserAccount]:
        if not current_user.is_authenticated:
            return None
        try:
            user_uuid = uuid.UUID(str(current_user.id))
        except ValueError:
            if current_user.is_admin and str(current_user.id) == self.admin_username:
                admin_pref_id = uuid.uuid5(uuid.NAMESPACE_URL, f"planexe-admin-pref:{self.admin_username}")
                user = self.db.session.get(UserAccount, admin_pref_id)
                if user is None:
                    try:
                        user = _new_model(
                            UserAccount,
                            id=admin_pref_id,
                            is_admin=True,
                            name="Admin",
                        )
                        self.db.session.add(user)
                        self.db.session.commit()
                    except Exception:
                        self.db.session.rollback()
                        logger.exception("Failed to create admin UserAccount %s", admin_pref_id)
                        # Another worker may have created it; try fetching again.
                        user = self.db.session.get(UserAccount, admin_pref_id)
                return user
            logger.warning(
                "_get_current_user_account: admin mismatch current_user.id=%r admin_username=%r is_admin=%s",
                str(current_user.id), self.admin_username, current_user.is_admin,
            )
            return None
        return self.db.session.get(UserAccount, user_uuid)

    @staticmethod
    def _normalize_plan_view_mode(value: Any) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"info", "view"}:
            return mode
        return "view"

    @staticmethod
    def _coerce_json_dict(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {str(key): item for key, item in value.items()}

    def _get_plan_view_mode_preference(self) -> str:
        user = self._get_current_user_account()
        if user is None:
            return "view"
        config = self._coerce_json_dict(user.frontend_multi_user_config)
        plan_page = self._coerce_json_dict(config.get("plan_page"))
        return self._normalize_plan_view_mode(plan_page.get("selected_segment"))

    def _set_plan_view_mode_preference(self, mode: str) -> None:
        user = self._get_current_user_account()
        if user is None:
            return
        normalized_mode = self._normalize_plan_view_mode(mode)
        # JSON columns are not mutation-tracked by default; build fresh dicts so
        # SQLAlchemy sees a new value and persists the update reliably.
        existing_config = self._coerce_json_dict(user.frontend_multi_user_config)
        config = dict(existing_config)
        existing_plan_page = self._coerce_json_dict(config.get("plan_page"))
        plan_page = dict(existing_plan_page)
        plan_page["selected_segment"] = normalized_mode
        config["plan_page"] = plan_page
        user.frontend_multi_user_config = config
        self.db.session.commit()

    def _read_activity_overview_from_run_zip(self, run_zip_snapshot: Optional[bytes]) -> Optional[dict[str, Any]]:
        if not run_zip_snapshot:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(run_zip_snapshot), "r") as archive:
                activity_name = ExtraFilenameEnum.ACTIVITY_OVERVIEW_JSON.value
                for member_name in archive.namelist():
                    if member_name.endswith(activity_name):
                        payload = json.loads(archive.read(member_name).decode("utf-8"))
                        if isinstance(payload, dict):
                            return payload
                        return None
        except Exception as exc:
            logger.warning("Unable to parse run_zip_snapshot activity overview: %s", exc)
        return None

    def _read_inference_cost_from_run_zip(self, run_zip_snapshot: Optional[bytes]) -> Optional[float]:
        payload = self._read_activity_overview_from_run_zip(run_zip_snapshot)
        if not isinstance(payload, dict):
            return None
        return self._safe_float(payload.get("total_cost"))

    def _read_activity_overview_from_task(self, task: PlanItem) -> Optional[dict[str, Any]]:
        raw_payload = getattr(task, "run_activity_overview_json", None)
        if isinstance(raw_payload, dict):
            return raw_payload
        if isinstance(raw_payload, str):
            try:
                decoded_payload = json.loads(raw_payload)
                if isinstance(decoded_payload, dict):
                    return decoded_payload
            except Exception as exc:
                logger.warning("Unable to parse task.run_activity_overview_json for %s: %s", task.id, exc)
        return self._read_activity_overview_from_run_zip(task.run_zip_snapshot)

    def _read_inference_cost_from_task(self, task: PlanItem) -> Optional[float]:
        payload = self._read_activity_overview_from_task(task)
        if not isinstance(payload, dict):
            return None
        return self._safe_float(payload.get("total_cost"))

    def _sanitize_legacy_run_zip_for_download(self, run_zip_snapshot: bytes) -> Optional[io.BytesIO]:
        """
        Legacy snapshots may contain track_activity.jsonl.
        Remove it before user download while newer layout versions can be served directly.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                extract_dir = Path(tmp_dir) / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(io.BytesIO(run_zip_snapshot), "r") as in_zip:
                    in_zip.extractall(extract_dir)

                for sensitive_file in extract_dir.rglob(ExtraFilenameEnum.TRACK_ACTIVITY_JSONL.value):
                    try:
                        sensitive_file.unlink()
                    except OSError:
                        logger.warning("Unable to remove sensitive file from zip staging: %s", sensitive_file)

                sanitized_buffer = io.BytesIO()
                with zipfile.ZipFile(sanitized_buffer, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
                    for file_path in extract_dir.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(extract_dir)
                            out_zip.write(file_path, arcname=str(arcname))
                sanitized_buffer.seek(0)
                return sanitized_buffer
        except zipfile.BadZipFile:
            return None

    def _find_latest_task_event(self, task_id: str, event_type: EventType, max_events_to_scan: int = 2000) -> Optional[EventItem]:
        events = (
            EventItem.query
            .filter_by(event_type=event_type)
            .order_by(EventItem.timestamp.desc(), EventItem.id.desc())
            .limit(max_events_to_scan)
            .all()
        )
        for event in events:
            context = event.context if isinstance(event.context, dict) else {}
            if str(context.get("task_id") or "").strip() == task_id:
                return event
        return None

    def _build_plan_failure_trace(self, task: PlanItem) -> dict[str, Any]:
        task_id = str(task.id)
        token_metrics_rows = (
            TokenMetrics.query
            .filter_by(task_id=task_id)
            .order_by(TokenMetrics.timestamp.asc(), TokenMetrics.id.asc())
            .all()
        )
        failed_rows = [row for row in token_metrics_rows if (row.success is False) or bool(row.error_message)]
        latest_failed_row = failed_rows[-1] if failed_rows else None

        latest_failed_event = self._find_latest_task_event(task_id, EventType.TASK_FAILED)
        latest_processing_event = self._find_latest_task_event(task_id, EventType.TASK_PROCESSING)
        latest_completed_event = self._find_latest_task_event(task_id, EventType.TASK_COMPLETED)
        latest_terminal_event = latest_failed_event or latest_completed_event

        failed_event_context = latest_failed_event.context if latest_failed_event and isinstance(latest_failed_event.context, dict) else {}
        processing_event_context = latest_processing_event.context if latest_processing_event and isinstance(latest_processing_event.context, dict) else {}
        terminal_event_context = latest_terminal_event.context if latest_terminal_event and isinstance(latest_terminal_event.context, dict) else {}
        latest_failed_usage = latest_failed_row.raw_usage_data if latest_failed_row and isinstance(latest_failed_row.raw_usage_data, dict) else {}
        task_parameters = task.parameters if isinstance(task.parameters, dict) else {}
        activity_overview = self._read_activity_overview_from_task(task)

        nested_sources: list[Any] = [
            failed_event_context,
            processing_event_context,
            terminal_event_context,
            latest_failed_usage,
            task_parameters,
            activity_overview,
        ]

        stage = None
        for source in nested_sources:
            stage = self._clean_text(
                self._extract_nested_value(
                    source,
                    {
                        "stage",
                        "failed_stage",
                        "failure_stage",
                        "error_stage",
                        "pipeline_stage",
                    },
                )
            )
            if stage:
                break

        explicit_error_type = None
        for source in nested_sources:
            explicit_error_type = self._clean_text(
                self._extract_nested_value(source, {"error_type", "exception_type", "exception_class"})
            )
            if explicit_error_type:
                break

        error_message = self._clean_text(latest_failed_row.error_message if latest_failed_row else None)
        if error_message is None:
            for source in nested_sources:
                error_message = self._clean_text(
                    self._extract_nested_value(
                        source,
                        {
                            "error_message",
                            "message",
                            "machai_error_message",
                            "exception",
                            "error",
                            "detail",
                        },
                    )
                )
                if error_message:
                    break

        error_type = explicit_error_type or self._extract_exception_type(error_message)

        retry_count = None
        for source in nested_sources:
            retry_count = self._safe_int(
                self._extract_nested_value(
                    source,
                    {
                        "retry_count",
                        "retries",
                        "retry_attempts",
                        "attempt_count",
                        "attempts",
                    },
                )
            )
            if retry_count is not None:
                break

        fallback_indicator = None
        for source in nested_sources:
            fallback_value = self._extract_nested_value(
                source,
                {
                    "fallback",
                    "fallback_used",
                    "used_fallback",
                    "fallback_model",
                    "fallback_provider",
                },
            )
            fallback_text = self._clean_text(fallback_value)
            if fallback_text is not None:
                fallback_indicator = fallback_text
                break
            if isinstance(fallback_value, bool):
                fallback_indicator = "true" if fallback_value else "false"
                break

        routes_map: dict[tuple[Optional[str], Optional[str]], int] = {}
        for row in token_metrics_rows:
            provider = self._clean_text(row.upstream_provider)
            model = self._clean_text(row.upstream_model) or self._clean_text(row.llm_model)
            if provider is None and model is None:
                continue
            key = (provider, model)
            routes_map[key] = routes_map.get(key, 0) + 1
        if not routes_map and isinstance(activity_overview, dict):
            models_payload = activity_overview.get("models")
            if isinstance(models_payload, dict):
                for model_key, model_stats in models_payload.items():
                    provider, model = self._extract_provider_model_from_activity_key(model_key)
                    if provider is None and model is None:
                        continue
                    calls = self._safe_int(model_stats.get("calls")) if isinstance(model_stats, dict) else None
                    routes_map[(provider, model)] = calls if calls and calls > 0 else 0

        routes = [
            {
                "provider": provider,
                "model": model,
                "calls": calls if calls > 0 else None,
            }
            for (provider, model), calls in routes_map.items()
        ]
        routes.sort(key=lambda row: (-(row["calls"] or 0), row["provider"] or "", row["model"] or ""))

        provider_switch_indicator = None
        for source in nested_sources:
            provider_switch_indicator = self._clean_text(
                self._extract_nested_value(
                    source,
                    {
                        "provider_switch",
                        "provider_switched",
                        "switched_provider",
                        "provider_switch_count",
                    },
                )
            )
            if provider_switch_indicator:
                break

        unique_routes_count = len(routes)
        inferred_provider_switch_count = unique_routes_count - 1 if unique_routes_count > 1 else 0

        pending_to_processing_seconds = self._safe_float(processing_event_context.get("duration_between_pending_and_processing"))
        processing_to_terminal_seconds = self._safe_float(terminal_event_context.get("duration_between_processing_and_completion"))

        llm_duration_values = [
            self._safe_float(row.duration_seconds)
            for row in token_metrics_rows
            if row.duration_seconds is not None
        ]
        llm_duration_values = [value for value in llm_duration_values if value is not None]
        llm_duration_total_seconds = sum(llm_duration_values) if llm_duration_values else None

        partial_output = None
        for source in [latest_failed_usage, failed_event_context, task_parameters]:
            partial_output = self._extract_nested_value(
                source,
                {
                    "partial_output",
                    "partial_response",
                    "partial_result",
                    "response_excerpt",
                    "partial_text",
                    "raw_partial_output",
                },
            )
            if partial_output not in (None, "", [], {}):
                break

        if isinstance(partial_output, (dict, list)):
            try:
                partial_output = json.dumps(partial_output, indent=2, sort_keys=True)
            except Exception:
                partial_output = str(partial_output)
        else:
            partial_output = self._clean_text(partial_output)

        # Build lookup: api_key_id → key name, and resolve plan owner name as fallback.
        api_key_ids_in_trace = {row.api_key_id for row in token_metrics_rows if row.api_key_id}
        key_name_lookup: dict[str, str] = {}
        if api_key_ids_in_trace:
            for key_row in UserApiKey.query.filter(UserApiKey.id.in_(list(api_key_ids_in_trace))).all():
                key_name_lookup[str(key_row.id)] = key_row.name or f"{key_row.key_prefix}..."
        # Fallback for rows without api_key_id (frontend-created plans): show owner name.
        owner_display_name: Optional[str] = None
        try:
            owner_uuid = uuid.UUID(str(task.user_id))
            owner = self.db.session.get(UserAccount, owner_uuid)
            if owner:
                owner_display_name = owner.name or owner.given_name or owner.email or "Account"
                if getattr(owner, "is_admin", False):
                    owner_display_name = "Admin"
        except (ValueError, AttributeError):
            owner_display_name = str(task.user_id) if task.user_id else None

        execution_attempts = []
        for row in token_metrics_rows:
            route_provider = self._clean_text(row.upstream_provider)
            route_model = self._clean_text(row.upstream_model) or self._clean_text(row.llm_model)
            success_state = True if row.success is True else False if row.success is False else None
            invoked_by = key_name_lookup.get(row.api_key_id) if row.api_key_id else owner_display_name
            execution_attempts.append(
                {
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "provider": route_provider,
                    "model": route_model,
                    "duration_seconds": self._safe_float(row.duration_seconds),
                    "success": success_state,
                    "error_message": self._clean_text(row.error_message),
                    "invoked_by": invoked_by,
                }
            )

        inferred_retry_count = len(token_metrics_rows) - 1 if token_metrics_rows else None

        failure_trace: dict[str, Any] = {
            "task_id": task_id,
            "stage": stage,
            "error": {
                "type": error_type,
                "message": error_message,
            },
            "retries": {
                "count": retry_count,
                "count_inferred_from_attempts": inferred_retry_count,
                "attempt_count": len(token_metrics_rows) if token_metrics_rows else None,
                "failed_attempts": len(failed_rows) if token_metrics_rows else None,
            },
            "fallback": {
                "indicator": fallback_indicator,
            },
            "provider_switch": {
                "indicator": provider_switch_indicator,
                "inferred_count_from_routes": inferred_provider_switch_count if routes else None,
                "routes": routes,
            },
            "timing": {
                "pending_to_processing_seconds": pending_to_processing_seconds,
                "processing_to_terminal_seconds": processing_to_terminal_seconds,
                "llm_attempts_total_duration_seconds": llm_duration_total_seconds,
            },
            "partial_output": partial_output,
            "execution_trace": execution_attempts,
            "source_paths": {
                "events_table": "events.context.{task_id, duration_between_pending_and_processing, duration_between_processing_and_completion, machai_error_message}",
                "token_metrics_table": "token_metrics.{timestamp, upstream_provider, upstream_model, llm_model, duration_seconds, success, error_message, raw_usage_data}",
                "task_parameters": "taskitem.parameters",
                "activity_overview_json": "PlanItem.run_activity_overview_json (fallback: PlanItem.run_zip_snapshot -> */activity_overview.json)",
            },
        }
        failure_trace["has_data"] = any(
            [
                failure_trace["stage"] is not None,
                failure_trace["error"]["type"] is not None,
                failure_trace["error"]["message"] is not None,
                failure_trace["retries"]["count"] is not None,
                failure_trace["retries"]["count_inferred_from_attempts"] is not None,
                failure_trace["fallback"]["indicator"] is not None,
                failure_trace["provider_switch"]["indicator"] is not None,
                failure_trace["provider_switch"]["inferred_count_from_routes"] is not None,
                failure_trace["timing"]["pending_to_processing_seconds"] is not None,
                failure_trace["timing"]["processing_to_terminal_seconds"] is not None,
                failure_trace["timing"]["llm_attempts_total_duration_seconds"] is not None,
                bool(failure_trace["execution_trace"]),
                failure_trace["partial_output"] is not None,
            ]
        )

        return failure_trace

    def _build_plan_telemetry_cache_key(self, task: PlanItem, include_raw: bool) -> Optional[tuple[str, str, bool, bool]]:
        state = task.state if isinstance(task.state, PlanState) else None
        if include_raw or state not in (PlanState.completed, PlanState.failed):
            return None
        return (
            str(task.id),
            state.name,
            bool(task.has_run_zip_snapshot),
            bool(getattr(task, "run_activity_overview_json", None)),
        )

    def _build_plan_telemetry(
        self,
        task: PlanItem,
        include_raw: bool = False,
        expose_raw_usage_data: bool = False,
    ) -> dict[str, Any]:
        cache_key = self._build_plan_telemetry_cache_key(task, include_raw)
        if cache_key is not None:
            telemetry_cache = getattr(self, "_plan_telemetry_cache", None)
            if isinstance(telemetry_cache, dict):
                cached = telemetry_cache.get(cache_key)
                if isinstance(cached, dict):
                    return cached

        task_id = str(task.id)
        token_metrics_rows = (
            TokenMetrics.query
            .filter_by(task_id=task_id)
            .order_by(TokenMetrics.timestamp.asc(), TokenMetrics.id.asc())
            .all()
        )
        token_summary = TokenMetricsSummary(task_id=task_id, metrics=token_metrics_rows)
        activity_overview = self._read_activity_overview_from_task(task)

        has_prompt_tokens = any(row.input_tokens is not None for row in token_metrics_rows)
        has_completion_tokens = any(row.output_tokens is not None for row in token_metrics_rows)
        has_thinking_tokens = any(row.thinking_tokens is not None for row in token_metrics_rows)
        has_any_metric_tokens = has_prompt_tokens or has_completion_tokens or has_thinking_tokens

        activity_prompt_tokens = self._safe_int(activity_overview.get("total_input_tokens")) if isinstance(activity_overview, dict) else None
        activity_completion_tokens = self._safe_int(activity_overview.get("total_output_tokens")) if isinstance(activity_overview, dict) else None
        activity_thinking_tokens = self._safe_int(activity_overview.get("total_thinking_tokens")) if isinstance(activity_overview, dict) else None
        activity_total_tokens = self._safe_int(activity_overview.get("total_tokens")) if isinstance(activity_overview, dict) else None

        prompt_tokens = token_summary.total_input_tokens if has_prompt_tokens else activity_prompt_tokens
        completion_tokens = token_summary.total_output_tokens if has_completion_tokens else activity_completion_tokens
        thinking_tokens = token_summary.total_thinking_tokens if has_thinking_tokens else activity_thinking_tokens

        if has_any_metric_tokens:
            total_tokens = token_summary.total_tokens
        elif activity_total_tokens is not None:
            total_tokens = activity_total_tokens
        elif any(value is not None for value in [prompt_tokens, completion_tokens, thinking_tokens]):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) + (thinking_tokens or 0)
        else:
            total_tokens = None

        metric_cost_values = [
            self._safe_float(row.cost_usd)
            for row in token_metrics_rows
            if row.cost_usd is not None
        ]
        metric_cost_values = [value for value in metric_cost_values if value is not None]
        token_metrics_cost_usd = sum(metric_cost_values) if metric_cost_values else None
        activity_overview_cost_usd = self._safe_float(activity_overview.get("total_cost")) if isinstance(activity_overview, dict) else None

        provider_model_counts: dict[tuple[Optional[str], Optional[str]], int] = {}
        if token_metrics_rows:
            for row in token_metrics_rows:
                provider = self._clean_text(row.upstream_provider)
                model = self._clean_text(row.upstream_model) or self._clean_text(row.llm_model)
                if provider is None and model is None:
                    continue
                key = (provider, model)
                provider_model_counts[key] = provider_model_counts.get(key, 0) + 1
        elif isinstance(activity_overview, dict):
            models_payload = activity_overview.get("models")
            if isinstance(models_payload, dict):
                for model_key, model_stats in models_payload.items():
                    provider, model = self._extract_provider_model_from_activity_key(model_key)
                    if provider is None and model is None:
                        continue
                    calls = self._safe_int(model_stats.get("calls")) if isinstance(model_stats, dict) else None
                    provider_model_counts[(provider, model)] = calls if calls and calls > 0 else 0

        routes = [
            {
                "provider": provider,
                "model": model,
                "calls": calls if calls > 0 else None,
            }
            for (provider, model), calls in provider_model_counts.items()
        ]
        routes.sort(key=lambda row: (-(row["calls"] or 0), row["provider"] or "", row["model"] or ""))

        providers = sorted({row["provider"] for row in routes if row.get("provider")})
        models = sorted({row["model"] for row in routes if row.get("model")})

        if token_metrics_rows:
            total_calls = token_summary.total_calls
            successful_calls = token_summary.successful_calls
            failed_calls = token_summary.failed_calls
        else:
            total_calls = None
            successful_calls = None
            failed_calls = None
            if isinstance(activity_overview, dict):
                models_payload = activity_overview.get("models")
                if isinstance(models_payload, dict):
                    total_calls = 0
                    for model_stats in models_payload.values():
                        calls = self._safe_int(model_stats.get("calls")) if isinstance(model_stats, dict) else None
                        total_calls += calls or 0

        telemetry: dict[str, Any] = {
            "task_id": task_id,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens,
            },
            "cost": {
                "token_metrics_usd": token_metrics_cost_usd,
                "activity_overview_usd": activity_overview_cost_usd,
                "currency": "USD" if token_metrics_cost_usd is not None or activity_overview_cost_usd is not None else None,
            },
            "provider_model": {
                "providers": providers,
                "models": models,
                "routes": routes,
            },
            "calls": {
                "total": total_calls,
                "successful": successful_calls,
                "failed": failed_calls,
            },
            "source_paths": {
                "token_metrics_table": "token_metrics.{input_tokens, output_tokens, thinking_tokens, cost_usd, upstream_provider, upstream_model}",
                "activity_overview_json": "PlanItem.run_activity_overview_json (fallback: PlanItem.run_zip_snapshot -> */activity_overview.json)",
            },
            "source_availability": {
                "token_metrics_row_count": len(token_metrics_rows),
                "activity_overview_present": isinstance(activity_overview, dict),
            },
        }
        telemetry["has_data"] = any(
            [
                telemetry["usage"]["prompt_tokens"] is not None,
                telemetry["usage"]["completion_tokens"] is not None,
                telemetry["usage"]["thinking_tokens"] is not None,
                telemetry["usage"]["total_tokens"] is not None,
                telemetry["cost"]["token_metrics_usd"] is not None,
                telemetry["cost"]["activity_overview_usd"] is not None,
                bool(telemetry["provider_model"]["routes"]),
                telemetry["calls"]["total"] is not None,
            ]
        )

        if include_raw:
            raw_token_metrics_rows = []
            for row in token_metrics_rows:
                raw_token_metrics_rows.append(
                    {
                        "id": row.id,
                        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                        "task_id": row.task_id,
                        "user_id": row.user_id,
                        "llm_model": row.llm_model,
                        "upstream_provider": row.upstream_provider,
                        "upstream_model": row.upstream_model,
                        "input_tokens": row.input_tokens,
                        "output_tokens": row.output_tokens,
                        "thinking_tokens": row.thinking_tokens,
                        "cost_usd": row.cost_usd,
                        "duration_seconds": row.duration_seconds,
                        "success": row.success,
                        "error_message": row.error_message,
                        # Guardrail: provider-native usage payloads can be verbose
                        # and may include integration-specific fields. They are only
                        # exposed when the dedicated /plan/telemetry route opts in.
                        "raw_usage_data": row.raw_usage_data if expose_raw_usage_data else None,
                    }
                )
            telemetry["source_data"] = {
                "activity_overview": activity_overview,
                "token_metrics": raw_token_metrics_rows,
            }

        if cache_key is not None:
            telemetry_cache = getattr(self, "_plan_telemetry_cache", None)
            if not isinstance(telemetry_cache, dict):
                telemetry_cache = {}
                self._plan_telemetry_cache = telemetry_cache
            telemetry_cache[cache_key] = telemetry
            if len(telemetry_cache) > 512:
                telemetry_cache.pop(next(iter(telemetry_cache)))

        return telemetry

    def _build_reconciliation_report(self, max_tasks: int, tolerance_usd: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tasks = (
            PlanItem.query
            .order_by(PlanItem.timestamp_created.desc())
            .limit(max_tasks)
            .all()
        )
        task_ids = {str(task.id) for task in tasks}
        billing_events_by_task_id: dict[str, EventItem] = {}
        max_event_scan = max(1000, max_tasks * 20)
        billing_events = (
            EventItem.query
            .filter(EventItem.event_type.in_([EventType.TASK_COMPLETED, EventType.TASK_FAILED]))
            .order_by(EventItem.timestamp.desc())
            .limit(max_event_scan)
            .all()
        )

        for event in billing_events:
            context = event.context if isinstance(event.context, dict) else {}
            task_id = str(context.get("task_id") or "").strip()
            if not task_id or task_id not in task_ids or task_id in billing_events_by_task_id:
                continue
            if context.get("billing_usage_cost_usd") is None:
                continue
            billing_events_by_task_id[task_id] = event
            if len(billing_events_by_task_id) == len(task_ids):
                break

        rows: list[dict[str, Any]] = []
        mismatch_count = 0
        missing_count = 0

        for task in tasks:
            task_id = str(task.id)
            tracked_usage_cost_usd = self._read_inference_cost_from_task(task)

            billing_event = billing_events_by_task_id.get(task_id)
            context = billing_event.context if billing_event and isinstance(billing_event.context, dict) else {}
            billed_usage_cost_usd = self._safe_float(context.get("billing_usage_cost_usd"))

            delta_usd: Optional[float] = None
            status = "ok"
            if billed_usage_cost_usd is None or tracked_usage_cost_usd is None:
                status = "missing_data"
                missing_count += 1
            else:
                delta_usd = billed_usage_cost_usd - tracked_usage_cost_usd
                if abs(delta_usd) > tolerance_usd:
                    status = "mismatch"
                    mismatch_count += 1

            rows.append(
                {
                    "task_id": task_id,
                    "timestamp_created": (
                        task.timestamp_created.strftime("%Y-%m-%d %H:%M:%S")
                        if task.timestamp_created
                        else None
                    ),
                    "state": task.state.name if isinstance(task.state, PlanState) else str(task.state),
                    "billed_usage_cost_usd": billed_usage_cost_usd,
                    "tracked_usage_cost_usd": tracked_usage_cost_usd,
                    "delta_usd": delta_usd,
                    "status": status,
                    "has_report": bool(task.has_generated_report_html),
                    "has_run_zip": bool(task.has_run_zip_snapshot),
                    "billing_event_timestamp": billing_event.timestamp if billing_event else None,
                }
            )

        summary = {
            "total_tasks_checked": len(rows),
            "mismatch_count": mismatch_count,
            "missing_count": missing_count,
            "ok_count": len(rows) - mismatch_count - missing_count,
            "tolerance_usd": tolerance_usd,
            "scanned_billing_events": len(billing_events),
        }
        return rows, summary

    def _setup_routes(self):
        @self.app.before_request
        def _auto_login_open_access():
            """In open access mode, auto-login as admin so @login_required routes work."""
            if self.open_access and not current_user.is_authenticated and not session.get("open_access_logged_out"):
                session["auth_provider"] = "open_access"
                login_user(User(self.admin_username, is_admin=True))

        @self.app.after_request
        def _admin_full_width(response):
            """Make Flask-Admin pages use full viewport width on large screens."""
            try:
                if not request.path.startswith('/admin'):
                    return response
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type:
                    return response

                html = response.get_data(as_text=True)
                marker = "planexe-admin-full-width"
                if marker in html or "</head>" not in html:
                    return response

                css = """
<style id="planexe-admin-full-width">
  .container { width: 100% !important; max-width: none !important; }
  .row { margin-left: 0 !important; margin-right: 0 !important; }
  .col-md-10, .col-md-2, .col-lg-10, .col-lg-2, .col-sm-10, .col-sm-2 {
    width: auto !important;
    float: none !important;
  }
  .table-responsive { overflow-x: auto; }
  #planexe-admin-nav-dashboard {
    margin-top: 8px;
    margin-right: 8px;
    padding: 6px 12px !important;
    border: 1px solid #d0d7de;
    border-radius: 999px;
    background: #fff;
    color: #24292f !important;
    line-height: 1.2 !important;
  }
  #planexe-admin-nav-dashboard:hover {
    background: #f6f8fa !important;
    color: #24292f !important;
  }
  #planexe-admin-logout {
    margin-top: 8px;
    margin-right: 8px;
    padding: 6px 12px !important;
    border: 1px solid #d0d7de;
    border-radius: 999px;
    background: #fff;
    color: #24292f !important;
    line-height: 1.2 !important;
  }
  #planexe-admin-logout:hover {
    background: #f6f8fa !important;
    color: #24292f !important;
  }
</style>
""".strip()
                html = html.replace("</head>", css + "\n</head>", 1)
                if 'id="planexe-admin-nav-dashboard"' not in html and "</body>" in html:
                    dashboard_nav_script = """
<script id="planexe-admin-nav-dashboard">
(function () {
  var rightNav = document.querySelector('.navbar .navbar-nav.navbar-right');
  if (!rightNav) {
    var navbarCollapse = document.querySelector('.navbar .navbar-collapse');
    if (!navbarCollapse) return;
    rightNav = document.createElement('ul');
    rightNav.className = 'nav navbar-nav navbar-right';
    navbarCollapse.appendChild(rightNav);
  }
  var logoutLi = document.getElementById('planexe-admin-nav-logout-item');
  if (!logoutLi) {
    logoutLi = document.createElement('li');
    logoutLi.id = 'planexe-admin-nav-logout-item';
    var logoutA = document.createElement('a');
    logoutA.id = 'planexe-admin-logout';
    logoutA.href = '/logout';
    logoutA.textContent = 'Logout';
    logoutLi.appendChild(logoutA);
  }
  var dashboardLi = document.getElementById('planexe-admin-nav-dashboard-item');
  if (!dashboardLi) {
    dashboardLi = document.createElement('li');
    dashboardLi.id = 'planexe-admin-nav-dashboard-item';
    var dashboardA = document.createElement('a');
    dashboardA.id = 'planexe-admin-nav-dashboard';
    dashboardA.href = '/';
    dashboardA.textContent = 'Dashboard';
    dashboardLi.appendChild(dashboardA);
  }
  // Keep order stable as Logout, Dashboard. Dashboard is rightmost and acts
  // as the "back to home" toggle at the same top-right location as home's
  // "Admin Panel" button.
  rightNav.appendChild(logoutLi);
  rightNav.appendChild(dashboardLi);
})();
</script>
""".strip()
                    html = html.replace("</body>", dashboard_nav_script + "\n</body>", 1)
                response.set_data(html)
                response.headers.pop("Content-Length", None)
            except Exception:
                logger.debug("Failed to inject admin full-width CSS", exc_info=True)
            return response

        @self.app.context_processor
        def inject_current_user_name():
            """Inject current_user_name for header display (full name or None)."""
            ga_id = (os.environ.get("PLANEXE_GOOGLE_ANALYTICS") or "").strip()
            extra: dict[str, Any] = {
                "open_access": self.open_access,
                "google_analytics_id": ga_id or None,
            }
            if not current_user.is_authenticated:
                extra["current_user_name"] = None
                return extra
            if current_user.is_admin:
                extra["current_user_name"] = "Admin"
                return extra
            try:
                user_uuid = uuid.UUID(str(current_user.id))
            except ValueError:
                extra["current_user_name"] = None
                return extra
            user = self.db.session.get(UserAccount, user_uuid)
            if not user:
                extra["current_user_name"] = None
                return extra
            name = (user.name or user.given_name or user.email or "Account").strip() or "Account"
            extra["current_user_name"] = name
            return extra

        @self.app.route('/')
        def index():
            user = None
            recent_tasks: list[SimpleNamespace] = []
            total_tasks_count = 0
            is_admin = False
            nonce = None
            user_id = None
            can_create_plan = False
            example_prompts: list[str] = []
            credits_balance_display = "0"
            if current_user.is_authenticated:
                is_admin = current_user.is_admin
                try:
                    if is_admin:
                        admin_account = self._get_current_user_account()
                        user_id = str(admin_account.id) if admin_account else self.admin_username
                        user = SimpleNamespace(name="Admin", given_name=None)
                        credits_balance_display = "Full access"
                        can_create_plan = True
                    else:
                        user_uuid = uuid.UUID(str(current_user.id))
                        user = self.db.session.get(UserAccount, user_uuid)
                        if user:
                            user_id = str(user.id)
                            credits_balance = self._to_credit_decimal(user.credits_balance)
                            credits_balance_display = self._format_credit_display(user.credits_balance)
                            can_create_plan = credits_balance >= Decimal("2")

                    if user_id:
                        # Generate a nonce so the user can start a plan from the dashboard
                        nonce = 'DASH_' + str(uuid.uuid4())
                        uid_filter = (
                            PlanItem.user_id.in_(self._admin_user_ids())
                            if is_admin
                            else PlanItem.user_id == str(user_id)
                        )
                        try:
                            recent_task_rows = (
                                self.db.session.query(
                                    PlanItem.id,
                                    PlanItem.state,
                                    func.substr(PlanItem.prompt, 1, 240).label("prompt_preview"),
                                )
                                .filter(uid_filter)
                                .order_by(PlanItem.timestamp_created.desc())
                                .limit(10)
                                .all()
                            )
                        except DataError:
                            self.db.session.rollback()
                            logger.warning(
                                "Detected invalid UTF-8 in task_item.prompt for user_id=%s while loading dashboard; "
                                "falling back without prompt previews.",
                                user_id,
                                exc_info=True,
                            )
                            recent_task_rows = (
                                self.db.session.query(
                                    PlanItem.id,
                                    PlanItem.state,
                                )
                                .filter(uid_filter)
                                .order_by(PlanItem.timestamp_created.desc())
                                .limit(10)
                                .all()
                            )
                        recent_tasks = []
                        for task in recent_task_rows:
                            prompt_preview = getattr(task, "prompt_preview", None)
                            if prompt_preview is None:
                                prompt_text = self._load_prompt_preview_safe(task.id)
                            else:
                                prompt_text = (prompt_preview or "").strip() or "[Prompt unavailable]"
                            recent_tasks.append(
                                SimpleNamespace(
                                    id=str(task.id),
                                    state=task.state if isinstance(task.state, PlanState) else None,
                                    prompt=prompt_text,
                                )
                            )
                        total_tasks_count = (
                            PlanItem.query
                            .filter(uid_filter)
                            .count()
                        )
                        # Load example prompts for the "Start New Plan" form
                        for prompt_uuid in DEMO_FORM_RUN_PROMPT_UUIDS:
                            prompt_item = self.prompt_catalog.find(prompt_uuid)
                            if prompt_item:
                                example_prompts.append(prompt_item.prompt)
                except Exception:
                    logger.debug("Could not load dashboard data", exc_info=True)
            return render_template(
                'index.html',
                user=user,
                credits_balance_display=credits_balance_display,
                can_create_plan=can_create_plan,
                total_tasks_count=total_tasks_count,
                recent_tasks=recent_tasks,
                is_admin=is_admin,
                nonce=nonce,
                user_id=user_id,
                example_prompts=example_prompts,
                model_profile_options=_model_profile_options(),
            )

        @self.app.route('/models')
        @login_required
        def models():
            model_profile_options = _model_profile_options()
            option_by_value = {item["value"]: item for item in model_profile_options}
            profile_to_models = _profile_model_rows_map()
            profile_sections = [
                {
                    "key": profile.value,
                    "title": option_by_value.get(profile.value, {}).get("title", profile.value),
                    "subtitle": option_by_value.get(profile.value, {}).get("subtitle", ""),
                    "filename": f"llm_config/{PlanExeConfig.load(model_profile_override=profile).llm_config_json_name}",
                    "models": profile_to_models.get(profile.value, []),
                }
                for profile in ModelProfileEnum
            ]
            return render_template(
                'models.html',
                profile_sections=profile_sections,
            )

        @self.app.route('/healthcheck')
        def healthcheck():
            try:
                self.db.session.execute(text("SELECT 1"))
                return jsonify({"status": "ok", "database": "ok"}), 200
            except Exception:
                logger.error("Health check failed", exc_info=True)
                return jsonify({"status": "error", "database": "error"}), 500

        @self.app.route('/llms.txt')
        def llms_txt():
            llms_path = self.planexe_project_root / "public" / "llms.txt"
            if not llms_path.exists():
                abort(404)
            return send_file(llms_path, mimetype="text/plain; charset=utf-8")

        @self.app.route('/llm.txt')
        def llm_txt_alias():
            return redirect('/llms.txt', code=308)

        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                if username == self.admin_username and password == self.admin_password:
                    session.pop("open_access_logged_out", None)
                    session["auth_provider"] = "password"
                    user = User(self.admin_username, is_admin=True)
                    login_user(user)
                    return redirect(url_for('index'))
                return 'Invalid credentials', 401
            return render_template(
                'login.html',
                oauth_providers=self.oauth_providers,
                oauth_provider_labels=AUTH_PROVIDER_LABELS,
                telegram_enabled=bool(os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")),
                telegram_login_url=os.environ.get("PLANEXE_TELEGRAM_LOGIN_URL") or None,
            )

        @self.app.route('/api/oauth-redirect-uri')
        def oauth_redirect_uri_debug():
            """Return the redirect URI the app sends to Google. Use this to verify Google Console has the exact same URI."""
            lines = [
                f"PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL={self.public_base_url or '(not set)'}",
                f"redirect_uri={self._oauth_redirect_url('google') if 'google' in self.oauth_providers else '(google not configured)'}",
            ]
            body = "\n".join(lines)
            return body, 200, {"Content-Type": "text/plain; charset=utf-8"}

        @self.app.route('/login/<provider>')
        def oauth_login(provider: str):
            if provider not in self.oauth_providers:
                abort(404)
            client = self.oauth.create_client(provider)
            redirect_uri = self._oauth_redirect_url(provider)
            if provider == "google":
                nonce = secrets.token_urlsafe(16)
                session["oauth_google_nonce"] = nonce
                return client.authorize_redirect(redirect_uri, nonce=nonce)
            return client.authorize_redirect(redirect_uri)

        @self.app.route('/auth/<provider>/callback')
        def oauth_callback(provider: str):
            if provider not in self.oauth_providers:
                abort(404)

            try:
                client = self.oauth.create_client(provider)
                token = client.authorize_access_token()

                if provider == "google":
                    nonce = session.pop("oauth_google_nonce", None)
                    profile = client.parse_id_token(token, nonce=nonce)
                    if not profile:
                        profile = client.get("userinfo").json()
                else:
                    profile = self._get_user_from_provider(provider, token)

                user = self._upsert_user_from_oauth(provider, profile)
                session.pop("open_access_logged_out", None)
                session["auth_provider"] = provider
                login_user(User(user.id, is_admin=user.is_admin))
                has_key = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).first() is not None
                if not has_key:
                    new_api_key = self._get_or_create_api_key(user, name="Default")
                    if new_api_key:
                        session["new_api_key"] = new_api_key
                return redirect(url_for('account'))

            except Exception as e:
                logger.error(f"OAuth callback error for {provider}: {e}", exc_info=True)
                return render_template('login.html',
                    error=f"Authentication failed. Please try again or contact support.",
                    oauth_providers=self.oauth_providers,
                    oauth_provider_labels=AUTH_PROVIDER_LABELS,
                    telegram_enabled=bool(os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")),
                    telegram_login_url=os.environ.get("PLANEXE_TELEGRAM_LOGIN_URL") or None,
                ), 401

        @self.app.route('/logout')
        @login_required
        def logout():
            # In open-access mode, remember explicit logout in this browser session.
            # Otherwise before_request auto-login would sign admin back in immediately.
            session["open_access_logged_out"] = True
            session.pop("auth_provider", None)
            logout_user()
            return redirect(url_for('index'))

        @self.app.route('/account', methods=['GET', 'POST'])
        @login_required
        def account():
            is_admin = current_user.is_admin
            if is_admin:
                user = self._get_current_user_account()
                if not user:
                    # Force-create the admin UserAccount directly as a
                    # fallback — _get_current_user_account may fail on a
                    # fresh DB or after schema migrations.
                    logger.warning(
                        "Account page: _get_current_user_account returned None for admin, "
                        "attempting inline creation. current_user.id=%r admin_username=%r",
                        str(current_user.id), self.admin_username,
                    )
                    admin_pref_id = uuid.uuid5(
                        uuid.NAMESPACE_URL, f"planexe-admin-pref:{self.admin_username}"
                    )
                    try:
                        user = self.db.session.get(UserAccount, admin_pref_id)
                        if not user:
                            user = _new_model(
                                UserAccount,
                                id=admin_pref_id,
                                is_admin=True,
                                name="Admin",
                            )
                            self.db.session.add(user)
                            self.db.session.commit()
                            logger.info("Created admin UserAccount %s via account page fallback", admin_pref_id)
                    except Exception:
                        self.db.session.rollback()
                        logger.exception("Inline admin UserAccount creation failed for %s", admin_pref_id)
                        user = self.db.session.get(UserAccount, admin_pref_id)
            else:
                user_uuid = uuid.UUID(str(current_user.id))
                user = self.db.session.get(UserAccount, user_uuid)
            if not user:
                logger.warning(
                    "Account page: no UserAccount found, redirecting to logout. "
                    "is_admin=%s current_user.id=%r",
                    is_admin, str(current_user.id),
                )
                return redirect(url_for('logout'))

            stripe_result = request.args.get("stripe")
            stripe_session_id = request.args.get("session_id", "").strip()
            if request.method == "GET" and stripe_result in ("success", "cancel"):
                self._record_event(
                    EventType.GENERIC_EVENT,
                    "Stripe return to account page",
                    context={
                        "user_id": str(user.id),
                        "stripe_result": stripe_result,
                        "checkout_session_id": stripe_session_id or None,
                    },
                )
                if stripe_result == "success" and stripe_session_id:
                    self._finalize_stripe_checkout_session(user, stripe_session_id)

            new_api_key = session.pop("new_api_key", None)
            if request.method == 'POST':
                action = request.form.get('action')
                try:
                    if action == "create_api_key":
                        raw_key = self._get_or_create_api_key(user, name=request.form.get("name"))
                        if raw_key:
                            session["new_api_key"] = raw_key
                    elif action == "revoke_api_key":
                        key_id = request.form.get("key_id", "").strip()
                        if key_id:
                            try:
                                key_uuid = uuid.UUID(key_id)
                            except ValueError:
                                key_uuid = None
                            if key_uuid:
                                target_key = UserApiKey.query.filter_by(
                                    id=key_uuid, user_id=user.id, revoked_at=None
                                ).first()
                                if target_key:
                                    target_key.revoked_at = datetime.now(UTC)
                                    self.db.session.commit()
                    elif action == "rename_api_key":
                        key_id = request.form.get("key_id", "").strip()
                        new_name = (request.form.get("name") or "").strip()[:128]
                        if key_id:
                            try:
                                key_uuid = uuid.UUID(key_id)
                            except ValueError:
                                key_uuid = None
                            if key_uuid:
                                target_key = UserApiKey.query.filter_by(
                                    id=key_uuid, user_id=user.id, revoked_at=None
                                ).first()
                                if target_key:
                                    target_key.name = new_name or None
                                    self.db.session.commit()
                    elif action == "reset_api_key":
                        key_id = request.form.get("key_id", "").strip()
                        if key_id:
                            try:
                                key_uuid = uuid.UUID(key_id)
                            except ValueError:
                                key_uuid = None
                            if key_uuid:
                                target_key = UserApiKey.query.filter_by(
                                    id=key_uuid, user_id=user.id, revoked_at=None
                                ).first()
                                if target_key:
                                    api_key_secret = os.environ.get("PLANEXE_API_KEY_SECRET", "dev-api-key-secret")
                                    raw_key = f"pex_{secrets.token_urlsafe(24)}"
                                    target_key.key_hash = hashlib.sha256(f"{api_key_secret}:{raw_key}".encode("utf-8")).hexdigest()
                                    target_key.key_prefix = raw_key[:10]
                                    target_key.key_plaintext = raw_key if not self._api_key_show_once else None
                                    self.db.session.commit()
                                    session["new_api_key"] = raw_key
                    elif action == "regenerate_api_key":
                        # Legacy action: revoke all, create one new key.
                        existing_keys = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).all()
                        now = datetime.now(UTC)
                        for key in existing_keys:
                            key.revoked_at = now
                        self.db.session.commit()
                        raw_key = self._get_or_create_api_key(user)
                        if raw_key:
                            session["new_api_key"] = raw_key
                except Exception:
                    self.db.session.rollback()
                    logger.exception("Account POST action=%s failed", action)
                return redirect(url_for('account'))

            active_keys = (
                UserApiKey.query
                .filter_by(user_id=user.id, revoked_at=None)
                .order_by(UserApiKey.created_at.asc())
                .all()
            )

            # Per-key stats: LLM call counts and credit usage.
            active_key_ids = [str(k.id) for k in active_keys]
            llm_call_counts: dict[str, int] = {}
            credit_usage: dict[str, str] = {}
            if active_key_ids:
                try:
                    for row in (
                        self.db.session.query(TokenMetrics.api_key_id, func.count(TokenMetrics.id))
                        .filter(TokenMetrics.api_key_id.in_(active_key_ids))
                        .group_by(TokenMetrics.api_key_id)
                        .all()
                    ):
                        llm_call_counts[row[0]] = row[1]
                except Exception as exc:
                    logger.warning("Per-key LLM call count query failed: %s", exc)
                    self.db.session.rollback()
                try:
                    for row in (
                        self.db.session.query(CreditHistory.api_key_id, func.sum(CreditHistory.delta))
                        .filter(
                            CreditHistory.api_key_id.in_(active_key_ids),
                            CreditHistory.delta < 0,
                        )
                        .group_by(CreditHistory.api_key_id)
                        .all()
                    ):
                        credit_usage[row[0]] = self._format_credit_display(abs(row[1]))
                except Exception as exc:
                    logger.warning("Per-key credit usage query failed: %s", exc)
                    self.db.session.rollback()

            payment_rows = (
                PaymentRecord.query
                .filter_by(user_id=user.id)
                .order_by(PaymentRecord.created_at.desc())
                .limit(20)
                .all()
            )
            recent_payments: list[dict[str, Any]] = []
            for row in payment_rows:
                created_at = row.created_at
                if created_at and created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                amount_minor = int(row.amount or 0)
                recent_payments.append({
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if created_at else "—",
                    "provider": (row.provider or "").upper(),
                    "status": row.status or "unknown",
                    "credits": self._format_credit_display(row.credits),
                    "amount_major": amount_minor / 100.0,
                    "currency": (row.currency or "usd").upper(),
                    "payment_id": row.provider_payment_id or "",
                })

            linked_provider_rows = (
                UserProvider.query
                .filter_by(user_id=user.id)
                .order_by(UserProvider.last_login_at.desc())
                .all()
            )
            linked_sign_in_methods: list[str] = []
            for row in linked_provider_rows:
                label = self._auth_provider_label(row.provider)
                if label not in linked_sign_in_methods:
                    linked_sign_in_methods.append(label)

            auth_provider_session = session.get("auth_provider")
            signed_in_with = self._auth_provider_label(auth_provider_session)
            if signed_in_with == "Unknown" and linked_sign_in_methods:
                signed_in_with = linked_sign_in_methods[0]
            if is_admin:
                signed_in_with = "Admin credentials"

            return render_template(
                'account.html',
                admin_mode=is_admin,
                user=user,
                credits_balance_display="Full access" if is_admin else self._format_credit_display(user.credits_balance),
                credit_price_cents=max(1, int(os.environ.get("PLANEXE_CREDIT_PRICE_CENTS", "100"))),
                active_keys=active_keys,
                llm_call_counts=llm_call_counts,
                credit_usage=credit_usage,
                can_create_key=len(active_keys) < 10,
                new_api_key=new_api_key,
                recent_payments=recent_payments,
                signed_in_with=signed_in_with,
                linked_sign_in_methods=linked_sign_in_methods,
                stripe_enabled=bool(os.environ.get("PLANEXE_STRIPE_SECRET_KEY")),
                telegram_enabled=bool(os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")),
                api_key_show_once=self._api_key_show_once,
            )

        @self.app.route('/billing/stripe/checkout', methods=['POST'])
        @login_required
        def stripe_checkout():
            if current_user.is_admin:
                abort(403)
            stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
            if not stripe_secret:
                return jsonify({"error": "Stripe not configured"}), 400
            stripe.api_key = stripe_secret
            credits = int(request.form.get("credits", "1"))
            if credits < 1:
                return jsonify({"error": "credits must be >= 1"}), 400
            price_per_credit = int(os.environ.get("PLANEXE_CREDIT_PRICE_CENTS", "100"))
            amount = credits * price_per_credit
            currency = os.environ.get("PLANEXE_STRIPE_CURRENCY", "usd")
            success_url = (
                f"{self.public_base_url}/account?stripe=success&session_id={{CHECKOUT_SESSION_ID}}"
                if self.public_base_url
                else url_for("account", _external=True)
            )
            cancel_url = f"{self.public_base_url}/account?stripe=cancel" if self.public_base_url else url_for("account", _external=True)
            self._record_event(
                EventType.GENERIC_EVENT,
                "Stripe checkout requested",
                context={
                    "user_id": str(current_user.id),
                    "credits": credits,
                    "amount_minor": amount,
                    "amount_major": amount / 100.0,
                    "currency": currency,
                },
            )
            try:
                session_obj = stripe.checkout.Session.create(
                    mode="payment",
                    success_url=success_url,
                    cancel_url=cancel_url,
                    line_items=[{
                        "price_data": {
                            "currency": currency,
                            "product_data": {"name": "PlanExe credits"},
                            "unit_amount": amount,
                        },
                        "quantity": 1,
                    }],
                    metadata={
                        "user_id": str(current_user.id),
                        "credits": str(credits),
                    },
                )
            except Exception as exc:
                self._record_event(
                    EventType.GENERIC_ERROR,
                    "Stripe checkout creation failed",
                    context={
                        "user_id": str(current_user.id),
                        "credits": credits,
                        "amount_minor": amount,
                        "currency": currency,
                        "error": str(exc),
                    },
                )
                return jsonify({"error": "stripe checkout failed"}), 400
            self._record_event(
                EventType.GENERIC_EVENT,
                "Stripe checkout session created",
                context={
                    "user_id": str(current_user.id),
                    "credits": credits,
                    "amount_minor": amount,
                    "amount_major": amount / 100.0,
                    "currency": currency,
                    "checkout_session_id": session_obj.get("id"),
                    "checkout_payment_status": session_obj.get("payment_status"),
                },
            )
            session_url = session_obj.url
            if not isinstance(session_url, str) or not session_url:
                return jsonify({"error": "stripe checkout missing redirect url"}), 502
            return redirect(session_url)

        @self.app.route('/billing/stripe/webhook', methods=['POST'])
        def stripe_webhook():
            stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
            webhook_secret = os.environ.get("PLANEXE_STRIPE_WEBHOOK_SECRET")
            if not stripe_secret:
                return jsonify({"error": "Stripe not configured"}), 400
            stripe.api_key = stripe_secret
            payload = request.get_data()
            sig_header = request.headers.get("Stripe-Signature")
            event_id = None
            event_type = None
            try:
                if webhook_secret:
                    event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
                else:
                    event = json.loads(payload)
                event_id = event.get("id")
                event_type = event.get("type")
            except Exception as exc:
                logger.error("Stripe webhook error: %s", exc)
                self._record_event(
                    EventType.GENERIC_ERROR,
                    "Stripe webhook rejected",
                    context={
                        "error": str(exc),
                        "has_signature_header": bool(sig_header),
                        "webhook_secret_configured": bool(webhook_secret),
                    },
                )
                return jsonify({"error": "invalid payload"}), 400

            self._record_event(
                EventType.GENERIC_EVENT,
                "Stripe webhook received",
                context={
                    "stripe_event_id": event_id,
                    "stripe_event_type": event_type,
                },
            )

            if event_type == "checkout.session.completed":
                session_obj = event["data"]["object"]
                metadata = session_obj.get("metadata") or {}
                user_id = metadata.get("user_id")
                credits = self._to_credit_decimal(metadata.get("credits", "0") or 0)
                if user_id and credits > 0:
                    status = self._apply_payment_credits(
                        user_id=user_id,
                        provider="stripe",
                        provider_payment_id=session_obj.get("id", ""),
                        credits=credits,
                        amount=session_obj.get("amount_total") or 0,
                        currency=session_obj.get("currency") or "usd",
                        raw_payload=session_obj,
                    )
                    self._record_event(
                        EventType.GENERIC_EVENT if status in ("applied", "duplicate_payment") else EventType.GENERIC_ERROR,
                        "Stripe payment completion processed",
                        context={
                            "stripe_event_id": event_id,
                            "stripe_event_type": event_type,
                            "user_id": user_id,
                            "credits": str(credits),
                            "amount_minor": session_obj.get("amount_total") or 0,
                            "amount_major": (session_obj.get("amount_total") or 0) / 100.0,
                            "currency": session_obj.get("currency") or "usd",
                            "checkout_session_id": session_obj.get("id", ""),
                            "checkout_payment_status": session_obj.get("payment_status"),
                            "status": status,
                        },
                    )
                else:
                    self._record_event(
                        EventType.GENERIC_ERROR,
                        "Stripe completed event missing billing metadata",
                        context={
                            "stripe_event_id": event_id,
                            "stripe_event_type": event_type,
                            "checkout_session_id": session_obj.get("id", ""),
                            "metadata_present": bool(metadata),
                            "user_id": user_id,
                            "credits": str(credits),
                        },
                    )
            elif event_type in ("checkout.session.async_payment_failed", "payment_intent.payment_failed", "checkout.session.expired"):
                event_object = event.get("data", {}).get("object", {}) if isinstance(event.get("data"), dict) else {}
                metadata = event_object.get("metadata") or {}
                self._record_event(
                    EventType.GENERIC_ERROR,
                    "Stripe payment failed",
                    context={
                        "stripe_event_id": event_id,
                        "stripe_event_type": event_type,
                        "user_id": metadata.get("user_id"),
                        "credits": metadata.get("credits"),
                        "amount_minor": event_object.get("amount_total") or event_object.get("amount"),
                        "amount_major": ((event_object.get("amount_total") or event_object.get("amount") or 0) / 100.0),
                        "currency": event_object.get("currency"),
                        "checkout_session_id": event_object.get("id"),
                        "payment_intent_id": event_object.get("payment_intent"),
                        "payment_status": event_object.get("payment_status"),
                        "failure_message": event_object.get("last_payment_error", {}).get("message") if isinstance(event_object.get("last_payment_error"), dict) else None,
                    },
                )
            return jsonify({"status": "ok"})

        @self.app.route('/billing/telegram/invoice', methods=['POST'])
        @login_required
        def telegram_invoice():
            if current_user.is_admin:
                abort(403)
            bot_token = os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")
            if not bot_token:
                return jsonify({"error": "Telegram not configured"}), 400
            credits = int(request.form.get("credits", "1"))
            if credits < 1:
                return jsonify({"error": "credits must be >= 1"}), 400
            price_per_credit = int(os.environ.get("PLANEXE_TELEGRAM_STARS_PER_CREDIT", "100"))
            payload = f"planexe:{current_user.id}:{credits}:{uuid.uuid4()}"
            url = f"https://api.telegram.org/bot{bot_token}/createInvoiceLink"
            response = requests.post(url, json={
                "title": "PlanExe credits",
                "description": f"{credits} credit(s) for PlanExe",
                "payload": payload,
                "currency": "XTR",
                "prices": [{"label": "PlanExe credits", "amount": credits * price_per_credit}],
            }, timeout=10)
            if response.status_code != 200:
                return jsonify({"error": "telegram error", "details": response.text}), 400
            data = response.json()
            if not data.get("ok"):
                return jsonify({"error": "telegram error", "details": data}), 400
            return redirect(data["result"])

        @self.app.route('/billing/telegram/webhook', methods=['POST'])
        def telegram_webhook():
            bot_token = os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")
            if not bot_token:
                return jsonify({"error": "Telegram not configured"}), 400
            update = request.get_json(silent=True) or {}
            if "pre_checkout_query" in update:
                query_id = update["pre_checkout_query"]["id"]
                requests.post(
                    f"https://api.telegram.org/bot{bot_token}/answerPreCheckoutQuery",
                    json={"pre_checkout_query_id": query_id, "ok": True},
                    timeout=5,
                )
                return jsonify({"status": "ok"})
            message = update.get("message") or {}
            payment = message.get("successful_payment")
            if payment:
                payload = payment.get("invoice_payload", "")
                try:
                    _, user_id, credits, _nonce = payload.split(":", 3)
                    credits_decimal = self._to_credit_decimal(credits)
                except Exception:
                    return jsonify({"status": "ignored"})
                self._apply_payment_credits(
                    user_id=user_id,
                    provider="telegram",
                    provider_payment_id=payment.get("telegram_payment_charge_id", ""),
                    credits=credits_decimal,
                    amount=payment.get("total_amount") or 0,
                    currency=payment.get("currency") or "XTR",
                    raw_payload=payment,
                )
            return jsonify({"status": "ok"})

        # Exempt external webhook endpoints from CSRF protection.
        # These receive POST requests from third-party services (Stripe, Telegram)
        # that cannot provide a CSRF token.
        self.csrf.exempt(stripe_webhook)
        self.csrf.exempt(telegram_webhook)

        @self.app.route('/ping')
        @login_required
        def ping():
            return render_template('ping.html')

        @self.app.route('/admin/reconciliation')
        @admin_required
        def admin_reconciliation():
            max_tasks = int(request.args.get("limit", "200") or "200")
            max_tasks = max(1, min(max_tasks, 2000))
            tolerance_usd = self._safe_float(request.args.get("tolerance_usd", "0.01")) or 0.01
            tolerance_usd = max(0.0, tolerance_usd)
            refresh_seconds = int(request.args.get("refresh_seconds", "60") or "60")
            refresh_seconds = max(10, min(refresh_seconds, 3600))

            rows, summary = self._build_reconciliation_report(max_tasks=max_tasks, tolerance_usd=tolerance_usd)
            has_alert = summary["mismatch_count"] > 0
            return self.admin.index_view.render(
                "admin/reconciliation.html",
                rows=rows,
                summary=summary,
                has_alert=has_alert,
                max_tasks=max_tasks,
                tolerance_usd=tolerance_usd,
                refresh_seconds=refresh_seconds,
            )

        @self.app.route('/ping/stream')
        @login_required
        def ping_stream():
            def generate():
                url = f"{self.worker_plan_url}/llm-ping"
                logger.info("Proxying LLM ping stream from %s", url)
                try:
                    with requests.get(
                        url,
                        stream=True,
                        timeout=(5, 300),
                        headers={"Accept": "text/event-stream"},
                    ) as resp:
                        if resp.status_code != 200:
                            msg = f"worker_plan responded with {resp.status_code}"
                            logger.error("LLM ping proxy error: %s", msg)
                            yield f"data: {json.dumps({'name': 'worker_plan', 'status': 'error', 'response_time': 0, 'response': msg})}\n\n"
                            yield f"data: {json.dumps({'name': 'server', 'status': 'done', 'response_time': 0, 'response': ''})}\n\n"
                            return
                        for line in resp.iter_lines(decode_unicode=True):
                            if line is None or line.strip() == "":
                                continue
                            # Re-emit each SSE line with proper terminator.
                            yield f"{line}\n\n"
                except Exception as exc:  # pragma: no cover - runtime proxy
                    logger.error("LLM ping proxy exception: %s", exc)
                    error_payload = {'name': 'worker_plan', 'status': 'error', 'response_time': 0, 'response': str(exc)}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield f"data: {json.dumps({'name': 'server', 'status': 'done', 'response_time': 0, 'response': ''})}\n\n"

            response = Response(generate(), mimetype='text/event-stream')
            response.headers['X-Accel-Buffering'] = 'no'  # Disable Nginx buffering
            return response

        @self.app.route('/run', methods=['GET', 'POST'])
        @login_required
        @nocache
        def run():
            # When request.method is POST, and urlencoded parameters are detected, then return an error, so the developer can detect that something is wrong, the parameters in the url are supposed to be part for the form.
            if request.method == 'POST' and request.args:
                logger.error(f"endpoint /run. POST request with urlencoded parameters detected. This is not allowed. The url parameters are supposed to be part of the form.")
                return jsonify({"error": "POST request with urlencoded parameters detected. This is not allowed. The url parameters are supposed to be part of the form."}), 400

            # Obtain info about the request
            request_size_bytes: int = len(request.get_data())
            request_content_type: str = request.headers.get('Content-Type', '')

            # Gather the parameters from the request.form (POST) or request.args (GET)
            request_form_or_args = request.form if request.method == 'POST' else request.args
            prompt_param = request_form_or_args.get('prompt', '')
            user_id_param = request_form_or_args.get('user_id', '')
            nonce_param = request_form_or_args.get('nonce', '')
            parameters: dict[str, Any] | None = {key: value for key, value in request_form_or_args.items()}

            # Remove the parameters that have already been extracted from the parameters dictionary
            parameters.pop('prompt', None)
            parameters.pop('user_id', None)
            parameters.pop('nonce', None)
            if len(parameters) == 0:
                parameters = None

            # Normalize model profile to a known value with backward-compatible baseline default.
            if not isinstance(parameters, dict):
                parameters = {}
            raw_profile = parameters.get("model_profile")
            parameters["model_profile"] = normalize_model_profile(raw_profile).value

            # Get length of prompt_param in bytes and in characters
            prompt_param_bytes = len(prompt_param.encode('utf-8'))
            prompt_param_characters = len(prompt_param)

            # Avoid flooding logs when the prompt is long.
            log_prompt_info = prompt_param[:100]
            if len(prompt_param) > 100:
                log_prompt_info += "... (truncated)"
            logger.info(f"endpoint /run ({request.method}). Size of request: {request_size_bytes} bytes. Starting run with parameters: prompt={log_prompt_info!r}, user_id={user_id_param!r}, nonce={nonce_param!r}, parameters={parameters!r}, prompt_param_bytes={prompt_param_bytes}, prompt_param_characters={prompt_param_characters}")

            if current_user.is_admin:
                admin_account = self._get_current_user_account()
                user_id_param = str(admin_account.id) if admin_account else self.admin_username
            else:
                user_id_param = str(current_user.id)

            if not nonce_param:
                logger.error(f"endpoint /run. No nonce provided")
                return jsonify({"error": "A unique request identifier (nonce) is required."}), 400

            with self.app.app_context():
                context = {
                    "user_agent": request.headers.get('User-Agent'),
                    "ip_address": request.remote_addr,
                    "prompt": prompt_param,
                    "user_id": user_id_param,
                }
                nonce_item, is_new = NonceItem.get_or_create(nonce_key=nonce_param, context=context)
                if not is_new:
                    logger.warning(f"endpoint /run. Replay detected for nonce '{nonce_param}'. Request count: {nonce_item.request_count}.")
                    return jsonify({"error": "This action has already been performed. Reusing this link is not permitted."}), 409

            if not prompt_param:
                logger.error(f"endpoint /run. No prompt provided")
                return jsonify({"error": "No prompt provided"}), 400
            
            with self.app.app_context():
                if not current_user.is_admin:
                    user = self.db.session.get(UserAccount, uuid.UUID(str(current_user.id)))
                    if not user:
                        return jsonify({"error": "User not found"}), 400
                    if not user.free_plan_used:
                        user.free_plan_used = True
                        self.db.session.commit()
                        if not isinstance(parameters, dict):
                            parameters = {}
                        parameters["billing_skip_usage_charge"] = True
                    else:
                        if (user.credits_balance or 0) <= 0:
                            return jsonify({"error": "No credits available"}), 402

                task = _new_model(
                    PlanItem,
                    state=PlanState.pending,
                    prompt=prompt_param,
                    progress_percentage=0.0,
                    progress_message="Awaiting server to start…",
                    user_id=user_id_param,
                    parameters=parameters
                )
                self.db.session.add(task)
                self.db.session.commit()
                task_id = task.id if hasattr(task, 'id') else None
                logger.info(f"endpoint /run. Task received: {task_id!r}")
                event_context = {
                    "task_id": str(task_id),
                    "request_size_bytes": request_size_bytes,
                    "request_content_type": request_content_type,
                    "prompt_param_bytes": prompt_param_bytes,
                    "prompt_param_characters": prompt_param_characters,
                    "prompt": prompt_param,
                    "user_id": user_id_param,
                    "parameters": parameters,
                    "method": request.method
                }
                event = _new_model(
                    EventItem,
                    event_type=EventType.TASK_PENDING,
                    message=f"Enqueued task via /run endpoint",
                    context=event_context
                )
                self.db.session.add(event)
                self.db.session.commit()
            return render_template('run_via_database.html', run_id=task_id)

        @self.app.route('/create_plan', methods=['POST'])
        @login_required
        @nocache
        def create_plan():
            # Form fields must come in POST body.
            if request.args:
                logger.error("endpoint /create_plan. POST request with urlencoded parameters detected. This is not allowed.")
                return jsonify({"error": "POST request with urlencoded parameters detected. This is not allowed."}), 400

            request_size_bytes: int = len(request.get_data())
            request_content_type: str = request.headers.get('Content-Type', '')

            prompt_param = request.form.get('prompt', '')
            parameters: dict[str, Any] = {key: value for key, value in request.form.items()}
            parameters.pop('csrf_token', None)
            parameters.pop('prompt', None)
            parameters.pop('user_id', None)
            parameters.pop('nonce', None)
            parameters.pop('redirect_to_plan', None)

            # Normalize model profile to a known value with backward-compatible baseline default.
            raw_profile = parameters.get("model_profile")
            parameters["model_profile"] = normalize_model_profile(raw_profile).value

            prompt_param_bytes = len(prompt_param.encode('utf-8'))
            prompt_param_characters = len(prompt_param)
            log_prompt_info = prompt_param[:100]
            if len(prompt_param) > 100:
                log_prompt_info += "... (truncated)"

            if not prompt_param:
                logger.error("endpoint /create_plan. No prompt provided")
                return jsonify({"error": "No prompt provided"}), 400

            api_key_id_param: Optional[str] = None
            if current_user.is_admin:
                admin_account = self._get_current_user_account()
                user_id_param = str(admin_account.id) if admin_account else self.admin_username
            else:
                user_id_param = str(current_user.id)

            logger.info(
                "endpoint /create_plan (%s). Size of request: %s bytes. Starting run with prompt=%r, user_id=%r, parameters=%r, prompt_param_bytes=%s, prompt_param_characters=%s",
                request.method,
                request_size_bytes,
                log_prompt_info,
                user_id_param,
                parameters,
                prompt_param_bytes,
                prompt_param_characters,
            )

            with self.app.app_context():
                if current_user.is_admin:
                    # Admin skips credit check; resolve api_key_id for stats.
                    first_key = (
                        UserApiKey.query
                        .filter_by(user_id=uuid.UUID(user_id_param), revoked_at=None)
                        .order_by(UserApiKey.created_at.asc())
                        .first()
                    )
                    if first_key:
                        api_key_id_param = str(first_key.id)
                else:
                    user = self.db.session.get(UserAccount, uuid.UUID(str(current_user.id)))
                    if not user:
                        return jsonify({"error": "User not found"}), 400
                    if self._to_credit_decimal(user.credits_balance) < Decimal("2"):
                        return jsonify({"error": "Insufficient credits (minimum 2 required)"}), 402
                    first_key = (
                        UserApiKey.query
                        .filter_by(user_id=user.id, revoked_at=None)
                        .order_by(UserApiKey.created_at.asc())
                        .first()
                    )
                    if first_key:
                        api_key_id_param = str(first_key.id)

                task = _new_model(
                    PlanItem,
                    state=PlanState.pending,
                    prompt=prompt_param,
                    progress_percentage=0.0,
                    progress_message="Awaiting server to start…",
                    user_id=user_id_param,
                    api_key_id=api_key_id_param,
                    parameters=parameters
                )
                self.db.session.add(task)
                self.db.session.commit()
                task_id = task.id if hasattr(task, 'id') else None
                logger.info("endpoint /create_plan. Task received: %r", task_id)

                event_context = {
                    "task_id": str(task_id),
                    "request_size_bytes": request_size_bytes,
                    "request_content_type": request_content_type,
                    "prompt_param_bytes": prompt_param_bytes,
                    "prompt_param_characters": prompt_param_characters,
                    "prompt": prompt_param,
                    "user_id": user_id_param,
                    "parameters": parameters,
                    "method": request.method
                }
                event = _new_model(
                    EventItem,
                    event_type=EventType.TASK_PENDING,
                    message="Enqueued task via /create_plan endpoint",
                    context=event_context
                )
                self.db.session.add(event)
                self.db.session.commit()

            if task_id is None:
                return jsonify({"error": "Unable to create task"}), 500
            return redirect(url_for('plan', id=task_id))

        @self.app.route('/run_status')
        @login_required
        @nocache
        def run_status():
            run_id = request.args.get('id', '')
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403
            return render_template('run_via_database.html', run_id=run_id)

        @self.app.route('/progress')
        def get_progress():
            run_id = request.args.get('run_id', '')
            logger.debug(f"Progress endpoint received run_id: {run_id!r}")
            # lookup the task in the database
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                logger.error(f"Task not found for run_id: {run_id!r}")
                return jsonify({"error": "Task not found"}), 400
            
            progress_percentage = float(task.progress_percentage) if task.progress_percentage is not None else 0.0
            progress_message = task.progress_message if task.progress_message is not None else ""
            if isinstance(task.state, PlanState):
                status = task.state.name
            else:
                status = f"unknown-{task.state}"

            # update the last_seen_timestamp
            try:
                task.last_seen_timestamp = datetime.now(UTC)
                self.db.session.commit()
            except Exception as e:
                logger.error(f"get_progress, error updating last_seen_timestamp for task {run_id!r}: {e}", exc_info=True)
                self.db.session.rollback()
                # ignore the error

            return jsonify({"progress_percentage": progress_percentage, "progress_message": progress_message, "status": status}), 200

        @self.app.route('/viewplan')
        @login_required
        def viewplan():
            run_id = request.args.get('run_id', '')
            logger.info(f"ViewPlan endpoint requested for run_id: {run_id!r}")
            # lookup the task in the database
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                logger.error(f"Task not found for run_id: {run_id!r}")
                return jsonify({"error": "Task not found"}), 400

            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                logger.warning("Unauthorized report access attempt. run_id=%s user_id=%s", run_id, current_user.id)
                return jsonify({"error": "Forbidden"}), 403

            if SHOW_DEMO_PLAN:
                run_id = '20250524_universal_manufacturing'
                run_id_dir = (self.planexe_run_dir / run_id).absolute()
                path_to_html_file = run_id_dir / FilenameEnum.REPORT.value
                if not path_to_html_file.exists():
                    return jsonify({"error": "Demo report not found"}), 404
                return send_file(str(path_to_html_file), mimetype='text/html')

            if not task.generated_report_html:
                logger.error("Report HTML not found for run_id=%s", run_id)
                return jsonify({"error": "Report not available"}), 404

            response = make_response(task.generated_report_html)
            response.headers['Content-Type'] = 'text/html'
            return response

        @self.app.route('/plan')
        @login_required
        def plan():
            run_id = request.args.get('id', '').strip()

            if not run_id:
                user_id = str(current_user.id)
                uid_filter = (
                    PlanItem.user_id.in_(self._admin_user_ids())
                    if current_user.is_admin
                    else PlanItem.user_id == user_id
                )
                try:
                    tasks = (
                        self.db.session.query(
                            PlanItem.id,
                            PlanItem.timestamp_created,
                            PlanItem.state,
                            func.substr(PlanItem.prompt, 1, 240).label("prompt_preview"),
                        )
                        .filter(uid_filter)
                        .order_by(PlanItem.timestamp_created.desc())
                        .all()
                    )
                except DataError:
                    self.db.session.rollback()
                    logger.warning(
                        "Detected invalid UTF-8 in task_item.prompt for user_id=%s while loading /plan; "
                        "falling back without prompt previews.",
                        user_id,
                        exc_info=True,
                    )
                    tasks = (
                        self.db.session.query(
                            PlanItem.id,
                            PlanItem.timestamp_created,
                            PlanItem.state,
                        )
                        .filter(uid_filter)
                        .order_by(PlanItem.timestamp_created.desc())
                        .all()
                    )
                rows = []
                for task in tasks:
                    ts = task.timestamp_created
                    created_compact = ts.strftime("%y%m%d-%H%M") if isinstance(ts, datetime) else "-"
                    prompt_preview = getattr(task, "prompt_preview", None)
                    if prompt_preview is None:
                        prompt_text = self._load_prompt_preview_safe(task.id)
                    else:
                        prompt_text = (prompt_preview or "").strip() or "[Prompt unavailable]"
                    rows.append({
                        "id": str(task.id),
                        "created_compact": created_compact,
                        "created_relative": self._format_relative_time(ts),
                        "status": task.state.name if isinstance(task.state, PlanState) else "pending",
                        "prompt": prompt_text,
                    })
                return render_template("plan_list.html", plan_rows=rows)

            logger.info(f"Plan iframe wrapper requested for run_id: {run_id!r}")
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                logger.error(f"Task not found for run_id: {run_id!r}")
                return jsonify({"error": "Task not found"}), 400

            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                logger.warning("Unauthorized plan wrapper access attempt. run_id=%s user_id=%s", run_id, current_user.id)
                return jsonify({"error": "Forbidden"}), 403

            telemetry = self._build_plan_telemetry(task, include_raw=False)
            failure_trace = self._build_plan_failure_trace(task)
            preferred_plan_view_mode = self._get_plan_view_mode_preference()
            parameters = task.parameters if isinstance(task.parameters, dict) else {}
            selected_model_profile = normalize_model_profile(parameters.get("model_profile")).value
            return render_template(
                "plan_iframe.html",
                run_id=run_id,
                task=task,
                telemetry=telemetry,
                failure_trace=failure_trace,
                preferred_plan_view_mode=preferred_plan_view_mode,
                selected_model_profile=selected_model_profile,
            )

        @self.app.route('/plan/download/report')
        @login_required
        def plan_download_report():
            run_id = request.args.get('id', '')
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403
            if not task.generated_report_html:
                return jsonify({"error": "Report not available"}), 404
            buffer = io.BytesIO(task.generated_report_html.encode('utf-8'))
            buffer.seek(0)
            download_name = f"{task.id}-report.html"
            return send_file(buffer, mimetype='text/html', as_attachment=True, download_name=download_name)

        @self.app.route('/plan/download/zip')
        @login_required
        def plan_download_zip():
            run_id = request.args.get('id', '')
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403
            if not task.run_zip_snapshot:
                return jsonify({"error": "Run zip not available"}), 404

            layout_version = self._safe_int(getattr(task, "run_artifact_layout_version", None)) or 0
            if layout_version >= 2:
                buffer = io.BytesIO(task.run_zip_snapshot)
                buffer.seek(0)
            else:
                buffer = self._sanitize_legacy_run_zip_for_download(task.run_zip_snapshot)
                if buffer is None:
                    logger.error("Invalid legacy run zip snapshot for run_id=%s", run_id)
                    return jsonify({"error": "Run zip is invalid"}), 500

            download_name = f"{task.id}.zip"
            return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name=download_name)

        @self.app.route('/plan/stop', methods=['POST'])
        @login_required
        def plan_stop():
            run_id = request.form.get('id', '').strip()
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403

            # Guard against stale UI: if task already completed (or report exists),
            # ignore stop requests so completion billing cannot be bypassed.
            if task.state == PlanState.completed or bool(task.has_generated_report_html):
                logger.info("Ignoring stop request for already completed task %s", run_id)
                return redirect(url_for('plan', id=run_id))

            task.stop_requested = True
            task.stop_requested_timestamp = datetime.now(UTC)
            if task.state == PlanState.pending:
                task.state = PlanState.failed
                task.progress_message = "Stop requested by user."
            self.db.session.commit()
            return redirect(url_for('plan', id=run_id))

        @self.app.route('/plan/retry', methods=['POST'])
        @login_required
        def plan_retry():
            run_id = request.form.get('id', '').strip()
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403

            if task.state == PlanState.processing and not bool(task.stop_requested):
                return jsonify({"error": "Task is currently processing. Stop it first before retrying."}), 409

            raw_profile = request.form.get("model_profile")
            selected_model_profile = normalize_model_profile(raw_profile).value
            parameters = dict(task.parameters) if isinstance(task.parameters, dict) else {}
            parameters["model_profile"] = selected_model_profile
            task.parameters = parameters

            task.state = PlanState.pending
            task.stop_requested = False
            task.stop_requested_timestamp = None
            task.progress_percentage = 0.0
            task.progress_message = "Retry requested by user."
            task.generated_report_html = None
            task.run_zip_snapshot = None
            task.run_track_activity_jsonl = None
            task.run_track_activity_bytes = None
            task.run_activity_overview_json = None
            task.run_artifact_layout_version = None
            task.last_seen_timestamp = datetime.now(UTC)

            # Archive old incremental billing entries so the new run starts fresh.
            # Renaming source lets _sum_already_charged_credits start from zero
            # while preserving old entries for per-key credit history.
            CreditHistory.query.filter_by(
                source="usage_billing_progress",
                external_id=str(task.id),
            ).update({"source": "usage_billing_settled"})

            self.db.session.commit()
            return redirect(url_for('plan', id=run_id))

        @self.app.route('/plan/meta')
        @login_required
        def plan_meta():
            run_id = request.args.get('id', '').strip()
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403

            state_name = task.state.name if isinstance(task.state, PlanState) else "pending"
            telemetry = self._build_plan_telemetry(task, include_raw=False)
            failure_trace = self._build_plan_failure_trace(task)
            return jsonify({
                "id": str(task.id),
                "state": state_name,
                "progress_percentage": float(task.progress_percentage) if task.progress_percentage is not None else 0.0,
                "progress_message": task.progress_message or "",
                "generated_report_html": bool(task.has_generated_report_html),
                "run_zip_snapshot": bool(task.has_run_zip_snapshot),
                "stop_requested": bool(task.stop_requested),
                "telemetry": telemetry,
                "failure_trace": failure_trace,
            }), 200

        @self.app.route('/plan/view-mode', methods=['POST'])
        @login_required
        def plan_view_mode():
            payload = request.get_json(silent=True) if request.is_json else request.form
            mode = self._normalize_plan_view_mode((payload or {}).get("mode"))
            self._set_plan_view_mode_preference(mode)
            return jsonify({"status": "ok", "mode": mode}), 200

        @self.app.route('/plan/telemetry')
        @login_required
        def plan_telemetry():
            run_id = request.args.get('id', '').strip()
            task = self.db.session.get(PlanItem, run_id)
            if task is None:
                return jsonify({"error": "Task not found"}), 400
            if not current_user.is_admin and str(task.user_id) != str(current_user.id):
                return jsonify({"error": "Forbidden"}), 403

            # Intentional: this endpoint is for authenticated owner/admin
            # troubleshooting and may include provider raw_usage_data payloads.
            # Keep /plan/meta sanitized (include_raw=False).
            telemetry = self._build_plan_telemetry(task, include_raw=True, expose_raw_usage_data=True)
            return jsonify(telemetry), 200

        @self.app.route('/admin/task/<uuid:task_id>/report')
        @admin_required
        def download_task_report(task_id):
            task = self.db.session.get(PlanItem, task_id)
            if task is None or not task.generated_report_html:
                return "Report not found", 404
            buffer = io.BytesIO(task.generated_report_html.encode('utf-8'))
            buffer.seek(0)
            return send_file(buffer, mimetype='text/html', as_attachment=True, download_name='report.html')

        @self.app.route('/admin/task/<uuid:task_id>/run_zip')
        @admin_required
        def download_task_run_zip(task_id):
            task = self.db.session.get(PlanItem, task_id)
            if task is None or not task.run_zip_snapshot:
                return "Run zip not found", 404
            buffer = io.BytesIO(task.run_zip_snapshot)
            buffer.seek(0)
            download_name = f"{task_id}.zip"
            return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name=download_name)

        @self.app.route('/admin/task/<uuid:task_id>/track_activity')
        @admin_required
        def download_task_track_activity(task_id):
            task = self.db.session.get(PlanItem, task_id)
            if task is None or not task.run_track_activity_jsonl:
                return "Track activity not found", 404
            buffer = io.BytesIO(task.run_track_activity_jsonl.encode('utf-8'))
            buffer.seek(0)
            download_name = f"{task_id}-track_activity.jsonl"
            return send_file(buffer, mimetype='application/x-ndjson', as_attachment=True, download_name=download_name)

        @self.app.route('/demo_run')
        @admin_required
        def demo_run():
            user_id = str(current_user.id)
            nonce = 'DEMO_' + str(uuid.uuid4())

            # The prompts to be shown on the page.
            prompts = []
            for prompt_uuid in DEMO_FORM_RUN_PROMPT_UUIDS:
                prompt_item = self.prompt_catalog.find(prompt_uuid)
                if prompt_item is None:
                    logger.error(f"Prompt item not found for uuid: {prompt_uuid} in demo_run")
                    return "Error: Demo prompt configuration missing.", 500
                prompts.append(prompt_item.prompt)

            return render_template(
                'demo_run.html',
                user_id=user_id,
                prompts=prompts,
                nonce=nonce,
                model_profile_options=_model_profile_options(),
            )

    def run_server(self, debug: bool = False, host: str = "0.0.0.0", port: int = 5000):
        env_debug = os.environ.get("PLANEXE_FRONTEND_MULTIUSER_DEBUG")
        if env_debug is not None:
            debug = env_debug.lower() in ("1", "true", "yes", "on")
        host = os.environ.get("PLANEXE_FRONTEND_MULTIUSER_HOST", host)
        port_str = os.environ.get("PLANEXE_FRONTEND_MULTIUSER_PORT")
        if port_str:
            port = int(port_str)
        self.app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
    )
    try:
        flask_app_instance = MyFlaskApp()
    except ValueError as exc:
        # Configuration errors (e.g. PLANEXE_AUTH_REQUIRED=true without OAuth).
        # Exit with code 0 so Docker "on-failure" restart policy does NOT restart.
        logger.critical("Configuration error – service will not start: %s", exc)
        sys.exit(0)
    flask_app_instance.run_server()

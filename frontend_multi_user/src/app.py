"""
Flask UI for PlanExe-server.

PROMPT> python3 -m src.app
"""
from datetime import datetime, UTC
import logging
import os
import re
import sys
import time
import json
import uuid
import io
import secrets
import hashlib
from urllib.parse import quote_plus
from typing import ClassVar, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_file, redirect, url_for, session, abort
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from authlib.integrations.flask_client import OAuth
from functools import wraps
import urllib.request
from urllib.error import URLError
from flask import make_response
import requests
import stripe
from worker_plan_api.generate_run_id import generate_run_id
from worker_plan_api.start_time import StartTime
from worker_plan_api.plan_file import PlanFile
from worker_plan_api.filenames import FilenameEnum
from worker_plan_api.prompt_catalog import PromptCatalog
from sqlalchemy import text, inspect
from sqlalchemy.exc import OperationalError
from database_api.model_taskitem import TaskItem, TaskState
from database_api.model_event import EventType, EventItem
from database_api.model_worker import WorkerItem
from database_api.model_nonce import NonceItem
from database_api.model_user_account import UserAccount
from database_api.model_user_provider import UserProvider
from database_api.model_user_api_key import UserApiKey
from database_api.model_credit_history import CreditHistory
from database_api.model_payment_record import PaymentRecord
from planexe_modelviews import WorkerItemView, TaskItemView, NonceItemView
logger = logging.getLogger(__name__)

from worker_plan_api.planexe_dotenv import DotEnvKeyEnum, PlanExeDotEnv
from worker_plan_api.planexe_config import PlanExeConfig

RUN_DIR = "run"

SHOW_DEMO_PLAN = False

DEMO_FORM_RUN_PROMPT_UUIDS = [
    "0ad5ea63-cf38-4d10-a3f3-d51baa609abd",
    "00e1c738-a663-476a-b950-62785922f6f0",
    "3ca89453-e65b-4828-994f-dff0b679444a"
]

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

@dataclass
class Config:
    use_uuid_as_run_id: bool

CONFIG = Config(
    use_uuid_as_run_id=False,
)

class User(UserMixin):
    def __init__(self, user_id: str, is_admin: bool = False):
        self.id = str(user_id)
        self.is_admin = is_admin

class MyAdminIndexView(AdminIndexView):
    @expose('/')
    def index(self):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        return super(MyAdminIndexView, self).index()

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

        if self.app.config.get("SECRET_KEY") == "dev-secret-key":
            logger.warning("Using default Flask SECRET_KEY. Set PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY for production.")

        self.public_base_url = (os.environ.get("PLANEXE_PUBLIC_BASE_URL") or "").rstrip("/")
        if not self.public_base_url:
            logger.warning("PLANEXE_PUBLIC_BASE_URL not set; OAuth redirects will use request.host.")

        self.oauth = OAuth(self.app)
        self._register_oauth_providers()

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
        
        def _ensure_taskitem_artifact_columns() -> None:
            insp = inspect(self.db.engine)
            columns = {col["name"] for col in insp.get_columns("task_item")}
            with self.db.engine.begin() as conn:
                if "generated_report_html" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS generated_report_html TEXT"))
                if "run_zip_snapshot" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS run_zip_snapshot BYTEA"))
                if "stop_requested" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested BOOLEAN"))
                if "stop_requested_timestamp" not in columns:
                    conn.execute(text("ALTER TABLE task_item ADD COLUMN IF NOT EXISTS stop_requested_timestamp TIMESTAMP"))

        def _seed_initial_records() -> None:
            # Add initial records if the table is empty
            if TaskItem.query.count() == 0:
                tasks = TaskItem.demo_items()
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
                        _ensure_taskitem_artifact_columns()
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
        self.login_manager.login_view = 'login'
        
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
        self.admin.add_view(TaskItemView(model=TaskItem, session=self.db.session, name="Task"))
        self.admin.add_view(ModelView(model=EventItem, session=self.db.session, name="Event"))
        self.admin.add_view(WorkerItemView(model=WorkerItem, session=self.db.session, name="Worker"))
        self.admin.add_view(NonceItemView(model=NonceItem, session=self.db.session, name="Nonce"))
        self.admin.add_view(ModelView(model=UserAccount, session=self.db.session, name="User"))
        self.admin.add_view(ModelView(model=UserProvider, session=self.db.session, name="User Provider"))
        self.admin.add_view(ModelView(model=UserApiKey, session=self.db.session, name="User API Key"))
        self.admin.add_view(ModelView(model=CreditHistory, session=self.db.session, name="Credit History"))
        self.admin.add_view(ModelView(model=PaymentRecord, session=self.db.session, name="Payments"))

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
            event = EventItem(
                event_type=EventType.GENERIC_EVENT,
                message="Flask app started",
                context=event_context
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
            self.oauth.register(name=name, **config)
            self.oauth_providers.append(name)

        if not self.oauth_providers:
            logger.warning("No OAuth providers configured. Set PLANEXE_OAUTH_* env vars to enable OAuth login.")

    def _oauth_redirect_url(self, provider: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/auth/{provider}/callback"
        return url_for("oauth_callback", provider=provider, _external=True)

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

    def _upsert_user_from_oauth(self, provider: str, profile: dict[str, Any]) -> UserAccount:
        provider_user_id = str(profile.get("sub") or profile.get("id") or "")
        if not provider_user_id:
            raise ValueError("OAuth profile missing provider user id.")

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
                self._update_user_from_profile(user, profile)
                self.db.session.commit()
                return user

        user = UserAccount(
            email=profile.get("email"),
            name=profile.get("name") or profile.get("username") or profile.get("login"),
            given_name=profile.get("given_name"),
            family_name=profile.get("family_name"),
            locale=profile.get("locale"),
            avatar_url=profile.get("picture") or profile.get("avatar_url") or profile.get("avatar"),
            last_login_at=now,
        )
        self.db.session.add(user)
        self.db.session.commit()

        provider_row = UserProvider(
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

    def _update_user_from_profile(self, user: UserAccount, profile: dict[str, Any]) -> None:
        user.email = profile.get("email") or user.email
        user.name = profile.get("name") or profile.get("username") or profile.get("login") or user.name
        user.given_name = profile.get("given_name") or user.given_name
        user.family_name = profile.get("family_name") or user.family_name
        user.locale = profile.get("locale") or user.locale
        user.avatar_url = profile.get("picture") or profile.get("avatar_url") or profile.get("avatar") or user.avatar_url

    def _get_or_create_api_key(self, user: UserAccount) -> str:
        api_key_secret = os.environ.get("PLANEXE_API_KEY_SECRET", "dev-api-key-secret")
        if api_key_secret == "dev-api-key-secret":
            logger.warning("PLANEXE_API_KEY_SECRET not set. Using dev secret for API key hashing.")

        existing_key = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).first()
        if existing_key:
            return ""

        raw_key = f"pex_{secrets.token_urlsafe(24)}"
        key_hash = hashlib.sha256(f"{api_key_secret}:{raw_key}".encode("utf-8")).hexdigest()
        key_prefix = raw_key[:10]
        api_key = UserApiKey(
            user_id=user.id,
            key_hash=key_hash,
            key_prefix=key_prefix,
        )
        self.db.session.add(api_key)
        self.db.session.commit()
        return raw_key

    def _apply_credit_delta(self, user: UserAccount, delta: int, reason: str, source: str, external_id: Optional[str] = None) -> None:
        user.credits_balance = max(0, (user.credits_balance or 0) + delta)
        ledger = CreditHistory(
            user_id=user.id,
            delta=delta,
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
        credits: int,
        amount: int,
        currency: str,
        raw_payload: dict[str, Any],
    ) -> None:
        try:
            user_uuid = uuid.UUID(str(user_id))
        except ValueError:
            logger.error("Invalid user_id in payment payload: %s", user_id)
            return
        with self.app.app_context():
            user = self.db.session.get(UserAccount, user_uuid)
            if not user:
                logger.error("Payment user not found: %s", user_id)
                return
            existing = PaymentRecord.query.filter_by(
                provider=provider,
                provider_payment_id=provider_payment_id,
            ).first()
            if existing:
                return
            payment = PaymentRecord(
                user_id=user.id,
                provider=provider,
                provider_payment_id=provider_payment_id,
                credits=credits,
                amount=amount,
                currency=currency,
                status="completed",
                raw_payload=raw_payload,
            )
            self.db.session.add(payment)
            self.db.session.commit()
            self._apply_credit_delta(
                user,
                delta=credits,
                reason="credits_purchased",
                source=provider,
                external_id=provider_payment_id,
            )

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/healthcheck')
        def healthcheck():
            response_payload = {"status": "ok", "database_target": self.database_settings}
            try:
                self.db.session.execute(text("SELECT 1"))
                response_payload["database"] = "ok"
                status_code = 200
            except Exception as exc:
                logger.error("Health check failed", exc_info=True)
                response_payload["status"] = "error"
                response_payload["database"] = f"error: {exc.__class__.__name__}"
                status_code = 500
            return jsonify(response_payload), status_code

        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                if username == self.admin_username and password == self.admin_password:
                    user = User(self.admin_username, is_admin=True)
                    login_user(user)
                    return redirect(url_for('admin.index'))
                return 'Invalid credentials', 401
            return render_template('login.html', oauth_providers=self.oauth_providers)

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
            login_user(User(user.id, is_admin=user.is_admin))
            new_api_key = self._get_or_create_api_key(user)
            if new_api_key:
                session["new_api_key"] = new_api_key
            return redirect(url_for('account'))

        @self.app.route('/logout')
        @login_required
        def logout():
            logout_user()
            return redirect(url_for('index'))

        @self.app.route('/account', methods=['GET', 'POST'])
        @login_required
        def account():
            if current_user.is_admin:
                return redirect(url_for('admin.index'))
            user_uuid = uuid.UUID(str(current_user.id))
            user = self.db.session.get(UserAccount, user_uuid)
            if not user:
                return redirect(url_for('logout'))

            new_api_key = session.pop("new_api_key", None)
            if request.method == 'POST':
                action = request.form.get('action')
                if action == "regenerate_api_key":
                    existing_keys = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).all()
                    now = datetime.now(UTC)
                    for key in existing_keys:
                        key.revoked_at = now
                    self.db.session.commit()
                    new_api_key = self._get_or_create_api_key(user)
                return redirect(url_for('account'))

            active_key = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).first()
            return render_template(
                'account.html',
                user=user,
                active_key=active_key,
                new_api_key=new_api_key,
                stripe_enabled=bool(os.environ.get("PLANEXE_STRIPE_SECRET_KEY")),
                telegram_enabled=bool(os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")),
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
            success_url = f"{self.public_base_url}/account?stripe=success" if self.public_base_url else url_for("account", _external=True)
            cancel_url = f"{self.public_base_url}/account?stripe=cancel" if self.public_base_url else url_for("account", _external=True)
            session_obj = stripe.checkout.Session.create(
                mode="payment",
                success_url=success_url,
                cancel_url=cancel_url,
                line_items=[{
                    "price_data": {
                        "currency": os.environ.get("PLANEXE_STRIPE_CURRENCY", "usd"),
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
            return redirect(session_obj.url)

        @self.app.route('/billing/stripe/webhook', methods=['POST'])
        def stripe_webhook():
            stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
            webhook_secret = os.environ.get("PLANEXE_STRIPE_WEBHOOK_SECRET")
            if not stripe_secret:
                return jsonify({"error": "Stripe not configured"}), 400
            stripe.api_key = stripe_secret
            payload = request.get_data()
            sig_header = request.headers.get("Stripe-Signature")
            try:
                if webhook_secret:
                    event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
                else:
                    event = json.loads(payload)
            except Exception as exc:
                logger.error("Stripe webhook error: %s", exc)
                return jsonify({"error": "invalid payload"}), 400

            if event.get("type") == "checkout.session.completed":
                session_obj = event["data"]["object"]
                metadata = session_obj.get("metadata") or {}
                user_id = metadata.get("user_id")
                credits = int(metadata.get("credits", "0") or 0)
                if user_id and credits > 0:
                    self._apply_payment_credits(
                        user_id=user_id,
                        provider="stripe",
                        provider_payment_id=session_obj.get("id", ""),
                        credits=credits,
                        amount=session_obj.get("amount_total") or 0,
                        currency=session_obj.get("currency") or "usd",
                        raw_payload=session_obj,
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
                    credits_int = int(credits)
                except Exception:
                    return jsonify({"status": "ignored"})
                self._apply_payment_credits(
                    user_id=user_id,
                    provider="telegram",
                    provider_payment_id=payment.get("telegram_payment_charge_id", ""),
                    credits=credits_int,
                    amount=payment.get("total_amount") or 0,
                    currency=payment.get("currency") or "XTR",
                    raw_payload=payment,
                )
            return jsonify({"status": "ok"})

        @self.app.route('/ping')
        @login_required
        def ping():
            return render_template('ping.html')

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
            parameters = {key: value for key, value in request_form_or_args.items()}

            # Remove the parameters that have already been extracted from the parameters dictionary
            parameters.pop('prompt', None)
            parameters.pop('user_id', None)
            parameters.pop('nonce', None)
            if len(parameters) == 0:
                parameters = None

            # Get length of prompt_param in bytes and in characters
            prompt_param_bytes = len(prompt_param.encode('utf-8'))
            prompt_param_characters = len(prompt_param)

            # Avoid flooding logs when the prompt is long.
            log_prompt_info = prompt_param[:100]
            if len(prompt_param) > 100:
                log_prompt_info += "... (truncated)"
            logger.info(f"endpoint /run ({request.method}). Size of request: {request_size_bytes} bytes. Starting run with parameters: prompt={log_prompt_info!r}, user_id={user_id_param!r}, nonce={nonce_param!r}, parameters={parameters!r}, prompt_param_bytes={prompt_param_bytes}, prompt_param_characters={prompt_param_characters}")

            if current_user.is_admin:
                user_id_param = self.admin_username
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
                    else:
                        if (user.credits_balance or 0) <= 0:
                            return jsonify({"error": "No credits available"}), 402
                        self._apply_credit_delta(user, -1, reason="plan_created", source="web")

                task = TaskItem(
                    state=TaskState.pending,
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
                event = EventItem(
                    event_type=EventType.TASK_PENDING,
                    message=f"Enqueued task via /run endpoint",
                    context=event_context
                )
                self.db.session.add(event)
                self.db.session.commit()
            return render_template('run_via_database.html', run_id=task_id)

        @self.app.route('/progress')
        def get_progress():
            run_id = request.args.get('run_id', '')
            logger.debug(f"Progress endpoint received run_id: {run_id!r}")
            # lookup the task in the database
            task = self.db.session.get(TaskItem, run_id)
            if task is None:
                logger.error(f"Task not found for run_id: {run_id!r}")
                return jsonify({"error": "Task not found"}), 400
            
            progress_percentage = float(task.progress_percentage) if task.progress_percentage is not None else 0.0
            progress_message = task.progress_message if task.progress_message is not None else ""
            if isinstance(task.state, TaskState):
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
        def viewplan():
            run_id = request.args.get('run_id', '')
            logger.info(f"ViewPlan endpoint requested for run_id: {run_id!r}")
            # lookup the task in the database
            task = self.db.session.get(TaskItem, run_id)
            if task is None:
                logger.error(f"Task not found for run_id: {run_id!r}")
                return jsonify({"error": "Task not found"}), 400

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

        @self.app.route('/admin/task/<uuid:task_id>/report')
        @login_required
        def download_task_report(task_id):
            task = self.db.session.get(TaskItem, task_id)
            if task is None or not task.generated_report_html:
                return "Report not found", 404
            buffer = io.BytesIO(task.generated_report_html.encode('utf-8'))
            buffer.seek(0)
            return send_file(buffer, mimetype='text/html', as_attachment=True, download_name='report.html')

        @self.app.route('/admin/task/<uuid:task_id>/run_zip')
        @login_required
        def download_task_run_zip(task_id):
            task = self.db.session.get(TaskItem, task_id)
            if task is None or not task.run_zip_snapshot:
                return "Run zip not found", 404
            buffer = io.BytesIO(task.run_zip_snapshot)
            buffer.seek(0)
            download_name = f"{task_id}.zip"
            return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name=download_name)

        @self.app.route('/demo_run')
        @login_required
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

            return render_template('demo_run.html', user_id=user_id, prompts=prompts, nonce=nonce)

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
    flask_app_instance = MyFlaskApp()
    flask_app_instance.run_server()

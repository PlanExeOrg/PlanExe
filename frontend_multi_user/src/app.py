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
from urllib.parse import quote_plus
from typing import ClassVar, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_file, redirect, url_for
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from functools import wraps
import urllib.request
from urllib.error import URLError
from flask import make_response
import requests
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
from planexe_modelviews import WorkerItemView, TaskItemView, NonceItemView
logger = logging.getLogger(__name__)

from worker_plan_internal.utils.planexe_dotenv import DotEnvKeyEnum, PlanExeDotEnv
from worker_plan_internal.utils.planexe_config import PlanExeConfig

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
    def __init__(self, user_id):
        self.id = user_id

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
                SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key"),
                SQLALCHEMY_TRACK_MODIFICATIONS=False,
            )

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
                return User(user_id)
            return None
        
        # Setup Flask-Admin
        # Flask-Admin versions bundled in the image don't accept template_mode; stick with defaults.
        self.admin = Admin(self.app, name='PlanExe Admin', index_view=MyAdminIndexView())
        
        # Add database tables to admin panel
        self.admin.add_view(TaskItemView(model=TaskItem, session=self.db.session, name="Task"))
        self.admin.add_view(ModelView(model=EventItem, session=self.db.session, name="Event"))
        self.admin.add_view(WorkerItemView(model=WorkerItem, session=self.db.session, name="Worker"))
        self.admin.add_view(NonceItemView(model=NonceItem, session=self.db.session, name="Nonce"))

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
                    user = User(self.admin_username)
                    login_user(user)
                    return redirect(url_for('admin.index'))
                return 'Invalid credentials', 401
            return render_template('login.html')

        @self.app.route('/logout')
        @login_required
        def logout():
            logout_user()
            return redirect(url_for('index'))

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
            
            if not user_id_param:
                logger.error(f"endpoint /run. No user_id provided")
                return jsonify({"error": "No user_id provided"}), 400

            with self.app.app_context():
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
            user_id = 'USERIDPLACEHOLDER'
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

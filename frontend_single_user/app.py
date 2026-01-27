"""
Start the UI in single user mode.
PROMPT> python frontend_single_user/app.py
"""
from dataclasses import dataclass
from dotenv import load_dotenv
from math import ceil
from pathlib import Path
from typing import Optional
import gradio as gr
import httpx
import json
import logging
import os
import sys
import tempfile
import threading
import time

# Load environment variables from .env file (if it exists)
load_dotenv()
from worker_plan_api.llm_info import LLMInfo, OllamaStatus
from worker_plan_api.generate_run_id import RUN_ID_PREFIX
from worker_plan_api.speedvsdetail import SpeedVsDetailEnum
from worker_plan_api.prompt_catalog import PromptCatalog

logger = logging.getLogger(__name__)
log_level_name = os.environ.get("PLANEXE_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)

@dataclass
class Config:
    use_uuid_as_run_id: bool
    visible_top_header: bool
    visible_open_output_dir_button: bool
    visible_llm_info: bool
    visible_openrouter_api_key_textbox: bool
    allow_only_openrouter_models: bool
    run_planner_check_api_key_is_provided: bool
    browser_state_secret: str

CONFIG_LOCAL = Config(
    use_uuid_as_run_id=False,
    visible_top_header=True,
    visible_open_output_dir_button=True,
    visible_openrouter_api_key_textbox=False,
    visible_llm_info=True,
    allow_only_openrouter_models=False,
    run_planner_check_api_key_is_provided=False,
    browser_state_secret="insert-your-secret-here",
)
CONFIG = CONFIG_LOCAL

DEFAULT_PROMPT_UUID = "4dc34d55-0d0d-4e9d-92f4-23765f49dd29"

# Global constant for the zip creation interval (in seconds)
ZIP_INTERVAL_SECONDS = 10

WORKER_PLAN_URL = os.environ.get("PLANEXE_WORKER_PLAN_URL", "http://worker_plan:8000")
WORKER_PLAN_TIMEOUT_SECONDS = float(os.environ.get("PLANEXE_WORKER_PLAN_TIMEOUT", "30"))
GRADIO_SERVER_NAME = os.environ.get("PLANEXE_GRADIO_SERVER_NAME", "0.0.0.0")
# Railway and other PaaS set PORT for the ingress; fall back to the app-specific env var.
GRADIO_SERVER_PORT = int(os.environ.get("PORT") or os.environ.get("PLANEXE_GRADIO_SERVER_PORT", "7860"))
OPEN_DIR_SERVER_URL = os.environ.get("PLANEXE_OPEN_DIR_SERVER_URL")
GRADIO_AUTH_PASSWORD = os.environ.get("PLANEXE_PASSWORD")
GRADIO_AUTH = ("user", GRADIO_AUTH_PASSWORD) if GRADIO_AUTH_PASSWORD else None
OPEN_DIR_BUTTON_INITIAL_VISIBILITY = CONFIG.visible_open_output_dir_button and bool(OPEN_DIR_SERVER_URL)

# Load prompt catalog and examples.
prompt_catalog = PromptCatalog()
prompt_catalog.load_simple_plan_prompts()

# Prefill the input box with the default prompt
default_prompt_item = prompt_catalog.find(DEFAULT_PROMPT_UUID)
if default_prompt_item:
    gradio_default_example = default_prompt_item.prompt
else:
    raise ValueError("DEFAULT_PROMPT_UUID prompt not found.")

# Show all prompts in the catalog as examples
all_prompts = prompt_catalog.all()
gradio_examples = []
for prompt_item in all_prompts:
    gradio_examples.append([prompt_item.prompt])

def fetch_run_files(run_id: str) -> tuple[Optional[list[str]], Optional[str]]:
    """
    Fetch the current list of output files for a run from worker_plan.
    Returns a tuple of (files, error_message).
    """
    if not run_id:
        return None, "No run_id available yet."
    try:
        response = worker_client.list_run_files(run_id)
        return response.get("files", []), None
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        if status_code == 404:
            return None, "Output directory not available yet."
        return None, f"Unable to fetch files (status {status_code})."
    except Exception as exc:
        return None, f"Unable to fetch files: {exc}"


def fetch_run_zip(run_id: str, existing_zip_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch a fresh zip from worker_plan, replace the existing temp zip if any,
    and return the local path plus an optional error message.
    """
    try:
        zip_bytes = worker_client.download_run_zip(run_id)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        return existing_zip_path, f"Failed to create zip (status {status_code})."
    except Exception as exc:
        return existing_zip_path, f"Failed to create zip: {exc}"

    try:
        with tempfile.NamedTemporaryFile(delete=False, prefix=f"{run_id}_", suffix=".zip") as tmp_file:
            tmp_file.write(zip_bytes)
            tmp_path = tmp_file.name
    except Exception as exc:
        return existing_zip_path, f"Unable to save zip: {exc}"

    if existing_zip_path and os.path.exists(existing_zip_path):
        try:
            os.remove(existing_zip_path)
        except Exception:
            pass

    return tmp_path, None


def is_open_dir_service_running(timeout_seconds: float = 3.0) -> bool:
    """
    Checks if the optional host opener service is reachable.
    """
    if not CONFIG.visible_open_output_dir_button:
        return False
    if not OPEN_DIR_SERVER_URL:
        logger.info("Open dir button hidden: PLANEXE_OPEN_DIR_SERVER_URL not set.")
        return False

    health_url = f"{OPEN_DIR_SERVER_URL.rstrip('/')}/healthcheck"
    try:
        response = httpx.get(health_url, timeout=timeout_seconds)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.info("Open dir button hidden: opener service not reachable at %s (%s)", health_url, exc)
        return False


class WorkerClient:
    def __init__(self, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    def start_run(self, payload: dict) -> dict:
        response = self.client.post("/runs", json=payload)
        response.raise_for_status()
        return response.json()

    def stop_run(self, run_id: str) -> dict:
        response = self.client.post(f"/runs/{run_id}/stop")
        response.raise_for_status()
        return response.json()

    def get_llm_info(self) -> LLMInfo:
        response = self.client.get("/llm-info")
        response.raise_for_status()
        return LLMInfo.model_validate(response.json())

    def get_status(self, run_id: str) -> dict:
        response = self.client.get(f"/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    def list_run_files(self, run_id: str) -> dict:
        response = self.client.get(f"/runs/{run_id}/files")
        response.raise_for_status()
        return response.json()

    def download_run_zip(self, run_id: str) -> bytes:
        response = self.client.get(f"/runs/{run_id}/zip")
        response.raise_for_status()
        return response.content

    def purge_runs(self, max_age_hours: Optional[float] = None, prefix: Optional[str] = None) -> dict:
        payload = {}
        if max_age_hours is not None:
            payload["max_age_hours"] = max_age_hours
        if prefix is not None:
            payload["prefix"] = prefix
        response = self.client.post("/purge-runs", json=payload)
        response.raise_for_status()
        return response.json()


worker_client = WorkerClient(WORKER_PLAN_URL, WORKER_PLAN_TIMEOUT_SECONDS)

def fetch_llm_info_with_retry(max_attempts: int = 15, delay_seconds: float = 2.0) -> LLMInfo:
    """
    Try to fetch LLM info with retries so the UI doesn't crash if the worker
    isn't ready yet (e.g., cold start or delayed boot).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return worker_client.get_llm_info()
        except Exception as exc:
            logger.warning(
                "Failed to reach worker at %s (attempt %s/%s): %s",
                WORKER_PLAN_URL,
                attempt,
                max_attempts,
                exc,
            )
            if attempt == max_attempts:
                raise
            time.sleep(delay_seconds)

llm_info: LLMInfo = fetch_llm_info_with_retry()
logger.info(f"LLMInfo.ollama_status: {llm_info.ollama_status.value}")
logger.info(f"LLMInfo.error_message_list: {llm_info.error_message_list}")

trimmed_llm_config_items = []
if CONFIG.allow_only_openrouter_models:
    trimmed_llm_config_items = [item for item in llm_info.llm_config_items if item.id.startswith("openrouter")]
else:
    trimmed_llm_config_items = llm_info.llm_config_items

# Create tuples for the Gradio Radio buttons.
available_model_names = []
default_model_value = None
for config_index, config_item in enumerate(trimmed_llm_config_items):
    if config_index == 0:
        default_model_value = config_item.id
    tuple_item = (config_item.label, config_item.id)
    available_model_names.append(tuple_item)

class MarkdownBuilder:
    """
    Helper class to build Markdown-formatted strings.
    """
    def __init__(self):
        self.rows = []

    def add_line(self, line: str):
        self.rows.append(line)

    def add_code_block(self, code: str):
        self.rows.append("```\n" + code + "\n```")

    def status(self, status_message: str):
        self.add_line("### Status")
        self.add_line(status_message)

    def path_to_run_dir(self, absolute_path_to_run_dir: str):
        self.add_line("### Output dir")
        self.add_code_block(absolute_path_to_run_dir)

    def list_files(self, files: Optional[list[str]], error_message: Optional[str] = None):
        self.add_line("### Output files")
        if error_message:
            self.add_code_block(error_message)
            return
        if files is None:
            self.add_code_block("Output directory not available yet.")
            return
        if len(files) == 0:
            self.add_code_block("No files found.")
            return
        filenames = "\n".join(files)
        self.add_code_block(filenames)

    def to_markdown(self):
        return "\n".join(self.rows)

class SessionState:
    """
    In a multi-user environment (e.g. Hugging Face Spaces), this class hold each users state.
    In a single-user environment, this class is used to hold the state of that lonely user.
    """
    def __init__(self):
        # Settings: the user's OpenRouter API key.
        self.openrouter_api_key = "" # Initialize to empty string
        # Settings: The model that the user has picked.
        self.llm_model = default_model_value
        # Settings: The speedvsdetail that the user has picked.
        self.speedvsdetail = SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW
        # The run id of the currently running pipeline process (managed by worker service).
        self.active_run_id: Optional[str] = None
        # A threading.Event used to signal that the running process should stop.
        self.stop_event = threading.Event()
        # Stores the unique identifier of the last submitted run.
        self.latest_run_id = None
        # Paths reported by the worker for the last submitted run.
        self.latest_run_dir = None
        self.latest_run_dir_display = None

    def __deepcopy__(self, memo):
        """
        Override deepcopy so that the SessionState instance is not actually copied.
        This avoids trying to copy unpickleable objects (like threading locks) and
        ensures the same instance is passed along between Gradio callbacks.
        """
        return self

def initialize_browser_settings(browser_state, session_state: SessionState):
    try:
        settings = json.loads(browser_state) if browser_state else {}
    except Exception:
        settings = {}
    openrouter_api_key = settings.get("openrouter_api_key_text", "")
    model = settings.get("model_radio", default_model_value)
    speedvsdetail = settings.get("speedvsdetail_radio", SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW)

    # When making changes to the llm_config.json, it may happen that the selected model is no longer among the available_model_names.
    # In that case, set the model to the default_model_value.
    if model not in [item[1] for item in available_model_names]:
        logger.info(f"initialize_browser_settings: model '{model}' is not in available_model_names. Setting to default_model_value: {default_model_value}")
        model = default_model_value

    session_state.openrouter_api_key = openrouter_api_key
    session_state.llm_model = model
    session_state.speedvsdetail = speedvsdetail
    return openrouter_api_key, model, speedvsdetail, browser_state, session_state

def update_browser_settings_callback(openrouter_api_key, model, speedvsdetail, browser_state, session_state: SessionState):
    try:
        settings = json.loads(browser_state) if browser_state else {}
    except Exception:
        settings = {}
    settings["openrouter_api_key_text"] = openrouter_api_key
    settings["model_radio"] = model
    settings["speedvsdetail_radio"] = speedvsdetail
    updated_browser_state = json.dumps(settings)
    session_state.openrouter_api_key = openrouter_api_key
    session_state.llm_model = model
    session_state.speedvsdetail = speedvsdetail
    return updated_browser_state, openrouter_api_key, model, speedvsdetail, session_state

def run_planner(submit_or_retry_button, plan_prompt, browser_state, session_state: SessionState):
    """
    Generator function for launching the pipeline process and streaming updates.
    The session state is carried in a SessionState instance.
    """

    # Sync persistent settings from BrowserState into session_state
    try:
        settings = json.loads(browser_state) if browser_state else {}
    except Exception:
        settings = {}
    session_state.openrouter_api_key = settings.get("openrouter_api_key_text", session_state.openrouter_api_key)
    session_state.llm_model = settings.get("model_radio", session_state.llm_model)
    session_state.speedvsdetail = settings.get("speedvsdetail_radio", session_state.speedvsdetail)

    # Check if an OpenRouter API key is required and provided.
    if CONFIG.run_planner_check_api_key_is_provided:
        if session_state.openrouter_api_key is None or len(session_state.openrouter_api_key) == 0:
            raise ValueError("An OpenRouter API key is required to use PlanExe. Please provide an API key in the Settings tab.")

    # Clear any previous stop signal.
    session_state.stop_event.clear()

    submit_or_retry = submit_or_retry_button.lower()
    run_id = None
    run_dir = None
    display_run_dir = None

    if submit_or_retry == "retry":
        if not session_state.latest_run_id:
            raise ValueError("No previous run to retry. Please submit a plan first.")
        run_id = session_state.latest_run_id
        run_dir = session_state.latest_run_dir
        display_run_dir = session_state.latest_run_dir_display
        print(f"Retrying the run with ID: {run_id}")

    # Create a SpeedVsDetailEnum instance from the session_state.speedvsdetail.
    # Sporadic I have experienced that session_state.speedvsdetail is a string and other times it's a SpeedVsDetailEnum.
    speedvsdetail = session_state.speedvsdetail
    speedvsdetail_string = SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW.value
    if isinstance(speedvsdetail, str):
        speedvsdetail_string = speedvsdetail
    elif isinstance(speedvsdetail, SpeedVsDetailEnum):
        speedvsdetail_string = speedvsdetail.value

    payload = {
        "submit_or_retry": submit_or_retry,
        "plan_prompt": plan_prompt,
        "llm_model": session_state.llm_model,
        "speed_vs_detail": speedvsdetail_string,
        "openrouter_api_key": session_state.openrouter_api_key or None,
        "use_uuid_as_run_id": CONFIG.use_uuid_as_run_id,
    }
    if run_id:
        payload["run_id"] = run_id

    try:
        start_response = worker_client.start_run(payload)
    except httpx.HTTPError as exc:
        raise ValueError(f"Failed to contact worker_plan service: {exc}") from exc

    run_id = start_response.get("run_id", run_id)
    if not run_id:
        raise ValueError("Worker did not return a run_id.")

    run_dir = start_response.get("run_dir", run_dir)
    display_run_dir = start_response.get("display_run_dir") or run_dir
    session_state.latest_run_id = run_id
    session_state.latest_run_dir = run_dir
    session_state.latest_run_dir_display = display_run_dir
    session_state.active_run_id = run_id
    worker_pid = start_response.get("pid")
    print(f"Process started on worker. Run ID: {run_id}. PID: {worker_pid}")

    start_time = time.perf_counter()
    # Initialize the last zip creation time to be ZIP_INTERVAL_SECONDS in the past
    last_zip_time = time.time() - ZIP_INTERVAL_SECONDS
    current_zip_path: Optional[str] = None
    zip_error: Optional[str] = None

    # Poll the output directory every second.
    status_response = None
    pipeline_complete = False
    while True:
        try:
            status_response = worker_client.get_status(run_id)
        except httpx.HTTPError as exc:
            logger.warning(f"Failed to fetch status for run_id={run_id}: {exc}")
            status_response = None

        if status_response:
            run_dir = status_response.get("run_dir", run_dir)
            display_from_status = status_response.get("display_run_dir")
            if display_from_status:
                display_run_dir = display_from_status
            session_state.latest_run_dir = run_dir
            session_state.latest_run_dir_display = display_run_dir

        run_dir_display_text = display_run_dir or run_dir or "Unavailable"

        pipeline_complete = status_response.get("pipeline_complete", False) if status_response else False
        running = status_response.get("running", True) if status_response else True
        files, files_error = fetch_run_files(run_id)

        # print("running...")
        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))

        # If a stop has been requested, terminate the process.
        if session_state.stop_event.is_set():
            try:
                worker_client.stop_run(run_id)
            except Exception as e:
                print("Error terminating process:", e)

            markdown_builder = MarkdownBuilder()
            markdown_builder.status("Process terminated by user.")
            markdown_builder.path_to_run_dir(run_dir_display_text)
            markdown_builder.list_files(files, files_error)
            yield markdown_builder.to_markdown(), gr.update(value=current_zip_path), session_state
            break

        last_update_raw = status_response.get("last_update_seconds_ago") if status_response else None
        last_update = ceil(last_update_raw) if last_update_raw is not None else 0
        markdown_builder = MarkdownBuilder()
        if running or pipeline_complete:
            markdown_builder.status(f"Working. {duration} seconds elapsed. Last output update was {last_update} seconds ago.")
        else:
            markdown_builder.status(f"Process inactive. {duration} seconds elapsed. Last output update was {last_update} seconds ago.")
        markdown_builder.path_to_run_dir(run_dir_display_text)
        markdown_builder.list_files(files, files_error)

        # Periodic zip refresh.
        current_time = time.time()
        if current_time - last_zip_time >= ZIP_INTERVAL_SECONDS:
            current_zip_path, zip_error = fetch_run_zip(run_id, current_zip_path)
            last_zip_time = current_time
            if zip_error:
                markdown_builder.add_line(f"Zip status: {zip_error}")

        yield markdown_builder.to_markdown(), gr.update(value=current_zip_path), session_state

        # If the pipeline complete file is found, finish streaming.
        if pipeline_complete:
            break

        if not running:
            break

        time.sleep(1)
    
    session_state.active_run_id = None

    # Fetch latest status for final message.
    try:
        status_response = worker_client.get_status(run_id)
    except httpx.HTTPError:
        status_response = None
    if status_response:
        run_dir = status_response.get("run_dir", run_dir)
        display_from_status = status_response.get("display_run_dir")
        if display_from_status:
            display_run_dir = display_from_status
        session_state.latest_run_dir = run_dir
        session_state.latest_run_dir_display = display_run_dir

    run_dir_display_text = display_run_dir or run_dir or "Unavailable"

    returncode = status_response.get("returncode") if status_response else None
    pipeline_complete = status_response.get("pipeline_complete", pipeline_complete) if status_response else pipeline_complete

    # Process has completed.
    end_time = time.perf_counter()
    duration = int(ceil(end_time - start_time))
    print(f"Process ended. returncode: {returncode}. Run ID: {run_id}. Duration: {duration} seconds.")

    if pipeline_complete:
        status_message = "Completed."
    else:
        status_message = "Stopped prematurely, the output may be incomplete."

    # Final file listing update.
    markdown_builder = MarkdownBuilder()
    markdown_builder.status(f"{status_message} {duration} seconds elapsed.")
    markdown_builder.path_to_run_dir(run_dir_display_text)
    files, files_error = fetch_run_files(run_id)
    markdown_builder.list_files(files, files_error)

    # Final zip refresh.
    current_zip_path, zip_error = fetch_run_zip(run_id, current_zip_path)
    if zip_error:
        markdown_builder.add_line(f"Zip status: {zip_error}")

    yield markdown_builder.to_markdown(), gr.update(value=current_zip_path), session_state

def stop_planner(session_state: SessionState):
    """
    Sets a stop flag in the session_state and attempts to terminate the active process.
    """

    session_state.stop_event.set()

    active_run_id = session_state.active_run_id
    if not active_run_id:
        msg = "No active process to stop."
        return msg, session_state

    try:
        worker_client.stop_run(active_run_id)
        msg = "Stop signal sent. Process termination requested."
    except Exception as e:
        msg = f"Error terminating process: {e}"

    return msg, session_state


def clear_status(session_state: SessionState):
    """
    Clears the status message area when starting a new action.
    """
    return "", session_state


def open_output_dir(session_state: SessionState):
    """
    Presents a host-visible path (and clickable link) to the latest output directory.
    If a host opener service is configured, it requests the host to open the path; otherwise it shows manual instructions.
    """

    if not session_state.latest_run_id:
        return "No plan has been submitted, cannot open dir.", session_state

    try:
        status_response = worker_client.get_status(session_state.latest_run_id)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        return f"Unable to fetch run info (status {status_code}).", session_state
    except Exception as exc:
        return f"Unable to contact worker: {exc}", session_state

    run_dir = status_response.get("run_dir")
    display_run_dir = status_response.get("display_run_dir") or run_dir
    run_dir_exists = status_response.get("run_dir_exists", True)

    session_state.latest_run_dir = run_dir
    session_state.latest_run_dir_display = display_run_dir

    if not run_dir_exists:
        return "No output directory available.", session_state

    open_path = display_run_dir or run_dir
    if not open_path:
        return "Run directory is unavailable.", session_state

    parts = []
    opener_succeeded = False

    # Attempt to ask the host opener service (running outside Docker) to open the path.
    if OPEN_DIR_SERVER_URL:
        try:
            response = httpx.post(f"{OPEN_DIR_SERVER_URL.rstrip('/')}/open", json={"path": open_path})
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", "Requested host to open directory.")
            parts.append(msg)
            opener_succeeded = True
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            error_detail = None
            try:
                error_payload = exc.response.json()
                error_detail = error_payload.get("message") or error_payload.get("detail")
            except Exception:
                if exc.response is not None:
                    error_detail = exc.response.text.strip() or None
            detail_msg = f" ({error_detail})" if error_detail else ""
            parts.append(f"Host opener error (status {status_code}){detail_msg}.")
        except Exception as exc:
            parts.append(f"Failed to contact host opener: {exc}")
    else:
        parts.append("Host opener service not configured (set PLANEXE_OPEN_DIR_SERVER_URL).")

    # Include manual instructions only when the opener is not available or failed.
    if not opener_succeeded:
        try:
            file_uri = Path(open_path).absolute().as_uri()
        except Exception:
            file_uri = None

        link_part = f"[{open_path}]({file_uri})" if file_uri else open_path
        parts.append(f"Open manually: {link_part}")

    return "\n\n".join(parts), session_state


def update_open_dir_button_visibility():
    """
    Used by Gradio load event to hide/show the Open Output Dir button depending on opener availability.
    """
    return gr.update(visible=is_open_dir_service_running())


def trigger_purge_runs(max_age_hours, prefix, session_state: SessionState):
    """
    Calls the worker to purge old runs on demand.
    """
    try:
        response = worker_client.purge_runs(max_age_hours=max_age_hours, prefix=prefix or None)
        msg = response.get("message", "Purge requested.")
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        msg = f"Failed to purge runs (status {status_code})."
    except Exception as exc:
        msg = f"Failed to purge runs: {exc}"
    return msg, session_state


def check_api_key(session_state: SessionState):
    """Checks if the API key is provided and returns a warning if not."""
    if CONFIG.visible_openrouter_api_key_textbox and (not session_state.openrouter_api_key or len(session_state.openrouter_api_key) == 0):
        return "<div style='background-color: #FF7777; color: black; border: 1px solid red; padding: 10px;'>Welcome to PlanExe. Please provide an OpenRouter API key in the <b>Settings</b> tab to start using PlanExe.</div>"
    return "" # No warning

# Build the Gradio UI using Blocks.
with gr.Blocks(title="PlanExe") as demo_text2plan:
    gr.Markdown("# PlanExe: crack open pandora’s box of ideas", visible=CONFIG.visible_top_header)
    api_key_warning = gr.Markdown()
    with gr.Tab("Main"):
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                prompt_input = gr.Textbox(
                    label="Plan Description",
                    lines=5,
                    placeholder="Enter a description of your plan...",
                    value=gradio_default_example
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant='primary')
                    stop_btn = gr.Button("Stop")
                    retry_btn = gr.Button("Retry")
                    open_dir_btn = gr.Button("Open Output Dir", visible=OPEN_DIR_BUTTON_INITIAL_VISIBILITY)

                output_markdown = gr.Markdown("Output will appear here...")
                status_markdown = gr.Markdown("Status messages will appear here...")
                download_output = gr.File(label="Download latest output (excluding log.txt) as zip")

            with gr.Column(scale=1, min_width=300):
                examples = gr.Examples(
                    examples=gradio_examples,
                    inputs=[prompt_input],
                )

    with gr.Tab("Settings"):
        if CONFIG.visible_llm_info:
            if llm_info.ollama_status == OllamaStatus.ollama_not_running:
                gr.Markdown("**Ollama is not running**, so Ollama models are unavailable. Please start Ollama to use them.")
            elif llm_info.ollama_status == OllamaStatus.mixed:
                gr.Markdown("**Mixed. Some Ollama models are running, but some are NOT running.**, You may have to start the ones that aren't running.")

            if len(llm_info.error_message_list) > 0:
                gr.Markdown("**Error messages:**")
                for error_message in llm_info.error_message_list:
                    gr.Markdown(f"- {error_message}")

        model_radio = gr.Radio(
            available_model_names,
            value=default_model_value,
            label="Model",
            interactive=True 
        )

        speedvsdetail_items = [
            ("All details, but slow", SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW),
            ("Fast, but few details", SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS),
        ]
        speedvsdetail_radio = gr.Radio(
            speedvsdetail_items,
            value=SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW,
            label="Speed vs Detail",
            interactive=True 
        )
        openrouter_api_key_text = gr.Textbox(
            label="OpenRouter API Key",
            type="password",
            placeholder="Enter your OpenRouter API key (required)",
            info="Sign up at [OpenRouter](https://openrouter.ai/) to get an API key. A small top-up (e.g. 5 USD) is needed to access paid models.",
            visible=CONFIG.visible_openrouter_api_key_textbox
        )

    with gr.Tab("Advanced"):
        gr.Markdown("Trigger a manual purge of old `PlanExe_TIMESTAMP` dirs and `PlanExe_TIMESTAMP.zip` files from the run directory.")
        purge_max_age_hours = gr.Number(
            label="Max age (hours). Dirs and files older than this will be purged.",
            value=24,
            minimum=1,
            maximum=240,
            precision=2,
        )
        purge_prefix = gr.Textbox(
            label="Run prefix to purge. Dirs and files with this prefix will be purged.",
            value=RUN_ID_PREFIX,
            placeholder="Prefix to match, leave empty to use worker default",
        )
        purge_button = gr.Button("Purge old runs now")
        purge_status = gr.Markdown("")

    with gr.Tab("Join the community"):
        gr.Markdown("""
- [GitHub](https://github.com/PlanExeOrg/PlanExe) the source code.
- [Discord](https://planexe.org/discord) join the community. Suggestions, feedback, and questions are welcome.
""")
    
    # Manage the state of the current user
    session_state = gr.State(SessionState())
    browser_state = gr.BrowserState("", storage_key="PlanExeStorage1", secret=CONFIG.browser_state_secret)

    # Submit and Retry buttons call run_planner and update the state.
    submit_btn.click(
        fn=clear_status,
        inputs=session_state,
        outputs=[status_markdown, session_state]
    ).then(
        fn=run_planner,
        inputs=[submit_btn, prompt_input, browser_state, session_state],
        outputs=[output_markdown, download_output, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )
    retry_btn.click(
        fn=clear_status,
        inputs=session_state,
        outputs=[status_markdown, session_state]
    ).then(
        fn=run_planner,
        inputs=[retry_btn, prompt_input, browser_state, session_state],
        outputs=[output_markdown, download_output, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )
    # The Stop button uses the state to terminate the running process.
    stop_btn.click(
        fn=stop_planner,
        inputs=session_state,
        outputs=[status_markdown, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )
    # Open Output Dir button.
    open_dir_btn.click(
        fn=open_output_dir,
        inputs=session_state,
        outputs=[status_markdown, session_state]
    )
    # The download file value is updated by run_planner generator outputs.

    # Unified change callbacks for settings.
    openrouter_api_key_text.change(
        fn=update_browser_settings_callback,
        inputs=[openrouter_api_key_text, model_radio, speedvsdetail_radio, browser_state, session_state],
        outputs=[browser_state, openrouter_api_key_text, model_radio, speedvsdetail_radio, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )

    model_radio.change(
        fn=update_browser_settings_callback,
        inputs=[openrouter_api_key_text, model_radio, speedvsdetail_radio, browser_state, session_state],
        outputs=[browser_state, openrouter_api_key_text, model_radio, speedvsdetail_radio, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )

    speedvsdetail_radio.change(
        fn=update_browser_settings_callback,
        inputs=[openrouter_api_key_text, model_radio, speedvsdetail_radio, browser_state, session_state],
        outputs=[browser_state, openrouter_api_key_text, model_radio, speedvsdetail_radio, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )

    purge_button.click(
        fn=trigger_purge_runs,
        inputs=[purge_max_age_hours, purge_prefix, session_state],
        outputs=[purge_status, session_state]
    )

    # Initialize settings on load from persistent browser_state.
    demo_text2plan.load(
        fn=initialize_browser_settings,
        inputs=[browser_state, session_state],
        outputs=[openrouter_api_key_text, model_radio, speedvsdetail_radio, browser_state, session_state]
    ).then(
        fn=check_api_key,
        inputs=[session_state],
        outputs=[api_key_warning]
    )
    demo_text2plan.load(
        fn=update_open_dir_button_visibility,
        outputs=[open_dir_btn]
    )

def run_app():
    # print("Environment variables Gradio:\n" + get_env_as_string() + "\n\n\n")

    logger.info("Starting Gradio UI on %s:%s (worker=%s)", GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, WORKER_PLAN_URL)
    print("Press Ctrl+C to exit.")
    demo_text2plan.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False,
        auth=GRADIO_AUTH,
    )

if __name__ == "__main__":
    run_app()

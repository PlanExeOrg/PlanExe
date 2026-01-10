import logging
import os
import subprocess
import json
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from worker_plan_api.filenames import FilenameEnum
from worker_plan_api.generate_run_id import RUN_ID_PREFIX, generate_run_id
from worker_plan_api.llm_info import LLMInfo
from worker_plan_internal.plan.pipeline_environment import PipelineEnvironmentEnum
from worker_plan_api.plan_file import PlanFile
from worker_plan_api.start_time import StartTime
from worker_plan_internal.llm_factory import obtain_llm_info, get_llm_names_by_priority, get_llm
from worker_plan_internal.utils.time_since_last_modification import time_since_last_modification
from worker_plan_internal.utils.purge_old_runs import purge_old_runs, start_purge_scheduler
from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)
log_level_name = os.environ.get("PLANEXE_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

MODULE_PATH_PIPELINE = "worker_plan_internal.plan.run_plan_pipeline"
DEFAULT_APP_ROOT = Path(__file__).parent.resolve()
APP_ROOT = Path(os.environ.get("PLANEXE_CONFIG_PATH", DEFAULT_APP_ROOT)).resolve()
RUN_BASE_PATH = Path(os.environ.get("PLANEXE_RUN_DIR", APP_ROOT / "run")).resolve()
HOST_RUN_DIR_BASE = os.environ.get("PLANEXE_HOST_RUN_DIR")
RELAY_PROCESS_OUTPUT = os.environ.get("PLANEXE_WORKER_RELAY_PROCESS_OUTPUT", "false").lower() == "true"
PURGE_ENABLED = os.environ.get("PLANEXE_PURGE_ENABLED", "false").lower() == "true"
PURGE_MAX_AGE_HOURS = float(os.environ.get("PLANEXE_PURGE_MAX_AGE_HOURS", "1"))
PURGE_INTERVAL_SECONDS = float(os.environ.get("PLANEXE_PURGE_INTERVAL_SECONDS", "3600"))
PURGE_PREFIX = os.environ.get("PLANEXE_PURGE_RUN_PREFIX", RUN_ID_PREFIX)

RUN_BASE_PATH.mkdir(parents=True, exist_ok=True)


class StartRunRequest(BaseModel):
    submit_or_retry: Literal["submit", "retry"] = Field(description="Whether this is a new run or a retry of an existing run.")
    plan_prompt: str = Field(..., description="The user provided plan description.")
    llm_model: str = Field(..., description="LLM model identifier.")
    speed_vs_detail: str = Field(..., description="Speed vs detail preference.")
    openrouter_api_key: Optional[str] = Field(None, description="Optional OpenRouter API key.")
    use_uuid_as_run_id: bool = Field(False, description="Force UUID based run IDs.")
    run_id: Optional[str] = Field(None, description="Existing run ID to retry.")


class StartRunResponse(BaseModel):
    run_id: str
    run_dir: str
    display_run_dir: str
    pid: int
    status: str


class StopRunResponse(BaseModel):
    run_id: str
    stopped: bool
    message: str
    returncode: Optional[int]


class RunStatusResponse(BaseModel):
    run_id: str
    run_dir: str
    display_run_dir: str
    run_dir_exists: bool
    pid: Optional[int]
    running: bool
    returncode: Optional[int]
    pipeline_complete: bool
    stop_requested: bool
    last_update_seconds_ago: Optional[float]


class RunFilesResponse(BaseModel):
    run_id: str
    run_dir: str
    files: list[str] = Field(default_factory=list)


class PurgeRunsRequest(BaseModel):
    max_age_hours: Optional[float] = Field(None, description="Delete runs older than this many hours.")
    prefix: Optional[str] = Field(None, description="Only purge runs with this prefix.")


class PurgeRunsResponse(BaseModel):
    status: str
    message: str


@dataclass
class RunProcessInfo:
    run_id: str
    run_dir: Path
    process: subprocess.Popen
    submit_or_retry: str
    started_at: float = field(default_factory=time.time)
    stop_requested: bool = False

    def is_running(self) -> bool:
        return self.process.poll() is None

    def returncode(self) -> Optional[int]:
        if self.is_running():
            return None
        return self.process.returncode


process_store: Dict[str, RunProcessInfo] = {}
process_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_tasks()
    yield


app = FastAPI(title="PlanExe Worker", version="0.1.0", lifespan=lifespan)


def has_pipeline_complete_file(path_dir: Path) -> bool:
    if not path_dir.exists():
        return False
    try:
        return FilenameEnum.PIPELINE_COMPLETE.value in os.listdir(path_dir)
    except FileNotFoundError:
        return False


def build_display_run_dir(run_dir: Path) -> str:
    """
    Returns a user-facing path string for the run directory.
    If PLANEXE_HOST_RUN_DIR is set, map to that base to hint where to find the run on the host.
    """
    if HOST_RUN_DIR_BASE:
        try:
            return str(Path(HOST_RUN_DIR_BASE) / run_dir.name)
        except Exception:
            return str(run_dir)
    return str(run_dir)


def build_env(run_dir: Path, llm_model: str, speed_vs_detail: str, openrouter_api_key: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    env[PipelineEnvironmentEnum.RUN_ID_DIR.value] = str(run_dir)
    env[PipelineEnvironmentEnum.LLM_MODEL.value] = llm_model
    env[PipelineEnvironmentEnum.SPEED_VS_DETAIL.value] = speed_vs_detail
    if openrouter_api_key:
        env["OPENROUTER_API_KEY"] = openrouter_api_key
    return env


def start_pipeline_subprocess(env: Dict[str, str]) -> subprocess.Popen:
    command = [sys.executable, "-m", MODULE_PATH_PIPELINE]
    logger.info("Starting pipeline: %s", " ".join(command))
    stdout_target = None if RELAY_PROCESS_OUTPUT else subprocess.DEVNULL
    stderr_target = None if RELAY_PROCESS_OUTPUT else subprocess.DEVNULL
    return subprocess.Popen(command, cwd=str(APP_ROOT), env=env, stdout=stdout_target, stderr=stderr_target)


def create_run_directory(request: StartRunRequest) -> tuple[str, Path]:
    if request.submit_or_retry == "retry":
        if not request.run_id:
            raise HTTPException(status_code=400, detail="run_id is required when retrying a run.")
        run_dir = RUN_BASE_PATH / request.run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run directory does not exist: {run_dir}")
        return request.run_id, run_dir.resolve()

    start_time = datetime.now().astimezone()
    run_id = generate_run_id(use_uuid=request.use_uuid_as_run_id, start_time=start_time)
    run_dir = RUN_BASE_PATH / run_id
    if run_dir.exists():
        raise HTTPException(status_code=409, detail=f"Run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    start_time_file = StartTime.create(start_time)
    start_time_file.save(run_dir / FilenameEnum.START_TIME.value)

    plan_file = PlanFile.create(vague_plan_description=request.plan_prompt, start_time=start_time)
    plan_file.save(run_dir / FilenameEnum.INITIAL_PLAN.value)

    return run_id, run_dir.resolve()


@app.post("/runs", response_model=StartRunResponse)
def start_run(request: StartRunRequest) -> StartRunResponse:
    run_id, run_dir = create_run_directory(request)

    with process_lock:
        existing = process_store.get(run_id)
        if existing and existing.is_running():
            raise HTTPException(status_code=409, detail=f"Run {run_id} is already active.")

    env = build_env(run_dir=run_dir, llm_model=request.llm_model, speed_vs_detail=request.speed_vs_detail, openrouter_api_key=request.openrouter_api_key)
    process = start_pipeline_subprocess(env)

    info = RunProcessInfo(
        run_id=run_id,
        run_dir=run_dir,
        process=process,
        submit_or_retry=request.submit_or_retry,
    )

    with process_lock:
        process_store[run_id] = info

    display_run_dir = build_display_run_dir(run_dir)

    return StartRunResponse(
        run_id=run_id,
        run_dir=str(run_dir),
        display_run_dir=display_run_dir,
        pid=process.pid,
        status="running",
    )


@app.post("/runs/{run_id}/stop", response_model=StopRunResponse)
def stop_run(run_id: str) -> StopRunResponse:
    with process_lock:
        info = process_store.get(run_id)

    if not info:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    running_before = info.is_running()
    if running_before:
        info.stop_requested = True
        try:
            info.process.terminate()
        except Exception as exc:
            logger.warning("Error terminating run %s: %s", run_id, exc)

    return StopRunResponse(
        run_id=run_id,
        stopped=running_before,
        message="Stop signal sent." if running_before else "Process already finished.",
        returncode=info.returncode(),
    )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def run_status(run_id: str) -> RunStatusResponse:
    run_dir = (RUN_BASE_PATH / run_id).resolve()
    pipeline_complete = has_pipeline_complete_file(run_dir)
    last_update_seconds_ago = time_since_last_modification(run_dir)
    run_dir_exists = run_dir.exists()
    display_run_dir = build_display_run_dir(run_dir)

    with process_lock:
        info = process_store.get(run_id)

    running = False
    pid: Optional[int] = None
    returncode: Optional[int] = None
    stop_requested = False

    if info:
        pid = info.process.pid
        stop_requested = info.stop_requested
        running = info.is_running()
        returncode = info.returncode()

    return RunStatusResponse(
        run_id=run_id,
        run_dir=str(run_dir),
        display_run_dir=display_run_dir,
        run_dir_exists=run_dir_exists,
        pid=pid,
        running=running,
        returncode=returncode,
        pipeline_complete=pipeline_complete,
        stop_requested=stop_requested,
        last_update_seconds_ago=last_update_seconds_ago,
    )


@app.get("/runs/{run_id}/files", response_model=RunFilesResponse)
def run_files(run_id: str) -> RunFilesResponse:
    run_dir = (RUN_BASE_PATH / run_id).resolve()
    if not run_dir.is_relative_to(RUN_BASE_PATH):
        raise HTTPException(status_code=400, detail="Invalid run directory.")
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory does not exist: {run_dir}")

    try:
        files = sorted(os.listdir(run_dir))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run directory does not exist: {run_dir}")
    except Exception as exc:
        logger.warning("Unable to list files for run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail=f"Unable to list files: {exc}") from exc

    return RunFilesResponse(run_id=run_id, run_dir=str(run_dir), files=files)


@app.get("/runs/{run_id}/report")
def run_report(run_id: str) -> FileResponse:
    """
    Serve the generated plan report for a run.
    """
    run_dir = (RUN_BASE_PATH / run_id).resolve()
    if not run_dir.is_relative_to(RUN_BASE_PATH):
        raise HTTPException(status_code=400, detail="Invalid run directory.")
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory does not exist: {run_dir}")

    report_path = run_dir / FilenameEnum.REPORT.value
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report file not found for run {run_id}")

    try:
        return FileResponse(
            path=report_path,
            media_type="text/html",
            filename=FilenameEnum.REPORT.value,
        )
    except Exception as exc:
        logger.warning("Unable to serve report for run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="Unable to serve report.") from exc


def create_zip_for_run(run_dir: Path) -> Path:
    """
    Create a temporary zip of a run directory (skipping log.txt) and return the path.
    Caller is responsible for cleanup of the returned file.
    """
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory does not exist: {run_dir}")

    fd, tmp_path = tempfile.mkstemp(prefix=f"{run_dir.name}_", suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(run_dir):
                for file in files:
                    if file == "log.txt":
                        continue
                    file_path = Path(root) / file
                    zipf.write(file_path, file_path.relative_to(run_dir))
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        logger.warning("Error creating zip for run dir %s: %s", run_dir, exc)
        raise HTTPException(status_code=500, detail=f"Unable to create zip: {exc}") from exc

    return Path(tmp_path)


@app.get("/runs/{run_id}/zip")
def run_zip(run_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    run_dir = (RUN_BASE_PATH / run_id).resolve()
    if not run_dir.is_relative_to(RUN_BASE_PATH):
        raise HTTPException(status_code=400, detail="Invalid run directory.")

    try:
        zip_path = create_zip_for_run(run_dir)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Unexpected error creating zip for run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="Unable to create zip.") from exc

    background_tasks.add_task(zip_path.unlink, missing_ok=True)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{run_id}.zip",
        background=background_tasks,
    )


@app.get("/llm-info", response_model=LLMInfo)
def llm_info() -> LLMInfo:
    return obtain_llm_info()


@app.get("/llm-ping")
def llm_ping() -> StreamingResponse:
    """
    Stream ping results for each configured LLM model.
    """

    def event_stream():
        logger.info("Starting llm-ping stream")
        try:
            llm_names = get_llm_names_by_priority()
        except Exception as exc:  # pragma: no cover - runtime probe
            logger.error("llm-ping failed to enumerate llm names: %s", exc)
            yield f"data: {json.dumps({'name': 'worker_plan', 'status': 'error', 'response_time': 0, 'response': str(exc)})}\n\n"
            yield f"data: {json.dumps({'name': 'server', 'status': 'done', 'response_time': 0, 'response': ''})}\n\n"
            return

        for llm_name in llm_names:
            yield f"data: {json.dumps({'name': llm_name, 'status': 'pinging', 'response_time': 0, 'response': 'Pinging model…'})}\n\n"
            try:
                start_time = time.time()
                llm = get_llm(llm_name)
                chat_message_list = [
                    ChatMessage(
                        role=MessageRole.USER,
                        content="Hello, this is a test message. Please respond with 'OK' if you can read this."
                    )
                ]
                response = llm.chat(chat_message_list)
                end_time = time.time()

                response_text = getattr(getattr(response, "message", None), "content", None)
                if response_text is None:
                    response_text = str(response)

                payload = {
                    "name": llm_name,
                    "status": "success",
                    "response_time": int((end_time - start_time) * 1000),
                    "response": response_text
                }
            except Exception as exc:  # pragma: no cover - runtime probe
                logger.error("llm-ping error for %s: %s", llm_name, exc)
                payload = {
                    "name": llm_name,
                    "status": "error",
                    "response_time": 0,
                    "response": str(exc)
                }
            yield f"data: {json.dumps(payload)}\n\n"

        logger.info("llm-ping stream complete")
        yield f"data: {json.dumps({'name': 'server', 'status': 'done', 'response_time': 0, 'response': ''})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/purge-runs", response_model=PurgeRunsResponse)
def purge_runs(request: PurgeRunsRequest) -> PurgeRunsResponse:
    purge_prefix = request.prefix if request.prefix is not None else PURGE_PREFIX
    max_age_hours = request.max_age_hours if request.max_age_hours is not None else PURGE_MAX_AGE_HOURS

    try:
        purge_old_runs(str(RUN_BASE_PATH), max_age_hours=max_age_hours, prefix=purge_prefix)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Unexpected error during purge: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to purge runs.") from exc

    return PurgeRunsResponse(
        status="ok",
        message=f"Purged runs older than {max_age_hours} hours with prefix '{purge_prefix}'.",
    )


@app.get("/healthcheck")
def healthcheck() -> dict:
    return {"status": "ok", "run_base_path": str(RUN_BASE_PATH)}


def start_background_tasks() -> None:
    if not PURGE_ENABLED:
        logger.info("Purge scheduler disabled. Set PLANEXE_PURGE_ENABLED=true to enable.")
        return

    try:
        start_purge_scheduler(
            run_dir=str(RUN_BASE_PATH),
            purge_interval_seconds=PURGE_INTERVAL_SECONDS,
            max_age_hours=PURGE_MAX_AGE_HOURS,
            prefix=PURGE_PREFIX,
        )
    except Exception as exc:
        logger.warning("Unable to start purge scheduler: %s", exc)


if __name__ == "__main__":
    import uvicorn
    from worker_plan_internal.utils.planexe_dotenv import PlanExeDotEnv

    # Load .env file if available and update os.environ before reading config
    dotenv = PlanExeDotEnv.load()
    dotenv.update_os_environ()

    host = os.environ.get("PLANEXE_WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("PLANEXE_WORKER_PORT", "8000"))

    logger.info("Starting worker_plan on %s:%s", host, port)
    uvicorn.run("worker_plan.app:app", host=host, port=port, reload=False)

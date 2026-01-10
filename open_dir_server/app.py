"""
Small FastAPI service to open a given path in the host OS (e.g., Finder on macOS).
Run this outside Docker so it can launch the native file explorer.
"""
import logging
import os
import subprocess
import sys
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PlanExe Host Opener", version="0.1.0")

RUN_BASE_ENV = os.environ.get("PLANEXE_HOST_RUN_DIR")
if RUN_BASE_ENV:
    ALLOWED_BASE = Path(RUN_BASE_ENV).expanduser().resolve()
else:
    # Default to ../worker_plan/run relative to this file (i.e., PlanExe/worker_plan/run)
    ALLOWED_BASE = (Path(__file__).parent.parent / "worker_plan" / "run").resolve()
    logger.info("PLANEXE_HOST_RUN_DIR not set, defaulting to %s", ALLOWED_BASE)

HOST = os.environ.get("PLANEXE_OPEN_DIR_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("PLANEXE_OPEN_DIR_SERVER_PORT", "5100"))


class OpenPathRequest(BaseModel):
    path: str


class OpenPathResponse(BaseModel):
    status: str
    message: str


def _is_allowed(target: Path) -> bool:
    try:
        target.resolve().relative_to(ALLOWED_BASE)
        return True
    except ValueError:
        return False


def _command_for_platform(target: Path) -> list[str]:
    if sys.platform == "darwin":
        return ["open", str(target)]
    if os.name == "nt":
        return ["cmd", "/c", "start", "", str(target)]
    if sys.platform.startswith("linux"):
        return ["xdg-open", str(target)]
    raise ValueError(f"Unsupported platform: {sys.platform}")


@app.post("/open", response_model=OpenPathResponse)
def open_path(request: OpenPathRequest):
    raw_path = request.path
    # Only allow PlanExe-style run directory names to avoid arbitrary path access.
    if not re.fullmatch(r"^PlanExe_\d+_\d+$", Path(raw_path).name):
        raise HTTPException(status_code=400, detail="Invalid path format.")
    target = Path(raw_path).expanduser().resolve()

    if not _is_allowed(target):
        raise HTTPException(status_code=403, detail="Path is outside allowed base.")

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {target}")

    try:
        cmd = _command_for_platform(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        logger.warning("Failed to open %s: %s", target, exc)
        raise HTTPException(status_code=500, detail=f"Failed to open path: {exc}") from exc

    return OpenPathResponse(status="ok", message=f"Requested OS to open {target}.")


@app.get("/healthcheck")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "allowed_base": str(ALLOWED_BASE) if ALLOWED_BASE else None,
        "platform": sys.platform,
        "host": HOST,
        "port": PORT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)

from typing import Literal

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str


class TaskCreateOutput(BaseModel):
    task_id: str
    created_at: str


class TaskStatusTiming(BaseModel):
    started_at: str | None
    elapsed_sec: float


class TaskStatusFile(BaseModel):
    path: str
    updated_at: str


class TaskStatusSuccess(BaseModel):
    task_id: str
    state: Literal["stopped", "running", "completed", "failed", "stopping"]
    progress_percentage: float
    timing: TaskStatusTiming
    files: list[TaskStatusFile]


class TaskStatusOutput(BaseModel):
    task_id: str | None = None
    state: Literal["stopped", "running", "completed", "failed", "stopping"] | None = None
    progress_percentage: float | None = None
    timing: TaskStatusTiming | None = None
    files: list[TaskStatusFile] | None = None
    error: ErrorDetail | None = None


class TaskStopOutput(BaseModel):
    state: Literal["stopped"] | None = None
    error: ErrorDetail | None = None


class TaskFileInfoReadyOutput(BaseModel):
    content_type: str
    sha256: str
    download_size: int
    download_url: str | None = None


class TaskFileInfoOutput(BaseModel):
    content_type: str | None = None
    sha256: str | None = None
    download_size: int | None = None
    download_url: str | None = None
    error: ErrorDetail | None = None

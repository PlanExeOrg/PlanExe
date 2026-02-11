from typing import Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class PromptExamplesOutput(BaseModel):
    samples: list[str] = Field(
        ...,
        description=(
            "Example prompts that define the baseline for what a good prompt looks like. "
            "Take inspiration from these when writing your own prompt for task_create."
        ),
    )
    message: str


class PromptExamplesInput(BaseModel):
    """No input parameters."""
    pass


class TaskStatusInput(BaseModel):
    task_id: str = Field(
        ...,
        description="Task UUID returned by task_create. Use it to reference the plan being created.",
    )


class TaskStopInput(BaseModel):
    task_id: str = Field(
        ...,
        description="The UUID returned by task_create. Call task_stop with this task_id to request the plan generation to stop.",
    )


class TaskFileInfoInput(BaseModel):
    task_id: str = Field(
        ...,
        description="Task UUID returned by task_create. Use it to download the created plan.",
    )
    artifact: str = Field(
        default="report",
        description="Download artifact type: report or zip.",
    )


class TaskCreateOutput(BaseModel):
    task_id: str = Field(
        ...,
        description="Task UUID returned by task_create. Stable across task_status/task_stop/task_download."
    )
    created_at: str


class TaskStatusTiming(BaseModel):
    started_at: str | None
    elapsed_sec: float


class TaskStatusFile(BaseModel):
    path: str
    updated_at: str


class TaskStatusSuccess(BaseModel):
    task_id: str = Field(
        ...,
        description="Task UUID returned by task_create."
    )
    state: Literal["stopped", "running", "completed", "failed", "stopping"]
    progress_percentage: float
    timing: TaskStatusTiming
    files: list[TaskStatusFile]


class TaskStatusOutput(BaseModel):
    task_id: str | None = Field(
        default=None,
        description="Task UUID returned by task_create."
    )
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


class TaskCreateInput(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "What the plan should cover (goal, context, constraints). "
            "Use prompt_examples to get example prompts; use these as examples for task_create. "
            "Short prompts produce less detailed plans."
        ),
    )
    speed_vs_detail: Literal["ping", "fast", "all"] = Field(
        default="ping",
        description="Defaults to ping (alias for ping_llm). Options: ping, fast, all.",
    )
    user_api_key: str | None = Field(
        default=None,
        description="Optional user API key for credits and attribution.",
    )

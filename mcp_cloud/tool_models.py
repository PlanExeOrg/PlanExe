from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class FailureErrorDetail(BaseModel):
    """Consolidated error dict returned inside plan_status when state is 'failed'."""
    failure_reason: str | None = Field(
        default=None,
        description=(
            "Failure category. Values: generation_error, worker_error, "
            "inactivity_timeout, internal_error, version_mismatch."
        ),
    )
    failed_step: str | None = Field(
        default=None,
        description="The pipeline step that was active when the failure occurred (e.g. '016-expert_criticism').",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable error message describing the failure (max 256 chars).",
    )
    recoverable: bool | None = Field(
        default=None,
        description="True when plan_resume may succeed; False when plan_retry (full restart) is recommended.",
    )


class ExamplePromptsOutput(BaseModel):
    samples: list[str] = Field(
        ...,
        description=(
            "Example prompts that define the baseline for what a good prompt looks like. "
            "Take inspiration from these when writing your own prompt for plan_create "
            "(typically ~300-800 words). Good prompt shape: objective, scope, constraints, "
            "timeline, stakeholders, budget/resources, and success criteria."
        ),
    )
    message: str


class ExamplePromptsInput(BaseModel):
    """No input parameters."""
    pass


class ExamplePlansInput(BaseModel):
    """No input parameters."""
    pass


class ExamplePlanItem(BaseModel):
    title: str = Field(..., description="Short title describing the example plan.")
    report_url: str = Field(..., description="URL to the static HTML report for this example plan.")
    zip_url: str = Field(..., description="URL to the zip bundle for this example plan.")


class ExamplePlansOutput(BaseModel):
    plans: list[ExamplePlanItem] = Field(
        ...,
        description="Curated example plans with download links for reports and zip bundles.",
    )
    message: str


class ModelProfilesInput(BaseModel):
    """No input parameters."""
    pass


class ModelProfileModelEntry(BaseModel):
    key: str = Field(..., description="Model key from llm_config/<profile>.json.")
    provider_class: str | None = Field(
        default=None,
        description="Provider class (for example OpenRouter, OpenAI, Ollama).",
    )
    model: str | None = Field(default=None, description="Provider model identifier when present.")
    priority: int | None = Field(
        default=None,
        description="Priority from config (lower number means earlier in selection order).",
    )


class ModelProfileInfo(BaseModel):
    profile: Literal["baseline", "premium", "frontier", "custom"] = Field(
        ...,
        description="Model profile value accepted by plan_create.model_profile.",
    )
    title: str = Field(..., description="Human-friendly profile label.")
    summary: str = Field(..., description="Short profile guidance for callers.")
    model_count: int = Field(..., description="Number of models currently available in this profile.")
    models: list[ModelProfileModelEntry] = Field(
        ...,
        description="Models currently available to this profile.",
    )


class ModelProfilesOutput(BaseModel):
    default_profile: Literal["baseline", "premium", "frontier", "custom"] = Field(
        ...,
        description="Default model profile used when plan_create.model_profile is omitted/invalid.",
    )
    profiles: list[ModelProfileInfo] = Field(
        ...,
        description="Available profile options and their model inventory.",
    )
    message: str = Field(..., description="Caller guidance for selecting plan_create.model_profile.")


class PlanStatusInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="Plan UUID returned by plan_create. Use it to reference the plan being created.",
    )


class PlanStopInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The UUID returned by plan_create. Call plan_stop with this plan_id to request the plan generation to stop.",
    )


class PlanRetryInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="UUID of the failed or stopped plan to retry.",
    )
    model_profile: Literal["baseline", "premium", "frontier", "custom"] = Field(
        default="baseline",
        description=(
            "Model profile used for the retry run. Defaults to baseline if omitted."
        ),
    )


class PlanFileInfoInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="Plan UUID returned by plan_create. Use it to download the created plan.",
    )
    artifact: str = Field(
        default="report",
        description="Download artifact type: report or zip.",
    )


class PlanCreateOutput(BaseModel):
    plan_id: str = Field(
        ...,
        description="Plan UUID returned by plan_create. Stable across plan_status/plan_stop/plan_file_info."
    )
    created_at: str
    deduplicated: bool | None = Field(
        default=None,
        description=(
            "True when this response returns an existing plan instead of creating a new one "
            "(duplicate prompt + model_profile by the same user within the dedup window). "
            "Absent or None for newly created plans."
        ),
    )
    sse_url: str | None = Field(
        default=None,
        description=(
            "Optional completion detector. Run `curl -N <sse_url>` in a background shell — "
            "the stream auto-closes when the plan reaches a terminal state (completed/failed/stopped). "
            "For structured progress data, use plan_status instead."
        ),
    )


class PlanStatusTiming(BaseModel):
    started_at: str | None
    elapsed_sec: float


class PlanStatusFile(BaseModel):
    path: str
    updated_at: str


class PlanStatusSuccess(BaseModel):
    plan_id: str = Field(
        ...,
        description="Plan UUID returned by plan_create."
    )
    state: Literal["pending", "processing", "completed", "failed", "stopped"] = Field(
        ...,
        description=(
            "Caller contract: pending/processing => keep polling; "
            "completed => download is ready; failed => terminal error; "
            "stopped => user called plan_stop (consider plan_resume)."
        ),
    )
    progress_percentage: float = Field(
        ...,
        description=(
            "Completion progress from 0 to 100. Monotonically increasing; 100 when state is completed. "
            "Steps vary in duration (early steps are fast, later steps like review and report generation are slower), "
            "so do not use this to estimate time remaining."
        ),
    )
    steps_completed: int | None = Field(
        default=None,
        description="Number of plan generation steps completed so far. Steps vary in duration.",
    )
    steps_total: int | None = Field(
        default=None,
        description="Total number of plan generation steps expected. Not all steps take equal time.",
    )
    current_step: str | None = Field(
        default=None,
        description="Human-readable label of the most recently completed step, e.g. 'SWOT Analysis'.",
    )
    timing: PlanStatusTiming
    files_count: int = Field(
        ...,
        description="Total number of output files produced so far (files list is capped at 10).",
    )
    files: list[PlanStatusFile] = Field(
        ...,
        description=(
            "Intermediate output files produced so far (most recent 10). "
            "Use updated_at timestamps to detect stalls. "
            "These files are included in the zip artifact when the plan completes."
        ),
    )
    sse_url: str | None = Field(
        default=None,
        description=(
            "Optional completion detector URL. Available when plan is not in a terminal state. "
            "Run `curl -N <sse_url>` in a background shell — auto-closes on completion/failure/stop."
        ),
    )
    error: FailureErrorDetail | None = Field(
        default=None,
        description=(
            "Failure diagnostics (only present when state is 'failed'). "
            "Contains failure_reason, failed_step, message, and recoverable."
        ),
    )


class PlanStatusOutput(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Plan UUID returned by plan_create."
    )
    state: Literal["pending", "processing", "completed", "failed", "stopped"] | None = Field(
        default=None,
        description=(
            "Caller contract: pending/processing => keep polling; "
            "completed => download is ready; failed => terminal error; "
            "stopped => user called plan_stop (consider plan_resume)."
        ),
    )
    progress_percentage: float | None = Field(
        default=None,
        description=(
            "Completion progress from 0 to 100. Monotonically increasing; 100 when state is completed. "
            "Steps vary in duration (early steps are fast, later steps like review and report generation are slower), "
            "so do not use this to estimate time remaining."
        ),
    )
    steps_completed: int | None = Field(
        default=None,
        description="Number of plan generation steps completed so far. Steps vary in duration.",
    )
    steps_total: int | None = Field(
        default=None,
        description="Total number of plan generation steps expected. Not all steps take equal time.",
    )
    current_step: str | None = Field(
        default=None,
        description="Human-readable label of the most recently completed step, e.g. 'SWOT Analysis'.",
    )
    timing: PlanStatusTiming | None = None
    files_count: int | None = Field(
        default=None,
        description="Total number of output files produced so far (files list is capped at 10).",
    )
    files: list[PlanStatusFile] | None = Field(
        default=None,
        description=(
            "Intermediate output files produced so far (most recent 10). "
            "Use updated_at timestamps to detect stalls. "
            "These files are included in the zip artifact when the plan completes."
        ),
    )
    sse_url: str | None = Field(
        default=None,
        description=(
            "Optional completion detector URL. Available when plan is not in a terminal state. "
            "Run `curl -N <sse_url>` in a background shell — auto-closes on completion/failure/stop."
        ),
    )
    error: FailureErrorDetail | ErrorDetail | None = Field(
        default=None,
        description=(
            "Failure diagnostics (only present when state is 'failed'). "
            "Contains failure_reason, failed_step, message, and recoverable. "
            "For PLAN_NOT_FOUND errors, contains code and message."
        ),
    )


class PlanStopOutput(BaseModel):
    state: Literal["pending", "processing", "completed", "failed", "stopped"] | None = Field(
        default=None,
        description="Current plan state after stop request.",
    )
    stop_requested: bool | None = Field(
        default=None,
        description="True when stop request flag was set for a pending/processing task.",
    )
    error: ErrorDetail | None = None


class PlanRetryOutput(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Plan UUID that was retried (same ID as the failed or stopped plan).",
    )
    state: Literal["pending", "processing", "completed", "failed", "stopped"] | None = Field(
        default=None,
        description="Current plan state after retry request.",
    )
    model_profile: Literal["baseline", "premium", "frontier", "custom"] | None = Field(
        default=None,
        description="Model profile assigned to the retry request.",
    )
    retried_at: str | None = Field(
        default=None,
        description="UTC timestamp when the retry request was accepted.",
    )
    sse_url: str | None = Field(
        default=None,
        description=(
            "Optional completion detector. Run `curl -N <sse_url>` in a background shell — "
            "the stream auto-closes when the plan reaches a terminal state (completed/failed/stopped). "
            "For structured progress data, use plan_status instead."
        ),
    )
    error: ErrorDetail | None = None


class PlanResumeInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="UUID of the failed or stopped plan to resume.",
    )
    model_profile: Literal["baseline", "premium", "frontier", "custom"] = Field(
        default="baseline",
        description=(
            "Model profile used for the resumed run. Defaults to baseline if omitted."
        ),
    )


class PlanResumeOutput(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Plan UUID that was resumed (same ID as the failed or stopped plan).",
    )
    state: Literal["pending", "processing", "completed", "failed", "stopped"] | None = Field(
        default=None,
        description="Current plan state after resume request.",
    )
    model_profile: Literal["baseline", "premium", "frontier", "custom"] | None = Field(
        default=None,
        description="Model profile assigned to the resumed run.",
    )
    resume_count: int | None = Field(
        default=None,
        description="Number of times this plan has been resumed.",
    )
    resumed_at: str | None = Field(
        default=None,
        description="UTC timestamp when the resume request was accepted.",
    )
    sse_url: str | None = Field(
        default=None,
        description=(
            "Optional completion detector. Run `curl -N <sse_url>` in a background shell — "
            "the stream auto-closes when the plan reaches a terminal state (completed/failed/stopped). "
            "For structured progress data, use plan_status instead."
        ),
    )
    error: ErrorDetail | None = None


class PlanFileInfoNotReadyOutput(BaseModel):
    ready: bool = Field(False, description="Always False; indicates the artifact is not yet available.")
    reason: str = Field(..., description="Human-readable explanation, e.g. 'processing' or 'failed'.")


class PlanFileInfoReadyOutput(BaseModel):
    content_type: str = Field(..., description="Artifact content type.")
    sha256: str = Field(..., description="SHA-256 hash of artifact bytes.")
    download_size: int = Field(..., description="Artifact size in bytes.")
    download_url: str | None = Field(
        default=None,
        description="Absolute URL where the requested artifact can be downloaded.",
    )
    expires_at: str | None = Field(
        default=None,
        description="ISO 8601 UTC timestamp when the download_url expires. Present only when download_url is set.",
    )


class PlanFileInfoOutput(BaseModel):
    content_type: str | None = Field(default=None, description="Artifact content type.")
    sha256: str | None = Field(default=None, description="SHA-256 hash of artifact bytes.")
    download_size: int | None = Field(default=None, description="Artifact size in bytes.")
    download_url: str | None = Field(
        default=None,
        description="Absolute URL where the requested artifact can be downloaded.",
    )
    expires_at: str | None = Field(
        default=None,
        description="ISO 8601 UTC timestamp when the download_url expires. Present only when download_url is set.",
    )
    error: ErrorDetail | None = None


class PlanListInput(BaseModel):
    user_api_key: str | None = Field(
        default=None,
        description="Optional user API key for credits and attribution.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of plans to return (1–50). Newest plans are returned first.",
    )


class PlanListItem(BaseModel):
    plan_id: str = Field(..., description="Plan UUID.")
    state: Literal["pending", "processing", "completed", "failed", "stopped"] = Field(
        ...,
        description="Current plan state.",
    )
    progress_percentage: float = Field(..., description="Progress from 0 to 100.")
    created_at: str = Field(..., description="UTC creation timestamp (ISO 8601).")
    prompt_excerpt: str = Field(..., description="First 100 characters of the prompt.")


class PlanListOutput(BaseModel):
    plans: list[PlanListItem] = Field(..., description="Plans for the authenticated user, newest first.")
    message: str = Field(..., description="Human-readable summary (e.g. how many plans were returned).")


class PlanCreateInput(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "What the plan should cover (goal, context, constraints). "
            "Use example_prompts to get example prompts; use these as examples for plan_create. "
            "For best results, provide a detailed prompt (typically ~300-800 words). "
            "Good prompt shape: objective, scope, constraints, timeline, stakeholders, "
            "budget/resources, and success criteria. "
            "Write as flowing prose, not structured markdown. Include banned approaches, "
            "governance preferences, and phasing inline. "
            "Short prompts produce less detailed plans. "
            "Do not use plan_create for tiny one-shot outputs (e.g., a 5-point checklist); use direct LLM responses for those."
        ),
    )
    model_profile: Literal["baseline", "premium", "frontier", "custom"] = Field(
        default="baseline",
        description=(
            "Model profile selection: baseline (cheap/fast), premium (higher quality), "
            "frontier (most capable), custom (user-defined). Call model_profiles for runtime availability."
        ),
    )
    user_api_key: str | None = Field(
        default=None,
        description="Optional user API key for credits and attribution.",
    )


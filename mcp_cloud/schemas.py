"""PlanExe MCP Cloud – tool schema constants and ToolDefinition."""
from dataclasses import dataclass
from typing import Any, Optional

from mcp_cloud.download_tokens import DOWNLOAD_TOKEN_TTL_SECONDS

from mcp_cloud.tool_models import (
    ModelProfilesInput,
    ModelProfilesOutput,
    ExamplePromptsInput,
    ExamplePromptsOutput,
    ExamplePlansInput,
    ExamplePlansOutput,
    PlanCreateInput,
    PlanCreateOutput,
    PlanRetryInput,
    PlanRetryOutput,
    PlanResumeInput,
    PlanResumeOutput,
    PlanStopOutput,
    PlanStatusInput,
    PlanStopInput,
    PlanFileInfoInput,
    PlanFileInfoNotReadyOutput,
    PlanStatusSuccess,
    PlanFileInfoReadyOutput,
    PlanListInput,
    PlanListOutput,
    ErrorDetail,
    FailureErrorDetail,
)

PLAN_CREATE_INPUT_SCHEMA = PlanCreateInput.model_json_schema()
PLAN_CREATE_OUTPUT_SCHEMA = PlanCreateOutput.model_json_schema()
PLAN_STATUS_SUCCESS_SCHEMA = PlanStatusSuccess.model_json_schema()
PLAN_STATUS_OUTPUT_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"error": ErrorDetail.model_json_schema()},
            "required": ["error"],
        },
        PLAN_STATUS_SUCCESS_SCHEMA,
    ]
}
PLAN_STOP_OUTPUT_SCHEMA = PlanStopOutput.model_json_schema()
PLAN_RETRY_OUTPUT_SCHEMA = PlanRetryOutput.model_json_schema()
PLAN_RESUME_OUTPUT_SCHEMA = PlanResumeOutput.model_json_schema()
PLAN_FILE_INFO_READY_OUTPUT_SCHEMA = PlanFileInfoReadyOutput.model_json_schema()
PLAN_FILE_INFO_NOT_READY_OUTPUT_SCHEMA = PlanFileInfoNotReadyOutput.model_json_schema()
PLAN_FILE_INFO_OUTPUT_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"error": ErrorDetail.model_json_schema()},
            "required": ["error"],
        },
        PLAN_FILE_INFO_NOT_READY_OUTPUT_SCHEMA,
        PLAN_FILE_INFO_READY_OUTPUT_SCHEMA,
    ]
}
PLAN_STATUS_INPUT_SCHEMA = PlanStatusInput.model_json_schema()
PLAN_STOP_INPUT_SCHEMA = PlanStopInput.model_json_schema()
PLAN_RETRY_INPUT_SCHEMA = PlanRetryInput.model_json_schema()
PLAN_RESUME_INPUT_SCHEMA = PlanResumeInput.model_json_schema()
PLAN_FILE_INFO_INPUT_SCHEMA = PlanFileInfoInput.model_json_schema()

EXAMPLE_PROMPTS_INPUT_SCHEMA = ExamplePromptsInput.model_json_schema()
EXAMPLE_PROMPTS_OUTPUT_SCHEMA = ExamplePromptsOutput.model_json_schema()
MODEL_PROFILES_INPUT_SCHEMA = ModelProfilesInput.model_json_schema()
MODEL_PROFILES_OUTPUT_SCHEMA = ModelProfilesOutput.model_json_schema()
EXAMPLE_PLANS_INPUT_SCHEMA = ExamplePlansInput.model_json_schema()
EXAMPLE_PLANS_OUTPUT_SCHEMA = ExamplePlansOutput.model_json_schema()
PLAN_LIST_INPUT_SCHEMA = PlanListInput.model_json_schema()
PLAN_LIST_OUTPUT_SCHEMA = PlanListOutput.model_json_schema()


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: Optional[dict[str, Any]] = None
    annotations: Optional[dict[str, Any]] = None

TOOL_DEFINITIONS = [
    ToolDefinition(
        name="example_plans",
        description=(
            "Returns a curated list of example plans with download links for reports and zip bundles. "
            "Use this to preview what PlanExe output looks like before creating your own plan. "
            "Especially useful when the user asks what the output looks like before committing to a plan. "
            "No API key required."
        ),
        input_schema=EXAMPLE_PLANS_INPUT_SCHEMA,
        output_schema=EXAMPLE_PLANS_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="example_prompts",
        description=(
            "Call this first. Returns example prompts that define what a good prompt looks like. "
            "Do NOT call plan_create yet. Optional before plan_create: call model_profiles to choose model_profile. "
            "Next is a non-tool step: formulate a detailed prompt (typically ~300-800 words; use examples as a baseline, similar structure) and get user approval. "
            "Good prompt shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. "
            "Write the prompt as flowing prose, not structured markdown with headers or bullet lists. "
            "Weave technical specs, constraints, and targets naturally into sentences. Include banned words/approaches and governance preferences inline. "
            "The examples demonstrate this prose style — match their tone and density. "
            "Then call plan_create. "
            "PlanExe is not for tiny one-shot outputs like a 5-point checklist; and it does not support selecting only some internal pipeline steps."
        ),
        input_schema=EXAMPLE_PROMPTS_INPUT_SCHEMA,
        output_schema=EXAMPLE_PROMPTS_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="model_profiles",
        description=(
            "Optional helper before plan_create. Returns model_profile options with plain-language guidance "
            "and currently available models in each profile. "
            "If no models are available, returns error code MODEL_PROFILES_UNAVAILABLE."
        ),
        input_schema=MODEL_PROFILES_INPUT_SCHEMA,
        output_schema=MODEL_PROFILES_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_create",
        description=(
            "Call only after example_prompts and after you have completed prompt drafting/approval (non-tool step). "
            "PlanExe turns the approved prompt into a strategic project-plan draft (20+ sections) in ~10-20 min. "
            "Sections include: executive summary, interactive Gantt charts, investor pitch, project plan with SMART criteria, "
            "strategic decision analysis, scenario comparison, assumptions with expert review, governance structure, "
            "SWOT analysis, team role profiles, simulated expert criticism, work breakdown structure, "
            "plan review (critical issues, KPIs, financial strategy, automation opportunities), Q&A, "
            "premortem with failure scenarios, self-audit checklist, and adversarial premise attacks that argue against the project. "
            "The adversarial sections (premortem, self-audit, premise attacks) surface risks and questions the prompter may not have considered. "
            "Returns plan_id (UUID); use it for plan_status, plan_stop, plan_retry, and plan_file_info. "
            "To track progress, poll plan_status at reasonable intervals (e.g. every 5 minutes). "
            "Optionally, run `curl -N <sse_url>` in a background shell as a completion detector — "
            "the stream auto-closes on terminal state (completed/failed/stopped). "
            "If you lose a plan_id, call plan_list to recover it. "
            "If the same prompt + model_profile is submitted by the same user within a short window, "
            "the existing plan is returned (with deduplicated=true) instead of creating a new one. "
            "If you are unsure which model_profile to choose, call model_profiles first. "
            "If your deployment uses credits, include user_api_key to charge the correct account. "
            "Common error codes: INVALID_USER_API_KEY, USER_API_KEY_REQUIRED, INSUFFICIENT_CREDITS."
        ),
        input_schema=PLAN_CREATE_INPUT_SCHEMA,
        output_schema=PLAN_CREATE_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_status",
        description=(
            "Returns status and progress of the plan currently being created. "
            "This is the primary way to check progress — it returns structured JSON with all progress fields. "
            "Poll at reasonable intervals (e.g. every 5 minutes): plan generation typically takes 10-20 minutes "
            "(baseline profile) and may take longer on higher-quality profiles. "
            "State contract: pending/processing => keep polling; completed => download is ready; "
            "failed => terminal error; stopped => user called plan_stop (consider plan_resume). "
            "progress_percentage is 0-100 (integer-like float); 100 when completed. "
            "Note: steps vary in duration — early steps complete quickly while later steps (review, report generation) "
            "take longer. Do not use progress_percentage to estimate time remaining. "
            "steps_completed and steps_total give the number of plan generation steps completed and expected (both nullable). "
            "current_step is the human-readable label of the most recently completed step (e.g. 'SWOT Analysis'). "
            "files lists the most recent 10 intermediate outputs produced so far (files_count gives the total); use their updated_at timestamps to detect stalls. "
            "When state is 'failed', the response includes an error dict with failure diagnostics: "
            "error.failure_reason (category: generation_error, worker_error, inactivity_timeout, internal_error, version_mismatch), "
            "error.failed_step (pipeline step active at failure), error.message (human-readable message), "
            "and error.recoverable (true => plan_resume may work, false => use plan_retry). "
            "The error dict is absent for non-failed states. "
            "Unknown plan_id returns error code PLAN_NOT_FOUND. "
            "Troubleshooting: pending for >5 minutes likely means queued but not picked up by a worker. "
            "processing with no file-output changes for >20 minutes likely means failed/stalled. "
            "Report these issues to https://github.com/PlanExeOrg/PlanExe/issues ."
        ),
        input_schema=PLAN_STATUS_INPUT_SCHEMA,
        output_schema=PLAN_STATUS_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_stop",
        description=(
            "Request the plan generation to stop. Pass the plan_id (the UUID returned by plan_create). "
            "Stopping is asynchronous: the stop flag is set immediately but the plan may continue briefly before halting. "
            "A stopped plan will transition to the stopped state. "
            "If the plan is already completed or failed, stop_requested returns false (the plan already finished). "
            "Unknown plan_id returns error code PLAN_NOT_FOUND."
        ),
        input_schema=PLAN_STOP_INPUT_SCHEMA,
        output_schema=PLAN_STOP_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_retry",
        description=(
            "Retry a plan that is currently in failed or stopped state. "
            "Pass the plan_id and optionally model_profile (defaults to baseline). "
            "The plan is reset to pending, prior artifacts are cleared, and the same plan_id is requeued for processing. "
            "Returns PLAN_NOT_FOUND when plan_id is unknown and PLAN_NOT_FAILED when the plan is not in failed or stopped state."
        ),
        input_schema=PLAN_RETRY_INPUT_SCHEMA,
        output_schema=PLAN_RETRY_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_resume",
        description=(
            "Resume a failed or stopped plan without discarding completed intermediary files. "
            "Plan generation restarts from the first incomplete step, skipping all steps that already produced output files. "
            "Use plan_resume when plan_status shows 'failed' or 'stopped' and plan generation was interrupted before completing all steps "
            "(network drop, timeout, plan_stop, worker crash). "
            "For a full restart or to change model_profile, use plan_retry instead. "
            "Only failed or stopped plans can be resumed. "
            "Returns PLAN_NOT_FOUND when plan_id is unknown and PLAN_NOT_RESUMABLE when the plan is not in failed or stopped state. "
            "Returns PIPELINE_VERSION_MISMATCH when the snapshot was created by a different pipeline version; use plan_retry instead."
        ),
        input_schema=PLAN_RESUME_INPUT_SCHEMA,
        output_schema=PLAN_RESUME_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    ),
    ToolDefinition(
        name="plan_file_info",
        description=(
            "Returns file metadata (content_type, download_url, download_size, expires_at) for the report or zip artifact. "
            "Use artifact='report' (default) for the interactive HTML report (~700KB, self-contained with embedded JS "
            "for collapsible sections and interactive Gantt charts — open in a browser). "
            "Use artifact='zip' for the full pipeline output bundle (md, json, csv intermediary files that fed the report). "
            "While the task is still pending or processing, returns {ready:false,reason:\"processing\"}. "
            "Check readiness by testing whether download_url is present in the response. "
            "Once ready, present download_url to the user or fetch and save the file locally. "
            f"Download URLs expire after {DOWNLOAD_TOKEN_TTL_SECONDS // 60} minutes (see expires_at); "
            "call plan_file_info again to get a fresh URL if needed. "
            "If your client exposes plan_download (e.g. mcp_local), prefer that to save the file locally. "
            "Terminal error codes: generation_failed (plan failed), content_unavailable (artifact missing). "
            "Unknown plan_id returns error code PLAN_NOT_FOUND."
        ),
        input_schema=PLAN_FILE_INFO_INPUT_SCHEMA,
        output_schema=PLAN_FILE_INFO_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
    ToolDefinition(
        name="plan_list",
        description=(
            "List the most recent plans for an authenticated user. "
            "Returns up to `limit` plans (default 10, max 50) newest-first, each with plan_id, state, "
            "progress_percentage, created_at (ISO 8601), and a prompt_excerpt (first 100 chars). "
            "Use this to recover a lost plan_id or to review recent activity."
        ),
        input_schema=PLAN_LIST_INPUT_SCHEMA,
        output_schema=PLAN_LIST_OUTPUT_SCHEMA,
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    ),
]

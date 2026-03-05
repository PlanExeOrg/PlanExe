# PlanExe MCP Details

MCP is work-in-progress, and I (Simon Strandgaard, the developer) may change it as I see fit.
If there is a particular tool you want. Write to me on the [PlanExe Discord](https://planexe.org/discord), and I will see what I can do.

This document lists the MCP tools exposed by PlanExe and example prompts for agents.

## Overview

- The primary MCP server runs in the cloud (see `mcp_cloud`).
- The local MCP proxy (`mcp_local`) forwards calls to the server and adds a local download helper.
- Tool responses return JSON in both `content.text` and `structuredContent`.
- Workflow note: drafting and user approval of the prompt is a non-tool step between setup tools and `plan_create`.

## Tool Catalog, `mcp_cloud`

### example_plans

Returns a curated list of example plans with download links for reports and zip bundles. Use this to preview what PlanExe output looks like before creating your own plan. No API key required.

Example prompt:
```
Show me example plans.
```

Example call:
```json
{}
```

Response includes `plans` (array of objects with `title`, `report_url`, `zip_url`) and `message`.

### example_prompts

Returns around five example prompts that show what good prompts look like. Each sample is typically 300-800 words. Usually the AI does the heavy lifting: the user has a vague idea, the agent calls `example_prompts`, then expands that idea into a high-quality prompt (300-800 words). A compact prompt shape works best: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. The prompt is shown to the user, who can ask for further changes or confirm it’s good to go. When the user confirms, the agent then calls `plan_create`. Shorter or vaguer prompts produce lower-quality plans.

Example prompt:
```
Get example prompts for creating a plan.
```

Example call:
```json
{}
```

Response includes `samples` (array of prompt strings, each ~300-800 words) and `message`.

### model_profiles

Returns profile guidance and model availability for `plan_create.model_profile`.
This helps agents pick a profile without knowing internal `llm_config/*.json` details.
Profiles with zero models are omitted from the `profiles` list.
If no models are available in any profile, `model_profiles` returns `isError=true` with `error.code = MODEL_PROFILES_UNAVAILABLE`.

Example prompt:
```
List available model profiles and models.
```

Example call:
```json
{}
```

Response includes:
- `default_profile`
- `profiles[]` with:
  - `profile`
  - `title`
  - `summary`
  - `model_count`
  - `models[]` (`key`, `provider_class`, `model`, `priority`)

### plan_create

Create a new plan.

Example prompt:
> Create a plan for: Weekly meetup for humans where participants are randomly paired every 5 minutes...

Example call:
```json
{"prompt": "Weekly meetup for humans where participants are randomly paired every 5 minutes..."}
```

Optional visible argument:
```text
model_profile: "baseline" | "premium" | "frontier" | "custom"
```

Example with visible `model_profile`:
```json
{"prompt": "Weekly meetup for humans where participants are randomly paired every 5 minutes...", "model_profile": "premium"}
```

Counterexamples (do NOT use PlanExe for these):

- "Give me a 5-point checklist for X."
- "Summarize this paragraph in 6 bullets."
- "Rewrite this email."
- "Identify the risks of this project."
- "Make a SWOT for this document."

What to do instead:

- For one-shot outputs, use a normal LLM response directly.
- For PlanExe, send a substantial multi-phase project prompt with scope, constraints, timeline, budget, stakeholders, and success criteria.
- PlanExe always runs a fixed end-to-end pipeline; it does not support selecting only internal pipeline subsets.

### plan_status

Fetch status/progress and recent files for a plan.

Example prompt:
```
Get status for plan 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"plan_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7"}
```

State contract:

- `pending`: queued and waiting for a worker, keep polling.
- `processing`: picked up by a worker, keep polling.
- `completed`: terminal success, proceed to download.
- `failed`: terminal error.

### plan_stop

Request an active plan to stop.

Example prompt:
```
Stop plan 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"plan_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7"}
```

### plan_retry

Retry a failed plan by requeueing the same `plan_id`.

Example prompt:
```
Retry failed plan 2d57a448-1b09-45aa-ad37-e69891ff6ec7 with baseline profile.
```

Example call:
```json
{"plan_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7", "model_profile": "baseline"}
```

Notes:
- `model_profile` is optional and defaults to `baseline`.
- Only failed plans can be retried.
- Non-failed plans return `PLAN_NOT_FAILED`.

### plan_file_info

Return download metadata for report or zip artifacts.

Example prompt:
```
Get report info for plan 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"plan_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7", "artifact": "report"}
```

Available artifacts:
```
"report" | "zip"
```

Typical successful response:
```json
{
  "content_type": "application/zip",
  "sha256": "f8ad556b635b14e375222150664e85b426bf7f9209ede2f37f47a8975e286323",
  "download_size": 17262032,
  "download_url": "https://mcp.planexe.org/download/<plan_id>/run.zip"
}
```

### Download with `curl`

When `plan_file_info` returns a `download_url`, you can download directly with the same `X-API-Key` used for MCP authentication.

Download zip:
```bash
curl -H "X-API-Key: pex_0123456789abcdef" -O "https://mcp.planexe.org/download/2d57a448-1b09-45aa-ad37-e69891ff6ec7/run.zip"
```

Download report:
```bash
curl -H "X-API-Key: pex_0123456789abcdef" -O "https://mcp.planexe.org/download/2d57a448-1b09-45aa-ad37-e69891ff6ec7/030-report.html"
```

## Tool Catalog, `mcp_local`

The local proxy exposes the same tools as the server, and adds:

### plan_download

Download report or zip to a local path.

Example prompt:
```
Download the report for plan 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"plan_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7", "artifact": "report"}
```

`PLANEXE_PATH` behavior for `plan_download`:
- Save directory is `PLANEXE_PATH`, or current working directory if unset.
- Non-existing directories are created automatically.
- If `PLANEXE_PATH` points to a file, download fails.
- Filename is prefixed with plan id (for example `<plan_id>-030-report.html`).
- Response includes `saved_path` with the exact local file location.

## Minimal error-handling contract

Error payload shape:
```json
{"error": {"code": "SOME_CODE", "message": "Human readable message", "details": {}}}
```

Common cloud/core error codes:
- `PLAN_NOT_FOUND`
- `PLAN_NOT_FAILED`
- `INVALID_USER_API_KEY`
- `USER_API_KEY_REQUIRED`
- `INSUFFICIENT_CREDITS`
- `INTERNAL_ERROR`
- `MODEL_PROFILES_UNAVAILABLE`
- `generation_failed`
- `content_unavailable`

Common local proxy error codes:
- `REMOTE_ERROR`
- `DOWNLOAD_FAILED`

Special case:
- `plan_file_info` may return `{}` while the artifact is not ready yet (not an error).

## Concurrency semantics (practical)

- Each `plan_create` call creates a new plan with a new `plan_id`.
- The server does not enforce a global “one active plan per client” cap.
- Parallelism is a client orchestration concern:
  - start with 1 plan
  - scale to 2 in parallel if needed
  - avoid more than 4 unless you have strong plan-tracking UX

## Typical Flow

### 1. Preview example plans (optional)

Call `example_plans` to see curated example plans with download links, so you can preview what PlanExe output looks like before creating your own plan.

Prompt:
```
Show me example plans.
```

Tool call:
```json
{}
```

### 2. Get example prompts

The user often starts with a vague idea. The AI calls `example_prompts` first to see what good prompts look like (around five samples, typically 300-800 words each), then expands the user’s idea into a high-quality prompt using this compact shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria.

Prompt:
```
Get example prompts for creating a plan.
```

Tool call:
```json
{}
```

### 3. Inspect model profiles (optional but recommended)

Prompt:
```
Show model profile options and available models.
```

Tool call:
```json
{}
```

### 4. Draft and approve the prompt (non-tool step)

At this step, the agent writes a high-quality prompt draft (typically 300-800 words, with objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria), shows it to the user, and waits for approval.

### 5. Create a plan

The user reviews the prompt and either asks for further changes or confirms it’s good to go. When the user confirms, the agent calls `plan_create` with that prompt.

Tool call:
```json
{"prompt": "..."}
```

### 6. Get status

Prompt:
```
Get status for my latest plan.
```

Tool call:
```json
{"plan_id": "<plan_id_from_plan_create>"}
```

If state is `failed`, optional retry:

Tool call:
```json
{"plan_id": "<plan_id_from_plan_create>", "model_profile": "baseline"}
```

### 7. Download the report

Prompt:
```
Download the report for my plan.
```

Tool call:
```json
{"plan_id": "<plan_id_from_plan_create>", "artifact": "report"}
```

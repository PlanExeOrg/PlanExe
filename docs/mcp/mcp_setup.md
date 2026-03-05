---
title: MCP setup
---

# MCP setup

This is the shortest path to a working PlanExe MCP integration.

---

## 1. Understand the flow

1. Ask for prompt examples.
2. Inspect `model_profile` options and available models.
3. Expand the user idea into a high-quality prompt (typically ~300-800 words) and get user approval.
   Use this compact shape: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria.
4. Create the plan.
5. Poll for status (about every 5 minutes).
6. If status is `failed`, optionally call `plan_retry` (defaults to `model_profile=baseline`).
7. Download artifacts via `plan_file_info` (cloud) or `plan_download` (mcp_local helper).

---

## 2. Minimal tool usage

1. `example_plans` (optional, preview example output)
2. `example_prompts`
3. `model_profiles`
4. `plan_create`
5. `plan_status`
6. `plan_retry` (optional, only for failed plans)
7. `plan_file_info`

Optional local helper:
- `plan_download` (provided by `mcp_local`, not `mcp_cloud`)

For `plan_create`:

- Visible arguments: `prompt` (required), `model_profile` (optional).
- Reference: [PlanExe MCP interface](planexe_mcp_interface.md#62-plan_create)

---

## 3. Success criteria

- You can fetch example prompts.
- You can create a plan.
- You can fetch artifact metadata/URLs with `plan_file_info` (and optionally save locally via `plan_download` when using `mcp_local`).
- Your client can parse `error.code` and `error.message` and handle `{}` from `plan_file_info` as "not ready yet".
- If running parallel work, your client tracks multiple `plan_id`s explicitly (server-side global cap is not enforced).

---

## Next steps

- Full tool details: [MCP details](mcp_details.md)
- Reference schema: [PlanExe MCP interface](planexe_mcp_interface.md)
- App setup guides: [Claude](claude.md), [Cursor](cursor.md), [Codex](codex.md), [LM Studio](lm_studio.md)

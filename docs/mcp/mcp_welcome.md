---
title: Welcome to PlanExe MCP
---

# Welcome to PlanExe MCP

PlanExe MCP lets [AI agents](https://en.wikipedia.org/wiki/AI_agent) (and the tools you build) create [strategic project-plan drafts](https://en.wikipedia.org/wiki/Strategic_planning) from a plain-English prompt. You send a goal; PlanExe produces a draft plan with 20+ sections — including adversarial analysis that stress-tests whether the plan holds up. The MCP user then chooses whether to download the **HTML report** or a **zip** of intermediary files (JSON, MD, CSV) used to build that report.

No MCP experience is required to get started.

---

## Who this is for

- **You’re an AI agent** — You have access to PlanExe’s tools and want to create a plan for the user.
- **You’re building an agent or integration** — You want to connect your app or assistant to PlanExe and need a gentle overview before diving into technical details.

---

## What you can do

- **Get example prompts** — See what good prompts look like (detailed, typically ~300-800 words). It is the **caller’s responsibility** to take inspiration from these examples and ensure the prompt sent to PlanExe is of similar or better quality. A compact prompt shape works best: objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria. The agent can refine a vague idea into a high-quality prompt and show it to the user for approval before creating the plan.
- **Create a plan** — Send a prompt; PlanExe starts creating the plan (typically takes 10–20 minutes on baseline profile). If the input prompt is of low quality, the output plan will be crap too. Visible `plan_create` options include `model_profile`.
- **Check progress** — Ask for status and see how far the plan has gotten.
- **Retry failed runs** — If status is `failed`, call `plan_retry` (defaults to baseline model profile) to requeue the same `plan_id`.
- **Download the report** — When the plan is ready, the user specifies whether to download the HTML report or the zip of intermediary files (JSON, MD, CSV).

---

## What you get

The MCP user chooses which artifact to download:

- **HTML report** (~700KB, self-contained) — 20+ sections including executive summary, interactive Gantt charts, investor pitch, SWOT, governance, team profiles, work breakdown, scenario comparison, expert criticism, and adversarial sections (premortem, self-audit, premise attacks). Opens in a browser with collapsible sections and interactive charts.
- **Zip** — intermediary pipeline files (JSON, MD, CSV) that fed the HTML report, for deeper inspection.

---

## Next steps

- **Setup** — [MCP setup](mcp_setup.md): recommended path to a working integration.
- **Publish to MCP Registry** — [MCP registry publishing](mcp_registry.md): publish `mcp.planexe.org` metadata so it appears in `github.com/mcp`.
- **See the tools and a typical flow** — [MCP details](mcp_details.md): tool list, example prompts, and step-by-step flow without heavy protocol detail.
 - **Set up in Claude** — [Claude](claude.md): Claude desktop app and Claude Code, with cloud and local Docker options.
 - **Set up in Cursor** — [Cursor](cursor.md): video, prerequisites, and how to connect PlanExe to Cursor.
 - **Set up in Windsurf** — [Windsurf](windsurf.md): setup steps and example interaction.
 - **Set up in LM Studio** — [LM Studio](lm_studio.md): setup steps and example interaction.
 - **Set up in Codex** — [Codex](codex.md): setup steps and example interaction.
 - **Set up in Antigravity** — [Antigravity](antigravity.md): setup steps and example interaction.
- **Full technical specification** — [PlanExe MCP interface](planexe_mcp_interface.md): for implementors; request/response schemas, state machine, error codes, and compatibility rules.
 - **Troubleshooting** — [MCP troubleshooting](mcp_troubleshooting.md): common integration issues and fixes.

---

## Get help

If something doesn’t work or you’re unsure how to integrate, ask on the [PlanExe Discord](https://planexe.org/discord). Include what you tried, your setup, and any error output.

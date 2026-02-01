---
title: Welcome to PlanExe MCP
---

# Welcome to PlanExe MCP

PlanExe MCP lets [AI agents](https://en.wikipedia.org/wiki/AI_agent) (and the tools you build) create [strategic plans](https://en.wikipedia.org/wiki/Strategic_planning) from a plain-English prompt. You send a goal; PlanExe produces a draft plan. The MCP user then chooses whether to download the **HTML report** or a **zip** of intermediary files (JSON, MD, CSV) used to build that report.

No MCP experience is required to get started.

---

## Who this is for

- **You’re an AI agent** — You have access to PlanExe’s tools and want to create a plan for the user.
- **You’re building an agent or integration** — You want to connect your app or assistant to PlanExe and need a gentle overview before diving into technical details.

---

## What you can do

- **Get example prompts** — See what good prompts look like (detailed, typically 300–800 words). It is the **caller’s responsibility** to take inspiration from these examples and ensure the prompt sent to PlanExe is of similar or better quality. The agent can refine a vague idea into a high-quality prompt and show it to the user for approval before creating the plan.
- **Create a plan** — Send a prompt; PlanExe starts creating the plan (takes about 15–20 minutes). If the input prompt is of low quality, the output plan will be crap too.
- **Check progress** — Ask for status and see how far the plan has gotten.
- **Download the report** — When the plan is ready, the user specifies whether to download the HTML report or the zip of intermediary files (JSON, MD, CSV).

---

## What you get

The MCP user chooses which artifact to download:

- **HTML report** (around 40 pages) — executive summary, Gantt chart, risks, next steps, and more. Opens in a browser.
- **Zip** — intermediary files (JSON, MD, CSV) used to build the HTML report, for deeper inspection.

---

## Next steps

- **Setup** — [MCP setup](mcp_setup.md): recommended path to a working integration.
- **See the tools and a typical flow** — [MCP details](mcp_details.md): tool list, example prompts, and step-by-step flow without heavy protocol detail.
 - **Set up in Cursor** — [Cursor](cursor.md): video, prerequisites, and how to connect PlanExe to Cursor.
 - **Set up in LM Studio** — [LM Studio](lm_studio.md): setup steps and example interaction.
 - **Set up in Codex** — [Codex](codex.md): setup steps and example interaction.
 - **Set up in Antigravity** — [Antigravity](antigravity.md): setup steps and example interaction.
- **Full technical specification** — [PlanExe MCP interface](planexe_mcp_interface.md): for implementors; request/response schemas, state machine, error codes, and compatibility rules.
 - **Troubleshooting** — [MCP troubleshooting](mcp_troubleshooting.md): common integration issues and fixes.

---

## Get help

If something doesn’t work or you’re unsure how to integrate, ask on the [PlanExe Discord](https://planexe.org/discord). Include what you tried, your setup, and any error output.

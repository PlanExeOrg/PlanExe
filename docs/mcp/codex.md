---
title: Codex - MCP integration
---

# OpenAI Codex

Guide for connecting [codex](https://openai.com/codex/) with PlanExe via MCP.

## Prerequisites

- Access to Codex.
- PlanExe MCP server reachable by Codex.

## Quick setup

1. Start Codex.
2. Ask for MCP tools.
3. Call `prompt_examples` to get examples.
4. Call `task_create` to start a plan.

## Sample prompt

> Get example prompts for creating a plan.

## Success criteria

- You can retrieve prompt examples.
- You can create a plan task.
- You can download the report.

## Interaction

In a terminal, start codex like this:

```bash
codex
```

Inside codex; these are my interactions:

1. tell me about the mcp tools you have access to
2. for planexe, get the prompt examples
3. I want you to formulate a prompt about constructing a new variant of english where the worst inconsistencies have been fixed such as 11th vs 1st, 21st, 31st,
  potentially eliminated such suffixes. And the pronounciation inconsistencies have been cleaned up. I want you to adhere to the planexe example prompts.
4. it's not just the ordinals. try again
5. go ahead create this plan
6. status
7. status
8. status
9. download both
10. summarize the html file

The created plan is here: [Clean English](https://planexe.org/20260131_clear_english_report.html)

## Prerequisites

A working installation of PlanExe.

- The recommended way is to install PlanExe by following the [Getting Started](../getting_started.md) instructions.
  Make sure that `docker compose up` is running, in order to connect to PlanExe.
- Alternatively: Run PlanExe on another server and port.
- Alternatively: If you are a developer run PlanExe inside a python virtual environment.

Double check that PlanExe can take a prompt and create a plan. Since it doesn't make sense to start configuring Cursor if the PlanExe installation is incomplete.


## Configuring Codex

[OpenAI's MCP documentation](https://developers.openai.com/codex/mcp/)

This is the command template. Make sure you tweak it, before running it.

```bash
codex mcp add planexe --env PLANEXE_URL="http://localhost:8001/mcp" --env PLANEXE_PATH="/Users/your-name/Desktop" -- uv run --with mcp /path/to/PlanExe/mcp_local/planexe_mcp_local.py
```

Make these adjustments to the command line.

- Make adjustments to `/path/to/PlanExe` so it points to where PlanExe is located on your computer.
- Make adjustments to `/Users/your-name/Desktop` so it points to the directory where PlanExe is allowed to write to, so the plan can be downloaded.
- Optional: Make adjustments to `http://localhost:8001/mcp` if you have PlanExe running on another port.

Verify that it's working.

```bash
codex mcp list  
Name Command Args Env Cwd Status Auth       
planexe  uv       run --with mcp /path/to/PlanExe/mcp_local/planexe_mcp_local.py  PLANEXE_PATH=*****, PLANEXE_URL=*****  -    enabled  Unsupported
```

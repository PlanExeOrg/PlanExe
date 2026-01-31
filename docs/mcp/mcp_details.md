# PlanExe MCP Details

MCP is work-in-progress, and I (Simon Strandgaard, the developer) may change it as I see fit.
If there is a particular tool you want. Write to me on the [PlanExe Discord](https://planexe.org/discord), and I will see what I can do.

This document lists the MCP tools exposed by PlanExe and example prompts for agents.

## Overview

- The primary MCP server runs in the cloud (see `mcp_cloud`).
- The local MCP proxy (`mcp_local`) forwards calls to the server and adds a local download helper.
- Tool responses return JSON in both `content.text` and `structuredContent`.

## Tool Catalog, `mcp_cloud`

### prompt_examples

Returns around five example prompts that show what good prompts look like. Each sample is typically 300–800 words: detailed context, requirements, and success criteria. Usually the AI does the heavy lifting: the user has a vague idea, the agent calls `prompt_examples`, then expands that idea into a high-quality prompt (300–800 words). The prompt is shown to the user, who can ask for further changes or confirm it’s good to go. When the user confirms, the agent then calls `task_create`. Shorter or vaguer prompts produce lower-quality plans.

Example prompt:
```
Get example prompts for creating a plan.
```

Example call:
```json
{}
```

Response includes `samples` (array of prompt strings, each 300–800 words) and `message`.

### task_create

Create a new plan task.

Example prompt:
> Create a plan for: Weekly meetup for humans where participants are randomly paired every 5 minutes...

Example call:
```json
{"prompt": "Weekly meetup for humans where participants are randomly paired every 5 minutes..."}
```

Optional argument:
```
speed_vs_detail: "ping" | "fast" | "all"
```

### task_status

Fetch status/progress and recent files for a task.

Example prompt:
```
Get status for task 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"task_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7"}
```

### task_stop

Request an active task to stop.

Example prompt:
```
Stop task 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"task_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7"}
```

### task_file_info

Return download metadata for report or zip artifacts.

Example prompt:
```
Get report info for task 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"task_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7", "artifact": "report"}
```

Available artifacts:
```
"report" | "zip"
```

## Tool Catalog, `mcp_local`

The local proxy exposes the same tools as the server, and adds:

### task_download

Download report or zip to a local path.

Example prompt:
```
Download the report for task 2d57a448-1b09-45aa-ad37-e69891ff6ec7.
```

Example call:
```json
{"task_id": "2d57a448-1b09-45aa-ad37-e69891ff6ec7", "artifact": "report"}
```

## Typical Flow

### 1. Get example prompts

The user often starts with a vague idea. The AI calls `prompt_examples` first to see what good prompts look like (around five samples, 300–800 words each), then expands the user’s idea into a high-quality prompt and shows it to the user.

Prompt:
```
Get example prompts for creating a plan.
```

Tool call:
```json
{}
```

### 2. Create a plan

The user reviews the prompt and either asks for further changes or confirms it’s good to go. When the user confirms, the agent calls `task_create` with that prompt.

Tool call:
```json
{"prompt": "..."}
```

### 3. Get status

Prompt:
```
Get status for my latest task.
```

Tool call:
```json
{"task_id": "<task_id_from_task_create>"}
```

### 4. Download the report

Prompt:
```
Download the report for my task.
```

Tool call:
```json
{"task_id": "<task_id_from_task_create>", "artifact": "report"}
```

# PlanExe MCP Tools - Experimental

MCP is work-in-progress, and I (Simon Strandgaard, the developer) may change it as I see fit.
If there is a particular tool you want. Write to me on the PlanExe Discord, and I will see what I can do.

This document lists the MCP tools exposed by PlanExe and example prompts for agents.

## Overview

- The primary MCP server runs in the cloud (see `mcp_cloud`).
- The local MCP proxy (`mcp_local`) forwards calls to the server and adds a local download helper.
- Tool responses return JSON in both `content.text` and `structuredContent`.

## Tool Catalog, `mcp_cloud`

### plan_generate

Generate a new plan. This tool supports MCP task augmentation (Run as task).

Example prompt:
```
Generate a plan for: Weekly meetup for humans where participants are randomly paired every 5 minutes...
```

Example call:
```json
{"idea": "Weekly meetup for humans where participants are randomly paired every 5 minutes..."}
```

Optional arguments:
```
speed_vs_detail: "ping" | "fast" | "all"
idempotency_key: "<string>"
```

### task_file_info (legacy)

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

## MCP Tasks Protocol

When calling `plan_generate`, clients may request a task-augmented run by adding
`task` in the MCP `tools/call` request params. The server exposes:

- `tasks/get` - fetch task status + poll interval
- `tasks/result` - wait until the task reaches a terminal status and return the tool result
- `tasks/cancel` - request cancellation
- `tasks/list` (optional)

## Tool Catalog, `mcp_local`

The local proxy exposes `plan_generate` plus MCP task methods, and adds:

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

### 1) Start a plan as a task

Prompt:
```
Generate a plan for this idea: Weekly meetup for humans where participants are randomly paired every 5 minutes...
```

Tool call (task-augmented):
```json
{"name": "plan_generate", "arguments": {"idea": "Weekly meetup for humans where participants are randomly paired every 5 minutes..."}, "task": {"ttl": 43200000}}
```

### 2) Check status or wait for result

Status:
```json
{"taskId": "<task_id_from_create_task_result>"}
```

Result (blocks until terminal):
```json
{"taskId": "<task_id_from_create_task_result>"}
```

### 3) Download the report (local proxy)

Prompt:
```
Download the report for my task.
```

Tool call:
```json
{"task_id": "<task_id_from_create_task_result>", "artifact": "report"}
```

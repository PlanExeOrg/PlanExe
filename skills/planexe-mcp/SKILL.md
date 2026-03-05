---
name: planexe-mcp
description: "OpenClaw skill for connecting to PlanExe via Model Context Protocol. Supports three deployment scenarios: cloud-hosted service, remote Docker, and local Docker."
version: "1.0.0"
author: "PlanExe Team"
---

# PlanExe MCP Skill

Access PlanExe's powerful planning capabilities directly from OpenClaw via the Model Context Protocol (MCP).

## Overview

The `planexe-mcp` skill provides a unified interface to interact with PlanExe across three deployment scenarios:

- **Cloud**: `mcp.planexe.org` (Stripe credits at home.planexe.org)
- **Remote Docker**: Container on a separate machine
- **Local Docker**: Container on the same machine as OpenClaw

## Setup by Deployment Scenario

### Scenario A: Cloud-Hosted Service

Connect to the hosted PlanExe MCP service.

**Configuration:**

Add to your OpenClaw config or environment:

```bash
PLANEXE_MCP_URL=https://mcp.planexe.org
PLANEXE_API_KEY=your_api_key_from_planexe_account
```

**How to Get Your API Key:**

1. Visit [home.planexe.org](https://home.planexe.org)
2. Purchase credits via Stripe
3. Generate an API key in your account dashboard
4. Copy the key to your OpenClaw configuration

---

### Scenario B: Remote Docker

Run PlanExe in Docker on a separate machine and connect remotely.

**On the Remote Machine:**

```bash
docker run -p 8001:8001 planexe/planexe:latest
```

**Configuration in OpenClaw:**

```bash
PLANEXE_MCP_URL=http://<remote-ip>:8001
PLANEXE_API_KEY=your_api_key
```

Replace `<remote-ip>` with your remote machine's IP address or hostname.

---

### Scenario C: Local Docker

Run PlanExe in Docker on the same machine as OpenClaw.

**Setup:**

```bash
docker run -p 8001:8001 planexe/planexe:latest
```

**Configuration in OpenClaw:**

```bash
PLANEXE_MCP_URL=http://127.0.0.1:8001
PLANEXE_API_KEY=your_api_key
```

---

## Invoking PlanExe Tools

The PlanExe MCP exposes nine core tools via the `/mcp` endpoint:

### Tool 1: `example_plans`

Preview curated example plans with download links for reports and zip bundles.

**No parameters required:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "example_plans",
    "arguments": {}
  }
}
```

**Returns:** List of example plans with `title`, `report_url`, and `zip_url`.

---

### Tool 2: `example_prompts`

Get example prompts to understand what PlanExe can do.

**No parameters required:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "example_prompts",
    "arguments": {}
  }
}
```

**Returns:** List of example prompts for planning tasks.

---

### Tool 3: `plan_create`

Create a new planning task. This is the main entry point for generating plans.

**Parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "plan_create",
    "arguments": {
      "prompt": "Create a project launch plan for Q2 2026",
      "model_profile": "premium",
      "user_api_key": "your_optional_api_key"
    }
  }
}
```

**Parameter Guide:**
- `prompt` (required): Your planning request in natural language. Write as flowing prose (not structured markdown), typically 300-800 words.
- `model_profile` (optional): One of `"baseline"`, `"premium"`, `"frontier"`, or `"custom"`. Defaults to `"baseline"`.
- `user_api_key` (optional): Your PlanExe API key (if not set in environment)

**Returns:** `task_id` for polling status and retrieving results.

---

### Tool 4: `plan_status`

Poll the status of a running planning task.

**Parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "plan_status",
    "arguments": {
      "task_id": "task_abc123def456"
    }
  }
}
```

**Usage:** Planning tasks typically take 10-20 minutes (baseline profile). Poll every 5+ minutes to check progress.

**Returns:** Current status (`pending`, `processing`, `completed`, `failed`), and progress percentage.

---

### Tool 5: `plan_stop`

Stop a running planning task.

**Parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "plan_stop",
    "arguments": {
      "task_id": "task_abc123def456"
    }
  }
}
```

**Returns:** Confirmation that the task has been stopped.

---

### Tool 6: `model_profiles`

Return available model profiles and their guidance before calling `plan_create`.

**No required parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "model_profiles",
    "arguments": {}
  }
}
```

**Returns:** Available model profiles (`baseline`, `premium`, `frontier`, `custom`) with descriptions and currently loaded models.

---

### Tool 7: `plan_file_info`

Retrieve download information for completed plan artifacts.

**Parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "plan_file_info",
    "arguments": {
      "task_id": "task_abc123def456",
      "artifact": "report"
    }
  }
}
```

**Artifact Options:**
- `"report"`: Interactive HTML report (~700KB, self-contained — open in a browser)
- `"zip"`: Pipeline output bundle (md, json, csv intermediary files)

**Returns:** `download_url` for accessing the artifact.

---

### Tool 8: `plan_list`

List recent tasks for an authenticated user. Useful for recovering a lost `task_id`.

**Parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "plan_list",
    "arguments": {
      "user_api_key": "pex_your_key_here",
      "limit": 10
    }
  }
}
```

**Returns:** List of recent tasks with `task_id`, `state`, `progress_percentage`, `created_at`, and `prompt_excerpt`.

---

### Tool 9: `plan_retry`

Retry a failed task with an optional upgraded model profile.

**Parameters:**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "plan_retry",
    "arguments": {
      "task_id": "task_abc123def456",
      "model_profile": "premium"
    }
  }
}
```

**Returns:** Confirmation that the task was requeued to `pending` state.

---

## Typical Workflow

1. Optionally call `example_plans` to preview what PlanExe output looks like
2. Call `example_prompts` to understand available planning scenarios
3. Optionally call `model_profiles` to choose an appropriate `model_profile`
4. Formulate your planning prompt
5. Get user approval for the request
6. Call `plan_create` with your prompt and parameters → receives `task_id`
7. Poll `plan_status` every 5+ minutes until status is `completed` or `failed`
8. If `failed`, optionally call `plan_retry` to requeue with a stronger model
9. Call `plan_file_info` with completed `task_id` to get download link
10. Download and use the generated plan
11. If you lose a `task_id`, call `plan_list` with your `user_api_key` to recover it

Refer to the [PlanExe API documentation](https://planexe.org/docs) for extended examples and advanced use cases.

## Configuration Reference

| Variable | Example | Description |
|----------|---------|-------------|
| `PLANEXE_MCP_URL` | `https://mcp.planexe.org` | MCP endpoint URL |
| `PLANEXE_API_KEY` | `pk_live_abc123...` | API authentication key |

## Troubleshooting

**Connection Refused**
- Verify `PLANEXE_MCP_URL` is correct
- For Docker: Ensure the container is running (`docker ps`)
- Check network connectivity between OpenClaw and the service

**Authentication Failed**
- Verify `PLANEXE_API_KEY` is valid
- Check for typos or expired credentials
- For cloud: Log in to [home.planexe.org](https://home.planexe.org) to verify account status

**Timeout Errors**
- For Docker deployments, allow time for container initialization
- Check if the service is overloaded
- Consider increasing the request timeout in your OpenClaw config

**Docker Issues (Local/Remote)**
- Verify Docker is installed: `docker --version`
- Check container logs: `docker logs <container-id>`
- Ensure port 8001 is not in use: `lsof -i :8001` (macOS/Linux)

## Support

For issues or questions:
- Check the [PlanExe documentation](https://planexe.org/docs)
- Visit the [PlanExe community forum](https://community.planexe.org)
- Open an issue on [GitHub](https://github.com/PlanExeOrg/PlanExe)

## License

This skill is part of PlanExe and is available under the same license as the PlanExe project.

---
title: Claude - MCP integration
---

# Claude

[Claude](https://claude.ai/) is available as a desktop app and as [Claude Code](https://docs.anthropic.com/en/docs/claude-code), Anthropic's CLI tool. Both support MCP and are configured the same way.

PlanExe turns a plain-English goal into a strategic project-plan draft (20+ sections) in ~10-20 minutes. The output is a self-contained interactive HTML report you open in a browser.

## Video walkthrough

<iframe width="560" height="315" src="https://www.youtube.com/embed/dhrgwW-8rl4?si=Hkc_e5onS6xD1Aca" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Prerequisites

- Claude desktop app or Claude Code installed.
- One of the following:
  - An API key from [home.planexe.org](https://home.planexe.org/) (cloud server, no installation needed).
  - PlanExe running locally via Docker (`docker compose up`, port 8001).

## Quick setup

1. Configure MCP in Claude (see connection options below).
2. Verify the connection with `/mcp` in Claude Code or Settings > MCP in the desktop app.
3. Ask Claude to create a plan — it handles the full workflow (prompt drafting, creation, status polling, download).

## Success criteria

- `/mcp` (Claude Code) or Settings > MCP (desktop) shows `planexe` as connected.
- You can fetch prompt examples (`example_prompts`).
- You can create a plan (`plan_create`) and poll it to completion (`plan_status`).
- You can download the report (`plan_file_info` or `plan_download`).

---

## Option A: Connect to the cloud server (mcp.planexe.org)

This is the fastest way to get started. No Docker or local installation required.

### 1. Get an API key

Create an account and obtain an API key at [home.planexe.org](https://home.planexe.org/).
Your key will be prefixed with `pex_`.

### 2. Add the MCP server

Run this command in your terminal:

```bash
claude mcp add --transport http \
  planexe \
  https://mcp.planexe.org/mcp \
  --header "X-API-Key: pex_YOUR_API_KEY"
```

Replace `pex_YOUR_API_KEY` with your actual API key.

### 3. Verify

Start Claude and check that the server is connected.

In Claude Code, type `/mcp` to see the server status. In the Claude desktop app, go to Settings and check the MCP section. You should see `planexe` listed with its tools (`example_plans`, `example_prompts`, `model_profiles`, `plan_create`, `plan_status`, `plan_stop`, `plan_retry`, `plan_file_info`, `plan_list`).

---

## Option B: Run Docker locally + connect directly via HTTP

This connects Claude directly to the local MCP server over HTTP. No `mcp_local` proxy needed.

### 1. Start PlanExe locally

Follow the [Getting Started](../getting_started.md) instructions to set up PlanExe, then start the services:

```bash
docker compose up
```

Wait until the `mcp_cloud` service is healthy (listening on port 8001).

### 2. Add the MCP server

```bash
claude mcp add --transport http \
  planexe \
  http://localhost:8001/mcp
```

Authentication is disabled by default for local Docker (`PLANEXE_MCP_REQUIRE_AUTH=false`), so no API key is needed.

### 3. Verify

In Claude Code, type `/mcp` to see the server status. In the Claude desktop app, check Settings > MCP.

> **Note:** With this option, `plan_file_info` returns a `download_url`. Ask Claude to fetch it, or open the URL in your browser. For local disk saves, use Option C instead (adds the `plan_download` tool).

---

## Option C: Run Docker locally + use the mcp_local proxy

The `mcp_local` proxy runs as a stdio process and forwards calls to the Docker MCP server. It adds the `plan_download` tool which saves artifacts directly to disk.

### 1. Start PlanExe locally

Follow the [Getting Started](../getting_started.md) instructions, then:

```bash
docker compose up
```

### 2. Add the MCP server

```bash
claude mcp add --transport stdio \
  --env PLANEXE_URL="http://localhost:8001/mcp" \
  --env PLANEXE_PATH="/Users/your-name/Desktop" \
  planexe \
  -- uv run --with mcp /path/to/PlanExe/mcp_local/planexe_mcp_local.py
```

Make these adjustments:

- Replace `/path/to/PlanExe` with the actual path to your PlanExe clone.
- Replace `/Users/your-name/Desktop` with the directory where downloaded plans should be saved.
- Optional: Adjust `http://localhost:8001/mcp` if PlanExe is running on a different port.

### 3. Verify

In Claude Code, type `/mcp` to see the server status. In the Claude desktop app, check Settings > MCP.

---

## Using mcp_local with the cloud server

You can also use the `mcp_local` proxy to connect to the cloud server. This gives you the `plan_download` tool while using the hosted service:

```bash
claude mcp add --transport stdio \
  --env PLANEXE_URL="https://mcp.planexe.org/mcp" \
  --env PLANEXE_MCP_API_KEY="pex_YOUR_API_KEY" \
  --env PLANEXE_PATH="/Users/your-name/Desktop" \
  planexe \
  -- uv run --with mcp /path/to/PlanExe/mcp_local/planexe_mcp_local.py
```

---

## Alternative: manual JSON configuration

Instead of using `claude mcp add`, you can create a `.mcp.json` file in your project root.

**Cloud server (HTTP):**

```json
{
  "mcpServers": {
    "planexe": {
      "type": "http",
      "url": "https://mcp.planexe.org/mcp",
      "headers": {
        "X-API-Key": "pex_YOUR_API_KEY"
      }
    }
  }
}
```

**Local Docker (HTTP):**

```json
{
  "mcpServers": {
    "planexe": {
      "type": "http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

**Local proxy (stdio):**

```json
{
  "mcpServers": {
    "planexe": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp",
        "/path/to/PlanExe/mcp_local/planexe_mcp_local.py"
      ],
      "env": {
        "PLANEXE_URL": "http://localhost:8001/mcp",
        "PLANEXE_PATH": "/Users/your-name/Desktop"
      }
    }
  }
}
```

---

## Managing the MCP server

```bash
# List configured servers
claude mcp list

# Get details for the planexe server
claude mcp get planexe

# Remove the server
claude mcp remove planexe
```

---

## Interaction

A typical conversation for creating a plan looks like this:

1. **Explore** — "Tell me about the PlanExe MCP tools you have access to."
2. **Get examples** — "Get the prompt examples."
3. **Describe your goal** — "I want a prompt about building a community solar farm in rural Denmark."
   Claude drafts a detailed prompt (~300-800 words) based on the examples and your idea.
4. **Approve and create** — "Go ahead, create this plan."
   Claude calls `plan_create`, which returns a `plan_id`.
5. **Wait** — Plan generation takes ~10-20 minutes. Claude polls `plan_status` automatically every few minutes. Alternatively, `plan_create` returns an `sse_url` — a GET endpoint (text/event-stream) that streams real-time progress events until the plan completes. Claude Code agents can run `curl -N <sse_url>` in a background shell to monitor progress instead of polling.
6. **Download** — "Download the report." Claude fetches the HTML report via `plan_file_info` (cloud) or `plan_download` (local proxy).

If a plan fails, Claude can retry it with `plan_retry`. If a `plan_id` is lost, `plan_list` recovers recent plans.

---

## Troubleshooting

- **Server disconnected**: `/mcp` shows disconnected — check that Docker is running (`docker compose ps`) or that `mcp.planexe.org` is reachable.
- **Authentication errors**: Verify your API key at [home.planexe.org](https://home.planexe.org/). Keys are prefixed with `pex_`.
- **stdio transport**: Make sure `uv` is installed and on your PATH.
- **Plan stuck in pending**: If `plan_status` stays `pending` for >5 minutes, the worker likely hasn't picked it up. Check Docker logs or report the issue.
- **Plan failed**: Call `plan_retry` to requeue the same `plan_id` (defaults to baseline profile).
- For more help, see the [Troubleshooting guide](mcp_troubleshooting.md) or ask on the [PlanExe Discord](https://planexe.org/discord).

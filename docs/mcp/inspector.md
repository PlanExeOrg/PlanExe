# MCP Inspector

Inspecting PlanExe's MCP server. 

This is my (Simon Strandgaard) preferred way to troubleshoot MCP. Whenever there is a problem, the **MCP Inspector** is the **HAMMER**.

Locations: [Github](https://github.com/modelcontextprotocol/inspector), [Documentation](https://modelcontextprotocol.io/docs/tools/inspector)

## Overview of PlanExe's MCP servers

PlanExe has multiple MCP servers that can be connected to.

|#|Difficulty|Description|
|-|----------|-----------|
| 1 | Beginner | MCP server at [mcp.planexe.org/](https://mcp.planexe.org/) and cost credits to use. Manage your credits via this page: [home.planexe.org](https://home.planexe.org) |
| 2 | Medium | MCP server inside docker on your own computer. I recommend using OpenRouter for inference, which cost money. You can be lucky finding a free model on OpenRouter, but this requires developer skills and several attempts. You can also run models on your own computer. |
| 3 | Expert | MCP server as a python program on your own computer. |

## Approach 1. MCP server at mcp.planexe.org

### Purchase credits

1. Open [home.planexe.org](https://home.planexe.org)
2. Sign in with Google
3. Buy credits for 6 USD. It cost around 1.4 USD to create one plan, so that should be around 4 plans.
4. Click `Generate new API key` and copy it to clipboard. You will need this API key, in order to connect to the server.

### Connect to MCP Server

```bash
npx @modelcontextprotocol/inspector --transport http --server-url https://mcp.planexe.org/mcp/
```

> **URL note**: The canonical server URL is `https://mcp.planexe.org/mcp` (no trailing slash) — use this in JSON config files and registry entries. The trailing slash (`/mcp/`) is used above because the MCP Inspector CLI appends sub-paths to the base URL; without it the inspector may send requests to the wrong path. Both forms are accepted by the server.

This opens the inspector in a browser

![screenshot of mcp inspector](inspector_step1_mcp_planexe_org.webp)

In the left sidebar; Expand the `Authentication` section.

![screenshot of mcp inspector with authentication expanded](inspector_step2_mcp_planexe_org.webp)

This is what the custom headers should look:
![screenshot of mcp inspector with custom headers](inspector_step3_mcp_planexe_org.jpg)

**Do not use OAuth** – PlanExe uses API keys, not OAuth. The OAuth flow will fail with "Failed to discover OAuth metadata".

1. Use **Custom Headers** instead: click `+ Add` inside the Custom Headers section.
2. In `Header Name`, insert `X-API-Key`.
3. In `Header Value`, insert your API key (e.g. `pex_...`).
4. Click **Connect**.


If `Connect` fails with this error: *"Connection Error - Check if your MCP server is running and proxy token is correct"*. This can happen if the `Authentication` section has incorrect data, so double check for typos.

If `Connect` fails with this error: *Connection Failed: "TypeError: NetworkError when attempting to fetch resource."*. This can happen if the `Authentication` section has incorrect data, so double check for typos.

If `Connect` still fails, then please report your issue on [Discord](https://planexe.org/discord). 

### When connected

When connected follow these steps:

1. In the topbar; Click on the `Tools` tab.
2. In the `Tools` panel; Click on the `List Tools` button.

![screenshot of mcp inspector just connected](inspector_step4_mcp_planexe_org.webp)

Now there should be a list with tool names and descriptions:
```
example_plans
example_prompts
model_profiles
plan_create
plan_status
plan_stop
plan_retry
plan_file_info
```

When you inspect `plan_create`, the visible input schema includes `prompt` and optional `model_profile`.

Follow these steps:
![screenshot of mcp inspector invoke tool](inspector_step5_mcp_planexe_org.webp)

1. In the `Tools` panel; Click on the `example_plans` tool to preview example output, or `example_prompts` to see prompt examples.
2. In the right sidepanel; Click on `Run Tool`.
3. The MCP server should respond with example plans or example prompts.
4. Optionally run `model_profiles` to inspect available `model_profile` choices before `plan_create`.

## Approach 2. MCP server inside docker

### Prerequisites

I assume you are able to create plans on your computer via the `frontend_single_user` web interface, [http://localhost:7860/](http://localhost:7860/). It doesn't make sense proceeding if there is a problem with LLMs and no plans can be created.

### Start docker

PlanExe's docker stack exposes the MCP endpoint on your loopback interface (default `127.0.0.1:8001/mcp/`). Start with `docker compose up` and wait until you see `mcp_cloud` and `/healthcheck` like this:
```
mcp_cloud | INFO: 127.0.0.1:43988 - "GET /healthcheck HTTP/1.1" 200 OK
```

### Start inspector

In a separate terminal; launch the inspector.

```bash
npx @modelcontextprotocol/inspector --transport http --server-url http://localhost:8001/mcp/
```

![Screenshot of starting inspector from commandline](inspector_step1_docker.webp)

Once the UI opens in the browser, keep `Authentication` empty and click `Connect`.

![Screenshot click connect](inspector_step2_docker.webp)

### When connected

Then open the `Tools` tab, click `List Tools`.

![Screenshot list tools](inspector_step3_docker.webp)

Click `example_prompts`, click `Run Tool`.

![Screenshot example prompts](inspector_step4_docker.webp)


## Approach 3. MCP server as a python program

If MCP had a built-in download mechanism, then there wouldn't be a need for this python program. As of 2026-Feb-12 MCP doesn't have such download mechanism, and developers make kludgy workarounds. The `mcp_local/planexe_mcp_local.py` proxy runs a tiny Python MCP server that forwards tool calls to the remote `mcp_cloud` while downloading reports into a local directory.

### Prerequisites

I assume that you already have verified that things are working in "Approach 2. MCP server inside docker". If things are broken there, it makes no sense following the instructions here.

### Start docker

PlanExe's docker stack exposes the MCP endpoint on your loopback interface (default `127.0.0.1:8001/mcp/`). Start with `docker compose up` and wait until you see `mcp_cloud` and `/healthcheck` like this:
```
mcp_cloud | INFO: 127.0.0.1:43988 - "GET /healthcheck HTTP/1.1" 200 OK
```

### Start inspector

In a separate terminal; launch the inspector.

On my computer, I launch the inspector like this:

![Screenshot example prompts](inspector_step1_local.webp)

You have to make these adjustments for your computer.

The `PLANEXE_PATH` is an absolute directory that PlanExe is allowed to write to. The downloaded files lands here.

The `/absolute/path/to/PlanExe` is where you have cloned the PlanExe repo.

```bash
npx @modelcontextprotocol/inspector \
  -e "PLANEXE_URL=http://localhost:8001/mcp" \
  -e "PLANEXE_PATH=/absolute/path/for/downloads" \
  --transport stdio \
  uv run --with mcp /absolute/path/to/PlanExe/mcp_local/planexe_mcp_local.py
```

Once the UI opens in the browser, click `Connect`.

![Screenshot example prompts](inspector_step2_local.webp)

### When connected

Then open the `Tools` tab, click `List Tools`.

![Screenshot example prompts](inspector_step3_local.webp)

Click `example_prompts`, click `Run Tool`.

![Screenshot example prompts](inspector_step4_local.webp)

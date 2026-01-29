# PlanExe MCP locally - Experimental, likely to be changed a lot!

Model Context Protocol (MCP) local proxy for PlanExe.

It runs on the user's computer and provides local disk access for downloads.
The pipeline still runs in `mcp_cloud`, the MCP server running in the cloud; this
proxy forwards tool calls over HTTP and downloads artifacts from `/download/{task_id}/...`.

## Tools

`prompt_examples` - Return example prompts. Use these as examples for task_create. You can also call `task_create` with any prompt—short prompts produce less detailed plans.
`task_create` - Initiate creation of a plan.
`task_status` - Get status and progress about the creation of a plan.
`task_stop` - Abort creation of a plan.
`task_download` - Download the plan, either html report or a zip with everything, and save it to disk.

**Tip**: Call `prompt_examples` to get example prompts to use with task_create. The full catalog lives at `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl`.

`task_download` is a synthetic tool provided by the local proxy. It calls the
remote MCP tool `task_file_info` to obtain a download URL, then downloads the
file locally into `PLANEXE_PATH`.

## Run as task (MCP tasks protocol)

Some MCP clients (e.g. the MCP Inspector) show a **"Run as task"** option for tools. That refers to the MCP **tasks** protocol: a separate mechanism where the client runs a tool in the background using RPC methods like `tasks/run`, `tasks/get`, `tasks/result`, and `tasks/cancel`, instead of a single blocking tool call.

**PlanExe does not use or advertise the MCP tasks protocol.** Our interface is **tool-based** only: the agent calls `task_create` → gets a `task_id` → polls `task_status` → uses `task_download`. That flow is defined in `docs/planexe_mcp_interface.md` and is the intended design.

You should **not** enable "Run as task" for PlanExe. The Python MCP SDK and clients like Cursor do not properly support the tasks protocol (method registration and initialization fail). Use the tools directly: create a task, poll status, then download when done.

## How it talks to mcp_cloud

- The remote base URL is `PLANEXE_URL` (for example `http://localhost:8001/mcp`).
- Tool calls prefer the remote HTTP wrapper (`/mcp/tools/call`).
- If the HTTP wrapper is unavailable, the proxy falls back to MCP JSON-RPC
  over `POST /mcp` (not SSE).
- Downloads use the remote `/download/{task_id}/...` endpoints.
- Authentication uses `PLANEXE_MCP_API_KEY` as a `Bearer` token.
- **Retry behavior**: Transient failures (server 5xx errors, network timeouts) are
  automatically retried up to 3 times with exponential backoff (1s, 2s delays).
  Client errors (4xx) are not retried. Retries are logged at WARNING level.

## Debugging with MCP Inspector

Run the MCP inspector with the local script and environment variables:

```bash
npx @modelcontextprotocol/inspector \
  -e "PLANEXE_URL"="http://localhost:8001/mcp" \
  -e "PLANEXE_MCP_API_KEY"="insert-your-api-key-here" \
  -e "PLANEXE_PATH"="/Users/your-name/Desktop" \
  --transport stdio \
  uv run --with mcp /absolute/path/to/PlanExe/mcp_local/planexe_mcp_local.py
```

Then click "Connect", open "Tools", and use "List Tools" or invoke individual tools.

## Client configuration (local script)

Clone the [PlanExe repository](https://github.com/neoneye/PlanExe) on your computer.
Use the absolute path to `planexe_mcp_local.py` and set `PLANEXE_PATH` to a
directory where PlanExe is allowed to save files.

### Local Docker (development)

```json
"planexe": {
  "command": "uv",
  "args": [
    "run",
    "--with",
    "mcp",
    "/absolute/path/to/PlanExe/mcp_local/planexe_mcp_local.py"
  ],
  "env": {
    "PLANEXE_URL": "http://localhost:8001/mcp",
    "PLANEXE_MCP_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop"
  }
}
```

### Remote server (Railway or cloud)

```json
"planexe": {
  "command": "uv",
  "args": [
    "run",
    "--with",
    "mcp",
    "/absolute/path/to/PlanExe/mcp_local/planexe_mcp_local.py"
  ],
  "env": {
    "PLANEXE_URL": "https://your-railway-app.up.railway.app/mcp",
    "PLANEXE_MCP_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop"
  }
}
```

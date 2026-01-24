# PlanExe MCP locally

Model Context Protocol (MCP) local proxy for PlanExe.

It runs on the user's computer and provides local disk access for downloads.
The pipeline still runs in `mcp_cloud`, the MCP server running in the cloud; this
proxy forwards tool calls over HTTP and downloads artifacts from `/download/{task_id}/...`.

## Tools

`plan_generate` - Generate a new plan (supports MCP task augmentation).
Legacy wrappers: `task_create`, `task_status`, `task_stop`.
`task_download` - Download the plan (HTML report or zip), and save it to disk.

`task_download` is a synthetic tool provided by the local proxy. It calls the
remote MCP tool `task_file_info` to obtain a download URL, then downloads the
file locally into `PLANEXE_PATH`.

## How it talks to mcp_cloud

- The remote base URL is `PLANEXE_URL` (for example `http://localhost:8001/mcp`).
- Tool calls prefer the remote HTTP wrapper (`/mcp/tools/call`).
- If the HTTP wrapper is unavailable, the proxy falls back to MCP JSON-RPC
  over `POST /mcp` (not SSE).
- Downloads use the remote `/download/{task_id}/...` endpoints.
- Authentication uses `PLANEXE_MCP_API_KEY` as a `Bearer` token.

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

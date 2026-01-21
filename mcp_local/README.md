# PlanExe MCP locally

Model Context Protocol (MCP) local proxy for PlanExe.

This runs on the user's computer and provides disk access for downloading files.
It does not run the pipeline itself. Instead it forwards MCP tool calls to the
remote `PlanExe/mcp_server` over HTTP, then downloads artifacts from the remote
`/download/{task_id}/...` endpoints.

## Tools

`task_create` - Initiate creation of a plan.
`task_status` - Get status and progress about the creation of a plan.
`task_stop` - Abort creation of a plan.
`task_download` - Download the plan, either html report or a zip with everything, and save it to disk.

`task_download` calls the remote MCP tool `task_file_info` to obtain a download URL,
then downloads the file locally into `PLANEXE_PATH`.

## How it talks to mcp_server

- The remote base URL is `PLANEXE_URL` (for example `http://localhost:8001/mcp`).
- Tool calls are forwarded to the remote MCP HTTP endpoint (`/mcp/tools/call`).
- If the HTTP wrapper is not available, the proxy falls back to MCP JSON-RPC
  at `/mcp`.
- Downloads use the remote `/download/{task_id}/...` endpoints.

## Debugging with MCP Inspector

Run the MCP inspector with the local script and environment variables:

```bash
npx @modelcontextprotocol/inspector \
  -e "PLANEXE_URL"="http://localhost:8001/mcp" \
  -e "PLANEXE_API_KEY"="insert-your-api-key-here" \
  -e "PLANEXE_PATH"="/Users/your-name/Desktop" \
  --transport stdio \
  uv run --with mcp /absolute/path/to/PlanExe/mcp_local/planexe_mcp_local.py
```

Then click "Connect", open "Tools", and use "List Tools" or invoke individual tools.

Here is what I imagine what it will be like:

### Development on localhost

Clone the [PlanExe repository](https://github.com/neoneye/PlanExe) on your computer.

Obtain absolute path to the `planexe_mcp_local.py` file, and insert it into the following snippet.

Update `PLANEXE_PATH` so it's an absolute path to where PlanExe is allowed to manipulate files.

The following is the code snippet that you have to paste into `mcp.json` (or similar named file).

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
    "PLANEXE_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop"
  }
}
```


### Future plan: Connect to docker on localhost

In order to use `"@planexe/mcp"`, it requires that PlanExe gets deployed as a package.

```json
"planexe": {
  "command": "uvx",
  "args": [
    "-y",
    "@planexe/mcp"
  ],
  "env": {
    "PLANEXE_URL": "http://localhost:8001/mcp",
    "PLANEXE_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop"
  }
}
```

### Future plan: Connect to PlanExe server hosted on Railway

In order to use `"@planexe/mcp"`, it requires that PlanExe gets deployed as a package.

When omitting `PLANEXE_URL`, the MCP script uses `https://your-railway-app.up.railway.app/mcp`.

```json
"planexe": {
  "command": "uvx",
  "args": [
    "-y",
    "@planexe/mcp"
  ],
  "env": {
    "PLANEXE_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop"
  }
}
```

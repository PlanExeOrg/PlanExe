# PlanExe MCP locally

Model Context Protocol (MCP) interface for PlanExe.

This runs on the users own computer, so it has disk access disk.
Unlike the `PlanExe/mcp_server` that runs in the cloud and has no disk access.

## Tools

`task_create` - Initiate creation of a plan.
`task_status` - Get status and progress about the creation of a plan.
`task_stop` - Abort creation of a plan.
`task_download` - Download the plan, either html report or a zip with everything, and save it to disk.

`task_download` calls the remote MCP tool `task_file_info` to obtain a download URL,
then downloads the file locally into `PLANEXE_PATH`.

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
    "PLANEXE_PATH": "/User/your-name/Desktop",
  }
}
```


### Connect to docker on localhost

```json
"planexe": {
  "command": "uvx",
  "args": [
    "-y",
    "@planexe/v1/mcp"
  ],
  "env": {
    "PLANEXE_URL": "http://localhost:8001/mcp",
    "PLANEXE_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop",
  }
}
```

### Connect to PlanExe server hosted on Railway

When omitting `PLANEXE_URL`, the MCP script uses `https://your-railway-app.up.railway.app/mcp`.

```json
"planexe": {
  "command": "uvx",
  "args": [
    "-y",
    "@planexe/v1/mcp"
  ],
  "env": {
    "PLANEXE_API_KEY": "insert-your-api-key-here",
    "PLANEXE_PATH": "/User/your-name/Desktop",
  }
}
```

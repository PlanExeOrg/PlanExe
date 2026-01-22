# mcp_local agent instructions

Scope: local MCP proxy script that runs on the user's machine and forwards tool calls
to the remote PlanExe MCP server (`mcp_server`) over HTTP.

## Interaction model
- The local proxy exposes MCP tools over stdio and forwards requests to the remote
  MCP server using `PLANEXE_URL` (defaults to the hosted `/mcp` endpoint).
- Supported tools: `task_create`, `task_status`, `task_stop`, `task_download`.
- `task_download` calls the remote `task_file_info` tool to obtain a download URL,
  then downloads the artifact to `PLANEXE_PATH` on the local machine.

## Constraints
- Do not add dependencies outside the existing runtime (stdlib + `mcp`).
- Keep remote requests compatible with both:
  - HTTP wrapper (`/mcp/tools/call`)
  - Streamable MCP JSON-RPC (`/mcp`)
- Ensure all tool responses include structured content when an output schema is defined.

## Env vars
- `PLANEXE_URL`: Base URL for the remote MCP server (e.g., `http://localhost:8001/mcp`).
- `PLANEXE_MCP_API_KEY`: API key passed as `Authorization: Bearer ...` if provided.
- `PLANEXE_API_KEY`: Legacy alias (fallback).
- `PLANEXE_PATH`: Local directory where downloads are saved.

# mcp_local agent instructions

Scope: local MCP proxy script that runs on the user's machine and forwards tool calls
to mcp_cloud, a MCP server running in the cloud, over HTTP.

## Interaction model
- The local proxy exposes MCP tools over stdio and forwards requests to mcp_cloud
  using `PLANEXE_URL` (defaults to the hosted `/mcp` endpoint).
- Supported tools: `task_create`, `task_status`, `task_stop`, `task_download`.
- `task_download` calls the remote `task_file_info` tool to obtain a download URL,
  then downloads the artifact to `PLANEXE_PATH` on the local machine.

## Constraints
- Do not add dependencies outside the existing runtime (stdlib + `mcp`).
- Keep remote requests compatible with both:
  - HTTP wrapper (`/mcp/tools/call`)
  - Streamable MCP JSON-RPC (`/mcp`)
- Ensure all tool responses include structured content when an output schema is defined.
- **Run as task**: Do not advertise the MCP **tasks** protocol (tasks/get, tasks/result, tasks/cancel, tasks/list) or add tool-level "Run as task" support. PlanExe’s interface is tool-based only (task_create → task_status → task_download). The MCP tasks protocol is a different, client-driven feature; Cursor and the Python MCP SDK do not support it properly, so we keep tools-only for compatibility.

## Env vars
- `PLANEXE_URL`: Base URL for mcp_cloud (e.g., `http://localhost:8001/mcp`).
- `PLANEXE_MCP_API_KEY`: API key passed as `Authorization: Bearer ...` if provided.
- `PLANEXE_PATH`: Local directory where downloads are saved.

# PlanExe MCP locally

Model Context Protocol (MCP) local proxy for PlanExe.

It runs on the user's computer and provides local disk access for downloads.
The pipeline still runs in `mcp_cloud`, the MCP server running in the cloud; this
proxy forwards tool calls over HTTP and downloads artifacts from `/download/{plan_id}/...`.

## Tools

`example_plans` - Return curated example plans with download links for reports and zip bundles.
`example_prompts` - Return example prompts. Use these as examples for plan_create. You can also call `plan_create` with any prompt—short prompts produce less detailed plans.
`model_profiles` - Show model_profile options and currently available models in each profile.
`plan_create` - Initiate creation of a plan.
`plan_status` - Get status and progress about the creation of a plan.
`plan_stop` - Abort creation of a plan.
`plan_retry` - Retry a failed plan using the same plan id (optional model_profile, defaults to baseline).
`plan_download` - Download the plan, either html report or a zip with everything, and save it to disk.

`plan_status` caller contract:
- `pending` / `processing`: keep polling.
- `completed`: terminal success, download is ready.
- `failed`: terminal error.

Concurrency semantics:
- Each `plan_create` call creates a new `plan_id`.
- `plan_retry` reuses the same failed `plan_id`.
- Server does not enforce a global one-plan-at-a-time cap per client.
- Local clients should track plan ids explicitly when running plans in parallel.

Minimal error contract:
- Tool errors use `{"error":{"code","message","details?"}}`.
- Common proxied cloud codes include: `PLAN_NOT_FOUND`, `INVALID_USER_API_KEY`, `USER_API_KEY_REQUIRED`, `INSUFFICIENT_CREDITS`, `INTERNAL_ERROR`, `generation_failed`, `content_unavailable`.
- `plan_retry` may return `PLAN_NOT_FAILED` if the task is not currently failed.
- Local proxy specific codes: `REMOTE_ERROR`, `DOWNLOAD_FAILED`.
- `plan_file_info` (called under the hood by plan_download) may return `{}` while output is not ready.

**Tip**: Call `example_plans` to preview example output, then call `example_prompts` to get example prompts to use with plan_create. The full catalog lives at `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl`.

`plan_download` is a synthetic tool provided by the local proxy. It calls the
remote MCP tool `plan_file_info` to obtain a download URL, then downloads the
file locally into `PLANEXE_PATH`.

`PLANEXE_PATH` behavior:
- If unset, downloads are saved to the current working directory.
- If the path does not exist, it is created.
- If the path points to a file (not a directory), download fails.
- Filenames are `<plan_id>-030-report.html` or `<plan_id>-run.zip` (with `-1`, `-2`, ... suffixes on collisions).
- `plan_download` returns `saved_path` with the final file location.

## Run as task (MCP tasks protocol)

Some MCP clients (e.g. the MCP Inspector) show a **"Run as task"** option for tools. That refers to the MCP **tasks** protocol: a separate mechanism where the client runs a tool in the background using RPC methods like `tasks/run`, `tasks/get`, `tasks/result`, and `tasks/cancel`, instead of a single blocking tool call.

**PlanExe does not use or advertise the MCP tasks protocol.** Our interface is **tool-based** only: the agent calls `example_plans`, `example_prompts`, and `model_profiles` for setup, completes a non-tool prompt drafting/approval step, then `plan_create` → gets a `plan_id` → polls `plan_status` → optionally calls `plan_retry` if failed → uses `plan_download`. That flow is defined in `docs/mcp/planexe_mcp_interface.md` and is the intended design.

You should **not** enable "Run as task" for PlanExe. The Python MCP SDK and clients like Cursor do not properly support the tasks protocol (method registration and initialization fail). Use the tools directly: create a task, poll status, then download when done.

## How it talks to mcp_cloud

- The remote base URL is `PLANEXE_URL` (for example `http://localhost:8001/mcp`).
- Tool calls prefer the remote HTTP wrapper (`/mcp/tools/call`).
- If the HTTP wrapper is unavailable, the proxy falls back to MCP JSON-RPC
  over `POST /mcp` (not SSE).
- Downloads use the remote `/download/{plan_id}/...` endpoints.
- Authentication uses `PLANEXE_MCP_API_KEY` as custom header `X-API-Key` (not OAuth/Bearer).
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

## Ideas

I’m considering dropping `mcp_local`. The benefit of having the `plan_download` is too small, vs. using `plan_file_info` and downloading the url afterwards. It saves 1 reasoning step. Alas the UX of configuring `mcp_local` is terrible. The documentation can be less confusing by not covering `mcp_local`.

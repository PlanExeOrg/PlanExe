---

## title: MCP Interface — Evaluation and Roadmap

date: 2026-02-26

# MCP Interface — Evaluation and Roadmap

An honest audit of the current MCP surface (`mcp_cloud` + `mcp_local`), followed by concrete improvements and promotion ideas.

**Revision history:**

- **2026-02-26 (rev 1):** Initial version after `task_`* → `plan_*` rename.
- **2026-02-26 (rev 2):** Updated after `app.py` refactor into modules, `plan_list` `user_api_key` made optional in schema (auto-injected by HTTP layer), and re-evaluation of all open issues.
- **2026-02-26 (rev 3):** Updated after completing 4.9 — all stale `task` variable names, request classes, helper functions, and backward-compat aliases renamed/removed across `mcp_cloud` and `mcp_local`. Test files renamed from `test_task_`* to `test_plan_*`.
- **2026-02-26 (rev 4):** Updated after completing 4.2 — added separate download rate limiter with configurable limits (default 10 req/60s).
- **2026-02-26 (rev 5):** Renamed external-facing fields: `task_id` → `plan_id`, `tasks` → `plans`, error codes `TASK_NOT_FOUND` → `PLAN_NOT_FOUND`, `TASK_NOT_FAILED` → `PLAN_NOT_FAILED`. Internal function names and download URL paths unchanged.
- **2026-03-02 (rev 6):** SSE progress streaming implemented (`mcp_cloud/sse.py`, `GET /sse/plan/{plan_id}`). Incorporated feedback from Claude Code agent evaluation of the MCP interface: improved SSE documentation across server instructions, tool descriptions, and field descriptions to be actionable for agents; added new proposed improvements for credit feedback, pipeline stage names, and files array completeness. Discovered and fixed BaseHTTPMiddleware blocking issue with SSE streams.

---

## 1. Current Tool Surface

Nine tools, split across two transports:


| Tool              | Cloud (`mcp_cloud`) | Local (`mcp_local`) | Auth     | Annotations             |
| ----------------- | ------------------- | ------------------- | -------- | ----------------------- |
| `example_prompts` | yes                 | yes                 | Public   | readOnly, idempotent    |
| `model_profiles`  | yes                 | yes                 | Public   | readOnly, idempotent    |
| `plan_create`     | yes                 | yes                 | Required | openWorld               |
| `plan_status`     | yes                 | yes                 | Required | readOnly, idempotent    |
| `plan_stop`       | yes                 | yes                 | Required | destructive, idempotent |
| `plan_retry`      | yes                 | yes                 | Required | openWorld               |
| `plan_file_info`  | yes                 | —                   | Required | readOnly, idempotent    |
| `plan_download`   | —                   | yes                 | Required | openWorld               |
| `plan_list`       | yes                 | yes                 | Required | readOnly, idempotent    |


`plan_download` is a local-only synthetic tool that internally proxies to `plan_file_info` on the cloud, then downloads and saves the artifact to the user's filesystem. This intentional asymmetry is tested in `test_tool_surface_consistency.py`.

**Auth model for `plan_create` and `plan_list`:** Both tools accept an optional `user_api_key` in the visible MCP input schema. When called over HTTP, the middleware authenticates the caller via the `X-API-Key` header and auto-injects `user_api_key` into handler arguments. This means MCP clients never need to pass `user_api_key` explicitly — the key is invisible in the tool's published schema but enforced at runtime. Both handlers return `USER_API_KEY_REQUIRED` if no key arrives by either path.

---

## 2. What's Working Well

**Dual transport.** `mcp_cloud` (stateless HTTP / Railway) and `mcp_local` (stdio proxy) cover the two major deployment patterns. Most users can pick one without reading source code.

**Clean module structure.** `mcp_cloud/app.py` is now a thin re-export facade (~195 lines). Logic lives in focused modules: `handlers.py` (tool handlers), `schemas.py` (tool definitions), `tool_models.py` (Pydantic models), `db_queries.py` (DB operations), `auth.py` (key hashing/user resolution), `download_tokens.py` (signed tokens), `model_profiles.py`, `worker_fetchers.py`, `zip_utils.py`, `example_prompts.py`. This makes PRs reviewable and bugs easy to isolate.

**Consistent `plan_`* naming throughout.** The rename from `task_`* to `plan_*` covers the full stack: external tool names, handler functions, request classes (`PlanCreateRequest`, etc.), DB query helpers (`_create_plan_sync`, `get_plan_by_id`, etc.), local variable names, and test file names. No backward-compat aliases remain.

**Layered authentication.** Two distinct auth paths — a server-wide `PLANEXE_MCP_API_KEY` for self-hosters, and per-user `pex_…` keys issued by home.planexe.org — are a good design. The key-normalisation helper (`_normalize_api_key_value` in `http_server.py`) handles common copy-paste artefacts (Bearer prefix, surrounding quotes, full header line pasted as value).

**Auto-injected `user_api_key`.** For `plan_create` and `plan_list`, the HTTP layer reads the authenticated user from the request context and injects `user_api_key` into handler arguments automatically. Callers never see `user_api_key` as a required field in the MCP schema — a clean separation between transport-level auth and tool-level logic.

**Structured output schemas.** Every tool declares an `output_schema`, so MCP clients can validate responses without guessing. `TestAllToolsHaveOutputSchema` enforces this at CI time.

**Tool annotations.** `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` are set on every tool and tested. This is ahead of most MCP servers.

`**plan_retry` with model_profile selection.** Allowing the caller to re-run a failed task with a stronger model (e.g. upgrade from `baseline` to `premium`) at retry time is genuinely useful.

**Signed download tokens.** `plan_file_info` returns download URLs with HMAC-SHA256 signed, time-limited tokens (15-min default TTL) scoped to one artifact (`task_id:filename:expiry`). Tokens work in a browser without an API key header. Defence-in-depth: the download endpoint re-validates even after middleware has passed the token. The secret fallback chain is: `PLANEXE_DOWNLOAD_TOKEN_SECRET` → `PLANEXE_API_KEY_SECRET` → per-process random (with warning).

**Glama + llms.txt.** Being listed in the Glama registry and providing `llms.txt` lowers the discovery barrier for new users.

**Rate limiting on all MCP endpoints.** `_enforce_rate_limit` in `http_server.py` applies to `/mcp`, `/mcp/`, and `/mcp/tools/call`. The default limit (60 req / 60 s per client, keyed by API key or IP) is high enough that normal `plan_status` polling is never affected.

**Prompt guidance in schema.** The `prompt` field description ("300–800 words … objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria") sets user expectations up front.

`**plan_list` for plan recovery.** Authenticated users can list their most recent plans (up to 50, newest-first) to recover a lost `plan_id`. Each entry includes `plan_id`, `state`, `progress_percentage`, `created_at`, and `prompt_excerpt`.

**Comprehensive test suite.** 12 test files covering tool surface consistency, auth key parsing, CORS config, download tokens, HTTP routing, and individual tool behaviour (`test_plan_create_tool.py`, `test_plan_status_tool.py`, `test_plan_retry_tool.py`, `test_plan_file_info_tool.py`, `test_model_profiles_tool.py`).

**SSE progress streaming.** `GET /sse/plan/{plan_id}` streams real-time progress as Server-Sent Events (`text/event-stream`). Emits `status` events on state/progress changes, `heartbeat` every ~20s, and a final `complete` event on terminal state. Deduplication avoids sending unchanged state. Connection tracking enforces per-client (5) and server-wide (200) limits. SSE is complementary to polling — both are fully supported. The SSE endpoint bypasses `BaseHTTPMiddleware` and handles auth inline to avoid a Starlette bug where long-lived streaming responses block concurrent requests through the middleware.

**Actionable SSE documentation.** Server instructions, tool descriptions (`plan_create`, `plan_status`), and field descriptions (`sse_url`) explain SSE in concrete terms: it's a GET endpoint returning `text/event-stream`, usable with `curl -N -H 'X-API-Key: <key>' <sse_url>`, and auto-closes on terminal state. This was revised based on feedback from a Claude Code agent evaluation that found the original "if your client supports SSE" phrasing too vague for autonomous agents.

**`example_plans` tool.** Returns curated example plans with download links for reports and zip bundles. Lets users preview what PlanExe output looks like before committing to a plan. No API key required.

---

## 3. What's Been Fixed (Previously Reported)

### 3.1 ~~`skills/planexe-mcp/SKILL.md` says "5 tools"~~ (FIXED)

Updated to nine tools; SKILL.md now lists all tools with example JSON-RPC calls.

### 3.2 ~~Trailing-slash inconsistency~~ (FIXED)

The canonical URL (`https://mcp.planexe.org/mcp`, no trailing slash) is used in all JSON config files and registry entries.

### 3.3 ~~`speed_vs_detail` documented but hidden from agents~~ (FIXED)

Removed entirely from the MCP interface.

### 3.4 ~~`plan_file_info` returns `{}` on success instead of `isError~~` (FIXED)

Now returns `{"ready": false, "reason": "processing"}` while running and `{"ready": false, "reason": "failed", "error": {...}}` on failure.

### 3.5 ~~Rate limiting covers REST but not Streamable HTTP `/mcp~~` (FIXED)

`_enforce_rate_limit` now covers `/mcp`, `/mcp/`, and `/mcp/tools/call`.

### 3.6 ~~No `plan_list` tool — lost `task_id` = lost task~~ (FIXED)

Added `plan_list` to both `mcp_cloud` and `mcp_local`. Returns up to 50 tasks newest-first.

### 3.7 ~~Signed, expiring download tokens~~ (FIXED)

HMAC-SHA256 tokens, 15-minute default TTL, scoped per-artifact.

### 3.8 ~~Tools used `task_`* prefix instead of `plan_*~~` (FIXED)

All external tool names renamed to `plan_*`.

### 3.9 `~~app.py` is a 76 KB monolith~~ (FIXED)

Refactored into 10+ focused modules (commit 9f1a7db9). `app.py` is now a thin re-export facade.

### 3.10 ~~`plan_list` requires `user_api_key` in visible MCP schema~~ (FIXED)

`user_api_key` is now optional in the `PlanListInput` schema (not in `required` list), matching `plan_create`. The HTTP layer auto-injects it from the `X-API-Key` header via `_get_authenticated_user_api_key()`. The handler still enforces the key at runtime (returns `USER_API_KEY_REQUIRED` if absent).

---

## 4. What's Broken or Inconsistent

### ~~4.1 Dev-secret fallback in production~~ (FIXED)

`auth.py` now exports `validate_api_key_secret()` which raises `RuntimeError` when `PLANEXE_API_KEY_SECRET` is not set. `download_tokens.py` exports `validate_download_token_secret()` which raises when neither `PLANEXE_DOWNLOAD_TOKEN_SECRET` nor `PLANEXE_API_KEY_SECRET` is set. Both are called at module level in `http_server.py` when `AUTH_REQUIRED` is true, so the server fails hard at startup instead of silently falling back to dev secrets. The existing runtime fallbacks (`"dev-api-key-secret"` and random per-process secret) remain for local development with `PLANEXE_MCP_REQUIRE_AUTH=false`.

### ~~4.2 `/download` endpoint not rate-limited~~ (FIXED)

A separate download rate limiter (`_enforce_download_rate_limit`) now covers `/download` paths with its own bucket and configurable limits: `PLANEXE_MCP_DOWNLOAD_RATE_LIMIT` (default 10 req) and `PLANEXE_MCP_DOWNLOAD_RATE_WINDOW_SECONDS` (default 60s). This is deliberately tighter than the MCP rate limit (60 req/60s) since download responses are 700KB–6MB. The sweep task cleans up download buckets alongside MCP buckets.

### ~~4.3 Body size validation only on REST endpoint~~ (FIXED)

`_enforce_body_size` now checks both `/mcp/tools/call` and `/mcp/` POST requests. The `Content-Length` requirement (411) is only enforced on the REST endpoint since Streamable HTTP may use chunked encoding without `Content-Length`; however, when `Content-Length` is present on either endpoint it is validated against `MAX_BODY_BYTES`.

### ~~4.4 `plan_file_info` silently defaults invalid artifact to `"report"`~~ (FIXED)

Both `handle_plan_file_info` (cloud) and `handle_plan_download` (local) now return `INVALID_ARGUMENT` with a descriptive message when the artifact value is not `"report"` or `"zip"`.

### ~~4.5 No dedicated `plan_list` test~~ (FIXED)

Added `mcp_cloud/tests/test_plan_list_tool.py` with 8 tests covering: tool listed, returns tasks, empty result, limit clamping (both directions), invalid API key, `USER_API_KEY_REQUIRED` when env requires key, no-key passthrough when not required (user_id=None), and default limit.

### ~~4.6 CORS default is wildcard~~ (FIXED)

When `AUTH_REQUIRED` is true and `PLANEXE_MCP_CORS_ORIGINS` is unset, the default is now `["https://mcp.planexe.org", "https://home.planexe.org"]` instead of `["*"]`. Wildcard CORS is only used in dev mode (`PLANEXE_MCP_REQUIRE_AUTH=false`) so browser-based tools like MCP Inspector work without extra configuration. Operators can override via `PLANEXE_MCP_CORS_ORIGINS`.

### ~~4.7 No request logging for successful tool calls~~ (FIXED)

`handle_call_tool` now logs every tool call at INFO level with tool name, result (ok/error/exception), and duration in milliseconds. Unknown tools are logged at WARNING. Format: `tool_call tool=<name> result=<ok|error|exception> duration_ms=<N>`.

### ~~4.8 Prompt excerpt length hardcoded~~ (FIXED)

Extracted to `PROMPT_EXCERPT_MAX_LENGTH = 100` at module level in `db_queries.py`.

### ~~4.9 Stale `task` variable names and backward-compat aliases~~ (FIXED)

All internal naming now uses `plan` consistently. Request classes renamed (`TaskCreateRequest` → `PlanCreateRequest`, etc.), DB query helpers renamed (`_create_task_sync` → `_create_plan_sync`, `get_task_by_id` → `get_plan_by_id`, etc.), local variables renamed (`task_snapshot` → `plan_snapshot`, etc.), all backward-compat aliases removed from `tool_models.py`, `schemas.py`, `handlers.py`, `app.py`, and `mcp_local/planexe_mcp_local.py` (~86 lines deleted). Test files renamed from `test_task_*.py` to `test_plan_*.py` with patch targets updated.

### ~~4.10 `plan_list` auth differs from `plan_create`~~ (FIXED)

`plan_list` now uses the same `PLANEXE_MCP_REQUIRE_USER_KEY` check as `plan_create`. When the key is not required and not provided, `plan_list` returns all tasks (no user scoping). `_list_tasks_sync` accepts `user_id=None` to support this.

---

## 5. Proposed Improvements

### ~~5.1 SSE progress streaming (UX)~~ (IMPLEMENTED)

`GET /sse/plan/{plan_id}` now streams real-time progress via Server-Sent Events. Implementation in `mcp_cloud/sse.py`. `plan_create` and `plan_status` responses include an `sse_url` field pointing to the endpoint. Events: `status` (state/progress changes), `heartbeat` (~20s silence), `complete` (terminal state), `error` (not found / timeout). Stream auto-closes on terminal state or after 60 minutes.

**Feedback note (Claude Code agent evaluation, 2026-03-02):** The original documentation said "if your client supports Server-Sent Events" — this was too vague for autonomous agents. MCP tools are request-response by design, so agents didn't know whether they "support" SSE or how to consume it. Documentation was rewritten to be actionable: SSE is a GET endpoint returning `text/event-stream`, agents can use `curl -N` via their Bash tool, and polling remains the simpler alternative. Both `plan_create` and `plan_status` tool descriptions now mention `sse_url` with usage examples.

**Implementation note:** The SSE endpoint is intentionally excluded from Starlette's `BaseHTTPMiddleware` (`enforce_api_key`). The middleware pipes response bodies through an internal `anyio.MemoryObjectStream`; for long-lived SSE streams this keeps the middleware's task-group alive indefinitely and starves concurrent requests. Auth is handled inline in the SSE endpoint instead.

### 5.2 Webhook / push notification (power users)

Add an optional `webhook_url` to `plan_create`. When the task transitions to `completed` or `failed`, POST a JSON summary to that URL. This removes the need for polling and enables CI/CD integrations.

### 5.3 API versioning

All tool names and schemas are currently unversioned. A future breaking change will silently break clients. Add a `server_version` field to the `plan_status` output and document a stability policy.

### 5.4 Startup environment validation

Add an explicit check at server startup that required secrets (`PLANEXE_API_KEY_SECRET`, `PLANEXE_DOWNLOAD_TOKEN_SECRET`) are set when auth is enabled. Fail loudly instead of falling back to dev defaults.

### 5.5 Credit/cost feedback in `plan_create` response

**Source:** Claude Code agent feedback (2026-03-02).

After `plan_create`, there is no indication of credits consumed or remaining. For a paid service, this transparency builds trust. Consider adding `credits_used` and `credits_remaining` fields to the `plan_create` response (or to `plan_status` on completion).

### 5.6 Pipeline stage names in progress reporting

**Source:** Claude Code agent feedback (2026-03-02).

`progress_percentage` jumps non-linearly (e.g. 80% → 83% over 4 minutes, then straight to 100%). The metric doesn't feel informative. Consider adding a `current_stage` field to `plan_status` (e.g. "generating SWOT analysis", "running premortem") so agents and users can see *what* is happening, not just a number.

### 5.7 Complete files array in `plan_status` for completed plans

**Source:** Claude Code agent feedback (2026-03-02).

The `files` array in `plan_status` only shows early pipeline files (001-, 002-), not the final report or zip. When `state` is `completed`, the full manifest of outputs should be visible. This helps agents verify what was produced before calling `plan_file_info`.

### 5.8 Prompt approval skip for agent-provided prompts

**Source:** Claude Code agent feedback (2026-03-02).

The server instructions mandate that the agent drafts a prompt and gets user approval before calling `plan_create`. When the user hands the agent a polished prompt file and says "use this", the extra ceremony adds friction. Consider a note in the instructions that agent-provided or user-approved prompts can go directly to `plan_create` without the full drafting cycle.

---

## 6. Promotion and Growth Strategies

### 6.1 MCP registries

- **Glama** — already listed
- **Smithery** — another fast-growing directory; supports one-click install
- **mcp.so** — submit `server.json`; high traffic from Claude desktop users
- **awesome-mcp-servers** (GitHub) — submit a PR; maintainers merge quickly
- **OpenTools** — focus on enterprise MCP discovery

#### 6.1.1 Glama

Already listed, but stuck in an `unclaimed` state, where I can't customize it.

https://glama.ai/mcp/connectors/io.github.PlanExeOrg/planexe

MCP Servers that have been claimed, can alter their profile, and assign an icon.

I'm seing any health `connectors` that have been claimed and customized.

https://glama.ai/blog/2025-10-22-what-are-mcp-connectors

https://glama.ai/mcp/connectors?attributes=status%3Ahealthy&sort=featured%3Adesc

Outstanding issues:

- Claim ownership of PlanExe inside Glama.ai
- Add a /.well-known/glama.json to claim ownership of planexe. No luck.
- Add a /glama.json to repo to claim ownership of planexe. No luck.
- Customize profile text, categories, favicon.

#### 6.1.2 Smithery

https://smithery.ai/servers/planexeorg/planexe

Smithery has problems updating the entry automatically. When I have made mcp interface changes, then I'm not seeing them show up in Smithery's UI.
Syncing is something I have to do manually, by going to the `Releases` page, and go through the `Publish` flow.
https://smithery.ai/servers/planexeorg/planexe/releases
That reloads the PlanExe data entry, by pulling it from the mcp.planexe.org/mcp, IMO something that should happen automatic.
It may be possible to force reload via CLI. I have not investigated this.
https://smithery.ai/docs/build/publish#cli-advanced

Smithery has no filtering. No sort by date or by name.

Smithery's one-click install is neat.

Outstanding issues:

- Automation. Whenever I make changes to MCP, I will have to manually update the PlanExe profile on Smithery. 
- improve on Smithery’s Quality Score. Currently it’s 81 of 100.

### 6.2 Content

- **Blog post: "From prompt to project plan in 60 seconds"** — a short walkthrough showing MCP Inspector → `plan_create` → `plan_status` → download. Publish on dev.to, Hacker News (Show HN), and the PlanExe GitHub Discussions.
- **YouTube demo (2–3 minutes)** — screen recording of Claude Desktop using PlanExe MCP end-to-end. Pin it to the README.
- **Twitter/X thread** — "I built an MCP server that turns a ~500-word prompt into a full project plan. Here's how it works:"

### 6.3 Community integrations

- **Claude Desktop config snippet** — provide a ready-to-paste `claude_desktop_config.json` block in the README.
- **Cursor / Windsurf rule** — provide a `.cursorrules` or `.windsurfrules` snippet that wires PlanExe MCP automatically.
- **GitHub Actions** — a reusable workflow `planexe/create-plan@v1` that runs `plan_create` and uploads the result as a release asset. This is a high-visibility integration channel.

### 6.4 Example prompt gallery

Add 10–15 high-quality example prompts (startup, research paper, home renovation, hiring plan, …) to `example_prompts`. Agents and users copy-paste these; each successful use is a social proof data point.

### 6.5 Observability / social proof

- Add a public counter to the homepage: "X plans created this week".
- Post a monthly changelog to GitHub Discussions so subscribers see activity.
- Badge in the README: `![Plans created](https://img.shields.io/badge/dynamic/json?url=https://mcp.planexe.org/stats&label=plans+created)`.

---

## 7. Quick-win Checklist


| Priority | Task                                                                      | Effort | Status |
| -------- | ------------------------------------------------------------------------- | ------ | ------ |
| P0       | ~~Fix SKILL.md tool count~~                                               | —      | DONE   |
| P0       | ~~Standardise URL trailing slash~~                                        | —      | DONE   |
| P0       | ~~Fix `speed_vs_detail` schema/docs mismatch~~                            | —      | DONE   |
| P0       | ~~Rename tools from `task_`* to `plan_*~~`                                | —      | DONE   |
| P1       | ~~Add `plan_list` tool~~                                                  | —      | DONE   |
| P1       | ~~Fix `plan_file_info` empty-dict response~~                              | —      | DONE   |
| P1       | ~~Add rate limiting to `/mcp` endpoint~~                                  | —      | DONE   |
| P1       | ~~Signed download tokens~~                                                | —      | DONE   |
| P1       | ~~Refactor `app.py` into modules~~                                        | —      | DONE   |
| P1       | ~~Remove `user_api_key` from `plan_list` visible schema~~                 | —      | DONE   |
| P1       | ~~Fail-hard on missing secrets in production (4.1)~~                      | —      | DONE   |
| P1       | ~~Rate-limit `/download` endpoint (4.2)~~                                 | —      | DONE   |
| P1       | ~~Add `plan_list` handler tests (4.5)~~                                   | —      | DONE   |
| P1       | Submit to mcp.so + Smithery                                               | 30 min |        |
| P1       | Write README demo GIF / YouTube link                                      | 1 h    |        |
| P2       | ~~Body size validation on Streamable HTTP (4.3)~~                         | —      | DONE   |
| P2       | ~~Return error for invalid artifact value (4.4)~~                         | —      | DONE   |
| P2       | ~~Add tool-call audit logging (4.7)~~                                     | —      | DONE   |
| P1       | ~~SSE progress streaming (5.1)~~                                           | —      | DONE   |
| P2       | Add `log_lines` to `plan_status`                                           | 4 h    |        |
| P2       | ~~Rename internal `task` variables/classes/helpers to `plan` (4.9)~~      | —      | DONE   |
| P2       | ~~Remove backward-compat `Task*`/`handle_task_*`/`TASK_*` aliases (4.9)~~ | —      | DONE   |
| P2       | ~~Rename test files from `test_task_*` to `test_plan_*` (4.9)~~           | —      | DONE   |
| P2       | ~~Tighten default CORS origins (4.6)~~                                    | —      | DONE   |
| P2       | ~~Align `plan_list` auth with `plan_create` (4.10)~~                      | —      | DONE   |
| P2       | Credit/cost feedback in `plan_create` response (5.5)                       | 4 h    |        |
| P2       | Pipeline stage names in progress reporting (5.6)                           | 4 h    |        |
| P2       | Complete files array in `plan_status` for completed plans (5.7)            | 2 h    |        |
| P3       | Prompt approval skip for agent-provided prompts (5.8)                      | 1 h    |        |
| P3       | Webhook support (5.2)                                                      | 1 day  |        |
| P3       | API versioning (5.3)                                                       | 4 h    |        |
| P3       | GitHub Actions integration (6.3)                                           | 1 day  |        |


---

## 8. Summary

The MCP surface is functionally solid and ahead of most MCP servers in terms of schema rigour, annotation coverage, and security (signed download tokens, layered auth, auto-injected user keys). The codebase has been significantly improved since rev 1: `app.py` was refactored from a 76 KB monolith into 10+ focused modules, `plan_list` now follows the same auth-injection pattern as `plan_create`, and all P0 issues are resolved.

All P1 code-quality issues are now resolved, including fail-hard on missing secrets in production (4.1). SSE progress streaming (5.1) is now implemented, providing real-time push updates as an alternative to polling. A Claude Code agent evaluation (2026-03-02) surfaced actionable feedback: SSE documentation was too vague for autonomous agents (now fixed), and four new improvements were identified — credit/cost feedback (5.5), pipeline stage names in progress (5.6), complete files array on completion (5.7), and prompt approval flexibility (5.8). The remaining checklist items are these feedback-driven improvements, promotion/growth tasks (mcp.so submission, README demo), and lower-priority enhancements (webhooks, API versioning).
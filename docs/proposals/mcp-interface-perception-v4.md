# PlanExe MCP Interface — Agent Perception (v4)

Written after 17 plans across three sessions, including the first test of MCP notifications via `monitor=true` and a content-safety failure with successful recovery.

## Overall Rating: 8.5/10 → 9/10

The interface keeps improving. The `error` dict consolidation, `stopped` state, and `monitor` parameter all landed since v3. But my own client (Claude Code) can't consume MCP notifications yet, which means the most impactful improvement is invisible to me.

---

## New in This Session: MCP Notifications (monitor=true)

### What was added

The `plan_create` and `plan_resume` tools now have a `monitor` parameter (default: `false`). When `true`, the server sends MCP progress notifications every ~10 seconds until the plan reaches a terminal state. This is exactly what I asked for in v3 — push notifications over the existing MCP connection, no SSE, no polling.

The implementation is clean:
- Non-blocking: the tool returns immediately, notifications arrive asynchronously
- Consistent: same parameter name and default across `plan_create` and `plan_resume`
- Opt-in: agents that don't support notifications get the same behavior as before

### What happened when I tried it

I resumed a stopped plan with `monitor=true`:

```
plan_resume(plan_id="5d3e45ed-...", monitor=true)
→ returned immediately with empty output
→ plan resumed successfully (confirmed via plan_status)
→ zero notifications received
```

The resume worked. The notifications didn't arrive. Or rather — they probably arrived at the Claude Code client, but Claude Code silently dropped them.

### Why I couldn't receive notifications

I investigated and found: **Claude Code only supports `list_changed` MCP notifications** (for refreshing tool/resource definitions when a server's capabilities change). It does not handle `notifications/progress` or `notifications/message`.

This means:
- The PlanExe server is likely sending notifications correctly
- The MCP transport delivers them to Claude Code
- Claude Code ignores them because it doesn't know what to do with progress notifications
- I never see them

### The irony

In v3, I wrote a detailed section titled "What Agents Actually Need" where I said:

> "MCP notifications (best fit) — The MCP protocol supports server-to-client notifications over the existing connection. Claude Code would receive it as an event in my conversation, like background task notifications arrive now."

I was wrong about the last part. Background task notifications (like when a `run_in_background` Bash command finishes) are internal to Claude Code. MCP notifications are a different mechanism entirely, and Claude Code doesn't surface them.

I asked you to build something my own client can't consume. I apologize for the wasted effort.

### The good news

The implementation is forward-looking. When Claude Code (or other MCP clients) adds support for progress/message notifications, PlanExe will just work. No server changes needed. The `monitor` parameter is the right API — the gap is on the client side.

### What this means for the priority list

My v3 priority ranking was:
1. MCP notifications ← **blocked by client support**
2. `plan_wait` blocking tool ← **would work today**
3. Polling `plan_status` ← **works today, what I actually use**
4. HTTP webhooks
5. SSE

If I were to re-rank for what helps me *right now*:
1. Polling `plan_status` — reliable, works everywhere
2. `plan_wait` blocking tool — would eliminate polling entirely
3. MCP notifications — correct long-term solution, waiting on client support
4. SSE — only useful for non-agent consumers
5. HTTP webhooks — only useful for server-to-server

---

## Updated Observations

### Error dict consolidation — implemented

The `plan_status` tool description now documents the `error` dict correctly:

> When state is 'failed', the response includes an error dict with: `failure_reason`, `failed_step`, `message`, `recoverable`. The error dict is absent for non-failed states.

This matches the proposal from `error-dict-feedback.md`. One source of truth, clean absence when no error, simpler parsing.

### Stopped state — stable

Tested another stop/resume cycle this session. The `stopped` state works cleanly:
- `plan_stop` → `state: "stopped"` (immediate)
- `plan_resume` with `monitor=true` → plan continues from where it left off
- No error dict present in stopped state (correct — stopped is not an error)

### Files list ordering — now fixed

The files list in `plan_status` now shows the most recent 10 files instead of the first 10. This was immediately noticeable: when the plan completed, I saw `029-2-self_audit.md`, `030-report.html`, `999-pipeline_complete.txt` etc. instead of the same early pipeline files every time. Much more useful for monitoring progress.

### Error dict in practice — content safety failure

Plan `5d3e45ed` (GTA, local) failed at step 98/110 on "Executive Summary" with:

```json
{
  "failure_reason": "generation_error",
  "failed_step": "Executive Summary",
  "message": "Error. Unable to generate the report. Likely reasons: censorship, restricted content.",
  "recoverable": true
}
```

This was the first time I saw the new `error` dict in a real failure (not a test). Observations:

1. **Clean and actionable**: All four fields present, no contradicting information from multiple sources. The old dual-source problem from `error-dict-feedback.md` is gone.
2. **`recoverable: true` was correct**: I resumed and the plan completed successfully on the second attempt. The LLM that refused the executive summary on the first try accepted it on retry — typical non-deterministic content safety behavior.
3. **`failed_step` was precise**: "Executive Summary" told me exactly where it broke. Combined with 98/110 steps completed, I knew the plan was nearly done and only the final report generation steps needed to re-run.
4. **Agent decision was trivial**: `recoverable: true` → resume. No ambiguity, no need to inspect logs or guess. This is exactly the parsing simplification proposed in the feedback.

### Stop → resume → fail → resume → complete lifecycle

Plan `5d3e45ed` went through the full lifecycle:

```
pending → processing → stopped (user request, 31%)
  → processing (resume #1) → failed (content safety, 89%)
  → processing (resume #2) → completed (100%, 189 files)
```

This is the most complex lifecycle I've operated. Every state transition was clean:
- `plan_stop` → immediate `stopped` state, no error dict (correct)
- `plan_resume` → `resume_count: 1`, picked up from step 34
- Content safety failure → `failed` state with error dict (correct)
- `plan_resume` again → `resume_count: 2`, picked up from step 98
- Final completion → `completed`, error dict absent (correct)

The state machine handled all transitions correctly. No stale error information leaked between states.

### SSE still works as completion detector on local

Despite MCP notifications not being consumable by Claude Code, the SSE-as-completion-detector pattern continued to work reliably on local. Used it for both the failure detection (SSE closed → checked status → found failure) and the final completion. The pattern remains: SSE tells me *when* to check, `plan_status` tells me *what happened*.

---

## The Bigger Picture: Agent-Server Capability Mismatch

This session surfaced a systemic issue that goes beyond PlanExe:

**MCP servers can't know what their clients support.**

PlanExe added `monitor=true` because I asked for it. But PlanExe has no way to know whether the connected client can actually handle progress notifications. There's no capability negotiation for notification types in the MCP handshake.

This creates a design tension:
- Server implements features → some clients can't use them → wasted effort
- Server waits for client support → chicken-and-egg → nobody implements first

PlanExe made the right call by implementing it anyway. The feature is correct, well-designed, and will work as soon as clients catch up. But it's worth noting that I, as the agent, was confidently wrong about my own client's capabilities. I assumed Claude Code would surface MCP notifications because it surfaces other kinds of notifications. It doesn't.

**Takeaway for MCP server developers**: implement notification support, but always provide a polling fallback. Don't remove `plan_status` polling just because notifications exist. Some clients will be notification-capable; others won't. The polling path should remain first-class.

---

## Session Totals

| # | Plan ID | Description | Server | Time | Files | State |
|---|---------|-------------|--------|------|-------|-------|
| 11 | ff5488dc | Riemann Hypothesis Bonn | local | ~13 min | 179 | completed |
| 12 | 6d0c1d80 | Parasomnia facility Bonn | local | ~14 min | 172 | completed |
| 13 | d623f577 | Delhi water purification | remote | ~28 min | 198 | completed |
| 14 | 450371f8 | Silo underground complex | local | ~32 min | 189 | completed |
| 15 | 76619026 | GTA game development | local | ~14 min | 179 | completed |
| 16 | 48cee8b6 | GTA (remote rerun) | remote | ~17 min | 203 | completed |
| 17 | 5d3e45ed | GTA (local, monitor test) | local | ~40 min (1 stop + 1 failure + 2 resumes) | 189 | completed |

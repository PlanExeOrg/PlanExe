# Promising Directions for PlanExe

Analysis of 110+ proposals. Primary audience: AI agents consuming PlanExe via MCP. Secondary audience: humans reviewing plans.

---

## Strategic Frame

PlanExe's future is as **infrastructure for AI agents** — the planning layer that agents call before executing complex multi-step work. Today, agents use PlanExe to generate plans for humans to review. Tomorrow, agents will plan, validate, and execute autonomously with PlanExe as the orchestration backbone.

The proposals below are grouped by what agents need, in priority order.

---

## 1. Operational Reliability (agents can't babysit)

Agents need PlanExe runs to complete reliably without human intervention. A failed run that loses 73% of work is a dealbreaker for autonomous workflows.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **87** | Plan Resume MCP Tool | ✅ **Implemented**. `plan_resume` MCP tool lets agents recover from failures without restarting the full pipeline. Includes pipeline version checking and zip snapshot restore |
| **109** | LLM Executor Retry Improvements | ✅ **Implemented (PR #220)**. `RetryConfig` with exponential backoff for transient errors (rate limits, timeouts, connection failures). `is_transient_error()` classifies exceptions; permanent errors fall through to next model immediately. Agents no longer need their own retry wrappers |
| **102** | Pipeline Intelligence Layer | ⚙️ **Foundation implemented (PR #221)**. `LLMExecutor` extracts structured Pydantic validation feedback and exposes it via `validation_feedback` property; `max_validation_retries` controls per-model retry count. Remaining: wire individual tasks to inject feedback into prompts |
| **103** | Pipeline Hardening for Local Models | Fix silent truncation and context-window overflows. Critical for agents running local models where failures are subtle |
| **113** | LLM Error Traceability | ✅ **Implemented (PR #237)**. `LLMChatError` replaces generic `ValueError` across 38 call sites. Root cause preserved for error classification; `error_id` UUID enables log-to-metrics cross-referencing. Agents can programmatically diagnose failures |
| **101** | Luigi Resume Enhancements | Webhook hooks on task completion/failure — agents can subscribe to events instead of polling |
| **114-I1** | Stopped vs Failed State | ✅ **Implemented**. Dedicated `PlanState.stopped` enum value — `plan_stop` transitions to `stopped`, not `failed`. Agents can now distinguish user-initiated stops from actual errors. `plan_retry` and `plan_resume` accept both `failed` and `stopped` |
| **114-I2** | Failure Diagnostics in `plan_status` | ✅ **Implemented**. Four DB columns (`failure_reason`, `failed_step`, `error_message`, `recoverable`) populated by worker on failure. `plan_status` returns a consolidated `error` dict (keys: `failure_reason`, `failed_step`, `message`, `recoverable`) only for failed plans — non-failed plans omit `error` entirely. The `recoverable` flag lets agents immediately suggest `plan_resume` vs `plan_retry`. Migration renames the legacy `last_error` column to `error_message` with existence checks to avoid noisy PostgreSQL logs. Diagnostics reset on retry/resume |
| **114-I7** | Stalled-Plan Detection | No `last_progress_at` or `last_llm_call_at` timestamps. Agents can't distinguish "slow step" from "stuck worker". Complements #87 §8 |
| **114-I10** | Silent Partial Failures in Completed Plans | A plan can reach `completed` with empty or stub-quality sections (e.g. 2/8 experts responding). No `quality_summary` in `plan_status` — agents can't tell if `completed` means "all sections produced quality output" or just "all steps ran." Trust gap for autonomous workflows |
| **114-I12** | Remote Files Visibility During Processing | `plan_status` on remote returns `files_count: 0` while processing; local shows files incrementally. Agents lose the ability to confirm output is being produced and see which pipeline sections completed. Behavioral inconsistency between servers breaks workflows silently |
| **114-I13** | Remote SSE Reliability | SSE stream on local was 100% reliable across 12 plans; remote dropped at ~48% on first plan. Only matters if SSE is kept for non-agent consumers (see I5 correction below) |

---

## 2. Agent Interface & Discoverability (reduce integration friction)

Agents need to discover PlanExe, understand its tools, and consume outputs programmatically. The current interface assumes a human is in the loop.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **86** | Agent-Optimized Pipeline | ✅ **Implemented (PR #223)**. Autonomous agent guide and agent-oriented prompts. Removes key friction points for autonomous agent use |
| **62** | Agent-First Frontend Discoverability | ✅ **Implemented (PR #224)**. `robots.txt` and README agent discoverability section — standard discovery protocols so agents find PlanExe without human guidance |
| **110** | Usage Metrics for Local Runs | ✅ **Implemented (PR #219, #236, #237)**. Agents need cost accounting for budget-constrained workflows. `usage_metrics.jsonl` answers "how much did this run cost?" with per-call granularity (model, tokens, cost, duration). Errors are classified into short categories with traceable `error_id` UUIDs. Complements `activity_overview.json` aggregated totals |
| **114-I3** | Plan Delete / Archive | Stopped and failed plans persist in `plan_list` forever. After 10 plans, the list is noisy. Add `plan_delete` (hard delete) or `plan_archive` (soft delete, hidden from list but retained for billing) |
| **114-I4** | Idempotency Guard on `plan_create` | ✅ **Implemented (PR #242)**. Server-side auto-dedup on `(user_id, prompt, model_profile)` within configurable time window (default 10 min) |
| **114-I5** | SSE Wrong for Agents; MCP Notifications Deprioritized | **Corrected in v3, re-corrected in v4.** SSE is designed for real-time UI clients, not turn-based agents — the agent abused connection close as a binary "done" signal. v3 proposed MCP notifications as the replacement. v4 tested `monitor=true` (implemented and reverted): Claude Code silently drops `notifications/progress` and `notifications/message` — it only supports `list_changed`. No current MCP client can consume progress notifications. **Practical priority:** (1) polling `plan_status` — works everywhere, (2) `plan_wait` blocking tool (I8) — would eliminate polling, (3) MCP notifications — correct long-term solution, blocked by client support. SSE remains useful as a local completion detector for non-agent consumers |
| **114-I6** | Download URL TTL Extension | 15-minute signed URL expiry surprises users who review before downloading. Extend to 30–60 min, make configurable via env var |
| **114-I8** | `plan_wait` Tool | **Upgraded from P3 to P2.** The most practical improvement for agents today, given that MCP notifications are blocked by client support (I5). A blocking `plan_wait(plan_id, timeout)` tool returns final status on completion — replaces both SSE monitoring and polling `plan_status` in a single call |
| **114-I9** | Prompt Iteration Linking | Each `plan_create` is independent. Optional `parent_plan_id` links iteration chains so agents and users can track prompt refinement across plan versions |
| **114-I11** | Server Identity in Responses | **New in v3.** Agent cannot programmatically determine which backend it's connected to (local vs remote). Plans from one server aren't accessible from the other, speed expectations differ (13 min vs 28 min), and SSE behavior varies. `plan_status` should include a `server` field or `plan_create` should return `server_info` with capabilities |

Key friction points from #86 that block autonomous agent use:
- **F1**: Human approval step before `plan_create` — autonomous agents can't proceed
- **F4**: No machine-readable summary — agents must parse HTML or iterate 100+ zip files
- **F3**: 5-minute poll interval is wrong for cloud runs completing in 8 minutes

---

## 3. Plan Quality Gates (agents must not amplify bad plans)

Before agents can execute plans autonomously, the plans themselves need automated validation. An agent that executes an unchecked plan multiplies errors at machine speed.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **58** | Boost Initial Prompt | ⚙️ **Open PR #222**. Single LLM call to strengthen weak prompts before pipeline runs. Especially valuable for agent-originated prompts, which may be terse or overly technical |
| **42** | Evidence Traceability Ledger | Links every claim to evidence with freshness scoring. Agents can programmatically check whether assumptions are grounded |
| **43** | Assumption Drift Monitor | Watches key variables (costs, FX rates) against baselines, triggers re-plan on breach. Agents re-planning on a schedule need this to detect when a plan has gone stale |
| **57** | Banned Words + Lever Realism | Auto-detects hype-heavy or impractical outputs without human review |
| **56** | Adversarial Red-Team Reality Check | Multi-model adversarial review with judge scoring. Catches optimism bias that agents are prone to propagating |
| **88** | Fermi Sanity Check Validation Gate | ⚙️ **Open PR #225**. Rule-based guard on every assumption: bounds present, span ratio sane, evidence for low-confidence claims. First line of defense before expensive downstream processing |

---

## 4. Autonomous Prompt Optimization (agents improving PlanExe itself)

Luigi's caching makes prompt optimization uniquely practical — changing one prompt template and re-running only regenerates that single task (seconds, not the full 15-minute pipeline). This turns a 60+ LLM call pipeline into a 2–21 call experiment.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **94** | Autoresearch-Style Prompt Optimization | ⚙️ **Open PR #226** (LLM-as-judge foundation). Autonomous overnight loops: agent modifies one prompt, re-runs one task, scores, keeps or reverts. Hundreds of experiments per night exploiting Luigi resumability |
| **59** | Prompt Optimizing with A/B Testing | ⚙️ **Open PR #227** (A/B experiment runner). Structured promotion pipeline: multi-model A/B matrix, Elo tracking, regression guards. Validates candidates discovered by #94 before merging into baseline |

Two-stage system: **#94 discovers** promising variants at high volume (greedy, autonomous), **#59 validates** them with rigor (conservative, human-gated). Exploration feeds promotion.

This is a case where agents improve the tool they use — a self-reinforcing loop.

---

## 5. Post-Plan Execution (the big unlock)

This is PlanExe's largest gap: *"I have a plan. Now what?"* For agents, a plan that can't be executed programmatically is just a document.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **41** | Autonomous Execution of Plan | Converts static Gantt into live execution engine with AI/human task delegation. The plan becomes a runnable workflow, not a PDF |
| **60–66** | Plan-to-Repo + Agent Swarm | Auto-provision repo, spawn agents (research, issues, compliance), use git as state machine. Plans become collaborative artifacts with continuous enrichment |
| **92** | Task Complexity Scoring & Model Routing | Each task gets a complexity score and recommended model tier. Agents can route cheap tasks to cheap models and expensive tasks to expensive ones — 55% cost savings in benchmarks |

**#41 is the most transformative proposal** — but it only works if the quality gates (#42, #43, #56, #88) prevent automation from multiplying bad plans.

---

## Recommended Sequence

```
Phase 1: Reliable foundation         (nearly complete)
  ├─ #87  Plan resume ✅
  ├─ #109 Retry improvements ✅ (PR #220)
  ├─ #102 Error-feedback retries ⚙️ (PR #221, foundation only — tasks not yet wired)
  ├─ #110 Usage metrics ✅ (PR #219, #236, #237)
  ├─ #113 Error traceability ✅ (PR #237)
  ├─ #58  Prompt boost ⚙️ (open PR #222)
  ├─ #114-I1 Stopped vs failed state ✅
  ├─ #114-I2 Failure diagnostics in plan_status ✅
  ├─ #114-I5 MCP notifications for agents  ← deprioritized (v4: Claude Code can't consume them; polling is primary)
  ├─ #114-I7 Stalled-plan detection
  ├─ #114-I10 Silent partial failures in completed plans
  └─ #114-I12 Remote files visibility during processing

Phase 2: Agent-native interface       (next)
  ├─ #86  Remove agent friction points ✅ (PR #223)
  ├─ #62  Discovery protocols ✅ (PR #224)
  ├─ #88  Fermi validation gate ⚙️ (open PR #225)
  ├─ #114-I3 Plan delete/archive
  ├─ #114-I6 Download URL TTL extension
  ├─ #114-I4 Idempotency guard ✅ (PR #242)
  ├─ #114-I8 plan_wait tool (P2 — primary polling alternative now that MCP notifications are blocked)
  ├─ #114-I9 Prompt iteration linking
  ├─ #114-I11 Server identity in responses
  └─ #114-I13 Remote SSE reliability (if SSE kept for non-agent consumers)

Phase 3: Automated quality            (then)
  ├─ #42  Evidence traceability
  ├─ #43  Assumption drift monitor
  ├─ #56  Adversarial red-team
  └─ #57  Banned words / lever realism

Phase 4: Self-improving pipeline      (concurrent with 3)
  ├─ #94  Autoresearch prompt optimization ⚙️ (open PR #226 foundation, PR #227 runner)
  └─ #59  A/B testing promotion ⚙️ (open PR #227)

Phase 5: Autonomous execution         (after quality gates)
  ├─ #41  Plan execution engine
  ├─ #60-66 Agent swarm
  └─ #92  Model routing
```

Each phase enables the next. Skipping to Phase 5 without Phase 3 means agents execute unchecked plans at scale — the worst outcome.

---

## Key Machine-Readable Outputs Agents Need

From proposal #86 (F4), the most agent-useful files in the zip artifact:

| File | Contents |
|------|----------|
| `assumptions/distilled_assumptions.json` | Key planning assumptions |
| `pre_project_assessment/pre_project_assessment.json` | Go/no-go recommendation |
| `negative_feedback/negative_feedback.json` | Risk register |
| `wbs/wbs_level2.json` | Work breakdown structure |

A future `plan_summary.json` collating these into a single file would eliminate the need for agents to parse multiple artifacts.

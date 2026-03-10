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
| **102** | Pipeline Intelligence Layer | Error-feedback retries — the LLM gets its own error message and retries with an adjusted approach. Eliminates a class of silent failures |
| **103** | Pipeline Hardening for Local Models | Fix silent truncation and context-window overflows. Critical for agents running local models where failures are subtle |
| **113** | LLM Error Traceability | ✅ **Implemented (PR #237)**. `LLMChatError` replaces generic `ValueError` across 38 call sites. Root cause preserved for error classification; `error_id` UUID enables log-to-metrics cross-referencing. Agents can programmatically diagnose failures |
| **101** | Luigi Resume Enhancements | Webhook hooks on task completion/failure — agents can subscribe to events instead of polling |

---

## 2. Agent Interface & Discoverability (reduce integration friction)

Agents need to discover PlanExe, understand its tools, and consume outputs programmatically. The current interface assumes a human is in the loop.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **86** | Agent-Optimized Pipeline | Removes the 5 key friction points for autonomous agent use: human approval gate, no agent prompt examples, poll intervals tuned for humans, no machine-readable output, no autonomous agent setup docs |
| **62** | Agent-First Frontend Discoverability | `llms.txt`, `/.well-known/mcp.json`, agent-readable README — standard discovery protocols so agents find PlanExe without human guidance |
| **110** | Usage Metrics for Local Runs | ✅ **Implemented (PR #219, #236, #237)**. Agents need cost accounting for budget-constrained workflows. `usage_metrics.jsonl` answers "how much did this run cost?" with per-call granularity (model, tokens, cost, duration). Errors are classified into short categories with traceable `error_id` UUIDs. Complements `activity_overview.json` aggregated totals |

Key friction points from #86 that block autonomous agent use:
- **F1**: Human approval step before `plan_create` — autonomous agents can't proceed
- **F4**: No machine-readable summary — agents must parse HTML or iterate 100+ zip files
- **F3**: 5-minute poll interval is wrong for cloud runs completing in 8 minutes

---

## 3. Plan Quality Gates (agents must not amplify bad plans)

Before agents can execute plans autonomously, the plans themselves need automated validation. An agent that executes an unchecked plan multiplies errors at machine speed.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **58** | Boost Initial Prompt | Single LLM call to strengthen weak prompts before pipeline runs. Especially valuable for agent-originated prompts, which may be terse or overly technical |
| **42** | Evidence Traceability Ledger | Links every claim to evidence with freshness scoring. Agents can programmatically check whether assumptions are grounded |
| **43** | Assumption Drift Monitor | Watches key variables (costs, FX rates) against baselines, triggers re-plan on breach. Agents re-planning on a schedule need this to detect when a plan has gone stale |
| **57** | Banned Words + Lever Realism | Auto-detects hype-heavy or impractical outputs without human review |
| **56** | Adversarial Red-Team Reality Check | Multi-model adversarial review with judge scoring. Catches optimism bias that agents are prone to propagating |
| **88** | Fermi Sanity Check Validation Gate | Rule-based guard on every assumption: bounds present, span ratio sane, evidence for low-confidence claims. First line of defense before expensive downstream processing |

---

## 4. Autonomous Prompt Optimization (agents improving PlanExe itself)

Luigi's caching makes prompt optimization uniquely practical — changing one prompt template and re-running only regenerates that single task (seconds, not the full 15-minute pipeline). This turns a 60+ LLM call pipeline into a 2–21 call experiment.

| # | Proposal | Agent Impact |
|---|----------|-------------|
| **94** | Autoresearch-Style Prompt Optimization | Autonomous overnight loops: agent modifies one prompt, re-runs one task, scores, keeps or reverts. Hundreds of experiments per night exploiting Luigi resumability |
| **59** | Prompt Optimizing with A/B Testing | Structured promotion pipeline: multi-model A/B matrix, Elo tracking, regression guards. Validates candidates discovered by #94 before merging into baseline |

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
Phase 1: Reliable foundation         (now)
  ├─ #87  Plan resume ✅
  ├─ #109 Retry improvements ✅ (PR #220)
  ├─ #102 Error-feedback retries
  ├─ #110 Usage metrics ✅ (PR #219, #236, #237)
  ├─ #113 Error traceability ✅ (PR #237)
  └─ #58  Prompt boost

Phase 2: Agent-native interface       (next)
  ├─ #86  Remove agent friction points
  ├─ #62  Discovery protocols
  └─ #88  Fermi validation gate

Phase 3: Automated quality            (then)
  ├─ #42  Evidence traceability
  ├─ #43  Assumption drift monitor
  ├─ #56  Adversarial red-team
  └─ #57  Banned words / lever realism

Phase 4: Self-improving pipeline      (concurrent with 3)
  ├─ #94  Autoresearch prompt optimization
  └─ #59  A/B testing promotion

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

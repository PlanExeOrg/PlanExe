# Proposal 73: Task Complexity Scoring & Model Routing

**Author:** Larry (Claude Sonnet 4.6), with input from Egon (Minimax M2.5) and Bubba (Haiku 4.5)  
**Date:** 2026-02-26  
**Status:** Draft — awaiting Simon's review  
**Type:** Docs-only proposal (no code until approved)

---

## Problem

PlanExe generates structured plans with tasks. Today, all tasks in a plan are executed on the same model — whatever the user has configured. This is wasteful: high-complexity tasks (architectural refactors, 1,000+ line files) genuinely need expensive models, while low-complexity tasks (docs, tests, small renames) can be handled by cheap models at 90–95% lower cost.

There is currently no mechanism for PlanExe to:
1. Assess how complex each generated task actually is
2. Recommend which model tier should handle it
3. Route tasks to appropriate models during execution
4. Estimate and track per-task token costs

## Opportunity

By scoring task complexity at plan-creation time, PlanExe can become an **AI workflow cost optimizer**, not just a plan generator. Opus writes the master plan (always — it has the full context). Each task gets a complexity score. Cheap models execute low-complexity tasks in parallel. Expensive models handle only what requires them.

**Real-world impact (from post-mortem of Simon's 26 Feb 2026 refactor):**
Simon's day involved 8 distinct task clusters. Estimated cost at Opus throughout: ~$18. Estimated cost with complexity routing: ~$8. That's ~55% savings — and on a larger refactor or repeated daily usage, the compounding effect is significant.

More importantly: the *stress* savings. Simon doesn't need to worry about which model to use for each task. PlanExe tells him.

---

## Proposed Solution

### 1. Complexity Scoring Model

Each task in a generated plan receives scores on four dimensions (1–5 Likert scale):

| Dimension | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|
| **File size** | <100 lines | 100–300 | 300–600 | 600–1000 | 1000+ |
| **Semantic complexity** | rename/replace | simple logic | new function | architectural | cross-file refactor w/ deps |
| **Ambiguity** | crystal clear + line numbers | minor choices | some design calls | significant decisions | open-ended |
| **Context dependency** | self-contained | 1 file | 1 module | multi-module | whole codebase |

Sum of scores maps to a recommended model tier:

| Score | Recommended Model | Rationale |
|-------|------------------|-----------|
| 4–7 | Minimax M2.5 ($0.30/$1.10 per 1M) | Mechanical execution: docs, tests, renames, config |
| 8–11 | Haiku 4.5 ($1.00/$5.00 per 1M) | Guided execution: bug fixes, small logic changes, deployment config |
| 12–15 | Sonnet 4.6 ($3.00/$15.00 per 1M) | Moderate complexity: medium files, known patterns, clear spec |
| 16–20 | Opus 4.6 ($5.00/$25.00 per 1M) | Planning + large files: architectural decisions, 1000+ line files, cross-codebase |

### 2. Additional Metadata Per Task

Beyond the base score, each task carries:

- **Estimated token budget** (input + output): rough estimate based on file sizes and task type
- **Estimated cost** at recommended model tier vs Opus
- **Confidence score** (1–5): how confident the scoring model is in its routing recommendation
- **Retry policy**: if the recommended model fails, escalate to next tier or re-attempt at same tier?
- **Session boundary flag**: should this task start a fresh session (Y/N)? Fresh sessions recommended when context exceeds ~150K tokens or when switching from planning to execution.
- **Parallelizable** (Y/N): can this task run concurrently with others?

### 3. The Two-Phase Execution Pattern

```
Phase 1 — Planning (always Opus, fresh session):
  Input: user prompt + relevant codebase context
  Output: structured plan with tasks, each scored on complexity rubric
  Session ends after plan is written.

Phase 2 — Execution (routed model, fresh session per task or task group):
  Input: plan document + only the files relevant to this task
  No context from Phase 1 carried over.
  Cheap models execute from the hit list Opus produced.
  High-complexity tasks (score 16–20) may use Opus for execution too.
```

Key rule: **Opus writes the plan. The plan must be specific enough (file paths, line numbers, exact changes, all decisions made) that a junior developer with no codebase knowledge could execute it.** If the plan has ambiguity, it's not done yet.

### 4. Session Hygiene Rules

- Start a new session after writing a plan (never execute in the planning session)
- Start a new session when context approaches 150K tokens (approaching price-doubling zone at Anthropic direct; avoids context drag even on flat-rate providers)
- Start a new session when switching task clusters (e.g., done with perf work, now doing docs)
- Parallelizable low-complexity tasks (docs, tests) can run in separate sessions simultaneously

---

## Calibration Problem & This Proposal's Data

The rubric scores above are hypotheses. Different models assess complexity differently — a model working near its capability ceiling perceives tasks as harder than a model with headroom. Without ground truth, the thresholds (4–7, 8–11, etc.) are arbitrary.

**This proposal includes three independent complexity assessments of Simon's 26 Feb 2026 refactor:**
- `91-complexity-assessment-larry-sonnet.md` — Larry's view (Claude Sonnet 4.6)
- `72-complexity-assessment-egon-minimax.md` — Egon's view (Minimax M2.5)
- `90-complexity-assessment-bubba-haiku.md` — Bubba's view (Haiku 4.5)

Each agent scored the same 8 task clusters independently, without seeing the others' assessments. Simon's feedback on these three documents is the first real calibration data for the rubric.

**Simon's feedback should answer:**
1. Which agent's complexity scores most closely matched the actual difficulty you experienced?
2. Were there tasks we all scored too high? Too low?
3. Where did complexity manifest in ways the rubric doesn't capture (e.g., debugging time, failed attempts)?
4. Would you have accepted a plan from Haiku for the security hardening cluster, or did it feel risky?

Over time, Simon's feedback plus real execution data (did the routed model succeed on the first attempt, or did it need escalation?) calibrates the rubric thresholds.

---

## Implementation Sketch (for future code proposal)

This is a docs-only proposal. No code changes are proposed here. If Simon approves the concept, the implementation proposal would cover:

1. **Schema changes:** Add `complexity_score`, `recommended_model`, `estimated_tokens`, `estimated_cost`, `is_parallelizable`, `new_session_before` fields to `PlanItem`
2. **Scoring logic:** During plan generation, Opus scores each task it creates using the rubric (it's already reading the relevant context to create the task — scoring is incremental)
3. **Routing API:** New MCP tool `plan_routing_config` or metadata on `plan_create` response
4. **Execution tracking:** Record actual model used, actual tokens consumed, success/failure per task — feeds calibration loop
5. **Cost dashboard:** Show per-task cost breakdown in plan status/output

---

## Why This Matters Beyond PlanExe

The complexity rubric + model routing pattern is a general solution to a problem every team building with AI faces: *how do I not spend Opus tokens on something Haiku can do?*

PlanExe is uniquely positioned to solve this because it already:
- Reads the relevant codebase context to create plans
- Structures work into discrete tasks
- Has an MCP interface that tool-calling clients can consume

Adding complexity scoring makes PlanExe the **intelligence layer** between a developer's prompt and the actual model execution — routing, cost estimation, session management, and parallelization all handled automatically.

---

## Next Steps (pending Simon's approval)

1. Simon reviews the three complexity assessments and provides calibration feedback
2. If rubric concept is approved, open separate code proposal for schema + scoring implementation
3. First implementation: static rubric (hardcoded thresholds, Opus scores tasks at creation time)
4. Second iteration: dynamic calibration (rubric weights adjust based on execution outcomes)

---

*Proposal authored by Larry. Rubric refinements by Egon. Authorized by Mark Barney.*

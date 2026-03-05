---
title: OpenClaw Agent Skill Integration
date: 2026-02-11
status: proposal
author: PlanExe Team
---

# OpenClaw Agent Skill Integration

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** OpenClaw Developers, Agent Architects

---

## Pitch
Package PlanExe as a standardized OpenClaw skill that turns agents into project managers: generate plans in the cloud, execute locally on edge devices, and report progress back into PlanExe.

## Why
Edge agents have sensors and actuators but low compute. Cloud agents can plan but lack physical access. A unified skill bridges this split and enables coordinated execution.

## Problem

- Edge agents lack LLM capacity to generate robust plans.
- Cloud agents cannot directly execute physical tasks.
- There is no consistent interface for plan generation and task execution.

## Proposed Solution
Create a $PlanExeSkill for OpenClaw that:

1. Drafts plans via PlanExe Cloud.
2. Breaks plans into executable tasks.
3. Routes tasks to edge or human executors.
4. Reports results and updates the plan state.

## Architecture

```text
OpenClaw Agent
  -> PlanExe Skill
     -> MCP Client
        -> PlanExe Cloud
           -> Plan JSON
     -> Task Executor
  -> Result Reporter
```

### Skill Manifest (`skill.json`)

```json
{
  "name": "PlanExe Project Manager",
  "version": "1.0.0",
  "description": "Gives the agent the ability to plan, budget, and track complex projects via the PlanExe Cloud.",
  "permissions": ["network_access"],
  "mcp_tools": [
    "example_prompts",
    "task_create",
    "task_status",
    "task_stop",
    "task_file_info"
  ]
}
```

## Skill Capabilities (MCP Tools)

The PlanExe MCP exposes the following real tools via `https://mcp.planexe.org/mcp`:

### `example_prompts()` — Get example prompts

- Input: none
- Output: List of example planning prompts
- Used in: Skill initialization and user guidance

### `task_create(prompt, speed_vs_detail, model_profile, user_api_key?)` — Start a planning task

- Input:
  - `prompt` (string): Natural language planning request
  - `speed_vs_detail` (enum): `"ping"` | `"fast"` | `"all"`
  - `model_profile` (enum): `"baseline"` | `"premium"` | `"frontier"` | `"custom"`
  - `user_api_key` (optional string): Override environment API key
- Output: `task_id` for polling and result retrieval
- Used in: Step 1 of planning workflow

### `task_status(task_id)` — Poll planning progress

- Input: `task_id` (string)
- Output: status (queued|running|completed|failed), progress %, estimated time remaining
- Used in: Polling loop every 5+ minutes (plans take 15-20+ min)

### `task_stop(task_id)` — Cancel a running task

- Input: `task_id` (string)
- Output: Confirmation of cancellation
- Used in: Early termination / user interruption

### `task_file_info(task_id, artifact)` — Get download link for results

- Input:
  - `task_id` (string)
  - `artifact` (enum): `"report"` | `"zip"`
- Output: `download_url`
- Used in: Retrieving completed plans after task finishes

## Agent-to-Agent Protocol

- **EdgeBot** detects a local issue (low water).
- **EdgeBot** requests a plan from **CloudBot**.
- **CloudBot** calls `task_create()` via PlanExe MCP and receives `task_id`.
- **CloudBot** polls `task_status(task_id)` every 5+ minutes.
- **CloudBot** calls `task_file_info(task_id, "report")` to get download URL when complete.
- **EdgeBot** executes local steps from the plan and reports back.
- **CloudBot** can call `task_stop(task_id)` if execution is cancelled.

## Integration Points

- Uses PlanExe MCP interface for plan creation.
- Feeds execution data to assumption drift and readiness scoring.
- Works with distributed physical task dispatch protocol.

## Success Metrics

- Installation rate: % of OpenClaw instances with PlanExe skill.
- Plan completion rate without human intervention.
- Mean time from goal to first actionable task.

## Risks

- Over-reliance on cloud connectivity.
- Misaligned task interfaces between cloud and edge.
- Skill misuse without governance or budget limits.

## Future Enhancements

- Offline plan caching for intermittent connectivity.
- Capability-aware task routing across multiple agents.
- Automatic escalation to humans for high-risk tasks.

## Detailed Implementation Plan

### Phase A — Skill Packaging and Contracts (1–2 weeks)

1. Define skill manifest and MCP tool bindings:
   - `example_prompts` — Provide example planning prompts
   - `task_create` — Initiate planning task with goal and parameters
   - `task_status` — Poll task progress (required for async workflows)
   - `task_stop` — Cancel long-running tasks
   - `task_file_info` — Retrieve generated plan artifacts

2. Add JSON schema validation for tool inputs/outputs.
3. Version the skill API separately from PlanExe core API.

### Phase B — MCP Bridge and Runtime Adapter (2–3 weeks)

1. Implement MCP client wrapper inside skill runtime.
2. Add retry/backoff and circuit-breaker handling for cloud calls.
3. Normalize errors into agent-friendly remediation messages.

### Phase C — Execution Loop and State Sync (2 weeks)

1. Build local task executor abstraction for edge environments.
2. Add plan-state sync protocol:
   - pull next action
   - execute/report
   - reconcile conflicts

3. Support partial completion and blocked states.

### Phase D — Safety, Budgets, and Governance (2 weeks)

1. Add execution policy profiles by risk tier.
2. Add per-run budget controls and token/cost caps.
3. Add human escalation hooks for high-impact tasks.

### Integration points

- OpenClaw skill registry
- PlanExe MCP cloud endpoints
- Execution readiness and evidence ledger modules

### Operational safeguards

- Offline queue with replay upon reconnect
- Idempotency keys for duplicate submissions
- Signed task receipts for non-repudiation

### Validation checklist

- End-to-end tool contract tests
- Network partition resilience tests
- Duplicate event/retry safety tests
- Human escalation path latency and success tests

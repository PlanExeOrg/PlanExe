---
title: Autonomous Execution of a Plan (AI + Human Delegation)
date: 2026-02-11
status: proposal
author: EgonBot on Raspberry Pi 4
---

# Autonomous Execution of a Plan (AI + Human Delegation)

## Pitch
Turn a PlanExe plan into a living execution program that runs autonomously where possible, delegates to humans where necessary, and continuously re-plans as reality changes.

## Why
A static plan is not enough. Real execution requires:

- resource-aware scheduling
- continuous evidence checks
- delegation to humans when AI capability is insufficient
- iterative re-planning based on new data and outcomes

This proposal extends plan generation into **plan execution**, with orchestration, governance, and adaptive control.

## Problem

- Plans often assume ideal conditions and static timelines.
- AI agents can execute some tasks, but not all (legal, physical, negotiation).
- Human resources are limited and unevenly available.
- Without re-planning, early deviations compound into failure.

## Proposed Solution
Build an **Execution Engine** that:

1. Converts plan JSON into an executable task graph.
2. Assigns tasks to AI agents or humans based on capability and risk.
3. Tracks progress, evidence, and constraints in real time.
4. Re-plans dynamically as facts change.
5. Produces audit-ready reports and governance checkpoints.

## Execution Architecture

```text
PlanExe JSON
  -> Task Graph Builder
  -> Orchestrator
     -> Agent Pool (AI)
     -> Human Queue
     -> Evidence Validator
     -> Risk Gates
  -> Progress + Metrics
  -> Adaptive Re-Plan
```

## Core Components

### 1) Task Graph Builder

- Parse PlanExe outputs into a DAG of tasks with:
  - dependencies
  - required inputs/outputs
  - estimated duration and cost
  - risk class and verification level

### 2) Capability Registry

- Agents and humans register their capabilities as schemas.
- Each task specifies the capability required.
- Orchestrator performs matching based on skills, availability, and cost.

### 3) Orchestrator

- Dispatches tasks to agents or humans.
- Manages retries, fallbacks, and escalation.
- Tracks state transitions (queued, running, blocked, done).

### 4) Human Task Queue

- Dedicated UI showing:
  - task description
  - required evidence
  - urgency and deadline
  - instructions and context
- Human actions logged with rationale.

### 5) Evidence Validator

- Every task output must pass a schema and evidence check.
- Outputs are tagged with confidence.
- Low-confidence or high-impact results trigger human review.

### 6) Risk Gates

- High-impact tasks require approval before execution.
- Risk gates enforce compliance, safety, and budget limits.

## Delegation Model

### Capability Matching

- Match task requirements to agent or human skill profiles.
- Consider risk class, data sensitivity, and cost.

### Decision Rules

- **AI-first** for low-risk, repeatable tasks.
- **Human-first** for legal, regulatory, and negotiation tasks.
- **Hybrid** for tasks requiring AI prep plus human judgment.

### Escalation

- If AI confidence falls below threshold, reroute to human.
- If humans reject, re-plan or adjust scope.

## Adaptive Re-Planning

Execution is not linear. The system must re-plan when:

- task outputs contradict assumptions
- dependencies are delayed
- budgets exceed thresholds
- external factors change (regulation, market, supplier)

### Re-Planning Loop

1. Detect deviation vs baseline.
2. Update plan assumptions and constraints.
3. Recompute schedule and resource allocation.
4. Issue new tasks or adjust milestones.

### Re-Plan Outputs

- Updated Gantt (baseline vs current)
- Revised risk register
- Decision log with rationale

## Resource and Capacity Management

- Track available AI resources and human availability.
- Enforce concurrency limits.
- Prioritize high-impact work when capacity is constrained.

### Capacity Profile Example

```json
{
  "ai_agents": 12,
  "human_fte": {
    "legal": 1,
    "engineering": 3,
    "procurement": 0.5
  },
  "max_parallel_tasks": 20
}
```

## Data Contracts

### Task Schema

```json
{
  "task_id": "t_001",
  "title": "Obtain environmental permit",
  "capability": "regulatory_filing",
  "priority": "high",
  "risk_class": "critical",
  "inputs": ["site_plan", "impact_assessment"],
  "outputs": ["permit_document"],
  "status": "queued"
}
```

### Execution Event Log

```json
{
  "event_id": "e_445",
  "task_id": "t_001",
  "actor": "human:legal_01",
  "action": "approved",
  "timestamp": "2026-02-11T10:22:00Z",
  "notes": "Permit submitted"
}
```

## Reporting Outputs

- Progress dashboard: status by milestone.
- Risk dashboard: unresolved risks and blockers.
- Evidence coverage: which claims are verified.
- Final execution report:
  - timeline vs baseline
  - deviations and corrective actions
  - decision log
  - confidence profile

## Example Execution Timeline (With Change Events)

**Baseline (Day 0):**

- Week 1-2: feasibility + regulatory scoping
- Week 3-4: procurement + vendor selection
- Week 5-8: build + integration
- Week 9: compliance review
- Week 10: launch

**Execution Events:**

- Day 9: Regulator requests additional environmental study.
- Day 10: Task graph updated, compliance lane becomes critical.
- Day 11: Human legal capacity bottleneck detected.
- Day 12: Orchestrator proposes contractor augmentation and re-baselines timeline.
- Day 14: Updated plan published with new dependencies and budget deltas.

**Output:** Updated Gantt, risk register, and decision log entry that explains the change and its impact.

## Failure Handling and Rollback

Autonomous execution must assume failure cases and encode recovery:

### Failure Types

- **Data failure:** missing or invalid inputs.
- **Task failure:** agent returns low-confidence or incorrect output.
- **External failure:** third-party dependency delays or rejects.
- **Policy failure:** execution violates budget, compliance, or ethics constraints.

### Recovery Actions

- **Retry with alternative agent** (for data and task failures).
- **Escalate to human** for high-risk or repeated failures.
- **Re-plan** to adjust dependencies and milestones.
- **Rollback** to last stable snapshot if downstream impacts are unsafe.

### Rollback Mechanism

- Every milestone creates a snapshot of plan state and evidence.
- Rollback reverts to the last validated snapshot.
- All downstream tasks are invalidated and re-queued with updated inputs.

## AI-to-Human Handoff Contract (SLA + Requirements)

For tasks that require human intervention, define a strict handoff contract:

### Required Fields

- Task purpose and required outcome.
- Evidence needed and acceptance criteria.
- Deadline and urgency.
- Context pack (inputs, dependencies, prior decisions).
- Suggested next steps and contact references.

### SLA Targets

- **Acknowledgement:** within 24 hours.
- **Completion:** within task-defined window (default 5 business days).
- **Escalation:** if no response within SLA, route to alternate human or reduce scope.

### Audit Requirement

Every handoff must record:

- human assignee
- response timestamp
- decision rationale
- evidence provided

## Safety and Governance

- Immutable audit log for all actions and outputs.
- Explicit sign-offs on high-impact tasks.
- Budget and compliance thresholds enforced by policy.
- Kill-switch to halt execution in emergencies.

## Success Metrics

- % tasks executed without manual intervention
- Median deviation vs baseline schedule
- % high-risk tasks with evidence verified
- Time saved vs manual execution coordination

## Risks

- Over-automation may hide human context.
- Insufficient human capacity causes bottlenecks.
- False confidence from AI outputs.

## Feasibility Tiers

Autonomous execution is feasible only with clear boundaries. Define tiers by risk and controllability:

- **Tier 1 (Feasible Now):** low-risk, repeatable tasks (data gathering, summarization, formatting, internal reporting).
- **Tier 2 (Partially Feasible):** tasks that can be AI-driven but require human approval (procurement drafts, compliance checklists, vendor shortlists).
- **Tier 3 (Not Feasible Without Human Lead):** legal filings, negotiations, physical execution, regulatory approvals, and high-stakes financial commitments.

## Feasibility Adjustments

To make autonomous execution practical, enforce the following adjustments:

- **Scope limits:** constrain execution to well-defined domains and task types.
- **Hard policy gates:** require explicit human approval for high-impact steps.
- **Evidence sufficiency checks:** block execution when inputs are unverified.
- **Capacity-aware scheduling:** align execution with real human availability.
- **Rollback readiness:** snapshot after each milestone and auto-revert on critical deviation.

## Staged Rollout Plan

### Phase 1: Autonomous Coordination

- Task graphing, scheduling, and progress tracking.
- Human execution for all tasks.
- Evidence collection and audit logging only.

### Phase 2: Low-Risk Autonomous Execution

- AI executes low-risk tasks under strict validation.
- Human approval for all medium and high risk tasks.
- Adaptive re-planning enabled.

### Phase 3: Selective Autonomous Execution

- AI executes a broader set of tasks with confidence thresholds.
- High-risk steps remain human-led.
- Continuous monitoring and rollback enforced.

## Scope Checklist (Tier Classification)

Use this checklist to classify each task into Tier 1/2/3. If any Tier 3 condition is true, the task is human-led.

### Tier 1 (AI-Executable)

- Internal analysis or data aggregation only.
- No external commitments or legal exposure.
- Outputs are reversible and low-cost to redo.
- Evidence inputs are verified and structured.

### Tier 2 (AI-With-Approval)

- External communication or procurement drafts.
- Moderate budget impact but reversible.
- Requires validation by a human reviewer.
- Evidence is strong but not fully audited.

### Tier 3 (Human-Led)

- Legal filings, regulatory submissions, or compliance sign-offs.
- Negotiations, contracts, or financial commitments.
- Physical execution or safety-critical actions.
- High budget impact or irreversible decisions.

## Roadmap

1. MVP orchestrator with basic DAG execution.
2. Capability registry and task schema.
3. Human task queue integration.
4. Evidence validator and risk gate layer.
5. Adaptive re-planning and reporting.
6. Pilot on a real plan with human oversight.

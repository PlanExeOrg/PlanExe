# Autonomous Execution of a Plan by a Team of AI Agents

## Overview
This proposal describes how a PlanExe‑generated strategic plan can be executed autonomously by a coordinated team of AI agents, while delegating any tasks that fall outside the agents’ capabilities to human operators.

## 1. Execution Engine
- **Orchestrator** – a lightweight service that reads the PlanExe JSON output, builds a task graph, and schedules work across agents.
- **Agent Types** – specialized micro‑services (e.g., data‑gathering, analysis, reporting) each exposing a standard RPC/REST interface.
- **Human‑in‑the‑Loop** – tasks marked `human_required` are routed to a task‑queue watched by human workers via the existing PlanExe UI.

## 2. High‑level Architecture
```
+----------------+      +----------------+      +----------------+
|   Planner      | ---> | Orchestrator   | ---> |   Agents       |
+----------------+      +----------------+      +----------------+
        |                         |                     |
        v                         v                     v
   Plan JSON                Task Graph          Execution Results
```
- The **Planner** (PlanExe) produces a JSON plan.
- The **Orchestrator** parses the plan, constructs a DAG of tasks, and assigns each task to an appropriate agent.
- **Agents** are independent services (LLM‑driven, data‑fetching, computation) that expose a uniform `run(task)` API.
- Human‑only tasks are sent to a **Human Queue** visible in the UI.

## 3. Delegation Flow
1. **Capability Matching** – each agent registers a schema of actions it can perform. The orchestrator matches plan steps to agents based on these schemas.
2. **Task Assignment** – the orchestrator sends the task payload to the chosen agent via RPC.
3. **Result Collection** – agents return JSON results plus a confidence score.
4. **Fallback** – if no agent matches, a human ticket is created; if an agent rejects, the orchestrator retries with an alternative or escalates.
5. **Human Review** – low‑confidence or high‑impact results trigger a human approval step before continuation.

## 4. Required Extensions
- **Capability Registry Service** – a tiny HTTP service where agents POST their `schema.json` and the orchestrator queries it.
- **Human Ticket Queue** – extend the existing PlanExe UI with a task list (`/tasks`) that shows pending human‑required steps.
- **Result Validator** – a shared library that checks confidence thresholds and flags anomalies for review.
- **Audit Logger** – immutable log (e.g., append‑only file or simple DB) recording every task dispatch, result, and reviewer decision.

## 5. Reporting – What the Pipeline Will Emit
- **Progress Dashboard** – real‑time status (queued, running, completed, failed) displayed in the PlanExe front‑end.
- **Intermediate Reports** – after each major milestone the orchestrator invokes `run_plan_pipeline.py` to generate updated Gantt charts, risk registers, and executive summaries.
- **Final Execution Report** – a consolidated PDF/HTML document containing:
  - Execution timeline
  - Deviations from the original plan
  - Human decisions and rationale
  - Confidence metrics per task
  - Audit log reference

## 6. Safety & Risk Mitigation
- **Explicit Risk Gates** – before any high‑impact step (budget allocation, regulatory filing) the orchestrator requires explicit human approval.
- **Audit Trail** – every action is signed with the agent’s identity and timestamped, enabling full traceability.
- **Existential‑Risk Checks** – a dedicated “risk‑assessment” agent runs scenario analysis on critical milestones and flags any existential‑risk concerns for senior review.
- **Rollback Capability** – because each milestone produces a snapshot, the plan can be rolled back to a safe state if a downstream failure is detected.

## 7. Roadmap
1. **Prototype Orchestrator** – FastAPI service with a simple DAG scheduler (MVP in 2 weeks).
2. **Define Agent Schema** – publish a JSON‑Schema for task capabilities; implement two example agents (data fetcher, LLM summarizer).
3. **Integrate Human Queue** – UI extension to show pending human tasks and allow approval/rejection.
4. **Implement Reporting Hooks** – call `run_plan_pipeline.py` after each milestone.
5. **Safety Review Layer** – add risk‑gate middleware and audit logger.
6. **Beta Test** – run on a real PlanExe generated plan, collect feedback, iterate.

---
*This document merges the earlier high‑level sections (Architecture, Delegation Flow, Extensions, Reporting, Safety, Roadmap) with the concrete execution details described previously.*

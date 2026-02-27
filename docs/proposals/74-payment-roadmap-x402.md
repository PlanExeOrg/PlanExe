# 74-payment-roadmap-x402.md — Roadmap for x402 & A2A plan economics

**Author:** Egon
**Date:** 2026-02-27

## Context
PlanExe already turns prompts into structured plans. The next frontier is turning those plans into self-financing workflows. Two related initiatives anchor the ecosystem:

- **x402** — an internal plan-execution credit system that tracks compute spend and offsets it with downstream value (AI request billing, customer chargebacks, or contribution bounties).
- **A2A (Agent-to-Agent Payments)** — a practical ledger for agents to invoice each other for tool use, compute cycles, or specialized expertise when orchestrating multi-agent workflows.

This document maps those programs into a single roadmap for charging, settling, and reinvesting the work that PlanExe automates.

## Principles
1. **Cost visibility first** — Every task in a PlanExe plan should surface the estimated compute cost (model, tokens, session length) and whether it falls on AWS, OpenRouter, or a local inference engine.
2. **Charge attribution** — Agents (human or software) that initiate, approve, or operate a plan should be able to pay a share of the x402 credit cost or receive credits when they deliver value.
3. **Automated settlements** — A2A payments should happen automatically when an agent hands off a plan step to another agent, with escrow for verification/review.
4. **Reinvestment loop** — Collected x402 credits feed the Hydra-Matic Fund that keeps the plan orchestration stack healthy for low-cost execution tiers (Minimax, local models, etc.).

## Roadmap
### Phase 1: Cost tagging (Weeks 0-2)
- Extend the task metadata with `estimated_cost`, `model_tier`, and `execution_mode` (`local`, `cloud`, `accelerated`).
- Record session length & token counter per plan segment (`input_tokens`, `output_tokens`, `context_tokens`).
- Push the data into a lightweight `x402_cost_events` table for billing transparency.

### Phase 2: x402 credit ledger (Weeks 2-4)
- Create the `x402_credit` concept: each plan run consumes credits proportional to compute cost.
- Agents can top up credits manually (wallet tied to GitHub identity) or automatically via organizational budgets.
- When a plan executes, x402 debits the initiator and credits contributors (approval, QA, execution). Credits accumulate in `PlanExeReserve`.

### Phase 3: A2A payments and invoices (Weeks 4-6)
- Introduce `agent_invoice` objects for handoffs: e.g., `PlanExecutorAgent` runs plan nodes and invoices the initiating agent for the tokens burned.
- Use lightweight verification: the next agent in the chain approves the invoice before execution continues.
- Support fixed-rate services (e.g., `security-review-service` always charges 0.15 credits per 1K tokens).

### Phase 4: Reinvestment + hybrid funding (Weeks 6-8)
- Collected x402 credits fund a `Hydra-Matic Fund` that subsidizes manual-mode optimization (local inference hardware, dedicated Minimax capacity).
- Track `return_on_plan`: if a plan generates a deliverable (report, code, doc) valued > computed cost, issue rebate credits to the plan owner.
- Enable `Plan Marketplace` where agents browse pooled credit balances for cross-team execution.

### Phase 5: Governance & reporting (Weeks 8-10)
- Publish weekly `x402_spend` dashboards showing per-team, per-plan cost, average model tier, and credit utilization.
- Introduce compliance workflows for A2A payments (manual overrides, dispute resolution, audit logs). Integrate with MCP logging for transparency.

## Closing the loop
* x402 = dollars → compute credits → Hydra-Matic Fund → lower-cost tiers.
* A2A = agent accountability + micro-payments for work handoffs.
* This roadmap ensures PlanExe doesn’t just plan for free; it charges, settles, and reinvests in the same session.

Next steps: draft implementation PRs for task metadata (#72), Hydra-Matic UI (#74), and accounting APIs (#75). Let me know if you want a companion doc on the credit ledger schema.
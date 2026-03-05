---
title: Business model (developer)
---

# Business model (developer)

This document describes how PlanExe monetization works in hosted mode, how charging is computed, and how self-hosted mode remains unaffected.

---

## Product surfaces

PlanExe currently has two hosted surfaces:

- `home.planexe.org`: authentication and credit purchase (Stripe/Telegram).
- `mcp.planexe.org`: MCP API for creating plans.

Users buy credits on `home.planexe.org`, then spend credits when plans run through hosted services.

---

## Charging model

Charging is based on **actual token inference cost** plus an optional fixed success fee.

### Definitions

- `inference_cost_usd`: total run inference cost from `activity_overview.json.total_cost`.
- `success_fee_usd`: fixed fee for successful plans. Default is `1.0` USD.

### Formula

- If plan succeeds: `charge_usd = inference_cost_usd + success_fee_usd`
- If plan fails: `charge_usd = inference_cost_usd`
- Exception for `speed_vs_detail=ping_llm`: `charge_usd = inference_cost_usd` (no success fee)

This means failed plans still pay for consumed tokens, but do not pay the success fee.

---

## Why this model

- Fair across model choices: expensive models consume more and cost more.
- Fair on failures: real inference usage is billed even if no report is produced.
- Predictable business unit on successful plans: fixed success fee per completed output.

---

## Credit conversion

Internal billing deducts **fractional** credits from `UserAccount.credits_balance`.

- `PLANEXE_CREDIT_PRICE_CENTS` defines the value of one credit.
- USD charge is converted by exact division:
  - `charge_credits = charge_usd / (PLANEXE_CREDIT_PRICE_CENTS / 100)`
- Credits are stored with decimal precision (`NUMERIC(18,9)`), so tiny token costs are preserved.

Example with `PLANEXE_CREDIT_PRICE_CENTS=100`:

- `$1.00` -> `1.0` credits
- `$1.31` -> `1.31` credits
- `$0.0000068` -> `0.0000068` credits

---

## Billing timing

Billing is applied at **plan completion time** in `worker_plan_database`, not at plan creation time.

This ensures we can bill based on final observed inference usage and success/failure outcome.

---

## Data source for inference cost

The worker reads per-run:

- `activity_overview.json`
- field: `total_cost`

`total_cost` is produced by the token/cost tracking pipeline and reflects aggregated provider-side inference cost.

Operationally, per-call diagnostics are available in `token_metrics`:

- `task_id` and `user_id` for support and billing triage
- routed provider/model (`upstream_provider`, `upstream_model`) for cost-variance analysis
- per-call token counts, duration, and `cost_usd` when present

---

## Hosted flow (high level)

1. User buys credits via Stripe/Telegram.
2. User starts a plan from web UI or MCP.
3. Worker runs pipeline and records token/cost activity.
4. On completion/failure, worker computes charge using formulas above.
5. Credits are deducted and appended to `CreditHistory` ledger.

Ledger entries use usage-billing metadata for auditability and idempotency.

---

## Self-hosted behavior (must remain unchanged)

PlanExe is open source and can be run via Docker Compose or local environments.

In self-hosted deployments:

- Users manage their own model/provider costs directly (OpenRouter, OpenAI-compatible providers, Ollama, etc.).
- PlanExe hosted credit billing is not required.
- If users run local Ollama models, their inference can be effectively free (excluding hardware/power).

Implementation rule: hosted credit billing only applies when a run maps to a real `UserAccount` in the hosted database. Non-hosted run identities are not billed through hosted credits.

---

## Environment variables

- `PLANEXE_SUCCESS_PLAN_FEE_USD` (default `1.0`): fixed fee added only on successful plans.
- `PLANEXE_CREDIT_PRICE_CENTS` (default `100`): cents per credit.

Related payment-side variables are documented in [Stripe](stripe.md).

---

## Free plan behavior

Hosted web UI supports one free plan per user account.

- First plan can be flagged to skip usage billing.
- Subsequent plans are usage-billed according to formulas above.

This is implemented as an explicit plan parameter so billing logic remains deterministic and auditable.

---

## Operational notes

- Billing is idempotent per plan: usage charge is applied once.
- Billing records should be traceable to plan id and run output.
- Payment/support investigations should join `CreditHistory` with `token_metrics` via plan context and `user_id`.
- If pricing policy changes, update this document and relevant env defaults together.

---

## Related docs

- [User accounts and billing (database)](user_accounts_and_billing.md)
- [Stripe (credits and local testing)](stripe.md)
- [Costs and models](costs_and_models.md)
- [Token counting implementation](token_counting.md)

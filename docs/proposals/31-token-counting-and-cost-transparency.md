---
title: Token Counting + Cost Transparency (Raw Provider Tokens)
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Token Counting + Cost Transparency (Raw Provider Tokens)

## Pitch
Expose per-plan token usage and cost breakdowns, using raw provider token counts to enable transparent budgeting, optimization, and governance.

## Why
Token costs are opaque and often underestimated. Transparent cost accounting is essential for budgeting, pricing, and scaling decisions.

## Problem

- Users cannot see cost drivers across steps.
- Internal teams cannot optimize prompt and model usage.
- Investors and operators lack visibility into plan-generation cost structure.

## Proposed Solution
Implement a token accounting layer that:

1. Captures raw provider token counts for every model call.
2. Maps tokens to cost using provider pricing tables.
3. Aggregates cost by plan stage, plugin, and model.
4. Surfaces a user-facing cost report.

## Data Model

### Token Event Schema

```json
{
  "plan_id": "plan_123",
  "stage": "assume",
  "model": "gpt-4o-mini",
  "input_tokens": 4200,
  "output_tokens": 900,
  "provider_cost_usd": 0.034
}
```

### Aggregation Schema

```json
{
  "plan_id": "plan_123",
  "total_cost_usd": 1.42,
  "by_stage": {
    "assume": 0.35,
    "risk": 0.22,
    "finance": 0.47
  },
  "by_model": {
    "gpt-4o-mini": 0.78,
    "gemini-2.0-flash": 0.64
  }
}
```

## Reporting Views

- **Plan Cost Summary:** total tokens, total cost, top cost drivers.
- **Stage Breakdown:** cost per pipeline stage.
- **Model Breakdown:** cost per model/provider.
- **Optimization Insights:** suggestions to reduce high-cost stages.

## Governance Features

- Cost caps per plan or per day.
- Alerts when costs exceed thresholds.
- Audit logs for cost anomalies.

## Integration Points

- Works with all pipeline stages and plugins.
- Feeds budgeting dashboards.
- Used in governance and allocation decisions.

## Success Metrics

- Cost visibility for 100% of plans.
- Reduction in cost per plan after optimization.
- Fewer cost overruns and unexpected bills.

## Risks

- Provider token counts may change or be inconsistent.
- Cost reporting overhead adds latency.
- Misinterpretation of cost data by users.

## Future Enhancements

- Per-user or per-team cost budgeting.
- Predictive cost estimation before plan generation.
- Multi-currency cost reporting.

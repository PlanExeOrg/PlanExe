---
title: Finance Analysis via Bottom-Up Estimation + Reconciliation
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Finance Analysis via Bottom-Up Estimation + Reconciliation

## Pitch
Build a bottom-up financial model from tasks, resources, and unit economics, then reconcile it against top-down estimates to surface gaps and improve accuracy.

## Why
Top-down estimates are fast but coarse. Bottom-up estimates are realistic but time-consuming. Combining both gives the speed of top-down with the credibility of bottom-up, while exposing unrealistic assumptions early.

## Problem

- Plans often include partial or inconsistent financials.
- Bottom-up models are missing or unstructured.
- Divergence between top-down and bottom-up is not tracked.

## Proposed Solution
Implement a bottom-up estimation module that:

1. Extracts work packages, resources, and timelines.
2. Builds cost and revenue from unit-level assumptions.
3. Aggregates to totals and cash flow.
4. Reconciles differences with top-down estimates.

## Bottom-Up Estimation Framework

### 1) Work Package Extraction
Identify:

- Tasks and milestones
- Deliverables and work packages
- Staffing requirements
- Duration and dependencies

### 2) Unit Cost Modeling
Attach costs per unit:

- Labor: role-based hourly or monthly rates
- Materials: quantity x price
- Infrastructure: cloud usage, hardware
- External services: contractors, vendors

### 3) Revenue Modeling
Build revenue from:

- Units sold x price
- Contract values and timelines
- Subscription tiers and churn
- Conversion funnel estimates

### 4) Aggregation
Produce:

- Project budget by phase
- Monthly burn and runway
- Break-even timing
- Profit and loss summary

## Reconciliation Layer

Compare bottom-up vs top-down outputs:

- Total revenue variance
- Margin variance
- Capex and opex mismatches
- Timeline inconsistencies

**Reconciliation output:**

- Variance report
- Recommended adjustments
- Updated confidence levels

## Output Schema

```json
{
  "bottom_up": {
    "total_cost": 2200000,
    "total_revenue": 4800000,
    "burn_rate_monthly": 180000
  },
  "top_down": {
    "total_cost": 1500000,
    "total_revenue": 5200000
  },
  "variance": {
    "cost_delta": 700000,
    "revenue_delta": -400000
  },
  "reconciliation_notes": [
    "Bottom-up assumes 12 engineers, top-down assumes 8",
    "Top-down margin range exceeds observed unit economics"
  ]
}
```

## Integration Points

- Uses CBS generation as input for cost categories.
- Feeds into investor thesis matching and risk scoring.
- Drives evidence-based adjustments in financial claims.

## Success Metrics

- Percentage of plans with bottom-up models.
- Reduction in financial variance after reconciliation.
- Investor confidence in financial projections.

## Risks

- High data requirements: mitigate with default benchmarks and missing info prompts.
- Estimation complexity: prioritize major cost drivers first.
- False precision: publish ranges and confidence scores.

## Future Enhancements

- Automated cost libraries by region and sector.
- Sensitivity analysis and scenario modeling.
- Learning system that updates estimates from real outcomes.

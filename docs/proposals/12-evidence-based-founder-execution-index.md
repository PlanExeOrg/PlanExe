# Evidence-Based Founder Execution Index

**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `execution`, `founders`, `signals`, `anti-bias`, `roi`

---

## Pitch

Replace charisma-heavy founder evaluation with an evidence-based execution index built from verifiable delivery signals, improving investor confidence in projected ROI.

## TL;DR

- Score execution capability from objective signals, not pitch performance.

- Use delivery history, milestone reliability, hiring quality, and speed of iteration.

- Produce an auditable execution score with confidence level.

- Feed the score into investor matching and return forecasts.

## Problem

Investors often overweight presentation quality and social proof. This creates two failures:

- Good operators with low visibility are underrated.

- Great storytellers with weak execution can be overrated.

Both reduce expected portfolio returns.

## Proposed Solution

Create a **Founder Execution Index (FEI)** calculated from measurable evidence:

1. Delivery reliability (planned vs actual milestones)

2. Resource efficiency (burn vs validated progress)

3. Learning velocity (hypothesis-test cycles per month)

4. Team assembly quality (critical roles filled, retention, seniority relevance)

5. Incident response quality (speed and effectiveness after setbacks)

## Architecture

```text
┌─────────────────────────────┐
│ Data Sources                │
│ - Plan milestones           │
│ - Repo/product telemetry    │
│ - Hiring timeline           │
│ - Financial updates         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Signal Normalization Layer  │
│ - Clean / impute            │
│ - Sector-specific baselines │
│ - Fraud/anomaly checks      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ FEI Scoring Service         │
│ - Subscores                 │
│ - Confidence interval       │
│ - Explainability            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Matching Engine Integration │
│ - ROI adjustment            │
│ - Rank updates              │
└─────────────────────────────┘
```

## Implementation

### Phase 1: Signal Schema

- Define FEI event model:

  - `milestone_declared`, `milestone_delivered`, `experiment_started`, `experiment_validated`, `key_hire_added`, `incident_resolved`.

- Build ingestion adapters for PlanExe plans and optional external tools.

### Phase 2: FEI Model

- Compute subscores in [0,100]:

  - Reliability, Efficiency, Learning, Team, Resilience.

- Aggregate into composite score with uncertainty:

  - `FEI = Σ(weight_i * subscore_i) * data_confidence_factor`

- Adjust weights by sector and stage.

### Phase 3: Product + Investor UX

- Show FEI trend over time (trajectory matters more than static value).

- Add “evidence behind score” view with source links.

- Integrate FEI into investor recommendation ordering.

## Success Metrics

- **Prediction Lift:** FEI improves 12-month milestone attainment prediction by ≥ 20% over baseline profile review.

- **Bias Reduction:** Lower correlation between match rank and non-performance proxies (social following, founder media exposure).

- **Decision Speed:** Investor screening time reduced by ≥ 25%.

- **Outcome Link:** FEI top quartile portfolios show higher realized MOIC than bottom quartile.

## Risks

- **Sparse data for early teams** → Use uncertainty-aware scoring; never hide confidence level.

- **Metric gaming** → Cross-validate with external evidence and consistency checks.

- **Signal inequity across sectors** → Use sector-normalized benchmarks.

- **Privacy concerns** → Explicit consent and scoped data sharing.

## Why This Matters

A transparent execution index gives investors a stronger ROI signal and gives disciplined builders a fairer path to capital, independent of pitch theatrics.
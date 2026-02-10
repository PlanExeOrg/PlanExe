# Portfolio-Aware Capital Allocation for Investor Matching

**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `portfolio`, `allocation`, `optimization`, `risk`, `roi`

---

## Pitch

Upgrade matching from single-deal recommendations to portfolio-aware allocation so each investor sees opportunities that improve total expected portfolio ROI under risk constraints.

## TL;DR

- Build optimizer that recommends not only “what to invest in,” but also “how much.”
- Use covariance, concentration, and liquidity constraints.
- Prioritize deals with positive marginal contribution to portfolio return.
- Increase IRR consistency while reducing downside clustering.

## Problem

Most matching systems rank opportunities independently. Investors, however, deploy capital at portfolio level. Independent rankings can cause:
- Sector overconcentration
- Correlated downside exposure
- Capital fragmentation into low-impact checks

## Proposed Solution

Add a **Portfolio Allocation Optimizer** on top of plan-investor fit scores.

For each investor:
1. Estimate expected return distribution per plan
2. Estimate cross-plan correlation using sector + macro + business-model features
3. Solve constrained optimization for check sizing
4. Output prioritized shortlist with recommended allocation ranges

## Architecture

```text
┌──────────────────────────────┐
│ Plan Return Forecasts        │
│ - Expected MOIC/IRR          │
│ - Volatility + downside      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Correlation Estimation       │
│ - Sector links               │
│ - Revenue-model similarity   │
│ - Macro factor exposure      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Allocation Optimizer         │
│ - Constraints                │
│ - Position sizing            │
│ - Efficient frontier         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Investor Decision UI         │
│ - Recommended checks         │
│ - Risk contribution chart    │
│ - Scenario stress tests      │
└──────────────────────────────┘
```

## Implementation

### Phase 1: Return and Risk Inputs
- Standardize plan-level return forecasts to common horizons.
- Add downside metrics: probability of loss, expected drawdown, time-to-liquidity.

### Phase 2: Optimizer Service
- Formulate as constrained optimization:
  - Maximize expected portfolio utility (`E[R] - λ*Risk`)
  - Subject to check size, sector cap, stage cap, and liquidity limits.
- Run weekly recalculation and event-triggered refreshes.

### Phase 3: Decision Layer
- Render “marginal portfolio impact” per candidate.
- Provide stress scenarios (recession, funding winter, supply shock).
- Expose allocation confidence intervals.

## Success Metrics

- **Portfolio Sharpe-like Improvement:** +15% relative to baseline manual allocation.
- **Concentration Control:** No sector > configured cap in 95% of portfolios.
- **Capital Efficiency:** Higher deployed capital per decision hour.
- **Downside Reduction:** Lower 24-month tail-loss percentile.

## Risks

- **False precision in early-stage forecasting** → Use wide intervals and robust optimization.
- **Correlation instability** → Re-estimate continuously and include regime-switch models.
- **User complexity fatigue** → Default to simple recommendations with optional advanced views.
- **Data lag** → Ingest milestone updates in near real time.

## Why This Matters

Investors care about total portfolio outcomes, not isolated deal quality. Portfolio-aware matching improves capital allocation quality and makes ROI predictions more actionable.
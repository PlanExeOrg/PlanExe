# Investor Thesis Matching Engine

**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `investors`, `matching`, `roi`, `ranking`, `marketplace`

---

## Pitch

Build a Kickstarter-like discovery and funding layer where projects are matched to investors by expected risk-adjusted ROI and explicit thesis fit, not by founder charisma or social reach.

## TL;DR

- Convert every plan into a normalized feature vector (market, margin, burn, moat, timeline, execution risk).

- Convert every investor into a thesis vector (stage, sector, check size, target return, risk appetite, hold period).

- Score plan↔investor fit using explainable ranking.

- Show both sides a transparent “why this match” report.

- Goal: improve conversion rate, reduce time-to-first-commitment, and increase realized IRR.

## Problem

Current startup discovery is noisy and personality-driven:

- Strong projects can be underfunded if founders are weak at storytelling.

- Investors spend too much time filtering poor-fit deals.

- Match quality is opaque; post-hoc outcome learning is weak.

## Proposed Solution

Introduce a deterministic, data-first matching service that ranks investor-project pairs using:

1. **Thesis compatibility** (hard constraints + soft preferences)

2. **Projected ROI** (expected value with uncertainty)

3. **Execution confidence** (evidence-weighted feasibility)

4. **Diversification impact** (marginal portfolio contribution)

## Architecture

```text
┌────────────────────────────┐
│ Plan Ingestion             │
│ - PlanExe structured plan  │
│ - Financial assumptions    │
│ - Milestones + risks       │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ Feature Engineering        │
│ - Unit economics           │
│ - Market indicators        │
│ - Risk factors             │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐      ┌──────────────────────────┐
│ Matching & Scoring API     │◄────►│ Investor Thesis Profiles │
│ - Constraint filtering     │      │ - Return targets         │
│ - Fit + ROI ranking        │      │ - Risk + sector rules    │
│ - Explainability layer     │      │ - Check size constraints │
└─────────────┬──────────────┘      └──────────────────────────┘
              │
              ▼
┌────────────────────────────┐
│ Marketplace UI             │
│ - Ranked opportunities     │
│ - Why-match report         │
│ - Confidence intervals     │
└────────────────────────────┘
```

## Implementation

### Phase 1: Data Model + Constraint Engine

- Extend plan schema with investor-relevant fields:

  - TAM/SAM/SOM, CAC, LTV, gross margin, payback period, capital required, runway, regulatory risk.

- Add investor profile schema:

  - sectors, geography, stage, check range, target MOIC/IRR, max drawdown tolerance.

- Implement hard-filter pass (exclude impossible matches first).

### Phase 2: ROI + Fit Scoring

- Create weighted scoring function:

  - `FinalScore = 0.45*ThesisFit + 0.35*RiskAdjustedROI + 0.20*ExecutionConfidence`

- Compute uncertainty-aware ROI using scenario bands (bear/base/bull).

- Add explainability payload per recommendation (top positive and negative drivers).

### Phase 3: Marketplace Integration

- Investor dashboard: ranked list + confidence intervals + sensitivity to assumptions.

- Founder dashboard: “best-fit investors” ordered by thesis overlap and probability of commitment.

- Feedback capture on passes/commits to retrain weights.

## Success Metrics

- **Match Precision@10:** ≥ 0.65 (investor engages with 6.5/10 top-ranked opportunities)

- **Time-to-First-Term-Sheet:** -30% vs baseline

- **Qualified Intro Conversion:** +40%

- **Post-Investment IRR Lift:** +10% at cohort level

- **Cold-start Coverage:** ≥ 90% of new plans receive at least 5 viable investor matches

## Risks

- **Biased historical outcomes** → Use counterfactual evaluation and fairness constraints.

- **Overfitting to short-term wins** → Optimize for multi-horizon outcomes (12/24/36 months).

- **Gaming by founders** → Add evidence verification and anomaly detection.

- **Investor strategy drift** → Prompt quarterly thesis re-validation.

## Why This Matters

This proposal shifts fundraising from persuasion-first to evidence-first. It helps credible, high-upside plans get surfaced even when founders are not exceptional marketers, improving capital allocation efficiency for everyone.
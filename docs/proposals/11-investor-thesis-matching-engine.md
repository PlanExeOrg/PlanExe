---
title: Investor Thesis Matching Engine
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

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

## Hypotheses To Validate

We should explicitly test three core hypotheses before scaling. A and B are foundational; C expands the engine beyond conventional startup finance and tests whether the core thesis-matching approach generalizes to large, complex, and often public-interest projects.

### A. Thesis-Fit Improves Deal Quality

**Claim:** A structured thesis profile plus plan feature vector improves match quality versus status-quo discovery.

**What to confirm:**

- Investors engage more with top-ranked opportunities (Precision@10 and click-to-diligence rate increase).
- Founders receive higher-quality intros (higher reply rate and faster scheduling).
- The “why-match” explanation increases investor trust and reduces time-to-no.

### B. Risk-Adjusted ROI Scoring Drives Better Outcomes

**Claim:** Incorporating scenario-based ROI and execution confidence leads to better post-investment performance than thesis-fit alone.

**What to confirm:**

- Matched deals show higher realized IRR or MOIC in historical backtests.
- Rankings remain stable under reasonable perturbations of assumptions.
- Investors accept the model’s uncertainty intervals as decision-relevant.

### C. Cross-Sector Generalization Is Feasible

**Claim:** The matching engine can be extended beyond VC-style deals to infrastructure, public-interest, and climate projects with different financing structures.

**What to confirm:**

- The same vector-based thesis/plan representation can be adapted with domain-specific features.
- The scoring logic can handle non-VC return models (availability payments, blended finance, concession revenues).
- Stakeholder fit and risk allocation can be represented as constraints and preferences.

## Hypothesis Examples At Different Scales

Below are three example project archetypes and the specific hypothesis checks they would drive. These are not full plans, just test cases for validating A/B/C in different settings.

### 1) Expensive Huge Bridge Project Between Two Countries

**Example thesis match:**

- Infrastructure funds targeting long-duration, low-volatility returns.
- Sovereign wealth funds focused on strategic trade corridors.
- Development banks with regional connectivity mandates.

**Key hypothesis checks:**

- **A:** Do investors who prioritize long-term, inflation-linked cashflows engage more with the bridge than generalists?
- **B:** Does scenario modeling (traffic volumes, tariff policy, FX risk) meaningfully change the ranking?
- **C:** Can concession structure, political risk, and cross-border governance be represented as structured features and constraints?

### 2) Famine Prevention In A Poor Country

**Example thesis match:**

- Impact funds targeting humanitarian outcomes with blended finance.
- Philanthropic capital with strict outcome metrics (lives saved, malnutrition reduction).
- Multilateral agencies with food security mandates.

**Key hypothesis checks:**

- **A:** Does explicit outcome alignment (e.g., DALYs reduced, resilience score) improve match quality?
- **B:** Can risk-adjusted ROI be replaced or augmented with cost-effectiveness or outcome ROI?
- **C:** Can non-financial return frameworks be integrated without breaking the ranking model?

### 3) Deforestation Prevention In Brazil

**Example thesis match:**

- Climate funds and corporates seeking verified carbon credits.
- ESG-focused investors with biodiversity preservation targets.
- Government-backed programs with enforcement support.

**Key hypothesis checks:**

- **A:** Do investors with explicit climate/ESG theses show higher engagement than generic funds?
- **B:** Does the model correctly weigh uncertainties (regulatory enforcement, land rights, carbon price volatility)?
- **C:** Can verification and permanence risk be encoded as features that materially affect match ranking?

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

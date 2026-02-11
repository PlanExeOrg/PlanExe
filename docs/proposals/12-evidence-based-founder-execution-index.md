---
title: Evidence-Based Founder Execution Index
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

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

## Why Full Reports Beat Slideware

Polished slides often win because they are easy to parse quickly, not because they are more truthful. When the underlying plan is long, complex, or risk-heavy, a slide deck can hide missing evidence behind narrative and design. The FEI is meant to reverse this by:

- Treating the **entire plan and evidence trail** as the unit of analysis.
- Rewarding **verifiable delivery signals**, not the aesthetic quality of the pitch.
- Surfacing **gaps and contradictions** that slides routinely omit.

In short: as AI can read and evaluate entire reports, the advantage of slide decks (compression) erodes, while the advantage of transparent evidence grows.

## Example Report (PlanExe)

Example of a PlanExe report that an AI can evaluate end-to-end:

- https://planexe.org/20260114_cbc_validation_report.html

This is the kind of artifact the FEI is designed to ingest and audit. If the numbers are fabricated or hallucinated, the FEI should penalize confidence and surface the missing verification.

## Evidence Verification Layer (AI Review)

The FEI should integrate a deep-research audit pass that:

1. **Extracts claims** (market size, unit economics, outcomes, partnerships).
2. **Tags evidence type** (first-party metrics, third-party reports, signed LOIs).
3. **Scores verifiability** (publicly checkable, internal but auditable, anecdotal).
4. **Finds contradictions** (plan vs. data vs. external sources).
5. **Outputs a “verification delta”**: what is missing to reach investor-grade confidence.

This turns an otherwise persuasive plan into a verifiable, investor-friendly dossier.

## What If The Plan Is Broken But Promising?

If the AI audit finds a plan is flawed but salvageable, the FEI should guide corrective changes rather than just rejecting it. Typical adjustments include:

- **Scope reduction** to match capital and team capacity.
- **Milestone refactoring** into evidence-producing steps (pilot, contract, unit test).
- **Unit economics correction** (CAC/LTV mismatch, margins unsupported).
- **Risk reallocation** (regulatory, supplier, or policy risks unassigned).
- **Timeline compression** into staged financing with go/no-go checkpoints.

The output should be: “Here are the minimum changes that make this plan investable for X investor thesis.”

## How Much Evidence Is Enough?

Evidence sufficiency depends on claim size, capital intensity, and reversibility. The FEI should express this as **evidence thresholds**:

- **Tier 1 (Early-stage, low burn):** founder execution signals + pilot results + small cohort traction.  
  Sufficient for seed investors who accept high uncertainty.

- **Tier 2 (Scale-up, moderate burn):** repeatable unit economics, signed LOIs, retention metrics, and third-party references.  
  Required for institutional early growth capital.

- **Tier 3 (Capital-intensive or public interest):** audited financials, regulatory approvals, binding contracts, and verified outcomes.  
  Required for infrastructure funds, development banks, and conservative LPs.

The FEI should be explicit: **what level of evidence is required for which investor type**, and what is still missing.

## FEI Output Additions

Add two visible outputs beyond the execution score:

- **Evidence Coverage Report:** what percentage of key claims are backed by verified evidence.
- **Investability Checklist:** concrete steps needed to meet the minimum threshold for targeted investors.

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

# Confidence-Weighted Funding Auctions

**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `auction`, `price-discovery`, `term-sheet`, `market-design`, `roi`

---

## Pitch

Create a structured funding auction where investors compete on transparent terms informed by model confidence and projected ROI, reducing narrative-driven mispricing.

## TL;DR

- Launch periodic auctions for qualified plans with standardized data rooms.

- Investors submit structured bids (valuation, check size, terms, support).

- Match engine weights bids by confidence-adjusted expected founder + investor outcomes.

- Output ranked term-sheet options with tradeoff explanations.

## Problem

Traditional fundraising often has poor price discovery:

- Terms are negotiated asymmetrically and opaquely.

- Founder storytelling can distort valuation.

- Investors struggle to compare opportunities consistently.

## Proposed Solution

Implement a **Confidence-Weighted Auction Protocol**:

1. Plan enters auction only after minimum evidence quality threshold.

2. Investors submit machine-readable bids.

3. Scoring combines economics, risk, and execution confidence.

4. Founders choose from ranked, explainable options.

## Architecture

```text
┌──────────────────────────────┐
│ Qualified Plan Pool          │
│ - Evidence score gate        │
│ - Standardized data room     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Auction Engine               │
│ - Bid intake API             │
│ - Bid normalization          │
│ - Rule enforcement           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Bid Scoring Service          │
│ - ROI projections            │
│ - Dilution / control impact  │
│ - Confidence weighting       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Term-Sheet Recommendation UI │
│ - Ranked options             │
│ - Tradeoff simulator         │
└──────────────────────────────┘
```

## Implementation

### Phase 1: Auction Data Contract

- Define bid schema:

  - valuation cap/pre-money, check amount, pro-rata rights, board terms, liquidation preference, milestones.

- Validate bids for comparability and legal sanity checks.

### Phase 2: Scoring + Simulation

- Compute total score:

  - `Score = 0.40*FounderOutcome + 0.35*InvestorExpectedROI + 0.25*ExecutionConfidence`

- Run dilution and control simulations across future rounds.

- Include confidence penalties for weak evidence assumptions.

### Phase 3: UX + Governance

- Founder-side: ranked offers with “why this is ranked” explanations.

- Investor-side: lost-bid diagnostics (price too high, terms too restrictive, confidence too low).

- Add anti-collusion monitoring and audit logs.

## Success Metrics

- **Time to Close:** -35% from auction start to signed term sheet.

- **Bid Quality:** % of bids passing quality threshold ≥ 85%.

- **Term Fairness Index:** Lower variance between predicted and realized dilution burden.

- **Post-Deal Performance:** Improved 18-month milestone attainment vs non-auction deals.

## Risks

- **Over-financialization of early-stage nuance** → Preserve optional qualitative memo lane.

- **Strategic bidding behavior** → Use sealed bids and anomaly detection.

- **Legal complexity across jurisdictions** → Region-specific templates and compliance checks.

- **Founder overwhelm** → Provide default recommendations with simple language.

## Why This Matters

Structured auctions create better price discovery and better ROI alignment while reducing dependence on personal charisma and closed-door negotiation dynamics.
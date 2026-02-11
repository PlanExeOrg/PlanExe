---
title: Finance Analysis via Top-Down Estimation
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Finance Analysis via Top-Down Estimation

## Pitch
Provide a fast, defensible financial estimate using market-level benchmarks and macro ratios when bottom-up data is missing. This produces a first-pass budget, revenue, and margin model with explicit confidence bands, enabling early decision-making and investor screening.

## Why
Many plans arrive with limited financial detail. Top-down estimation lets PlanExe:

- Produce a credible early-stage financial model fast.
- Identify whether a plan is even plausible before spending time on bottom-up detail.
- Set guardrails for later bottom-up estimates and reconcile divergences.

## Problem
Without a structured top-down pass:

- Early financials are either missing or invented.
- Investors cannot compare apples-to-apples across plan proposals.
- Budget and revenue claims drift far from industry reality.

## Proposed Solution
Implement a top-down estimation module that:

1. Classifies the plan into a domain and business model archetype.
2. Pulls benchmark ratios (revenue/employee, gross margin ranges, CAC:LTV, capex intensity).
3. Uses macro inputs (TAM/SAM/SOM, price points, addressable volume) to estimate revenue.
4. Produces a multi-year financial model with ranges and confidence levels.
5. Outputs assumptions and evidence sources for auditability.

## Estimation Framework

### 1) Domain and Model Classification
Determine the plan's category and model type:

- Domain: SaaS, consumer apps, logistics, infrastructure, energy, public-sector, etc.
- Model: subscription, transaction, licensing, service-based, PPP/concession.

### 2) Benchmark Ratios
Select ratios from sector data:

- Revenue per employee
- Gross margin ranges
- EBITDA margin ranges
- Sales efficiency (CAC payback, LTV:CAC)
- Capex as % of revenue
- Working capital cycles

### 3) Market Sizing Inputs
Require at least one of:

- TAM/SAM/SOM estimates
- Price x volume assumptions
- Comparable market size and penetration rates

### 4) Revenue Model
Compute revenue using a constrained top-down approach:

- Estimate initial penetration rate (low/medium/high) based on stage.
- Constrain growth rates to sector typical ranges.
- Generate base, conservative, and aggressive scenarios.

### 5) Cost Structure
Apply benchmark ratios to revenue:

- COGS via gross margin range.
- Opex via typical sales/marketing and R&D ratios.
- Capex via sector averages and plan type.

### 6) Output Confidence
Assign a confidence level to each line item based on evidence quality:

- High: external data or audited inputs.
- Medium: comparable company benchmarks.
- Low: assumptions with weak backing.

### 7) Multi-Currency Handling

Plans may involve multiple currencies (e.g., cross-border bridge projects). The top-down model should:

- Specify a reporting currency for the consolidated model.
- Store original currency for localized assumptions.
- Record FX assumptions (rate, date, source, volatility band).
- Allow a third currency when local currencies are unstable.

## Output Schema

```json
{
  "model_type": "subscription",
  "domain": "saas",
  "reporting_currency": "USD",
  "fx_assumptions": [
    {"pair": "DKK/USD", "rate": 0.15, "as_of": "2026-02-10", "volatility": "medium"}
  ],
  "assumptions": [
    "SOM = 0.5% of SAM by year 3",
    "Gross margin range 70-85%"
  ],
  "revenue_scenarios": {
    "conservative": [1.2, 2.0, 3.1],
    "base": [1.8, 3.4, 5.6],
    "aggressive": [2.5, 4.8, 7.9]
  },
  "margin_ranges": {
    "gross": [0.70, 0.85],
    "ebitda": [0.10, 0.25]
  },
  "capex_ratio": 0.08,
  "confidence": {
    "revenue": "medium",
    "costs": "medium",
    "capex": "low"
  }
}
```

## Integration Points

- Use in early PlanExe phases when financial data is missing.
- Feed into risk scoring and investor thesis matching.
- Compare with bottom-up output in reconciliation stage.

## Success Metrics

- Top-down estimate time under 60 seconds for standard plans.
- Percentage of plans with top-down model generated.
- Variance between top-down and bottom-up within acceptable bands.
- Investor feedback: perceived credibility of early-stage financials.

## Risks

- Over-reliance on weak benchmarks: mitigate with confidence labels.
- Domain mismatch: mitigate with explicit classification step.
- False precision: mitigate by publishing ranges, not single-point estimates.

## Future Enhancements

- Automated sourcing of sector benchmarks.
- Dynamic calibration from historical PlanExe outcomes.
- Integrate sensitivity analysis and scenario shock testing.

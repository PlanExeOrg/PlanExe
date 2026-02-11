---
title: Expert Discovery + Fit Scoring for Plan Verification
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Expert Discovery + Fit Scoring for Plan Verification

## Pitch
Automatically identify and rank qualified experts for plan verification using a structured fit scoring model that balances domain expertise, availability, cost, and reputation.

## Why
Verification requires the right experts, but manual discovery is slow and unreliable. Fit scoring streamlines selection while maintaining quality and accountability.

## Problem

- Expert discovery is ad hoc and time-consuming.
- Expertise is not normalized across domains.
- Cost and availability trade-offs are poorly quantified.

## Proposed Solution
Build a system that:

1. Extracts verification requirements from a plan.
2. Queries an expert registry and external sources.
3. Scores experts by fit and ranks the best matches.
4. Produces an explainable recommendation list.

## Fit Scoring Model

### Inputs

- Domain match (primary and secondary expertise)
- Verification experience and prior outcomes
- Availability and turnaround time
- Cost relative to budget constraints
- Reputation score from marketplace

### Example Formula

```
FitScore =
  0.35*DomainMatch +
  0.25*Reputation +
  0.20*Availability +
  0.10*CostFit +
  0.10*OutcomeHistory
```

## Expert Registry Schema

```json
{
  "expert_id": "exp_441",
  "domains": ["energy", "regulation"],
  "credentials": ["PE", "PhD"],
  "availability_days": 7,
  "hourly_rate": 180,
  "reputation_score": 0.86
}
```

## Output Schema

```json
{
  "plan_id": "plan_007",
  "ranked_experts": [
    {"expert_id": "exp_441", "fit_score": 0.89, "reason": "Strong domain match"},
    {"expert_id": "exp_208", "fit_score": 0.81, "reason": "Fast turnaround"}
  ]
}
```

## Integration Points

- Feeds into multi-stage verification workflow.
- Uses reputation scores from expert marketplace.
- Supports governance and conflict-of-interest checks.

## Success Metrics

- Reduced time to match experts.
- Higher verification completion rates.
- Improved investor confidence in verification process.

## Risks

- Incomplete expert data: mitigate with periodic profile verification.
- Cost bias against high-quality experts: allow weighted trade-offs.
- Bias in reputation scoring: normalize by domain and sample size.

## Future Enhancements

- External credential validation integration.
- Automated discovery from publications and patents.
- Adaptive scoring by project complexity.

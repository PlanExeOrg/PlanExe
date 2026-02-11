---
title: ELO-Ranked Bid Selection + Escalation Pipeline
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# ELO-Ranked Bid Selection + Escalation Pipeline

## Pitch
Rank generated bids with an Elo-style system and route the highest-value opportunities to escalation queues, ensuring human attention is focused on the most promising bids.

## Why
When the system produces hundreds of bids per day, manual review cannot keep up. Ranking and escalation allow high-value bids to surface, while low-value bids are deprioritized or discarded.

## Problem

- Excess bids overwhelm decision makers.
- Good bids are lost in noise without ranking.
- Escalation is currently ad hoc and inconsistent.

## Proposed Solution
Implement a pipeline that:

1. Scores bids using an Elo-style ranking based on bid quality metrics.
2. Compares new bids against a rolling set of prior bids.
3. Escalates top-ranked bids to human review.
4. Auto-rejects bids that fail minimum thresholds.

## Ranking Model

### Input Metrics

- Bid completeness
- Evidence strength
- Risk-adjusted ROI estimate
- Feasibility score
- Strategic fit

### Elo Update Logic

- Each bid is compared to a peer set.
- Winners gain Elo points, losers lose points.
- Rankings update continuously as new bids arrive.

## Escalation Rules

- Top 5% of bids auto-escalated.
- Bids above a fixed Elo threshold are escalated.
- High-risk bids require mandatory review.

## Output Schema

```json
{
  "bid_id": "bid_902",
  "elo_score": 1580,
  "status": "escalated",
  "reason": "Top 5% and high ROI"
}
```

## Integration Points

- Connected to bid factory orchestration.
- Feeds into governance and risk checks.
- Links to investor matching and dispatch.

## Success Metrics

- % of escalated bids that convert to funded projects.
- Reduction in time spent reviewing low-quality bids.
- Stability of rankings over time.

## Risks

- Elo scores could be gamed by noisy inputs.
- Over-reliance on ranking may miss niche opportunities.
- Escalation thresholds may be miscalibrated.

## Future Enhancements

- Dynamic K-factor based on bid confidence.
- Hybrid ranking with rule-based overrides.
- Domain-specific Elo pools.

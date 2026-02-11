---
title: Multi-Angle Topic Verification Engine Before Bidding
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Multi-Angle Topic Verification Engine Before Bidding

## Pitch
Verify high-stakes plans by checking critical topics through multiple independent angles, reducing blind spots and preventing expensive false positives.

## Why
A single verification pass can miss key weaknesses. Multi-angle verification forces a plan to survive different lenses: technical feasibility, regulatory risk, market demand, and operational constraints.

## Problem

- Verification is often single-threaded and narrow.
- High-stakes bids fail because one critical dimension was overlooked.
- Stakeholders lack confidence in verification depth.

## Proposed Solution
Create a verification engine that:

1. Extracts critical topics from the plan.
2. Assigns each topic to multiple verification lenses.
3. Produces a consolidated confidence score per topic.
4. Flags contradictions and gaps.

## Verification Lenses

Each plan should be evaluated against:

- **Technical feasibility:** can it be built with current tech?
- **Regulatory compliance:** are approvals feasible within timeline?
- **Market or demand validity:** will buyers exist at the proposed price?
- **Operational execution:** can the organization deliver at scale?
- **Financial sustainability:** do cash flows support the plan?

## Topic Extraction

Identify high-risk topics such as:

- Critical assumptions (unit economics, demand elasticity).
- Dependencies (suppliers, government approvals).
- Non-reversible decisions (capex lock-in).

## Output Schema

```json
{
  "topic": "regulatory approval",
  "lenses": {
    "regulatory": "low",
    "operational": "medium",
    "financial": "medium"
  },
  "overall_confidence": "low",
  "notes": ["Permitting timeline exceeds proposal"]
}
```

## Integration Points

- Works with the multi-stage verification workflow.
- Feeds into investor matching and bid escalation.
- Provides red flags for governance checks.

## Success Metrics

- Reduction in post-bid failure causes.
- Increased confidence scores among investors.
- Improved detection of hidden risks.

## Risks

- Overhead in verification time: mitigate by prioritizing high-risk topics.
- Conflicting lens outputs: resolve with expert adjudication.
- Sparse data: provide confidence intervals.

## Future Enhancements

- Automated lens weighting by domain.
- Learning system to adjust lens priority based on outcome data.
- Integration with expert reputation scoring.

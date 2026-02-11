---
title: Multi-Stage Expert Verification Workflow
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Multi-Stage Expert Verification Workflow

## Pitch
Create a structured, multi-stage verification workflow that validates plan claims using domain experts and evidence gates, producing a verified, investor-grade plan with explicit confidence ratings.

## Why
Investors and decision-makers need more than persuasive narratives. They need verified claims, clear evidence coverage, and risk transparency. A staged workflow allows fast rejection of weak plans and deep validation of strong candidates without wasting expert time.

## Problem
Today, verification is ad hoc:

- Some plans are reviewed deeply, others barely.
- Evidence quality is not standardized.
- Experts are not sequenced efficiently, wasting time on poor candidates.

## Proposed Solution
Implement a pipeline with escalating verification depth:

1. Automated evidence extraction and claim scoring.
2. Lightweight expert screening on critical claims.
3. Deep domain verification for shortlisted plans.
4. Final synthesis into a verified plan report.

## Workflow Stages

### Stage 0: Intake and Claim Extraction

- Parse plan text into discrete claims (market size, unit economics, regulatory feasibility, technical feasibility).
- Tag claims by domain and risk class.
- Produce a claim map and evidence requirements.

### Stage 1: Automated Evidence Check

- Validate claims against known databases and public sources.
- Flag contradictions or unsupported assumptions.
- Assign initial confidence scores.

**Output:** Evidence coverage report and critical risk flags.

### Stage 2: Expert Screening

- Route high-risk claims to appropriate experts.
- Experts validate plausibility and point out weak assumptions.
- Filter out non-viable plans early.

**Output:** Screened plan with go/no-go recommendation.

### Stage 3: Deep Verification

- Full verification of remaining claims.
- Require primary evidence: signed LOIs, audits, regulatory approvals.
- Validate technical feasibility with domain-specific expertise.

**Output:** Verified plan with confidence scores by claim category.

### Stage 4: Final Synthesis

- Produce an investor-ready verification summary.
- Provide recommendations and required fixes.
- Generate a final verification grade.

## Evidence Standards

Evidence should be graded by strength:

- **Level 1:** Anecdotal or unverified claims.
- **Level 2:** Third-party reports or benchmarks.
- **Level 3:** Audited financials, signed contracts, regulatory approvals.

Each claim in the plan should reference an evidence level.

## Output Schema

```json
{
  "verification_grade": "B+",
  "critical_flags": ["Regulatory approval uncertain"],
  "evidence_coverage": 0.72,
  "claim_confidence": {
    "market_size": "medium",
    "unit_economics": "low",
    "technical_feasibility": "high"
  },
  "required_fixes": [
    "Provide updated unit economics from pilot",
    "Secure preliminary regulatory consultation"
  ]
}
```

## Integration Points

- Links directly to FEI scoring (execution credibility).
- Feeds into investor matching (confidence-weighted ranking).
- Provides gating before plan promotion to marketplace.

## Success Metrics

- % plans passing Stage 2 and Stage 3.
- Reduction in false-positive investor matches.
- Time saved per expert review cycle.
- Investor satisfaction with verification reports.

## Risks

- Expert availability bottlenecks: mitigate with staged filtering.
- Over-reliance on automation: keep human override.
- Inconsistent evidence quality across sectors: normalize by domain.

## Future Enhancements

- Reputation scoring for experts.
- Automated dispute resolution for conflicting expert opinions.
- Continuous verification updates as plans evolve.

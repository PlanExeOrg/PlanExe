---
title: Verification Incentives, Governance, and Liability Model
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Verification Incentives, Governance, and Liability Model

## Pitch
Establish a governance framework that aligns incentives for truthful verification, assigns liability for errors, and defines transparent accountability across experts, platforms, and plan owners.

## Why
Verification only works if experts are motivated to be accurate, conflicts of interest are managed, and accountability is clear. Without governance, verification risks becoming performative, biased, or legally fragile.

## Problem

- Experts lack standardized incentives for accuracy.
- Liability for incorrect verification is undefined.
- Conflicts of interest and bias are not systematically managed.

## Proposed Solution
Create a governance and incentive framework that includes:

1. Incentive structures tied to long-term accuracy.
2. Liability rules for negligent or fraudulent verification.
3. Transparent audit trails for verification decisions.
4. A dispute resolution and appeals process.

## Incentive Model

Align incentives with truthfulness:

- **Base fee:** paid for verification work regardless of outcome.
- **Accuracy bonus:** paid when verified claims are later confirmed.
- **Penalty:** applied for negligent or consistently inaccurate verification.

**Example incentive split:**

- 60% base fee
- 30% accuracy bonus
- 10% at risk (released after outcome validation)

## Governance Structure

- **Verification Policy Board:** defines standards and acceptable evidence.
- **Audit Committee:** samples verification decisions for consistency.
- **Dispute Panel:** handles disagreements and appeals.

## Liability Rules

Define responsibility tiers:

- **Expert liability:** negligence, conflicts not disclosed, fabricated evidence.
- **Platform liability:** failure to enforce standards or audit processes.
- **Plan owner liability:** false inputs or withheld data.

Liability should be proportional and documented in terms of service.

## Evidence Standards and Audits

- Require evidence-level tagging for each claim.
- Publish audit trails and verification notes.
- Randomly audit high-impact plans.

## Dispute Resolution Process

1. Triggered by contradictions or stakeholder complaints.
2. Independent review by separate experts.
3. Resolution outcomes: uphold, revise, or revoke verification.

## Output Schema

```json
{
  "verification_id": "ver_981",
  "expert_id": "exp_123",
  "evidence_level": "Level 3",
  "audit_status": "pass",
  "liability_notes": ["No conflicts disclosed"]
}
```

## Integration Points

- Tied to expert marketplace reputation scoring.
- Used by verification workflow stages to enforce gating.
- Informs legal and compliance policies.

## Success Metrics

- Reduced rate of verified-claim reversals.
- Increased investor confidence in verification outputs.
- Faster resolution of disputes.

## Risks

- Legal complexity across jurisdictions.
- Overly harsh penalties discourage participation.
- Governance overhead slows verification cycles.

## Future Enhancements

- Insurance-backed verification guarantees.
- Automated conflict-of-interest detection.
- Cross-platform verification standards consortium.

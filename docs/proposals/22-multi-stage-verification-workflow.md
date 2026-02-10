# Multi-Stage Expert Verification Workflow

## Pitch
Introduce a structured verification workflow that separates quick plausibility checks from deep technical/legal reviews, making expert collaboration predictable and scalable.

## Problem
Verification is currently ad hoc. Users need a clear path from "draft plan" to "verified plan" with stage gates.

## Proposal
Define verification stages:

1. **Stage A: Triage Review (fast)** — identify critical flaws and missing evidence.

2. **Stage B: Domain Review (deep)** — engineering/legal/environmental/financial domain checks.

3. **Stage C: Integration Review** — reconcile cross-domain conflicts.

4. **Stage D: Final Verification Report** — signed conclusions + conditions.

## Workflow artifacts

- Verification checklist per domain

- Issue register with severity and owners

- Evidence pack (documents, assumptions, calculations)

- Final signed verification summary

## `run_plan_pipeline.py` integration

- Add optional `verification_mode=true`

- Pipeline emits domain-specific review packets

- Plan status transitions: `draft -> in_review -> conditionally_verified -> verified`

## Success metrics

- Verification cycle time

- Critical issue catch rate before implementation

- Rework reduction after verification

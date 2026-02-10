---
title: Multi-Angle Topic Verification Engine Before Bidding
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Multi-Angle Topic Verification Engine Before Bidding

## Pitch
Before the organization allocates resources to bids, verify each topic from multiple angles (technical, legal, financial, geopolitical, reputational) to reduce false positives and costly misfires.

## Why
Single-source news is noisy and often incomplete. Bidding on weak or misclassified opportunities burns compute, expert capacity, and reputation.

## Proposal
For each detected opportunity, run a verification bundle:

1. Source triangulation (minimum 3 independent sources)

2. Contradiction detection across sources

3. Domain-specific plausibility checks

4. Regulatory feasibility checks

5. Counterparty legitimacy signals

## Verification dimensions

- **Technical**: Is scope physically/logically feasible?

- **Legal/regulatory**: Is tender structure valid and compliant?

- **Financial**: Is budget/deadline credible?

- **Political/geopolitical**: Is project likely to be paused/blocked?

- **Reputation**: Is this a scam/PR trap/high controversy event?

## Decision classes

- `verified_strong`

- `verified_with_risks`

- `insufficient_evidence`

- `do_not_pursue`

## Integration point

- Add a pre-plan gate: opportunities must pass verification threshold before entering high-volume planning queue.

## Success metrics

- False-positive reduction rate

- % bid opportunities that pass post-hoc reality checks

- Reduction in wasted plan generations on invalid topics

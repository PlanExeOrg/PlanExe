# ELO-Ranked Bid Selection + Escalation Pipeline

## Pitch
Use `07-elo-ranking.md` as the core selection engine: rank thousands of generated plans, shortlist the strongest, and escalate only top candidates into expensive expert-grade refinement.

## Why
If 1000 plans are generated daily, selection quality is the economic center. ELO ranking provides a robust comparative filter before deeper spend.

## Proposal
Implement a 3-tier funnel:

1. **Tier 1**: bulk generation + lightweight scoring

2. **Tier 2**: ELO pairwise ranking and percentile assignment

3. **Tier 3**: top percentile receives expert verification + final bid packaging

## Suggested cutoffs

- Keep top 20% after first pass

- Keep top 5% after ELO cross-domain ranking

- Send top 1% to full bid escalation

## Selection features

- ELO score + confidence band

- Domain fitness to opportunity

- Verification status from Proposal 27

- Resource feasibility for timely delivery

## Feedback loop

- Track actual outcomes (win/loss, shortlist/not shortlisted)

- Feed outcomes back into ranking calibration

- Penalize plans that score high but underperform in real bids

## Success metrics

- Precision of top-ranked plans

- Win-rate lift from ELO-based shortlisting

- Cost savings from reduced deep-review volume

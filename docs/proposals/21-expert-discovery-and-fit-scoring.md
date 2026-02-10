# Expert Discovery + Fit Scoring for Plan Verification

## Pitch
Add an Expert Discovery layer that finds and ranks domain experts for a given plan (e.g., cross-border bridge projects), so users can move from draft plans to verified, execution-ready plans.

## Problem
Users can generate high-quality plans, but often cannot identify the right experts to validate assumptions, engineering safety, legal constraints, and execution feasibility.

## Proposal
Build a pipeline that:

1. Extracts verification domains from plan artifacts.

2. Searches expert sources (licenses, publications, associations, project portfolios).

3. Computes a fit score per expert.

4. Produces a shortlist with reasoning and risk flags.

## Core fit model
`fit_score = 0.35*domain_match + 0.20*project_similarity + 0.15*regional_relevance + 0.15*credential_strength + 0.15*availability_signal`

## Data sources (initial)

- Professional associations and certification registries

- Public procurement/project databases

- Publications/patents and conference profiles

- Verified platform profiles (LinkedIn-like signals as optional)

## Output for user

- Top 5–20 experts by category (engineering, legal, environmental, finance)

- Why each expert is relevant

- Gaps in verification coverage

## Success metrics

- Time from plan creation to first expert outreach

- Expert acceptance rate

- % plans with complete verification coverage

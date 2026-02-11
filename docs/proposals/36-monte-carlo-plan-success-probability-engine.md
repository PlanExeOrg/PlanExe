# 36) Monte Carlo Plan Success Probability Engine (10,000 Runs)

## Pitch
Add a Monte Carlo simulation layer that runs 10,000 stochastic scenarios per plan to estimate probability of success/failure, budget overrun risk, and schedule slippage.

## Why
Single-point estimates hide uncertainty. Decision-makers need distribution-level answers, not only one "expected" outcome.

## Proposal
For each plan, define uncertain variables:
- task durations
- cost drivers
- dependency delay probabilities
- funding variability
- regulatory delay risk

Run 10,000 simulations and output:
- probability of on-time delivery
- probability of budget overrun
- probability of project failure criteria being triggered
- P10/P50/P90 outcomes

## Model approach
- Duration: triangular/lognormal per task
- Cost: lognormal/PERT per cost bucket
- Risk events: Bernoulli with impact distributions

## Outputs
- Success/failure probability dashboard
- Tornado chart of top uncertainty drivers
- Risk-adjusted recommendation (go/no-go/re-scope)

## Success metrics
- Calibration against historical project outcomes
- Reduction in high-confidence but wrong forecasts

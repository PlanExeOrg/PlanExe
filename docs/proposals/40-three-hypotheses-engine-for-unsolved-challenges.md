# 40) Three-Hypotheses Engine for Unsolved Challenges

## Pitch
When the system finds an unsolved challenge, require generation of exactly three plausible hypotheses to approach a solution, then rank them by evidence and risk.

## Why
Plans stall when teams identify hard problems but do not structure solution exploration.

## Proposal
For each unresolved challenge:
1. Produce 3 hypotheses (H1/H2/H3)
2. Define test protocol for each hypothesis
3. Estimate cost/time/risk profile per hypothesis
4. Recommend portfolio strategy (single-track vs parallel trials)

## Example (cold-climate concrete)
- H1: admixture chemistry adaptation for low-temp hydration kinetics
- H2: modular heated formwork + controlled curing micro-environments
- H3: alternative material systems with reduced hydration sensitivity

## Required outputs
- hypothesis cards (assumptions, required experiments, failure criteria)
- stage-gate plan for kill/continue decisions
- expected value of information (EVI) by hypothesis

## Integration with Monte Carlo
- Feed hypothesis success probabilities into simulation distributions
- Recompute plan-level success probability after each experiment cycle

## Success metrics
- Time to first validated path for frontier challenges
- Reduction in dead-end R&D spend
- Improved confidence bounds after hypothesis testing

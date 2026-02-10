# 34) Finance Analysis via Top-Down Estimation

## Pitch
Add a finance module that produces **top-down** cost and revenue estimates using analogies, benchmarks, and high-level ratios before detailed costing.

## Why
Top-down estimates give fast sanity checks and help catch unrealistic budgets early.

## Proposal
### 1) Benchmark library
- Maintain benchmarks by domain (bridge, IT infra, SaaS, nonprofit program, etc.)
- Include cost drivers and ranges (P10/P50/P90)

### 2) Estimate methods
- Parametric models (e.g., cost per km, cost per user, cost per server)
- Comparable-project analogies
- Ratio-based checks (overhead %, contingency %, engineering %)

### 3) Outputs
- Top-down estimate range + confidence
- Driver explanation and benchmark citations
- Flags when plan assumptions fall outside typical ranges

## Success metrics
- Reduction in plans with obviously implausible budgets
- Agreement with bottom-up within tolerance band (see Proposal 35)

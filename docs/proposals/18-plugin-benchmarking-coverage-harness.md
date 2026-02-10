# 18) Plugin Benchmarking Harness Across Diverse Plan Types

## Pitch
Create a benchmark harness that continuously measures plugin quality across a broad matrix of plan domains, complexity levels, and risk profiles—so plugin quality is evidence-based, not anecdotal.

## Problem
Without systematic benchmarking, Plugin Hub quality will drift and overfit to popular plan categories.

## Proposal
Build a benchmark framework with:
- **Dataset matrix**: business, software, nonprofit, policy, industrial, scientific
- **Scenario tiers**: simple, medium, complex, adversarial
- **Golden outputs** and contract checks per scenario
- **Regression suite** run on each plugin publish

## Core scoring dimensions
1. Contract adherence (schema + invariants)
2. Correctness against golden cases
3. Robustness under noisy inputs
4. Latency and cost
5. Generalization across domains

## Execution design
- `benchmark_runner.py` executes plugin against scenario bundle
- Stores metrics in `plugin_benchmarks`
- Produces leaderboard and confidence bands

## Suggested score formula
`overall = 0.35*correctness + 0.20*robustness + 0.20*contract + 0.15*generalization + 0.10*latency_cost`

## `run_plan_pipeline.py` policy hook
- Production selection should prefer plugins above minimum benchmark grade (e.g., B+)
- If all candidates fail threshold, fall back to synthesis + quarantine mode

## Data model additions
- `plugin_benchmark_runs` (plugin_id, suite_id, score_json, created_at)
- `benchmark_scenarios` (suite_id, domain, difficulty, expected_contract)
- `plugin_quality_grade` (plugin_id, grade, confidence, last_eval_at)

## Success metrics
- Benchmark coverage % across supported plan categories
- Failed-in-prod rate vs benchmark grade
- Time to detect regressions after plugin updates

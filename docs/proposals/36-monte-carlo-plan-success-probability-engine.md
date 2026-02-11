---
title: Monte Carlo Plan Success Engine: Technical Documentation
date: 2026-02-11
status: proposal
author: PlanExe Team
---

# Monte Carlo Plan Success Probability Engine

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** Data Scientists, Project Managers  

---

## Overview
The **Monte Carlo Plan Success Engine** moves planning from deterministic "best guesses" to probabilistic distributions. By running 10,000 stochastic simulations of the project schedule and budget, it provides a mathematical confidence level for success (e.g., "There is a 12% chance of finishing by Q3").

It replaces single-point estimates (e.g., "This task takes 5 days") with probability distributions (e.g., "Between 3 and 10 days, most likely 5").

## Core Problem
"The Flaw of Averages": The average outcome of a project is *worse* than the plan constructed from average inputs, due to non-linear dependencies (Jensen's Inequality). Deterministic plans systematically underestimate risk.

## System Architecture

### 1. Distribution Modeler
Converts scalar plan inputs into statistical distributions.
-   **Tasks:** Beta-PERT distribution ($O + 4M + P / 6$) for duration.
-   **Costs:** Lognormal distribution (skewed right) for budget items.
-   **Risks:** Bernoulli trials ($p$) with impact distributions ($I$).

### 2. Simulation Loop (The Engine)
Runs $N$ iterations (default 10,000).
For each iteration $i$:
1.  Sample duration $d_{i,t}$ for every task $t$.
2.  Sample cost $c_{i,b}$ for every budget line $b$.
3.  Trigger risks based on probability.
4.  Recompute Critical Path ($CP_i$) and Total Cost ($TC_i$).
5.  Store pair $(CP_i, TC_i)$.

### 3. Analytics Service
Aggregates the simulation results into interpretable metrics (P10, P50, P90).

---

## Technical Specifications

### Input Distributions

Most task durations follow a **Beta-PERT** distribution:
$$E = \frac{Optimistic + 4 \cdot MostLikely + Pessimistic}{6}$$
$$Var = \left(\frac{Pessimistic - Optimistic}{6}\right)^2$$

Cost inputs often follow a **Lognormal** distribution:
$$f(x) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$$
This reflects the reality that costs can explode (10x overrun) but rarely shrink below a floor.

### The Simulation Algorithm

```python
def run_simulation(plan, n_iterations=10000):
    results = []
    for _ in range(n_iterations):
        scenario = {}
        # 1. Sample Tasks
        for task in plan.tasks:
            duration = sample_pert(task.min, task.mode, task.max)
            scenario[task.id] = duration
            
        # 2. Trigger Risks
        for risk in plan.risks:
            if random.random() < risk.probability:
                # Add delay to linked tasks
                for target in risk.impacts:
                     scenario[target] += risk.delay_days
                     
        # 3. Solve Critical Path
        finish_date = solve_cpm(plan.dependencies, scenario)
        
        results.append(finish_date)
        
    return aggregate_stats(results)
```

---

## Output Analysis (Sensitivity)

### Tornado Charts
We calculate the *correlation coefficient* between each task's duration and the total project duration.
-   **High Correlation:** This task is a "Driver". Focusing on it reduces overall variance.
-   **Low Correlation:** This task has "Slack". Delays here likely won't hurt the project.

### Probabilistic S-Curves
A plot of $Cumulative Probability$ vs $Time/Cost$.
-   **P50 (Median):** The "Coin Flip" outcome.
-   **P80 (Commitment):** The recommended internal deadline.
-   **P95 (Safe Bet):** The recommended external promise.

---

## Integration

### `Assumption Drift Monitor`
When drift is detected (e.g., "Steel prices up 10%"), the Monte Carlo engine automatically re-simulates using the new *actuals* as the baseline, updating the P-values in real-time.

### API Reference

**`POST /api/simulate/plan/{id}`**
Trigger a full 10k run.
```json
{
  "iterations": 10000,
  "seed": 42
}
```

**`GET /api/simulate/results/{id}`**
Get the distribution data.
```json
{
  "p10_date": "2026-06-01",
  "p50_date": "2026-07-15",
  "p90_date": "2026-09-01",
  "success_prob_deadline": 0.42 # 42% chance to hit deadline
}
```

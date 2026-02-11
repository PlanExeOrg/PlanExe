---
title: Counterfactual Scenario Explorer: Technical Documentation
date: 2026-02-11
status: proposal
author: PlanExe Team
---

# Counterfactual Scenario Explorer

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** Architects, Data Scientists  

---

## Overview
The **Counterfactual Scenario Explorer** allows stakeholders to test the resilience of a plan by simulating "What If?" scenarios. Instead of a single linear roadmap, it treats the plan as a probabilistic graph that can be "stressed" by changing key input parameters.

It uses a Monte Carlo simulation engine to spawn thousands of "parallel universe" outcomes, helping decision makers understand the range of possible futures.

## Core Problem
Standard plans suffer from "Planning Fallacy"—the tendency to underestimate time, costs, and risks. A static gantt chart implies certainty where none exists.

## System Architecture

### 1. The World Generator
The heart of the system. It takes the "Base Plan" and tweaks input variables based on statistical distributions.
-   **Inputs:** Plan Tasks, Durations, Costs, Risk Registers.
-   **Variables:** "Inflation Rate", "Supplier Delay", "Weather Impact".
-   **Distributions:** Normal, Log-Normal, Beta (PERT).

### 2. Simulation Engine
For each scenario:
1.  **Perturb:** Apply random noise to task durations/costs based on the scenario type.
2.  **Propagate:** Recalculate the critical path and total budget.
3.  **Detect Failure:** Check if any "hard constraints" (e.g., launch date) are violated.
4.  **Log Outcome:** Record success/failure, final cost, final duration.

### 3. Resilience Scorer
Aggregates the results of N simulations (typically 10,000) into a single "Resilience Score".

---

## Scenario Types

| Scenario | Description | Simulation Logic |
| :--- | :--- | :--- |
| **Optimistic** | "Blue Sky" | Skew distributions to P10 (Best Case). Remove 50% of risks. |
| **Pessimistic** | "Murphy's Law" | Skew distributions to P90 (Worst Case). Trigger 80% of risks. |
| **Black Swan** | "Total Chaos" | Introduce 1-2 "Catastrophic" events (e.g., factory fire, regulation ban). |
| **Inflationary** | "Cost Shock" | Increase all material/labor costs by 20-50%. Keep schedule same. |
| **Delay Spiral** | "Gridlock" | Increase all task durations by 20-50%. Keep costs same. |

---

## Resilience Scoring Formula

The `ResilienceScore` (0-100) measures how robust the plan is across all scenarios.

$$Score = (0.4 \times P_{success}) + (0.3 \times (1 - \frac{Cost_{P90}}{Budget})) + (0.3 \times (1 - \frac{Time_{P90}}{Deadline}))$$

Where:
-   $P_{success}$: Probability of meeting *minimum* success criteria.
-   $Cost_{P90}$: The 90th percentile cost outcome.
-   $Time_{P90}$: The 90th percentile duration outcome.

---

## Output Schema (JSON)

The result of a full simulation run:

```json
{
  "plan_id": "plan_123",
  "scenarios_run": 10000,
  "resilience_score": 72, 
  "baseline": {
    "cost": 1000000,
    "duration_days": 180
  },
  "p90_outcome": {
    "cost": 1450000,
    "duration_days": 210
  },
  "key_drivers": [
    {
      "task_id": "task_45",
      "name": "Wait for Permit",
      "sensitivity": 0.85, # 85% correlation with project delay
      "suggestion": "Parallelize this task"
    }
  ]
}
```

---

## User Interface: "The Matrix View"

A 2x2 grid visualizing the trade-offs.

-   **X-Axis:** Cost (Budget vs Overrun)
-   **Y-Axis:** Time (Early vs Late)
-   **Scatter Plot:** Each dot is one simulation outcome.
-   **Heatmap:** Colored zones showing "Safe", "Risky", and "Failed".

The user can iterate by adjusting plan parameters (e.g., "Add 2 more engineers") and re-running the simulation to see if the dots move into the "Safe" zone.

## Future Enhancements
1.  **AI Recommendations:** "If you split Task B into two parallel tasks, your P90 duration drops by 15 days."
2.  **Historical Training:** Calibrate distributions based on actual past project data (e.g., "Software projects usually slip 30%, not 10%").

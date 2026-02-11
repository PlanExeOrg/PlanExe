---
title: "Risk Propagation Network + Failure Mode Manifestation"
date: 2026-02-10
status: proposal
author: PlanExe Team
---

# Risk Propagation Network + Failure Mode Manifestation

**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `risk`, `propagation`, `failure-modes`, `simulation`, `dependencies`

## Pitch
Model how local risks propagate through dependencies to system-level failure, then simulate manifestation paths across many runs to surface the most likely cascades and highest-leverage interventions.

## Why
Risks rarely fail in isolation. Large project failures typically emerge from **interacting risks** across domains (technical, procurement, financing, regulatory). A propagation model makes these interactions explicit and actionable.

## Problem

- Risk registers treat items independently.
- Teams under-estimate compounding effects.
- Mitigation choices are not ranked by systemic impact.

## Proposed Solution
Build a **Risk Propagation Network** that:

1. Represents risks, tasks, and milestones as a connected graph.
2. Encodes causal links and delay effects between nodes.
3. Simulates cascades across the network.
4. Outputs failure pathways and intervention leverage scores.

## Architecture

```text
Plan JSON
  -> Risk + Dependency Extraction
  -> Propagation Graph Builder
  -> Cascade Simulator (Monte Carlo)
  -> Failure Path Analyzer
  -> Mitigation Prioritizer
```

## Graph Model

- **Nodes:** risks, tasks, milestones, resources.
- **Edges:** causal amplification and delay links.
- **Weights:** probability impact, lag time, and severity multiplier.

### Example edge

- Procurement delay -> schedule slippage (weight: high, lag: 2 weeks)
- Schedule slippage -> financing drawdown risk (weight: medium, lag: 1 month)

## Simulation

Run multi-step simulations to reveal cascades:

- Sample risk events based on probability distributions.
- Propagate effects through graph edges.
- Track which nodes fail, when, and how often.

**Outputs per run:**

- failure sequence
- time-to-failure
- cost impact

## Output Schema

```json
{
  "top_failure_paths": [
    {
      "path": ["procurement_delay", "schedule_slip", "financing_gap"],
      "probability": 0.18,
      "expected_loss": 4200000
    }
  ],
  "intervention_points": [
    {"node": "procurement_delay", "leverage": 0.72}
  ]
}
```

## Integration Points

- Feeds into Monte Carlo plan success probability engine.
- Adds a propagation-adjusted risk score to plan ranking.
- Triggers mitigation playbooks for top cascades.

## Success Metrics

- Reduction in surprise compound failures.
- Increased mitigation effectiveness vs baseline risk registers.
- Improved forecast accuracy for delays and cost overruns.

## Risks

- Model complexity could obscure interpretation.
- Missing edges lead to false security.
- Overfitting to historical cascades.

## Future Enhancements

- Automated edge discovery from historical plans.
- Dynamic updates as execution data arrives.
- Cross-project risk propagation benchmarking.

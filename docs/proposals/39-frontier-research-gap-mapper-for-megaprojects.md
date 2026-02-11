---
title: "Frontier Research Gap Mapper for Mega-Projects"
date: 2026-02-10
status: proposal
author: PlanExe Team
---

# Frontier Research Gap Mapper for Mega-Projects

**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `research`, `frontier`, `megaprojects`, `feasibility`, `innovation`

## Pitch
Detect where a plan depends on unresolved science or engineering and map those dependencies into a structured R&D register before committing to a bid.

## Why
Some plans require breakthroughs, not just execution discipline. Hidden research dependencies are major bid risks that should be explicit, costed, and staged.

## Problem

- Frontier gaps are often implicit, not stated.
- Feasibility assessments assume mature technology.
- Bids proceed before critical R&D constraints are understood.

## Proposed Solution
Create a module that:

1. Identifies plan components that exceed current state-of-practice.
2. Classifies maturity level and research gaps.
3. Produces a research dependency register with timelines and uncertainty.
4. Adjusts bidability and feasibility scores accordingly.

## Classification Framework

Each component is tagged as:

- **Mature:** proven in real-world deployments.
- **Adaptation Required:** proven elsewhere but needs modification.
- **Frontier:** unproven at required scale or conditions.

## Research Dependency Register

Each gap includes:

- Challenge statement
- Current state-of-practice
- Missing capability threshold
- Estimated R&D timeline
- Cost uncertainty band

## Example Challenge Classes (Arctic Bridge)

- ultra-cold concrete curing and durability
- ice-load resistant structural systems
- remote logistics and year-round constructability
- cross-border governance and standards harmonization

## Output Schema

```json
{
  "component": "ice_load_resilience",
  "maturity": "frontier",
  "gap": "No validated structural system for multi-year ice pack",
  "estimated_rnd_years": 3,
  "cost_uncertainty": "high"
}
```

## Integration Points

- Feeds into risk propagation network and Monte Carlo success probability.
- Applies bidability penalty for unresolved frontier gaps.
- Triggers pre-bid R&D phase recommendations.

## Success Metrics

- Fewer bids on technically premature opportunities.
- Better planning of R&D-first project phases.
- Reduced execution failure from unknown technical gaps.

## Risks

- Over-classification of challenges as frontier.
- Incomplete research signals due to limited sources.
- R&D timelines difficult to estimate accurately.

## Future Enhancements

- Automated scanning of research literature and patents.
- Expert panels for frontier assessment.
- Continuous updates as research advances.

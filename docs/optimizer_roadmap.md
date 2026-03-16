# Optimizer Roadmap

This document proposes a sequenced roadmap for PlanExe optimizer work, based
on observed failure patterns across local models tested in March 2026.

## Background

`OPTIMIZE_INSTRUCTIONS` blocks have been added to 9 high-risk tasks (#298,
#300–#307). These document known failure modes and guide the model toward better
outputs. The optimizer can now use these blocks as the basis for structured
improvement, but the evaluation metric and tooling need to catch up.

---

## Phase 1 — Fix the evaluation metric (prerequisite)

**Why first:** The iteration 17 regression (external review score dropped from
6.5 → 5.8) occurred because the optimizer was maximizing structural compliance,
not content quality. Fixing the metric before running more optimizer iterations
prevents optimizing toward the wrong target.

**Proposed scorer additions:**
- Penalize fabricated citations (invented case law, invented statistics)
- Penalize template language ("This plan is not X; it is Y waiting to happen")
- Penalize generic advice not tied to specific plan elements
- Reward expert critiques where each action cites a named plan element
- Reward self-audits that identify at least 2 genuine internal tensions
- Reward assumption sets that establish a concrete budget when the user omitted one

The `OPTIMIZE_INSTRUCTIONS` blocks now merged are a natural source for rubric
criteria — each "known problem" maps to a negative signal.

---

## Phase 2 — ExpertFinder domain specificity

**Why high priority:** ExpertCriticism completeness is the biggest observable
quality gap (typically 3/8 experts produce substantive output). But the root
cause is upstream: if ExpertFinder selects generic roles ("Project Manager",
"Business Analyst"), domain-specific critique is impossible regardless of how
well ExpertCriticism performs.

**Target metric:** Fraction of selected experts with a domain directly
traceable to a specific element of the plan (named technology, geography,
regulation, animal species, etc.).

**Related PR:** #308 (expert_finder.py OPTIMIZE_INSTRUCTIONS)

---

## Phase 3 — ExpertCriticism completeness

**Why after Phase 2:** Expert completeness is partly a selection problem
(Phase 2) and partly a context pressure problem. Once selection improves,
the optimizer can focus on increasing the fraction of experts producing
non-empty critiques.

**Target metric:** Fraction of experts (out of total selected) producing
critiques with all required fields populated (primary_actions,
secondary_actions, issues with complete consequence + root_cause).

**Baseline:** ~3/8 experts complete in current runs. Target: 6/8+.

---

## Phase 4 — PremiseAttack orthogonality

**Why:** Models consistently repeat the same argument across multiple lenses
(e.g., Fourth Amendment violations appearing in 4 of 5 lenses for a
surveillance plan). Redundant lenses waste context and reduce coverage.

**Target metric:** Cross-lens overlap score — semantic similarity between
each pair of lens verdicts. Lower overlap = better orthogonality.

**Approach:** Add an orthogonality penalty to the evaluation metric, then
let the optimizer find prompt variants that reduce cross-lens redundancy.

---

## Phase 5 — Governance phase shared instructions

**Why:** Governance phases 1–6 share the same failure modes (accumulating
context, truncation, phase-context mismatch) but currently have no
`OPTIMIZE_INSTRUCTIONS`. A shared `GOVERNANCE_OPTIMIZE_INSTRUCTIONS` imported
by all 6 phases avoids drift between phases.

**Proposed pattern:**
```python
# governance/base.py
GOVERNANCE_OPTIMIZE_INSTRUCTIONS = """\
Goal: produce governance structures grounded in this plan's actual
stakeholders, jurisdiction, and risk profile...
"""
```

Each phase imports and extends with phase-specific guidance.

---

## Phase 6 — SelfAudit calibration

**Current state:** SelfAudit passes most plans without flagging genuine issues.
This is a calibration problem — the model is rubber-stamping rather than
auditing.

**Target metric:** Fraction of completed audits that identify at least 2
genuine internal tensions (budget vs. timeline conflicts, team capacity vs.
scope, regulatory constraints vs. stated approach).

---

## Deferred (lower priority)

- `QuestionsAndAnswersTask` / `ReviewPlanTask` — high context pressure but
  not a primary failure gate in current runs
- `EstimateTaskDurationsTask` — chunked, moderately reliable on tested models
- `CandidateScenariosTask` — JSON schema validation errors observed in practice

---

## Notes

- All phases assume a complete local model run exists
  as the baseline reference for "what good looks like" on local models
- The swarm-coordination `planexe-runs/` archive is the source of truth for
  observed failures; new optimizer runs should be archived there for comparison
- Governance phases should be addressed as a unit, not one phase at a time

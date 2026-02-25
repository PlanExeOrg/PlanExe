---
title: PlanExe Proposal Triage — 80/20 Landscape
date: 2026-02-25
status: working note
author: Egon + Larry
---

# Overview

Simon asked us to triage the proposal space with an 80:20 lens. The goal of this note is to capture:
1. Which proposals deliver outsized value (the 20% that unlock 80% of the architecture)
2. Which other proposals are nearby in the graph and could reuse their artifacts or reasoning
3. High-leverage parameter tweaks, code tweaks, and second/third order effects
4. Gaps in the current docs and ideas for new proposals
5. Relevant questions/tasks you might not have asked yet

We focused on the most recent proposals ("67+" cluster) plus the ones directly touching the validation/orchestration story that FermiSanityCheck will unlock.

# High-Leverage Proposals (the 20%)

1. **#07 Elo Ranking System (1,751 lines)** – Core ranking mechanism for comparing idea variants, plan quality, and post-plan summaries. Louis-level heuristics here inform nearly every downstream comparison use case.
2. **#63 Luigi Agent Integration & #64 Post-plan Orchestration Layer** – These three documents (#63, #64, #66) describe how PlanExe schedules, retries, and enriches its Luigi DAG. Any change to the DAG (including FermiSanityCheck or arcgentica-style loops) ripples through this cluster.
3. **#62 Agent-first Frontend Discoverability (609 lines)** – Defines the agent UX, which depends on the scoring/ranking engine (#07) and the reliability signals that our validation cluster will provide.
4. **#69 Arcgentica Agent Patterns (279 lines)** – The arcgentica comparison is already referencing our validation work and sets the guardrails for self-evaluation/soft-autonomy.
5. **#41 Autonomous Execution of Plan & #05 Semantic Plan Search Graph** – These represent core system-level capabilities (distributed execution and semantic search) whose outputs feed the ranking and reporting layers.

These documents together unlock most of the architectural work. They interlock around: planning quality signals (#07, #69, Fermi), orchestration (#63, #64, #66), and the interfaces (#62, #41, #05).

# Related Proposals & Reuse Opportunities

- **#07 Elo Ranking + #62 Agent-first Frontend** can share heuristics. Instead of reinventing ranking weights in #62, reuse the cost/feasibility tradeoffs defined in #07 plus FermiSanityCheck flags as features.
- **#63-66 orchestration cluster** already describe Luigi tasks. The validation loop doc should be cross-referenced there to show where FermiSanityCheck sits in the DAG and how downstream tasks like WBS, Scheduler, ExpertOrchestrator should consume the validation report.
- **#69 + #56 (Adversarial Red Team) + #43 (Assumption Drift Monitor)** form a validation cluster. FermiSanityCheck is the front line; these others are observers (red team, drift monitor) that should consume the validation report and escalate to human review.
- **#32 Gantt Parallelization & #33 CBS** could re-use the same thresholds as FermiSanityCheck when calculating duration plausibility (e.g., if duration falls outside the published feasible range, highlight the same issue in the Gantt UI).

# 80:20 Tweaks & Parameter Changes

- **Ranking weights (#07)** – adjust cost vs. feasibility vs. confidence to surface plans that pass quantitative grounding. No rewrite needed; just new weights (e.g., penalize plans where FermiSanityCheck flags >3 assumptions).
- **Batch size thresholds (#63)** – the Luigi DAG currently runs every task. We can gate the WBS tasks with a flag that only fires if FermiSanityCheck passes or fails softly, enabling a smaller workflow for low-risk inputs without re-architecting.
- **Risk terminology alignment (#38 & #44)** – harmonize the words used in the risk propagation network and investor audit pack so they can share visualization tooling, reducing duplicate explanations.

# Second/Third Order Effects

- **Validation loop → downstream trust**: Once FermiSanityCheck is in place, client reports (e.g., #60 plan-to-repo, #41 autonomous execution) can annotate numbers with the validation status, reducing rework.
- **Arcgentica/agent patterns**: Hardening PlanExe encourages stricter typed outputs (#69). This lets the UI (#08) and ranking engine (#07) rely on structured data instead of parsing Markdown.
- **Quantitative grounding improves ranking** (#07, #62) which in turn makes downstream dashboards (#60, #62) more actionable and reduces QA overhead.
- **Clustering proposals** (#63-66, #69, #56) around validation/orchestration helps the next human reviewer (Simon) make a single decision that affects multiple docs.

# Gaps & Future Proposal Ideas

- **FermiSanityCheck Implementation Roadmap** – Document how MakeAssumptions output becomes QuantifiedAssumption, where heuristics live, and how Luigi tasks consume the validation_report. (We have the spec in `planexe-validation-loop-spec.md` but not a public proposal yet.)
- **Validation Observability Dashboard** – A proposal capturing how the validation report is surfaced to humans (per #44, #60). Could cover alerts (Slack/Discord) when FermiSanityCheck fails or when repeated fails accumulate.
- **Arbitration Workflow** – When FermiSanityCheck fails and ReviewPlan still thinks the plan is OK, we need a human-in-the-loop workflow. This is not yet documented anywhere.

# Questions You Might Not Be Asking

1. What are the acceptance criteria for FermiSanityCheck? (confidence levels, heuristics, why 100× spans?)
2. Who owns the validation report downstream? Should ExpertOrchestrator or Governance phases be responsible for acting on it?
3. Does FermiSanityCheck expire per run or is it stored for audit trails (per #42 evidence traceability)?
4. Can we reuse the same heuristics for other tasks (#32 Gantt, #34 finance) to maximize payoff?
5. How do we rank the outputs once FermiSanityCheck is added? Should ranking (#07) penalize low confidence even if the costs look good?
6. Do we need a battle plan for manual overrides when FermiSanityCheck is overzealous (e.g., ROI assumptions where domain experts know the average is >100×)?

# Tasks We Can Own Now

- Extract the QuantifiedAssumption schema (claim, lower_bound, upper_bound, unit, confidence, evidence) and add it to PlanExe’s assumption bundle.
- Implement a FermiSanityCheck Luigi task that runs immediately after MakeAssumptions and produces validation_report.json.
- Hook the validation report into DistillAssumptions / ReviewAssumptions by adding a `validation_passed` flag.
- Update #69 and #56 docs with references to the validation report to keep the narrative cohesive.
- Create the proposed dashboard proposal (validation observability) to track how many plans fail numeric sanity each week.

# Summary

The high-leverage 20% of proposals are: ranking (#07), orchestration (#63-66), UI (#62), arcgentica patterns (#69), and autonomous execution/search (#41, #05). We can activate them by implementing FermiSanityCheck, aligning their heuristics, and surfacing the new validation signals in the UI/dashboards. The docs already cover most of the research; now we need a short, focused proposal/clustering doc (this one) plus the Fermi implementation and dashboards. After Simon approves, we can execute the chosen cluster.

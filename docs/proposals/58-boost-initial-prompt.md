---
title: Boost Initial Prompt
date: 2026-02-18
status: proposal
author: Simon Strandgaard
---

# Boost Initial Prompt

**Author:** Simon Strandgaard  
**Date:** 2026-02-18  
**Status:** Proposal  
**Tags:** `prompting`, `quality`, `normalization`, `assumptions`, `guardrails`

---

## Pitch
Add a pre-planning stage that rewrites weak user input into a stronger, concise, execution-ready initial prompt before the normal PlanExe pipeline starts.

The goal is not to overwrite user intent, but to preserve intent while repairing missing constraints, unrealistic parameters, and ambiguous wording.

This should be a first-class UX in plan creation UI, not only a backend/MCP behavior.

## Problem
The initial prompt has disproportionate impact on downstream output quality.

Current failure patterns:

- Missing key fields (location, budget range, timeline realism, resource constraints).
- Unrealistic values (for example near-zero budget for multi-month, multi-person execution).
- Vague or noisy language that produces weak assumptions and low-quality levers.
- Overly specific or contradictory details that anchor the plan in non-critical noise.

When this happens, later stages can look polished but still be impractical because the seed prompt is weak.

There is already a quality gap between two prompt sources:

- MCP tool-driven prompt assembly that follows `example_prompts` (high structure, better constraints).
- Direct human input that is often shorter, incomplete, or inconsistent.

This proposal targets that gap by lifting weak human prompts toward the same baseline used in MCP flows.

## Feasibility
This is feasible as an additive step before `find_plan_prompt` and early assumption tasks.

Why now:

- We already have high-quality prompt examples in `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl`.
- We already document strong prompt shape in `docs/prompt_writing_guide.md`.
- MCP usage already enforces prompt-quality workflow via `docs/mcp/planexe_mcp_interface.md` (`example_prompts` -> formulate -> `task_create`).
- PlanExe already contains assumption-oriented components (`assume/*`) that can consume cleaner input.
- The rewrite stage can be bounded, deterministic in structure, and audited with artifacts.

Constraints:

- Must preserve user intent and avoid silent scope changes.
- Must clearly label inferred fields versus user-provided fields.
- Must cap rewrite iterations to avoid latency/cost blowout.

## Proposal
Introduce a **Boost Initial Prompt** module with three steps.

### 1) Extract
Parse user input into a structured draft.

- sector
- goal/outcome
- location(s)
- budget + currency
- timeline
- audience/stakeholders
- user role + experience
- hard constraints

### 2) Repair
Apply bounded transformations.

- Fill missing high-impact fields via explicit assumptions.
- Normalize units/currency and clarify timeframe granularity.
- Detect unrealistic budget-time-scope combinations and propose realistic alternatives.
- Remove low-signal verbosity while preserving domain details that affect execution.

### 3) Rewrite
Produce normalized prompt artifacts.

- `boosted_prompt`: concise, execution-ready initial prompt.
- `change_log`: what changed and why.
- `assumption_flags`: inferred values requiring user confirmation or low-confidence handling.

## UI Prompt Boost Loop
Add a dedicated step in the create-plan flow: **Optimize Prompt**.

Flow:

1. User enters initial prompt.
2. System runs critique and scoring.
3. System generates exactly 3 improved prompt proposals.
4. System ranks the 3 proposals plus the original prompt.
5. User picks one candidate (or edits manually), then starts plan generation.

This gives a controlled back-and-forth loop before `task_create`/plan execution.

## Critique and Ranking Mechanism
For each candidate (including original), produce a compact scorecard:

- completeness
- realism
- clarity
- constraint coverage (budget, timeline, location, scope)
- risk of contradiction

Return:

- `overall_score` (0-100)
- `strengths` (short bullets)
- `weaknesses` (short bullets)
- `highest_risk_gap` (single biggest issue)

Generation policy for 3 proposals:

- **Proposal A (Conservative):** minimal edits, preserve original phrasing style.
- **Proposal B (Balanced):** strongest overall quality with moderate rewrites.
- **Proposal C (Aggressive):** larger structural rewrite for maximum clarity/feasibility.

Selection default:

- Preselect top-ranked candidate.
- Always show why it ranked highest.
- Keep original prompt selectable to preserve user control.

## Workflow
Suggested flow:

1. Receive raw user prompt.
2. Run structure extraction.
3. Score prompt quality (completeness + realism + clarity).
4. If score below threshold, run repair + rewrite once.
5. Re-score; if still below threshold, run one final constrained rewrite.
6. Pass `boosted_prompt` into existing planning pipeline.
7. Persist artifacts for debugging and A/B testing.

UI variant:

1. User submits initial prompt in UI.
2. Run critique + generate 3 candidate improvements.
3. Rank original + 3 candidates.
4. User chooses one and confirms.
5. Send selected prompt into normal pipeline.

## Prompt Quality Score
Use a transparent score to gate rewrites:

- **Completeness (0-40):** key fields present and parseable.
- **Realism (0-35):** budget/timeline/scope coherence.
- **Clarity (0-25):** concise, non-contradictory, actionable wording.

Decision rule:

- `score >= 75`: use original (or minimal normalization only).
- `score < 75`: trigger boost stage.

## Integration Points
- Entry point before `worker_plan_internal/plan/find_plan_prompt.py`.
- Shared assumptions path with `worker_plan_internal/assume/make_assumptions.py`.
- Optional report section in plan output: “Initial Prompt Boost Summary”.
- Prompt catalog logging for A/B comparisons.
- Prompt-shape alignment with:
  - `docs/prompt_writing_guide.md`
  - `docs/mcp/planexe_mcp_interface.md`
  - `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl` (including MCP-curated examples)

## MCP Baseline Alignment
Use MCP `example_prompts` as the reference quality target for rewritten human prompts.

Concretely:

- Extract structural patterns from MCP examples (scope, budget, timeline, location, success criteria).
- Rewrite weak human prompts to match that structure without changing core intent.
- Track “distance-to-baseline” before and after rewrite for A/B analysis.

## Data Artifacts
Add run artifacts:

- `prompt/raw_prompt.txt`
- `prompt/boosted_prompt.txt`
- `prompt/boost_change_log.json`
- `prompt/boost_quality_score.json`
- `prompt/boost_candidates.json`
- `prompt/boost_ranking.json`

Recommended fields in `boost_change_log.json`:

- `field`
- `original_value`
- `new_value`
- `reason`
- `confidence`
- `requires_user_confirmation`

Recommended fields in `boost_candidates.json`:

- `candidate_id` (`original`, `A`, `B`, `C`)
- `strategy` (`conservative`, `balanced`, `aggressive`)
- `prompt_text`
- `scorecard`

Recommended fields in `boost_ranking.json`:

- ordered candidate list
- score deltas
- top-choice rationale

## Phased Implementation
### Phase A: Baseline Booster
- Implement extraction, single-pass rewrite, and quality scoring.
- Log artifacts and route boosted prompt into pipeline.

### Phase B: Realism Guardrails
- Add budget-time-scope plausibility checks with bounded alternatives.
- Add low-confidence flags for missing critical context.

### Phase C: UI Optimization Loop
- Add create-plan UI step for critique, 3 proposals, and ranking.
- Allow user selection and final manual edits before generation.

### Phase D: Adaptive Improvement
- Run A/B tests: raw prompt vs boosted prompt on identical tasks.
- Promote rewrite patterns that improve objective quality metrics.

## Success Metrics
- Higher average plan quality rating for low-quality user inputs.
- Reduced rate of plans with obvious feasibility mismatches.
- Reduced manual prompt rewriting done by humans before run.
- Improved downstream stability (fewer contradiction flags in assumptions/review).
- Controlled overhead: boost stage adds limited latency and token cost.
- Percentage of UI users selecting one of the 3 boosted proposals.
- Win rate of top-ranked candidate versus original in downstream plan-quality evaluation.

## Risks
- Over-normalization may remove useful nuance.
- Rewrite model may inject incorrect assumptions.
- Extra stage may increase cost/latency without sufficient quality gain.

Mitigations:

- Preserve intent constraints as highest priority.
- Require explicit marking of inferred values.
- Bound rewrite iterations to max 2 passes.
- Keep rollback option: run pipeline on original prompt when confidence is low.

## Open Questions
- Should low-confidence inferred fields block execution or continue with warnings?
- Should users see and approve boosted prompts in UI before plan generation?
- Which quality metric should be canonical for A/B promotion decisions?

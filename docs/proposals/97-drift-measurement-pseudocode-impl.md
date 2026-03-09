# Proposal 84: Pseudocode Implementation — Drift Measurement Task

**Author:** EgonBot  
**Date:** 2026-03-07  
**Status:** Proposal (pseudocode only — full impl pending review)  
**Depends on:** Proposal 82 (framework), Proposal 83 (agent specification)  
**Scope:** One new Luigi task, two Pydantic models, one LLM prompt sequence  
**Files touched (when implemented):** 3–4 new files; zero changes to existing tasks

---

## 1. Purpose

This document presents a pseudocode implementation for `DriftEvaluationTask` — a post-pipeline Luigi task that measures how faithfully a generated plan represents the original user prompt.

It is intended for neoneye review **before** any real code is written.

Nothing here is executable. All class names, field names, and LLM prompt text are illustrative and subject to change.

---

## 2. Placement in the Pipeline

The task runs **after** the pipeline has fully completed. It is an optional post-processing step, not part of the core generation DAG.

```
[StartTimeTask]
     |
[... all generation tasks ...]
     |
[FinalReportTask]
     |
[DriftEvaluationTask]  ← new, optional
```

Inputs required:
- `001-2-initial_plan.txt` — the original user prompt (already exists)
- `final-report.md` or equivalent final plan artifact (already exists)

Output:
- `drift-evaluation.json` — structured drift report
- `drift-evaluation.md` — human-readable verdict

---

## 3. Pydantic Models (pseudocode)

```python
# FILE: worker_plan/worker_plan_internal/plan/drift_models.py
# (pseudocode — not executable)

class DriftIncident(BaseModel):
    drift_type: str          # TypeA–TypeJ from proposal 82 section 6
    severity: int            # 0–4
    section: str             # which plan section
    source_reference: str    # what the prompt said (or didn't say)
    output_claim: str        # what the plan claimed
    explanation: str         # why this is drift

class DimensionScores(BaseModel):
    scope_fidelity: int               # 0–5
    constraint_fidelity: int          # 0–5
    claim_strength_fidelity: int      # 0–5
    evidence_grounding_fidelity: int  # 0–5
    entity_fidelity: int              # 0–5
    causal_fidelity: int              # 0–5
    epistemic_fidelity: int           # 0–5
    source_trace_fidelity: int        # 0–5
    structural_priority_fidelity: int # 0–5
    language_posture_fidelity: int    # 0–5

class PromptContract(BaseModel):
    core_intent: str
    primary_problem: str
    proposed_solution: str
    non_goals: list[str]
    constraints: list[str]
    core_entities: dict[str, str]   # e.g. {"buyer": "...", "user": "..."}
    optional_features: list[str]
    uncertainties: list[str]
    success_metrics: list[str]

class DriftEvaluationResult(BaseModel):
    prompt_contract: PromptContract
    dimension_scores: DimensionScores
    drift_incidents: list[DriftIncident]
    overall_fidelity_score: float    # weighted, 0–5
    overall_drift_risk: str          # "low" | "medium" | "high" | "critical"
    critical_drift_count: int
    unsupported_claim_count: int
    constraint_violation_count: int
    confidence_inflation_count: int
    usable_as_is: bool
    verdict_preserved_well: list[str]
    verdict_major_failures: list[str]
    verdict_recommended_actions: list[str]
```

---

## 4. LLM Prompt Sequence (pseudocode)

The evaluation is split into three sequential LLM calls. Each is structured output.

### Call 1: Extract Prompt Contract

```
SYSTEM:
  You are a strict prompt analyst.
  Your job is to extract a structured contract from a user's planning prompt.
  Do not add anything not in the prompt.
  Do not invent goals, constraints, or entities.
  If something is not stated, leave the field empty or say "not specified".

USER:
  Here is the initial user prompt:
  ---
  {initial_plan_text}
  ---
  Extract the prompt contract. Output ONLY the JSON.

EXPECTED OUTPUT: PromptContract
```

### Call 2: Identify Drift Incidents

```
SYSTEM:
  You are a strict drift evaluator.
  You have been given:
    1. A prompt contract (the structured version of the original user prompt)
    2. A generated plan
  Your job is to identify every place the generated plan departs from the prompt contract.
  Use the drift type taxonomy from the instructions below:
    TypeA = scope expansion
    TypeB = constraint erosion
    TypeC = unsupported invention
    TypeD = confidence inflation
    TypeE = business model drift
    TypeF = customer drift
    TypeG = mechanism drift
    TypeH = priority drift
    TypeI = governance/regulatory drift
    TypeJ = style-induced semantic drift
  Severity scale: 0=no drift, 1=minor, 2=moderate, 3=major, 4=critical

USER:
  Prompt contract:
  ---
  {prompt_contract_json}
  ---
  Generated plan:
  ---
  {final_report_text}
  ---
  Identify all drift incidents. Output ONLY the JSON list.

EXPECTED OUTPUT: list[DriftIncident]
```

### Call 3: Score and Verdict

```
SYSTEM:
  You are a strict drift scoring judge.
  You have been given:
    1. A prompt contract
    2. A list of drift incidents with severities
  Your job is to assign fidelity scores across 10 dimensions and produce a final verdict.
  Scoring: 5=excellent fidelity, 4=good, 3=mixed, 2=weak, 1=severe drift, 0=failed.
  Weighted fidelity score formula:
    (constraint_fidelity * 0.20)
    + (scope_fidelity * 0.15)
    + (evidence_grounding_fidelity * 0.15)
    + (causal_fidelity * 0.10)
    + (entity_fidelity * 0.10)
    + (epistemic_fidelity * 0.10)
    + (structural_priority_fidelity * 0.08)
    + (claim_strength_fidelity * 0.05)
    + (source_trace_fidelity * 0.04)
    + (language_posture_fidelity * 0.03)
  Drift risk: critical if any severity-4 incident, else high if overall_fidelity_score < 2.5,
              medium if < 3.5, else low.
  Disqualifying conditions (force usable_as_is=false regardless of score):
    - any explicit banned concept reintroduced
    - target customer materially changed
    - business model materially changed
    - multiple critical unsupported numerical claims
    - explicit non-goals violated

USER:
  Prompt contract:
  ---
  {prompt_contract_json}
  ---
  Drift incidents:
  ---
  {drift_incidents_json}
  ---
  Produce dimension scores and verdict. Output ONLY the JSON.

EXPECTED OUTPUT: DriftEvaluationResult (dimension_scores + verdict fields)
```

---

## 5. Luigi Task (pseudocode)

```python
# FILE: worker_plan/worker_plan_internal/plan/run_plan_pipeline.py
# (pseudocode — shows where the task fits, not production code)

class DriftEvaluationTask(PlanTask):
    """
    Post-pipeline task: evaluate how faithfully the generated plan
    represents the original user prompt.
    Optional — does not block report generation.
    """

    def requires(self):
        return {
            'prompt': self.clone(SetupTask),          # 001-2-initial_plan.txt
            'report': self.clone(FinalReportTask),     # or equivalent final artifact
        }

    def output(self):
        return {
            'json': self.local_target('drift-evaluation.json'),
            'markdown': self.local_target('drift-evaluation.md'),
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs
        initial_plan_text = read(self.input()['prompt'])
        final_report_text = read(self.input()['report'])

        # Call 1: extract prompt contract
        prompt_contract = call_llm_structured(
            llm,
            system=SYSTEM_PROMPT_CONTRACT_EXTRACTION,
            user=initial_plan_text,
            output_model=PromptContract,
        )

        # Call 2: identify drift incidents
        drift_incidents = call_llm_structured(
            llm,
            system=SYSTEM_DRIFT_INCIDENT_DETECTION,
            user=format(prompt_contract, final_report_text),
            output_model=list[DriftIncident],
        )

        # Call 3: score and verdict
        result = call_llm_structured(
            llm,
            system=SYSTEM_DRIFT_SCORING,
            user=format(prompt_contract, drift_incidents),
            output_model=DriftEvaluationResult,
        )

        # Merge all three call results into final output
        result.prompt_contract = prompt_contract
        result.drift_incidents = drift_incidents
        result.overall_fidelity_score = compute_weighted_score(result.dimension_scores)
        result.overall_drift_risk = classify_risk(result)

        # Write outputs
        write_json(self.output()['json'], result)
        write_markdown(self.output()['markdown'], render_verdict(result))
```

---

## 6. Output File: `drift-evaluation.md` (example render)

```markdown
# Drift Evaluation Report

**Overall Fidelity Score:** 3.8 / 5.0
**Drift Risk:** medium
**Usable As-Is:** yes

## What Was Preserved Well
- Core intent (HVT drone paintball simulation) unchanged
- Customer definition intact (players, event organisers)
- Budget uncertainty preserved

## Major Failures
- Confidence inflation: "may reduce setup time" became "will eliminate logistics overhead"
- Unsupported invention: specific vendor names added (2 incidents, severity 3)

## Recommended Actions
- Restore modal language in logistics section
- Remove unsupported vendor claims or flag as speculative

## Dimension Scores
| Dimension | Score |
|---|---|
| Scope fidelity | 4 |
| Constraint fidelity | 4 |
| Evidence grounding | 3 |
| Entity fidelity | 5 |
| Epistemic fidelity | 3 |
| ... | ... |

## Drift Incidents (3 total)
### Incident 1 — Severity 3 (TypeC: Unsupported Invention)
...
```

---

## 7. Open Questions for neoneye

1. **Which final artifact to read?** The plan has multiple output files. Should this task read `final-report.md`, the full HTML report, or a specific intermediate markdown artifact? The HTML is ~700KB; a markdown section file may be more appropriate.

2. **Is this task mandatory or always-optional?** Proposal suggests optional (does not block report). Confirm?

3. **Three LLM calls per evaluation.** For large plans this is expensive on frontier models. Should `DriftEvaluationTask` use a cheaper model profile regardless of what profile ran the rest of the pipeline?

4. **Scope of the incident list.** A 20-section plan could produce 50+ incidents. Should Call 2 be limited to top-N most severe, or exhaustive?

5. **`list[DriftIncident]` as structured output.** This is an unbounded list. For local models this may hit token limits. Should the schema cap at e.g. `max_items=20`?

---

## 8. What This Proposal Does NOT Include

- No production code
- No changes to existing tasks
- No changes to `run_plan_pipeline.py` beyond the new task class
- No changes to report rendering
- No API endpoint changes
- No frontend changes

Full implementation will be a separate PR after this proposal is approved.

# Pipeline Hardening Roadmap for Local-Model Reliability (Implementation Spec)

Date: 2026-03-07  
Author: EgonBot  
Reviewers: neoneye, Bubba

---

## 0) Purpose of this document

This is not a high-level vision note. This is an implementation spec for improving local-model reliability in PlanExe’s structured-output pipeline.

It defines:
- exactly **what to change**,
- **where** to change it,
- **how** to validate each change,
- **what evidence** is required before merge.

The target is practical: reduce repeated run failures caused by structured-output truncation/type drift while preserving data quality signals.

---

## 1) Current state and known facts

### 1.1 Confirmed, merged fixes

- **PR #153 merged** with two minimal fixes:
  1) `select_scenario.py`: default for `holistic_profile_of_the_plan` (truncation unblock).  
  2) `run_plan_pipeline.py`: `start_time_dict.get('server_iso_utc', '')` (CLI KeyError unblock).

### 1.2 Current top failure after #153

- Pipeline now advances beyond `SelectScenarioTask`.
- Next hard stop: `PreProjectAssessmentTask` with missing required tail fields in `ExpertDetails`:
  - `combined_summary`
  - `go_no_go_recommendation`

### 1.3 Root failure classes

1. **Tail-field truncation** (required fields at end of schema missing).  
2. **Type drift** (wrong primitive/list/object type).  
3. **Schema echo** (model repeats schema shape/descriptions, not values).  
4. **Code-path assumption bugs** (`KeyError` on non-API path).

---

## 2) Guardrails and merge policy

### 2.1 Scope guard (mandatory)

Every PR must include in description:
- “This PR intentionally changes only X files.”
- “Failure class targeted: <one class only>.”
- “Out of scope: <explicit list>.”

### 2.2 Evidence guard (mandatory)

Before merge, attach:
- before/after task failure location,
- exact error signature change,
- `git diff --name-only` output,
- resume-run proof (Luigi resume path).

### 2.3 Data integrity guard

Defaults are allowed only for low-risk synthesis fields in containment phase. If default used, add trace marker in logs in Phase 1.

---

## 3) Work package sequence (concrete)

## WP-1 — PreProjectAssessmentTask truncation containment

### Objective
Unblock current failure gate with minimal code change.

### Files to modify
- `worker_plan/worker_plan_internal/expert/pre_project_assessment.py`

### Change
In `ExpertDetails` model:
- make `combined_summary` default to `""`
- make `go_no_go_recommendation` default to `""`

### Example patch shape
```python
class ExpertDetails(BaseModel):
    feedback: list[FeedbackItem]
    combined_summary: str = Field(default="", description="...")
    go_no_go_recommendation: str = Field(default="", description="...")
```

### Acceptance criteria
- Resume run no longer dies at `PreProjectAssessmentTask` with missing-field error.
- If it fails, error class must be different (e.g., type drift, not missing tail fields).

### PR constraints
- One file only.
- No prompt rewrites in same PR.
- No config changes in same PR.

---

## WP-2 — Retry behavior upgrade in LLM executor (failure-intelligent retries)

### Objective
Replace blind same-input retries with targeted retries informed by validation errors.

### Files to modify
- `worker_plan/worker_plan_internal/llm_util/llm_executor.py`
- (if needed) small helper module under `worker_plan/worker_plan_internal/llm_util/`

### Current behavior problem
Retries repeat with nearly identical prompt/context and often produce identical failure.

### Required behavior
On parse/validation failure:
1. Extract compact error summary:
   - missing fields,
   - invalid field types,
   - top-level schema name.
2. Build a retry suffix instruction:
   - “Return valid JSON object only.”
   - “Fix only these fields: …”
   - “Do not remove valid fields already present.”
3. Append suffix only for retry attempts (attempt > 1).

### Retry instruction template (exact)
```text
Validation failed for schema: {schema_name}.
Fix the JSON by correcting only these issues:
- Missing fields: {missing_fields_csv}
- Invalid fields/types: {invalid_fields_csv}
Return ONE JSON object only, no markdown, no explanation.
Preserve all previously valid fields.
```

### Implementation notes
- Keep max retries unchanged initially.
- Do not alter model fallback chain in this PR.
- Keep this patch isolated to retry message construction and logging.

### Acceptance criteria
- At least one known truncation/type failure case shows improved recovery rate vs baseline.
- Logs show per-attempt error summary and retry instruction injection.

---

## WP-3 — Structured failure telemetry

### Objective
Make failures diagnosable without guessing.

### Files to modify
- `worker_plan/worker_plan_internal/llm_util/llm_executor.py`
- optional: create `worker_plan/worker_plan_internal/llm_util/llm_failure_logging.py`

### Required telemetry record (per failed attempt)
Persist/log fields:
- `task_name`
- `schema_name`
- `attempt_index`
- `model_id`
- `missing_fields` (list)
- `invalid_fields` (list)
- `raw_response_preview` (bounded char length)
- `token_metrics_snapshot` (if available)

### Log format
Prefer structured JSON log line, example:
```json
{
  "event": "structured_parse_failure",
  "task_name": "PreProjectAssessmentTask",
  "schema_name": "ExpertDetails",
  "attempt_index": 2,
  "missing_fields": ["combined_summary"],
  "invalid_fields": ["feedback[2].description:type"],
  "model_id": "qwen/qwen3.5-35b-a3b"
}
```

### Acceptance criteria
- Same failure can be categorized instantly as truncation/type/schema-echo.
- No sensitive full payload dumps; preview length bounded.

---

## WP-4 — Task strictness matrix (hard vs soft fields)

### Objective
Prevent accidental silent corruption while keeping pipeline progress possible.

### Deliverable
A documented matrix (new doc section or file) listing for each high-risk schema:
- hard-required fields (must fail if absent)
- soft-required fields (may default with trace)

### Initial coverage
- `SelectScenarioTask` schema
- `PreProjectAssessmentTask` (`ExpertDetails`)
- `CreateWBSLevel3Task` details schema

### Acceptance criteria
- Every defaulting decision in code points to matrix rationale.
- Reviewers can verify why a field is soft/hard.

---

## WP-5 — Local reliability profile + preflight smoke gate

### Objective
Fail fast before expensive full pipeline if model/profile is incompatible.

### Files (proposed)
- `worker_plan/worker_plan_internal/plan/run_plan_pipeline.py` (hook)
- new utility: `worker_plan/worker_plan_internal/llm_util/preflight_smoke.py`
- docs update in provider docs and/or `docs/proposals` follow-up

### Preflight checks (minimal)
Run 3 cheap checks before full execution:
1. Scenario parse check.
2. Expert assessment parse check.
3. One WBS detail parse check.

### Behavior
- If any check fails: stop with explicit action list:
  - increase `num_output`,
  - switch model profile,
  - run with fallback-capable model.

### Acceptance criteria
- Reduction in long-run failures that die at first schema-heavy stages.

---

## 4) Test plan (must run per work package)

## For WP-1 (PreProjectAssessment containment)
- Run Luigi resume on previously failing run inputs.
- Verify `PreProjectAssessmentTask` completes.
- Capture next failure point (if any).

## For WP-2 and WP-3 (retry + telemetry)
- Reproduce known validation failure case.
- Confirm retry prompt contains error guidance.
- Confirm telemetry line appears with expected keys.

## For WP-5 (preflight)
- Test failing model profile: preflight must block full run.
- Test passing profile: preflight must allow run to start.

---

## 5) Rollout and rollback

### Rollout order
1. WP-1 (containment, unblock)
2. WP-2 (retry quality)
3. WP-3 (telemetry)
4. WP-4 (strictness matrix)
5. WP-5 (preflight controls)

### Rollback strategy
- Each WP in separate PR and commit series.
- If regression appears, revert only that WP PR.
- Never bundle two WPs in one PR.

---

## 6) Definition of done (program-level)

This roadmap is considered implemented when:
1. Pipeline clears current truncation gates on local profile in repeated runs.
2. Failures, when present, are classified and traceable.
3. Retry path demonstrably outperforms blind retries on at least one known case.
4. Preflight gate prevents at least one doomed full run class.
5. Docs include reproducible operator playbook and strictness rationale.

---

## 7) Immediate next action

Create **WP-1 PR now**: `PreProjectAssessmentTask` tail-field default containment only, then rerun via Luigi resume and report whether first-failure location moves downstream.

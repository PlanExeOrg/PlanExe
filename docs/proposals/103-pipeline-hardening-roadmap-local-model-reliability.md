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

### 1.4 Local model testing findings: llama_index silent prompt truncation (March 2026)

#### Root cause: llama_index default constants

Testing on Mac Mini M4 Pro (March 4-6, 2026) with qwen/qwen3.5-35b-a3b via LM Studio revealed the underlying cause of silent structured-output failures:

**File**: `llama_index/core/constants.py`
```python
DEFAULT_CONTEXT_WINDOW = 3900  # applies to ALL OpenAILike models unless overridden
DEFAULT_NUM_OUTPUT = 256       # affects PromptHelper budget, NOT model output limit
```

These defaults apply to **all OpenAILike models** unless explicitly overridden in the model config. The `PromptHelper` computes:
```
available_prompt_budget = context_window - num_output = 3900 - 256 = 3644 tokens
```

#### The failure mode

`DeduplicateLeversTask` input was ~8,400 tokens. With the default budget of 3644 tokens, the prompt was **silently truncated to 3644 tokens** before being sent to the model. The model received a mangled, mid-sentence prompt, produced thinking text instead of valid JSON, and the pipeline failed with `ValueError: Could not extract json string`.

This is **silent** — no warning, no log entry, no error at truncation time. The failure manifests downstream as a JSON extraction error, making it extremely hard to diagnose.

#### Critical insight: `num_output` does NOT limit model output

A common misunderstanding: setting `num_output: 4096` does NOT cap the model's actual output length. The LM Studio API generates freely (10K+ tokens observed). The limit ONLY affects llama_index's internal `PromptHelper` budget calculation for **input** truncation. This is the root cause of misdiagnosis.

#### The fix

Add explicit overrides to each LM Studio model entry in `llm_config/custom.json`:
```json
"context_window": 8192,
"num_output": 4096
```

This expands the available prompt budget from 3644 to 4096 tokens, accommodating tasks with large combined inputs. For models with larger context windows (e.g., 32K), use `context_window: 32768, num_output: 4096`.

#### Tasks most vulnerable to silent truncation

Tasks with large combined inputs (multiple prior outputs concatenated):
- `DeduplicateLeversTask` — ~8,400 tokens input (confirmed failure point)
- `PremortemTask` / `ReviewPlanTask` / `QuestionsAndAnswersTask` — 10-12 docs concatenated
- `GovernancePhase4-6` — accumulating context chain

#### Detection method

To verify if your deployment is vulnerable:
```bash
grep -r "DEFAULT_CONTEXT_WINDOW\|DEFAULT_NUM_OUTPUT" $(pip show llama-index-core | grep Location | cut -d' ' -f2)/llama_index/core/constants.py
```

If values are 3900/256 and your model config doesn't override them, you're vulnerable to silent truncation.

#### Relationship to this roadmap

Silent truncation is exactly the class of mid-pipeline failures that the containment and telemetry work packages (WP-1, WP-2, WP-3) are designed to catch and surface. By adding structured failure logging (WP-3), future truncation events will be diagnosable instantly. By adding a preflight smoke test (WP-5), bad model profiles can be caught before expensive full-pipeline runs.

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

# Pipeline Intelligence Layer — Proposal

**Date:** 2026-03-07  
**Author:** Bubba (Mac Mini agent)  
**Context:** Second-order proposal. The first proposal (PR #157) covered operational tooling. This one covers the deeper architectural gap: the pipeline treats model failures as terminal events, when they should be learning opportunities.

---

## The Core Problem

The current retry loop in `llm_executor.py` does something fundamentally broken: when a model fails Pydantic validation, it retries the **identical prompt** with the **identical model**. This is not a retry — it's repetition. It produces the same output and fails the same way.

Every truncation failure we fixed tonight with `default=""` is a symptom of this. We applied a bandage. The wound is that the pipeline has no way to tell the model what it did wrong and ask it to try again with that information.

This proposal is about giving the pipeline a real feedback loop.

---

## 1. Error-Feedback Retries

### Current behavior

```
Attempt 1: send prompt → model omits holistic_profile_of_the_plan → Pydantic fails
Attempt 2: send identical prompt → model omits holistic_profile_of_the_plan → Pydantic fails
Attempt 3: exhausted → pipeline dies
```

The model receives no information about what it did wrong.

### Proposed behavior

On Pydantic validation failure, extract structured error information and inject it into the next attempt:

```
Attempt 1: send prompt → model omits holistic_profile_of_the_plan → Pydantic fails
           → extract: missing field "holistic_profile_of_the_plan" in PlanCharacteristics

Attempt 2: send prompt + correction message:
  "Your previous response was missing the required field 'holistic_profile_of_the_plan' 
   in PlanCharacteristics. Please regenerate the complete JSON, including this field: 
   a concise holistic summary synthesizing the four characteristics above."
   → model generates complete JSON → Pydantic succeeds
```

### Implementation

In `llm_executor.py`, after catching a `ValidationError`:

1. Extract missing fields and invalid types from the Pydantic error
2. Generate a compact correction message (< 200 tokens)
3. Append as a new `ChatMessage(role=ASSISTANT, content=<bad_response>)` + `ChatMessage(role=USER, content=<correction>)`
4. Retry with the augmented message history

This converts a blind retry into a self-correcting dialogue. The model sees its own bad output and a precise instruction to fix it.

### Expected impact

Tasks that currently require `default=""` workarounds may self-correct in attempt 2 instead of failing silently with an empty field. The `default=""` fixes are still useful as a last resort — but they should be the fallback, not the primary recovery path.

---

## 2. Task-Level Output Validation Beyond Pydantic

### The problem

Pydantic validates structure. It does not validate content. A model can return `holistic_profile_of_the_plan: ""` (which our fix now allows) and the pipeline continues happily with an empty field. The plan output is now corrupted in a way that Pydantic cannot detect.

More broadly: a model can return `go_no_go_recommendation: "I'll go ahead and recommend proceeding"` instead of `"Go"` or `"No Go"`. Structurally valid, semantically wrong.

### Proposed solution

A lightweight post-validation layer per task that checks content constraints:

```python
class ContentConstraint:
    field: str
    constraint_type: Literal["non_empty", "one_of", "min_length", "max_length", "regex"]
    params: dict
    severity: Literal["warn", "retry", "fail"]
```

For `ExpertDetails`:
```python
constraints = [
    ContentConstraint("combined_summary", "min_length", {"chars": 100}, "retry"),
    ContentConstraint("go_no_go_recommendation", "one_of", 
                      {"values": ["Go", "No Go", "Execute Immediately", 
                                  "Proceed with Caution", "Do Not Execute"]}, "retry"),
]
```

On `severity="retry"`, run the error-feedback retry loop with the content violation as the correction message. On `severity="warn"`, log and continue. On `severity="fail"`, treat as a hard failure.

This closes the gap between structural validity (Pydantic) and semantic validity (what the pipeline actually needs).

---

## 3. Adaptive Model Selection

### The problem

The current model config has a fixed priority list. If model A fails task X, the pipeline falls back to model B — but model B may be known to fail task X for a completely different reason. There is no memory of which models succeed at which tasks.

### Proposed solution

A `model_task_matrix.json` that records observed task outcomes per model:

```json
{
  "lmstudio-qwen3.5-35b-a3b": {
    "SelectScenarioTask": "pass",
    "PreProjectAssessmentTask": "pass",
    "CreateWBSLevel3Task": "unknown"
  },
  "zai-org/glm-4.6v-flash": {
    "SelectScenarioTask": "pass",
    "PreProjectAssessmentTask": "fail:schema_echo",
    "CreateWBSLevel3Task": "fail:schema_echo"
  }
}
```

The pipeline updates this file after each task attempt. On the next run, when building the model fallback order for a given task, it consults the matrix and skips models with `fail:*` outcomes for that task.

This creates an empirical model scorecard that improves over time without requiring human intervention.

---

## 4. Human-in-the-Loop Gates

### The problem

Some decisions in the pipeline should not be automated. The `RedlineGateTask` already exists as a safety check — but it auto-continues. There is no mechanism to pause the pipeline and wait for a human to review output before downstream tasks consume it.

For an agent running an overnight pipeline, this is critical: if `go_no_go_recommendation` comes back `"Do Not Execute"`, the pipeline should pause and alert the human rather than continuing to generate a 60-task project plan for something the expert assessment said was infeasible.

### Proposed solution

A `PLANEXE_PAUSE_ON` environment variable (comma-separated list of task names or field values):

```
PLANEXE_PAUSE_ON=go_no_go_recommendation:Do Not Execute,RedlineGateTask:blocked
```

When a listed condition is met:
1. Pipeline writes a `pause.json` to the run directory with the condition details
2. If a webhook is configured, fires a `pipeline.paused` event
3. Pipeline waits for `resume.json` to appear (human creates it to continue) or `abort.json` (human creates it to stop)
4. `planexe resume --run-dir ./run/X` creates the `resume.json` file

This gives humans meaningful control over consequential decision points without requiring them to babysit every task.

---

## 5. Streaming Output Monitor

### The problem

For long-running LLM calls (2–4 minutes for expert assessment tasks), the pipeline is completely opaque. You cannot tell whether the model is generating useful output, hallucinating, or stuck. You only know the result when it finishes — by which point it has already timed out or produced bad output.

### Proposed solution

A streaming preview mode that logs model output tokens to a `stream.log` file in the run directory as they arrive:

```
PLANEXE_STREAM_LOG=true
```

This does not change pipeline behavior — it adds a side channel. An agent or human can `tail -f ./run/Qwen_Clean_v1/stream.log` to watch output in real time and detect early if a model is going off the rails (e.g., generating markdown prose instead of JSON, or repeating the schema back instead of filling it).

For agent monitoring, this enables early abort: if the stream shows the model is not producing JSON after the first 500 tokens, the agent can kill the request and retry before the full timeout expires.

---

## 6. Run Comparison

### The problem

After applying a fix and re-running, there is no easy way to see what changed in task outputs between run A and run B. Did the model generate more complete output after the `default=""` fix? Did the scenario selection change? Did the expert assessment reach a different conclusion?

### Proposed solution

```bash
planexe diff --run-a ./run/Qwen_Only_Clean_v1 --run-b ./run/Qwen_Clean_v1 --task PreProjectAssessmentTask
```

Output:

```diff
PreProjectAssessmentTask output diff:

  combined_summary:
-   (empty — field was missing)
+   The project faces three critical blockers: insufficient capital runway...

  go_no_go_recommendation:
-   (empty — field was missing)  
+   Proceed with Caution — the core concept is viable but...
```

This makes it immediately clear whether a fix improved output quality, changed conclusions, or introduced regressions. Essential for PR evidence and model comparison.

---

## Priority

| Priority | Feature | Complexity | Impact |
|----------|---------|-----------|--------|
| 1 | Error-feedback retries | Medium | Eliminates most `default=""` workarounds |
| 2 | Human-in-the-loop gates | Low | Safety for consequential decisions |
| 3 | Content validation beyond Pydantic | Medium | Closes silent corruption gap |
| 4 | Streaming output monitor | Medium | Enables early abort, faster debug |
| 5 | Adaptive model selection | Medium | Self-improving failure recovery |
| 6 | Run comparison | Low | Evidence-based development |

---

## Relationship to PR #157

PR #157 covers operational tooling (status, invalidation, webhooks, timeouts). This proposal covers the intelligence layer — making the pipeline smarter about failure and recovery. Both are necessary. #157 makes the pipeline observable and controllable. This proposal makes it self-correcting.

The highest-value item here is error-feedback retries (#1). Implementing it would make several of the `default=""` fixes in PRs #153, #155, #156 unnecessary — or demote them from primary fixes to last-resort fallbacks.

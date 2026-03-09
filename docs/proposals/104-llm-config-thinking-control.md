# llm_config Thinking-Control Rework (Egon's Proposal)

Date: 2026-03-07  
Author: EgonBot  
Context: PlanExe local-model runs reveal that some models (Qwen 35B, GLM) emit chain-of-thought reasoning that destroys structured JSON output on schema-heavy tasks. Current workarounds are model-specific hacks baked into task code. This proposal replaces them with a clean model-level configuration system.

---

## 1) The actual problem

When `ReviewPlanTask` accumulates multi-turn context and the model begins outputting reasoning before JSON, the pipeline breaks. The current fix attempts to suppress this per-model (`/no_think` for Qwen, explicit system prompt text). These hacks:
- Assume knowledge of the active model inside task code,
- Work differently (or not at all) across different models,
- Are invisible to operators configuring a new model profile.

The real fix: **model profiles should declare how they handle thinking/reasoning, and the task should declare what it needs. The infrastructure resolves the match.**

---

## 2) Design principle

Separation of concerns:
- **Task code** declares a `thinking_mode` requirement (e.g., "suppress thinking for clean JSON output").
- **Model profiles** in `llm_config.json` declare their thinking behavior and how to suppress it.
- **`llm_factory`** applies the right suppression mechanism for the active model when a task requires it.

No model-specific logic in task code. No task-specific logic in model config.

---

## 3) Proposed llm_config.json extension

Add a `thinking` block to each model profile that has thinking/reasoning behavior:

```json
"lmstudio-qwen3.5-35b-a3b": {
    "class": "LMStudio",
    "arguments": {
        "model_name": "qwen/qwen3.5-35b-a3b",
        "context_window": 32768,
        "num_output": 16384,
        "temperature": 0.2,
        "request_timeout": 300.0,
        "is_function_calling_model": false
    },
    "thinking": {
        "mode": "auto",
        "suppress_token": "/no_think",
        "suppress_system_prompt": "Output ONLY the JSON object. Do not include any reasoning, thinking steps, or explanations."
    }
}
```

Fields:
- `mode`: `"auto"` (default, model chooses), `"always"` (model always thinks), `"never"` (model never thinks unless explicitly triggered).
- `suppress_token`: if set, append this token to user messages when thinking suppression is requested by a task.
- `suppress_system_prompt`: if set, append this instruction to the system prompt when thinking suppression is requested by a task.

For models without thinking behavior (e.g., standard OpenRouter models), omit the `thinking` block entirely.

---

## 4) Proposed task-level declaration

Tasks that need clean JSON output (no reasoning prefix) declare their requirement at construction time:

```python
class ReviewPlanTask(PlanTask):
    thinking_mode_required = "suppress"
```

Or passed directly to `LLMExecutor`:

```python
executor = LLMExecutor(
    llm_models=llm_models,
    thinking_mode="suppress",
)
```

---

## 5) How llm_factory applies the suppression

When `get_llm()` is called and the model profile has a `thinking` block, and the caller requests `thinking_mode="suppress"`:

1. If `suppress_token` is set: add it as a wrapper or injection point for user messages.
2. If `suppress_system_prompt` is set: append to the system prompt being constructed.
3. If neither is set: log a warning that the model doesn't support thinking suppression.

The factory always applies the **profile's declared mechanism**. Task code requests suppression generically and never knows which model is active.

---

## 6) How this handles model diversity

| Model | Has thinking | Suppress mechanism |
|---|---|---|
| Qwen 35B | Yes (auto) | `/no_think` token + system prompt |
| GLM 4.7 | Yes (manual) | system prompt only |
| OpenAI o3 | Yes | `thinking: {budget_tokens: 0}` via API |
| Standard OpenRouter | No | nothing applied |

Each profile declares what works for it. Task code is identical for all.

---

## 7) Migration path

1. Add `thinking` block to Qwen and GLM profiles in `llm_config.json` (config-only, no code change).
2. Add `thinking_mode` parameter to `get_llm()` and `LLMExecutor` (small code change in `llm_factory.py`).
3. Apply suppression mechanism in factory when `thinking_mode="suppress"` is requested.
4. Update `ReviewPlanTask` (and any other affected task) to declare `thinking_mode="suppress"`.
5. Remove hardcoded `/no_think` and model-specific system prompt hacks from all task code.

Each step is an independently reviewable PR.

---

## 8) What this doesn't solve

- Models that cannot suppress thinking at all (architectural issue, not config).
- Tasks that *want* model reasoning to improve output quality (those would request `thinking_mode="auto"` or `"always"`).
- Token budget control for thinking models (separate concern — use `num_output` for now).

---

## 9) Immediate next action

Before this full rework lands, the interim fix (generic system prompt instruction in `ReviewPlanTask`) is correct and should stay. This proposal is a clean-up effort, not an emergency patch.

Draft implementation order:
1. Config extension (no code, pure JSON schema addition + sample profiles).
2. `llm_factory` suppression hook (small, ~30 lines).
3. `LLMExecutor` pass-through for `thinking_mode` parameter.
4. Task declaration update + cleanup of existing hacks.

---
title: LLM config (llm_config.json)
---

# LLM config (llm_config.json)

This file defines which LLM providers and models PlanExe can use. Each top‑level key is a model id used in the UI and pipeline.

`llm_config.json` lives in the PlanExe repo root and is read at runtime. Environment variables are substituted from `.env`.

---

## File structure

```json
{
  "model-id": {
    "comment": "Human description",
    "priority": 1,
    "luigi_workers": 4,
    "class": "OpenRouter",
    "arguments": {
      "model": "google/gemini-2.0-flash-001",
      "api_key": "${OPENROUTER_API_KEY}",
      "temperature": 0.1,
      "timeout": 60.0,
      "is_function_calling_model": false,
      "is_chat_model": true,
      "max_tokens": 8192,
      "max_retries": 5
    }
  }
}
```

---

## Top-level fields

- **comment**: Plain‑text description for humans. Optional.
- **priority**: Lower number = higher priority when `auto` is selected. Optional.
- **luigi_workers**: Number of Luigi workers used for this model. Use `1` for local models (Ollama/LM Studio).
- **class**: Provider class name (e.g., `OpenRouter`, `OpenAI`, `Ollama`, `LMStudio`, `OpenAILike`).
- **arguments**: Provider‑specific settings passed to the LLM client.

---

## Common arguments

These keys are common across most providers:

- **model** / **model_name**: Provider model identifier.
- **api_key**: API key reference (usually `${ENV_VAR}`).
- **base_url** / **api_base**: Override the provider base URL.
- **temperature**: Controls randomness. Lower is more deterministic.
- **timeout** / **request_timeout**: Max time per request in seconds.
- **max_tokens** / **max_completion_tokens**: Output token limit (provider specific).
- **max_retries**: Retry count on transient errors.
- **is_function_calling_model**: Whether the model supports structured/tool output.
- **is_chat_model**: Whether the model uses chat format.

---

## Choosing values

- Use **luigi_workers = 1** for local models (Ollama / LM Studio).
- Use **luigi_workers > 1** for cloud models if you want parallel tasks.
- Keep **timeout** higher for slower models.

---

## Notes

- If `llm_config.json` is missing, PlanExe logs a warning and proceeds with defaults.
- Changes to `llm_config.json` require a container restart (or rebuild if baked into the image).

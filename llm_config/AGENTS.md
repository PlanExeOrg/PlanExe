# llm_config agent instructions

Scope: JSON configuration files that define which LLM providers and models
PlanExe can use. Each profile targets a different use case (cost, quality,
local-only). The active profile is selected at runtime via the
`PLANEXE_MODEL_PROFILE` environment variable.

## File map

| File | Profile | Purpose |
|------|---------|---------|
| `baseline.json` | `baseline` (default) | Mix of cheap cloud models + local Ollama/LM Studio entries. Default when no env var is set. |
| `custom.json` | `custom` | User-curated selection. Filename overridable via `PLANEXE_LLM_CONFIG_CUSTOM_FILENAME`. |
| `premium.json` | `premium` | Higher-quality (and higher-cost) cloud models. |
| `frontier.json` | `frontier` | Cutting-edge models for maximum quality. |
| `anthropic_claude.json` | `custom` | Anthropic-only config (use with `PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=anthropic_claude.json`). |
| `local.json` | `custom` | LM Studio local models only (no API keys needed). |
| `custom_thinking_test.json` | `custom` | LM Studio models with thinking/reasoning tokens enabled. |

## Entry schema

Every key in a config file is a model entry. Required and optional fields:

```jsonc
{
  "config-key": {
    "comment": "Human-readable description with pricing info",  // optional
    "priority": 1,               // optional — lower = tried first in Auto mode
    "luigi_workers": 4,          // required — max parallel Luigi tasks for this model
    "class": "OpenRouter",       // required — provider class (see below)
    "arguments": {               // required — passed to the llama_index LLM constructor
      "model": "provider/model-id",
      "api_key": "${ENV_VAR}",
      "temperature": 0.1,
      "timeout": 60.0,
      "max_tokens": 8192,
      "max_retries": 5,
      "is_function_calling_model": false,
      "is_chat_model": true
    },
    "model_info_url": "https://...",  // optional — link to model docs
    "pricing": {                       // optional — fallback cost estimation
      "input_per_million_tokens": 0.15,
      "output_per_million_tokens": 0.60
    },
    "pricing_kind": "paid"            // required — "paid" or "free"
  }
}
```

## Provider classes

| Class | Provider | API key env var | Notes |
|-------|----------|-----------------|-------|
| `OpenRouter` | OpenRouter (routes to many providers) | `OPENROUTER_API_KEY` | Returns cost in usage response; pricing field still useful as a cross-check |
| `OpenAI` | OpenAI direct | `OPENAI_API_KEY` | Does NOT return cost — `pricing` field required for cost tracking |
| `Anthropic` | Anthropic direct | `ANTHROPIC_API_KEY` | Does NOT return cost — `pricing` field required. Supports OAuth tokens (`sk-ant-oat*`) |
| `OpenAILike` | Any OpenAI-compatible API | varies | Used for Alibaba DashScope, LM Studio, etc. |
| `Ollama` | Local Ollama | none | Free; set `base_url` for Docker |
| `LMStudio` | Local LM Studio | none | Free |
| `ThinkingAwareOpenAILike` | LM Studio with thinking tokens | none | Strips thinking tokens before JSON parsing |

## Guidelines

- **Do not remove entries** without checking that no users depend on them.
  Deprecate by moving to a less prominent profile first.
- **Always include `pricing`** for paid models. Without it, `activity_overview.json`
  will show `total_cost: 0.0` for providers that don't report cost (OpenAI,
  Anthropic, OpenAILike).
- **Keep pricing rates current.** Rates are sourced from provider pricing pages.
  When updating, also update the human-readable `comment`.
- **Environment variable references** use `${VAR_NAME}` syntax. They are
  resolved at load time from `.env` via `PlanExeDotEnv`.
- **`priority`** controls Auto mode fallback order (lower = higher priority).
  Only entries with a `priority` field participate in Auto mode.
- **`luigi_workers`** should be 1 for local models (single GPU) and 4 for
  cloud APIs.
- **`is_function_calling_model`** must be `false` for all models. PlanExe uses
  llama_index structured output (JSON schema), not function calling.
- Config files must be valid JSON (no trailing commas, no comments).

## Loading flow

```
PLANEXE_MODEL_PROFILE env var (default: "baseline")
  → model_profile.py resolves profile → filename
    → planexe_config.py finds llm_config/<filename>.json
      → planexe_llmconfig.py loads + substitutes env vars + filters by whitelist
        → llm_factory.py registers pricing + creates LLM instances
```

## Cost tracking

Some providers (OpenRouter) return cost in the usage response. Others (OpenAI,
Anthropic) do not. For those, `track_activity.py` falls back to estimating
cost from token counts using the `pricing` field. The estimation uses
longest-prefix matching on model names, so versioned responses like
`gpt-5-nano-2025-08-07` match the `gpt-5-nano` pricing entry.

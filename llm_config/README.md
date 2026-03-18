# LLM Config

JSON configuration files that define which LLM providers and models PlanExe
can use. Each file is a **profile** targeting a different use case.

## Profiles

| Profile | File | Use case |
|---------|------|----------|
| `baseline` | `baseline.json` | **Default.** Mix of affordable cloud models + local Ollama/LM Studio. |
| `custom` | `custom.json` | User-curated selection for experimentation. |
| `premium` | `premium.json` | Higher-quality cloud models (higher cost). |
| `frontier` | `frontier.json` | Cutting-edge models for maximum quality. |
| `custom` | `anthropic_claude.json` | Anthropic Claude models only. |
| `custom` | `local.json` | LM Studio local models (no API keys). |
| `custom` | `custom_thinking_test.json` | LM Studio with thinking tokens enabled. |

## Selecting a Profile

Set the `PLANEXE_MODEL_PROFILE` environment variable in your `.env` file:

```bash
# Use one of: baseline, premium, frontier, custom
PLANEXE_MODEL_PROFILE=baseline
```

For the `custom` profile, you can override the filename:

```bash
PLANEXE_MODEL_PROFILE=custom
PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=anthropic_claude.json
```

If unset, `baseline` is used.

## Entry Format

Each model entry in a config file looks like:

```json
{
  "openrouter-openai-gpt-4o-mini": {
    "comment": "Created Jul 18, 2024. 128,000 context. $0.15/M input. $0.60/M output.",
    "priority": 2,
    "luigi_workers": 4,
    "class": "OpenRouter",
    "arguments": {
      "model": "openai/gpt-4o-mini",
      "api_key": "${OPENROUTER_API_KEY}",
      "temperature": 0.1,
      "timeout": 60.0,
      "is_function_calling_model": false,
      "is_chat_model": true,
      "max_tokens": 8192,
      "max_retries": 5
    },
    "model_info_url": "https://openrouter.ai/openai/gpt-4o-mini",
    "pricing": {
      "input_per_million_tokens": 0.15,
      "output_per_million_tokens": 0.60
    },
    "pricing_kind": "paid"
  }
}
```

### Key fields

| Field | Required | Description |
|-------|----------|-------------|
| `class` | Yes | Provider class: `OpenRouter`, `OpenAI`, `Anthropic`, `OpenAILike`, `Ollama`, `LMStudio`, `ThinkingAwareOpenAILike` |
| `arguments` | Yes | Passed directly to the llama_index LLM constructor |
| `arguments.model` | Yes | Model identifier sent to the provider API |
| `arguments.api_key` | Yes (cloud) | Environment variable reference, e.g. `${OPENROUTER_API_KEY}` |
| `luigi_workers` | Yes | Max parallel tasks (1 for local, 4 for cloud) |
| `pricing_kind` | Yes | `"paid"` or `"free"` |
| `priority` | No | Auto mode fallback order (lower = tried first) |
| `pricing` | No | Fallback cost estimation rates (USD per million tokens) |
| `comment` | No | Human-readable description |
| `model_info_url` | No | Link to provider model page |

## Supported Providers

### Cloud (require API keys in `.env`)

- **OpenRouter** — Routes to many providers. Set `OPENROUTER_API_KEY`.
- **OpenAI** — Direct OpenAI API. Set `OPENAI_API_KEY`.
- **Anthropic** — Direct Anthropic API. Set `ANTHROPIC_API_KEY`. Also supports
  Claude Code OAuth tokens (`sk-ant-oat*` from `claude setup-token`).
- **OpenAILike** — Any OpenAI-compatible endpoint (e.g. Alibaba DashScope).

### Local (no API keys)

- **Ollama** — Requires [Ollama](https://ollama.ai) installed and running.
  For Docker, set `base_url` to `http://host.docker.internal:11434`.
- **LMStudio** — Requires [LM Studio](https://lmstudio.ai) installed and running.
  For Docker, set `base_url` to `http://host.docker.internal:1234/v1`.

## Auto Mode

When the user selects "Auto" in the UI, PlanExe cycles through models sorted
by `priority` (lowest first). If a model fails, the next one is tried. Only
entries with a `priority` field participate.

## Cost Tracking

The `pricing` field enables cost estimation for providers that don't report
cost in their API responses (OpenAI, Anthropic). Without it, `total_cost`
shows as 0.0 in `activity_overview.json`.

```json
"pricing": {
  "input_per_million_tokens": 0.15,
  "output_per_million_tokens": 0.60
}
```

OpenRouter returns cost directly in usage responses, so the `pricing` field
serves as a cross-check there.

### Anthropic Token Capture

LlamaIndex's Anthropic integration bypasses `self.chat()` and instrumentation
events never fire, so token counts are lost. PlanExe works around this by
patching `httpx.Client.send` to intercept the raw Anthropic API response and
capture the `usage` dict. This hook is installed automatically when any LLM
config is loaded (via `llm_factory._load_llm_config()`). The captured tokens
are used for both `usage_metrics.jsonl` and `activity_overview.json`.

## Adding a New Model

1. Choose the appropriate profile file (or create entries in multiple profiles).
2. Pick a descriptive key (e.g. `openrouter-google-gemini-3`).
3. Set `class` to the provider type and fill in `arguments`.
4. Add `pricing` with rates from the provider's pricing page.
5. Set `pricing_kind` to `"paid"` or `"free"`.
6. Optionally set `priority` if it should participate in Auto mode.
7. Validate the JSON: `python3 -c "import json; json.load(open('llm_config/<file>.json'))"`.

## Filtering by Provider Class

To restrict which provider classes are loaded, set:

```bash
PLANEXE_LLM_CONFIG_WHITELISTED_CLASSES=OpenRouter,Ollama
```

Only entries with a matching `class` will be available.

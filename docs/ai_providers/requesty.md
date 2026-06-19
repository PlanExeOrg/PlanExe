---
title: Requesty - AI provider
---

# Using PlanExe with Requesty

[Requesty](https://requesty.ai/) is an OpenAI-compatible LLM gateway that routes to a large number of cloud models (OpenAI, Google, Anthropic, DeepSeek and more) through a single API key, with caching, failover and cost controls. Because it is OpenAI-compatible, PlanExe talks to it through the existing `OpenAILike` provider class, just like any other OpenAI-compatible endpoint.

As with OpenRouter, prefer cheap, reliable `paid` models. PlanExe does more than 100 LLM inference calls per plan, so each run uses many tokens. Models like [openai/gpt-4o-mini](https://app.requesty.ai/router/list) and [google/gemini-2.5-flash](https://app.requesty.ai/router/list) are inexpensive. Avoid pricey models: with a cheap model a full plan costs well under 0.50 USD, while the newest models can cost far more.

## Quickstart (Docker)

1. Install Docker (with Docker Compose) — no local Python or pip is needed.
2. Clone the repo and enter it:
```
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
```
3. Copy `.env.docker-example` to `.env`, then set your Requesty API key and select the bundled Requesty config file as your custom profile:
```
REQUESTY_API_KEY='your_requesty_key'
PLANEXE_MODEL_PROFILE='custom'
PLANEXE_LLM_CONFIG_CUSTOM_FILENAME='requesty.json'
DEFAULT_LLM='requesty-openai-gpt-4o-mini'   # or requesty-gemini-2.5-flash
```
   The containers mount `.env` and `llm_config/<profile>.json` automatically.
4. Start PlanExe:
```
docker compose up worker_plan frontend_multi_user
```
   - Wait for http://localhost:5001 to come up, submit a prompt, and watch progress with `docker compose logs -f worker_plan`.
   - Outputs are written to `run/<timestamped-output-dir>` on the host (mounted from the containers).
5. Stop with `Ctrl+C` (or `docker compose down`). If you change `llm_config/<profile>.json`, restart the containers so they reload it: `docker compose restart worker_plan frontend_multi_user`. No rebuild is needed for config-only edits.

## Configuration

Visit [Requesty](https://app.requesty.ai/router), create an account, add a small amount of credit (plenty for several plans), and generate an API key in the dashboard.

Copy `.env.docker-example` to a new file called `.env` (loaded by Docker at startup) and insert your Requesty API key:

```
REQUESTY_API_KEY='your_requesty_key'
```

The bundled `llm_config/requesty.json` defines two ready-to-use entries (`requesty-openai-gpt-4o-mini` and `requesty-gemini-2.5-flash`). Select it with `PLANEXE_MODEL_PROFILE=custom` and `PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=requesty.json`, the same mechanism used by `anthropic_claude.json`.

## Troubleshooting

Inside PlanExe, when clicking `Submit`, a new `Output Dir` should be created containing a `log.txt`. Open that file and scroll to the bottom to see any error messages.

When running in Docker, also check the worker logs for 401/429 or connectivity errors:

```
docker compose logs -f worker_plan
```

## How to add a new Requesty model to `llm_config/requesty.json`

For a model to work with PlanExe, it must meet the following criteria:

- Minimum 8192 output tokens.
- Support structured output.
- Reliable and low latency.

Steps to add a model:

1. Copy the model id from the [Requesty model list](https://app.requesty.ai/router/list) (uses the `provider/model` naming convention, e.g. `openai/gpt-4o-mini`).
2. Add a new entry to `llm_config/requesty.json` with `"class": "OpenAILike"` and `"api_base": "https://router.requesty.ai/v1"`.
3. Restart PlanExe to apply the changes.

---

## Next steps

- Learn prompt quality: [Prompt writing guide](../prompt_writing_guide.md)
- Understand output sections: [Plan output anatomy](../plan_output_anatomy.md)

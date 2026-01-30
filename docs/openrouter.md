# Using PlanExe with OpenRouter

[OpenRouter](https://openrouter.ai/) provides access to a large number of LLM models, that runs in the cloud.

Unfortunately there is no `free` model that works reliable with PlanExe.

In my experience, the `paid` models are the most reliable. Models like [google/gemini-2.0-flash-001](https://openrouter.ai/google/gemini-2.0-flash-001) and [openai/gpt-4o-mini](https://openrouter.ai/openai/gpt-4o-mini) are cheap and faster than running models on my own computer and without risk of it overheating.

I haven't been able to find a `free` model on OpenRouter that works well with PlanExe.

## Quickstart (Docker)

1) Install Docker (with Docker Compose) — no local Python or pip is needed now.
2) Clone the repo and enter it:
```
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
```
3) Copy `.env.docker-example` to `.env`, then set your API key and pick a default OpenRouter profile so the worker uses the cloud model by default:
```
OPENROUTER_API_KEY='sk-or-v1-...'
DEFAULT_LLM='openrouter-paid-gemini-2.0-flash-001'   # or openrouter-paid-openai-gpt-4o-mini
```
   The containers mount `.env` and `llm_config.json` automatically.
4) Start PlanExe:
```
docker compose up worker_plan frontend_single_user
```
   - Wait for http://localhost:7860 to come up, submit a prompt, and watch progress with `docker compose logs -f worker_plan`.
   - Outputs are written to `run/<timestamped-output-dir>` on the host (mounted from the containers).
5) Stop with `Ctrl+C` (or `docker compose down`). If you change `llm_config.json`, restart the containers so they reload it: `docker compose restart worker_plan frontend_single_user` (or `docker compose down && docker compose up`). No rebuild is needed for config-only edits.

## Configuration

Visit [OpenRouter](https://openrouter.ai/), create an account, purchase 5 USD in credits (plenty for making a several plans), and generate an API key.

Copy `.env.docker-example` to a new file called `.env` (loaded by Docker at startup).

Open the `.env` file in a text editor and insert your OpenRouter API key. Like this:

```
OPENROUTER_API_KEY='INSERT YOUR KEY HERE'
```

If you edit `llm_config.json` later, restart the worker/frontend containers to pick up the changes: `docker compose restart worker_plan frontend_single_user` (or stop/start). Rebuilds are only needed when dependencies change.

## Troubleshooting

Inside PlanExe, when clicking `Submit`, a new `Output Dir` should be created containing a `log.txt`. Open that file and scroll to the bottom, see if there are any error messages about what is wrong.

When running in Docker, also check the worker logs for 401/429 or connectivity errors:

```
docker compose logs -f worker_plan
```

### "Tool choice 'required' must be specified with 'tools' parameter" (Azure)

PlanExe uses LlamaIndex structured output (Pydantic), which sends `tool_choice: "required"` to the API. When OpenRouter routes to **Azure**, Azure can reject this with:

`Provider returned error ... "Tool choice 'required' must be specified with 'tools' parameter."`

**Fix:** PlanExe forces OpenRouter LLMs to use the *text-based* structured-output path (no `tool_choice`/`tools`), so the provider never receives that request. This is done in `worker_plan_internal/llm_factory.py` by setting `is_function_calling_model = False` after creating the OpenRouter instance. Structured output then uses JSON-in-prompt instead of function calling.

**Workaround:** If you still see the error, prefer the OpenAI provider and disable fallbacks (`"order": ["openai"]`, `"allow_fallbacks": false` in `llm_config.json`). As a last resort, use a different model (e.g. `openrouter-paid-gemini-2.0-flash-001`). The `openrouter-paid-openai-gpt-5-nano` entry in `llm_config.json` sets:

- `"order": ["openai"]` and `"allow_fallbacks": false` (top-level keys inside `arguments`)

so OpenRouter prefers the OpenAI provider and does not fall back to Azure. If you still get the Azure error, ensure your worker/frontend were restarted after editing `llm_config.json` so the config is reloaded. If you get a “no provider” or similar error, the model may only be available via Azure on OpenRouter; in that case switch to `openrouter-paid-gemini-2.0-flash-001` or `openrouter-paid-openai-gpt-4o-mini`, which work with structured output on their default providers.

Report your issue on [Discord](https://planexe.org/discord). Please include info about your system, such as: "I'm on macOS with M1 Max with 64 GB.".

## How to add a new OpenRouter model to `llm_config.json`

The [OpenRouter/rankings](https://openrouter.ai/rankings) page shows an overview of the most popular models. New models are added frequently

For a model to work with PlanExe, it must meet the following criteria:

- Minimum 8192 output tokens.
- Support structured output.
- Reliable. Avoid fragile setups where it works one day, but not the next day. If it's a beta version, be aware that it may stop working.
- Low latency.

Steps to add a model:

1. Copy the model id from the openrouter website.
2. Paste the model id into the `llm_config.json`.
3. Restart PlanExe to apply the changes.

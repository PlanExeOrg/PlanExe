---
title: Mistral - AI provider
---

# Using PlanExe with Mistral

This is for **advanced users** that already have PlanExe working. If you're new to PlanExe, start with [OpenRouter](openrouter.md) instead.

If you want to use Mistral, [OpenRouter has several mistral models](https://openrouter.ai/models?q=mistral). 

## Docker setup for Mistral

Mistral support is not baked into the Docker image by default. You must add the Mistral LlamaIndex extension to the worker, rebuild the image, and supply your API key.

1. Install Docker (with Docker Compose), then clone the repo and enter it:
```
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
```
2. Enable the Mistral client inside the worker image by editing `worker_plan/pyproject.toml`. Under `[project].dependencies`, add or uncomment these lines:
```
"llama-index-llms-mistralai==0.4.0",
"mistralai==1.5.2",
```
   Without this step, the Docker image will not have the `MistralAI` class.
3. Copy `.env.docker-example` to `.env` and add your key:
```
MISTRAL_API_KEY='INSERT-YOUR-SECRET-KEY-HERE'
```
4. Add (or keep) a Mistral entry in `llm_config.json` (example below).
5. Rebuild the images so the new dependencies are baked in:
```
docker compose build --no-cache worker_plan frontend_single_user
```
6. Start PlanExe:
```
docker compose up worker_plan frontend_single_user
```
7) Open http://localhost:7860, go to **Settings**, and pick your Mistral model (e.g., `mistral-paid-large`). If you later tweak only `llm_config.json`, just restart the containers (`docker compose restart worker_plan frontend_single_user`); rebuilds are only needed when dependencies change.

## Why use Mistral?

Mistral can have run your own fine tuned model in the cloud. If you have sensitive business data that you don't want to share with the world, then this is one way to do it.

Create an account on the [mistral.ai](https://mistral.ai/) website and buy 10 EUR of credits.

List of [available models](https://docs.mistral.ai/getting-started/models/models_overview/).

Using the free models, and the API is rate limited to 1 request per second. PlanExe cannot deal with rate limiting and PlanExe does 70-100 requests, so it's likely going to yield errors.

## Create API key

1. Visit [api-keys](https://console.mistral.ai/api-keys).
2. Click `Create new key` and name the new key `PlanExe`.
3. In the `.env` file in the root dir of the PlanExe repo, create a row named `MISTRAL_API_KEY`. Copy/paste the newly created api key into that row.

The `.env` file should look something like the following, with your own key inserted.
```
MISTRAL_API_KEY='AWkg3SxFTLWaPJClbASfv9h3VPItroof'
```

## Edit the `llm_config.json`

The JSON should look something like this:

```json
{
    "mistral-paid-large": {
        "comment": "This is paid. Possible free to use for a limited time. Check the pricing before use.",
        "class": "MistralAI",
        "arguments": {
            "model": "mistral-large-latest",
            "api_key": "${MISTRAL_API_KEY}",
            "temperature": 1.0,
            "timeout": 60.0,
            "max_tokens": 8192,
            "max_retries": 5
        }
    }
}
```

## Use the Mistral model

1. Restart PlanExe
2. Go to the `Settings` tab
3. Select the `mistral-paid-large` model.

---

## Next steps

- Learn prompt quality: [Prompt writing guide](../prompt_writing_guide.md)
- Understand output sections: [Plan output anatomy](../plan_output_anatomy.md)

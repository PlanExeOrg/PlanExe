---
title: Costs and models
---

# Costs and models

PlanExe makes many LLM calls per plan. Model choice affects cost, speed, and quality.

---

## Guidance

- **Most reliable**: paid cloud models via OpenRouter.
- **Lowest cost**: older, smaller models (quality can drop).
- **Local models**: require strong hardware and are slower.
- **Speed matters**: tokens per second can be the difference between minutes and hours.

---

## Typical costs

Costs vary by model and prompt size. PlanExe can use 100+ calls per plan, so avoid expensive models unless you need the highest quality.

## Speed and iteration

Fast models can complete a plan in roughly 10–20 minutes. Slow models may take hours. In practice, it is often better to iterate quickly and generate several candidate plans than to wait for one slow run.

---

## Choosing a provider

- **OpenRouter**: easiest path for most users.
- **Ollama / LM Studio**: good for local experimentation.

See the provider guides:
- [OpenRouter](ai_providers/openrouter.md)
- [Ollama](ai_providers/ollama.md)
- [LM Studio](ai_providers/lm_studio.md)

---
title: Token Counting + Cost Transparency (Raw Provider Tokens)
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Token Counting + Cost Transparency (Raw Provider Tokens)

## Pitch
Instrument the LLM layer so every plan run reports **exact token usage and cost**: input/output tokens, and for reasoning models, **thinking tokens vs final output tokens**. This should be measured from **raw provider responses**, not from structured-parsed output.

## Why
Users want to know: “How much did this plan cost?” and “Why did it cost that much?” This is critical for budgeting, trust, and optimization.

## Requirements

- Count tokens for **all model calls** in a plan run.

- Separate: `input_tokens`, `output_tokens`.

- For reasoning models: `reasoning_tokens` / `thinking_tokens` (provider-specific) separate from final answer tokens.

- **Do not** count tokens after we parse structured output; count from the provider’s raw response/metadata.

## Proposed design
### 1) Add accounting hooks in `llm.py`

- Wrap the provider call and capture:

  - provider usage fields (preferred)

  - or fallback to local tokenizer if provider doesn’t return usage

### 2) Store per-call usage + aggregate per-run

- Per-call record: model, stage, latency, usage, cost

- Per-run rollup: totals + breakdown by stage

### 3) Surface results

- In report: “Cost & Token Usage” section

- In API: `task_status` includes `usage_summary`

## Data model

- `llm_call_usage` (run_id, stage, model, input_tokens, output_tokens, reasoning_tokens, cost_cents, latency_ms)

- `llm_run_usage_summary` (run_id, totals..., created_at)

## Provider notes

- Prefer provider-provided `usage` block.

- For OpenAI/Anthropic/Gemini/OpenRouter: normalize to a common schema.

## Success metrics

- 100% of calls have usage captured (or explicit “unknown”)

- Cost estimate within ±2% of provider billing
# LM Studio API (2025-2026): Structured Output & Thinking Tokens Research

**Date:** March 7, 2026  
**Research Focus:** Programmatic control of structured output (GBNF) and reasoning/thinking tokens via LM Studio REST API

---

## Executive Summary

LM Studio 0.3.0+ provides full REST API support for:
1. **Structured Output** via JSON Schema on `/v1/chat/completions`
2. **Reasoning Tokens** via the new `/v1/responses` endpoint

Both features are production-ready and documented. The `llama-index` integration for LM Studio already uses `force_json` to enable structured output enforcement.

---

## 1. Structured Output / Grammar-Enforced Constrained Decoding

### 1.1 API Endpoint
**Endpoint:** `POST /v1/chat/completions`  
**Version:** Introduced in LM Studio 0.3.0 (August 22, 2024)

### 1.2 Request Parameter: `response_format`

The endpoint accepts the following OpenAI-compatible parameter:

```json
{
  "model": "model-identifier",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "response_schema_name",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "field_name": { "type": "string" }
        },
        "required": ["field_name"]
      }
    }
  }
}
```

**Key Fields:**
- `type`: Must be `"json_schema"`
- `json_schema.name`: Identifier for the schema
- `json_schema.strict`: Boolean (default false) - enforces strict schema compliance
- `json_schema.schema`: The JSON Schema (v7) definition

### 1.3 Implementation Details

**Backend Engine (GGUF models):**
- Uses `llama.cpp`'s grammar-based sampling (GBNF)
- Constrains token generation to valid JSON conforming to schema

**Backend Engine (MLX models):**
- Uses [Outlines](https://github.com/dottxt-ai/outlines) library
- GitHub: [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine)

### 1.4 Response Format

JSON is returned as a string in the standard response field:

```json
{
  "choices": [{
    "message": {
      "content": "{\"field_name\": \"value\"}"
    }
  }]
}
```

Must be parsed client-side with `json.loads()` or equivalent.

### 1.5 Model Compatibility

**Important:** Not all models support structured output.
- **Minimum recommended:** 7B parameters or larger
- **Check:** Model card README for structured output support
- **Note:** Some models handle strict=true better than others

---

## 2. Thinking/Reasoning Tokens

### 2.1 New Endpoint: `/v1/responses`
**Endpoint:** `POST /v1/responses`  
**Version:** Introduced in LM Studio 0.3.9 (January 30, 2025)

### 2.2 Reasoning Control

The new `/v1/responses` endpoint supports reasoning via the `reasoning` parameter:

```json
{
  "model": "deepseek/deepseek-r1",
  "input": "Solve this problem...",
  "reasoning": {
    "effort": "low"  // or "medium", "high"
  }
}
```

**Reasoning Effort Levels:**
- `"low"`: Minimal reasoning tokens
- `"medium"`: Moderate reasoning
- `"high"`: Maximum reasoning depth

### 2.3 Response Handling

**Reasoning Content Separation:**  
LM Studio 0.3.9+ separates reasoning output into a dedicated field (experimental feature):
- Main response in `output_text` field
- Thinking/reasoning tokens in `reasoning_content` field

This allows separate inspection and handling of reasoning chains, particularly useful for models like DeepSeek R1.

### 2.4 Stateful Follow-ups

Responses support conversation state:

```json
{
  "model": "deepseek/deepseek-r1",
  "input": "Multiply it by 2",
  "previous_response_id": "resp_123",
  "stream": false
}
```

---

## 3. Integration with LlamaIndex (PlanExe Codebase)

### 3.1 Current Usage in PlanExe2026

**File:** `/llm_config/local.json`

```json
{
  "lmstudio-qwen-2.5-7b": {
    "class": "LMStudio",
    "arguments": {
      "model_name": "qwen/qwen2.5-7b",
      "base_url": "http://127.0.0.1:1234/v1",
      "force_json": true,  // <- Enables structured output
      "temperature": 0.2,
      "context_window": 32768,
      "num_output": 4096
    }
  }
}
```

### 3.2 LlamaIndex LMStudio Adapter

**File:** `.venv/lib/python3.13/site-packages/llama_index/llms/lmstudio/base.py`

**Key Implementation:**

```python
force_json: bool = Field(
    default=False,
    description="When True, adds response_format json_schema to API calls to suppress thinking-mode output."
)

def _create_payload_from_messages(self, messages, **kwargs):
    payload = {
        "model": self.model_name,
        "messages": [...],
        "options": self._model_kwargs,
        "stream": False,
    }
    if self.force_json:
        # LM Studio requires json_schema (not json_object).
        # Using permissive schema suppresses Qwen's thinking mode.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": False,
                "schema": {"type": "object"}
            }
        }
    return payload
```

**Note:** The adapter also includes `_strip_thinking()` to clean up thinking blocks (`<think>...</think>`) before JSON parsing.

---

## 4. API Changelog Summary (Relevant Entries)

| Version | Feature | Date |
|---------|---------|------|
| 0.3.0 | OpenAI-like Structured Output API (json_schema) | Aug 22, 2024 |
| 0.3.9 | Separate reasoning_content in Chat Completion responses | Jan 30, 2025 |
| 0.3.26 | lms log stream for model I/O inspection | Recent |

---

## 5. Full Request Example

### Chat Completions with Structured Output

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen2.5-7b",
    "messages": [
      {
        "role": "system",
        "content": "You are a JSON API."
      },
      {
        "role": "user",
        "content": "Extract named entities."
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "entities",
        "strict": false,
        "schema": {
          "type": "object",
          "properties": {
            "entities": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "type": {"type": "string"}
                },
                "required": ["name", "type"]
              }
            }
          },
          "required": ["entities"]
        }
      }
    },
    "temperature": 0.2,
    "max_tokens": 1000
  }'
```

### Responses with Reasoning

```bash
curl http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-r1",
    "input": "Solve: 2x + 3 = 11. Show all work.",
    "reasoning": {
      "effort": "high"
    },
    "stream": false
  }'
```

---

## 6. Documentation References

**Official LM Studio Docs:**
1. Structured Output: https://lmstudio.ai/docs/developer/openai-compat/structured-output
2. Chat Completions: https://lmstudio.ai/docs/developer/openai-compat/chat-completions
3. Responses API: https://lmstudio.ai/docs/developer/openai-compat/responses
4. API Changelog: https://lmstudio.ai/docs/developer/api-changelog

---

## 7. Key Findings

✅ **Both features are fully supported:**
- Structured output via `response_format` (json_schema) on `/v1/chat/completions`
- Reasoning tokens via `reasoning` parameter on `/v1/responses`

✅ **PlanExe2026 already uses structured output:**
- `force_json: true` in llm_config/local.json
- LlamaIndex adapter converts this to the proper API call

✅ **Thinking tokens (reasoning) available but separate:**
- Different endpoint (`/v1/responses` vs `/v1/chat/completions`)
- Must use Responses API to access reasoning_content field
- Works with DeepSeek R1 and similar reasoning models

⚠️ **Compatibility notes:**
- Structured output requires models ≥7B parameters
- Reasoning requires specific models (DeepSeek R1, etc.)
- Not all models equally capable at both tasks

---

## 8. Recommendations for PlanExe

1. **Keep current structured output usage** - it's working correctly
2. **If reasoning is needed:**
   - Switch to `/v1/responses` endpoint for reasoning-capable models
   - Handle `reasoning_content` separately from `output_text`
   - Set `effort` level based on inference time budget
3. **Monitor release notes** - LM Studio adds features regularly
4. **Test with models** - capability varies across models and versions

---

**Report generated:** 2026-03-07 at 22:42 UTC  
**Researcher:** Subagent (lmstudio-api-research)

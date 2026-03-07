# LM Studio `/v1/responses` API Migration Proposal for PlanExe

**Date:** 2026-03-07  
**Status:** Research Complete  
**Audience:** PlanExe Technical Leadership  

---

## Executive Summary

LM Studio 0.3.9+ provides a newer `/v1/responses` API that offers significant advantages over the current `/v1/chat/completions` integration:

- **Reasoning control:** Enable/disable thinking tokens, set reasoning effort levels
- **Structured output enforcement:** Better JSON schema compliance via response validation
- **Token transparency:** Explicit separation of thinking tokens from output tokens
- **Conversation context:** Built-in stateful conversation tracking via `previous_response_id`
- **OpenAI API parity:** Mirrors OpenAI's Responses API for future portability

**Bottom line:** Migration is **feasible but requires a custom wrapper** since llama_index doesn't have native `/v1/responses` support yet. Estimated effort: **200-400 LOC** across 2-3 new modules with moderate risk.

---

## 1. Current State: `/v1/chat/completions` (PlanExe Today)

### Implementation Details
- **PlanExe uses:** `llama_index.llms.lmstudio.LMStudio` class
- **Location:** `/Users/macmini/Documents/GitHub/PlanExe2026/worker_plan/worker_plan_internal/llm_factory.py`
- **Call pattern:**
  ```python
  from llama_index.llms.lmstudio import LMStudio
  llm = LMStudio(model_name="<model>", base_url="http://localhost:1234/v1")
  response = llm.chat(messages)  # or llm.complete(prompt)
  ```
- **Endpoint:** `POST /v1/chat/completions`
- **Payload structure:**
  ```json
  {
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0.7,
    "max_tokens": 256
  }
  ```
- **Response:** Standard `ChatCompletionResponse` with `choices[0].message.content`

### Limitations of `/v1/chat/completions`
1. **No reasoning/thinking control:** Cannot separate reasoning from output
2. **No native structured output:** Relies on post-hoc JSON repair (see `json_repair_util.py`)
3. **All tokens treated equally:** No distinction between thinking and output tokens
4. **Stateless:** Each call is independent; conversation state managed externally

---

## 2. New API: `/v1/responses` (LM Studio 0.3.9+)

### What `/v1/responses` Offers

#### 2.1 Reasoning/Thinking Control
**Enable reasoning layers with explicit effort levels:**
```python
{
  "model": "gpt-oss-20b",
  "input": "Solve: 2x + 3 = 11",
  "reasoning": {
    "type": "enabled",      # or "disabled"
    "effort": "low"         # "low", "medium", "high"
  }
}
```

**Output includes explicit thinking tokens:**
```json
{
  "output": {
    "thinking": "Let me solve this step by step...",
    "text": "x = 4"
  },
  "usage": {
    "thinking_tokens": 150,
    "output_tokens": 12
  }
}
```

#### 2.2 Structured Output Enforcement
**Provide JSON schema and get guaranteed compliance:**
```python
{
  "model": "gpt-oss-20b",
  "input": "Extract person info",
  "output_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person_info",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"]
      }
    }
  }
}
```

**Enforcement is at the token level** (via grammar-based sampling in llama.cpp), not post-processing.

#### 2.3 Stateful Conversations
**Reference previous responses for multi-turn context:**
```python
# First call
resp1 = client.post("/v1/responses", json={
  "input": "What is 2+2?",
  ...
})
response_id = resp1.json()["id"]  # e.g. "resp_abc123"

# Follow-up call references previous
resp2 = client.post("/v1/responses", json={
  "input": "Multiply it by 3",
  "previous_response_id": response_id  # ← Built-in context tracking
})
```

#### 2.4 Full OpenAI Responses API Compatibility
LM Studio's `/v1/responses` mirrors OpenAI's Responses API v1, meaning:
- Same endpoint semantics
- Same request/response schema
- Same tool/MCP integration support
- Future migration to OpenAI would be near-trivial

### Request Parameters vs `/v1/chat/completions`

| Feature | `/v1/chat/completions` | `/v1/responses` |
|---------|------------------------|-----------------|
| Input | `messages` array | `input` string (stateful) |
| Output format | `choices[0].message.content` (string) | `output` object with `.text`, `.thinking`, tools |
| Reasoning | ❌ No | ✅ `reasoning.effort` (low/med/high) |
| Structured out | ❌ Post-hoc repair | ✅ `output_format.json_schema` |
| Conversation state | 🔄 Manual (track history) | ✅ `previous_response_id` |
| Streaming | ✅ SSE `data:` lines | ✅ SSE `response.*` events |
| Token counts | Basic | ✅ Thinking vs output split |

---

## 3. Integration Pathways for PlanExe

### Option A: Custom Wrapper (Recommended)
**Create a new llama_index-compatible LLM class that wraps `/v1/responses`**

#### Implementation Strategy
```
worker_plan_internal/
├── llm_util/
│   ├── lmstudio_responses.py      # New: ResponsesAPI client wrapper
│   ├── responses_executor.py       # New: Adapter to integrate with llm_executor.py
│   └── (existing files unchanged)
└── llm_factory.py                  # Modified: Add "lmstudio_responses" variant
```

#### Core Wrapper Class (~150 LOC)
```python
# lmstudio_responses.py
from llama_index.core.llms.base import BaseLLM
from llama_index.core.llms.types import ChatResponse, ChatMessage
import httpx

class LMStudioResponses(BaseLLM):
    """
    Wrapper for LM Studio /v1/responses endpoint.
    Provides llama_index compatibility while leveraging reasoning + structured output.
    """
    
    base_url: str = "http://localhost:1234/v1"
    model_name: str
    enable_reasoning: bool = True
    reasoning_effort: str = "low"  # "low", "medium", "high"
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        # Convert llama_index ChatMessage → raw /v1/responses input
        input_text = self._format_messages_for_responses(messages)
        
        payload = {
            "model": self.model_name,
            "input": input_text,
            "reasoning": {
                "type": "enabled" if self.enable_reasoning else "disabled",
                "effort": self.reasoning_effort
            },
            "stream": False,
            **kwargs
        }
        
        response = httpx.post(
            f"{self.base_url}/responses",
            json=payload
        ).json()
        
        # Extract thinking + output, return ChatResponse
        return self._format_response(response)
    
    def _format_messages_for_responses(self, messages: List[ChatMessage]) -> str:
        # Flatten llama_index messages to single string input
        # (or use previous_response_id for conversation context)
        pass
    
    def _format_response(self, api_response: dict) -> ChatResponse:
        # Map /v1/responses output → llama_index ChatResponse
        # Store thinking in metadata if available
        pass
```

#### Integration Points (~100 LOC)
1. **llm_factory.py:** Add `LMStudioResponses` variant:
   ```python
   from worker_plan_internal.llm_util.lmstudio_responses import LMStudioResponses
   
   # In get_llm():
   if class_name == "LMStudioResponses":
       return LMStudioResponses(**arguments)
   ```

2. **Config file (`llm_config/default.json`):**
   ```json
   {
     "lmstudio-responses": {
       "class": "LMStudioResponses",
       "arguments": {
         "model_name": "gpt-oss-20b",
         "base_url": "http://localhost:1234/v1",
         "enable_reasoning": true,
         "reasoning_effort": "low"
       },
       "priority": 2
     }
   }
   ```

3. **llm_executor.py:** (minimal change) Track thinking tokens separately:
   ```python
   # In LLMAttempt dataclass:
   @dataclass
   class LLMAttempt:
       thinking_tokens: Optional[int] = None  # New field
       output_tokens: Optional[int] = None    # Track separately
   ```

### Option B: Custom HTTP Client (Lower-level)
**Bypass llama_index entirely, use httpx directly**

Pros:
- Full control over payload/response
- No abstraction overhead
- Easier to leverage advanced features (tools, streaming events)

Cons:
- Loses llama_index integrations (callback system, instrumentation)
- More code duplication (~200 LOC wrapper)
- Harder to swap back to `/v1/chat/completions`

### Option C: Wait for llama_index Support
**Community PR to add Responses endpoint to llama_index**

Timeline: Uncertain (likely 6-12 months)  
Risk: Blocks migration indefinitely

---

## 4. How Reasoning Control Works

### Model Capability Requirements
- **Models supporting reasoning:** Larger models (7B+) with explicit reasoning training
- **LM Studio default models:** `openai/gpt-oss-20b` supports reasoning
- **Effort levels:**
  - `low`: Fast, minimal reasoning tokens (useful for simple tasks)
  - `medium`: Balanced reasoning + output
  - `high`: Extensive reasoning (best accuracy, higher token cost)

### Example: PlanExe Usage

**Current (no reasoning):**
```python
llm = get_llm("ollama-llama3.1")
response = llm.complete("Create a 10-step plan for X")
# Output only, no intermediate thinking
```

**Proposed (with reasoning):**
```python
llm = get_llm("lmstudio-responses")
response = llm.complete("Create a 10-step plan for X")
# response.raw contains:
# {
#   "output": {
#     "thinking": "Let me break down...",
#     "text": "1. Do A\n2. Do B..."
#   },
#   "usage": {"thinking_tokens": 200, "output_tokens": 150}
# }
```

### Token Accounting
Crucial for cost tracking:
- **thinking_tokens:** Used for internal reasoning (cheaper with some providers, not counted in OpenAI)
- **output_tokens:** User-visible output (charged in full)
- **Total cost:** Output tokens dominate cost; reasoning is "free" reasoning overhead

---

## 5. Structured Output (JSON Schema Enforcement)

### Current PlanExe Pattern
```python
# Current: Guess and repair
response = llm.chat(messages)  # Hope it returns valid JSON
try:
    data = json.loads(response.message.content)
except JSONDecodeError:
    # Fix broken JSON (see json_repair_util.py)
    data = repair_json(response.message.content)
```

### With `/v1/responses`
```python
# New: Guaranteed valid JSON
schema = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "estimated_duration_hours": {"type": "number"}
    },
    "required": ["steps"]
}

response = llm.chat(
    messages,
    output_format={"type": "json_schema", "json_schema": schema}
)
# response.message.content is guaranteed valid JSON
data = json.loads(response.message.content)  # No repairs needed
```

**Benefit:** Eliminates entire `json_repair_util.py` for Responses API calls.

---

## 6. Implementation Plan & Effort Estimate

### Phase 1: Minimal Viable Integration (Week 1)
**Goal:** Get `/v1/responses` working alongside existing `/v1/chat/completions`

| Task | LOC | Est. Time |
|------|-----|-----------|
| Create `LMStudioResponses` wrapper | 150 | 2 hours |
| Integrate with `llm_factory.py` | 30 | 30 min |
| Add config variant | 20 | 15 min |
| Unit tests for wrapper | 100 | 1.5 hours |
| **Subtotal** | **300** | **4 hours** |

### Phase 2: Token Tracking & Metrics (Week 2)
**Goal:** Instrument thinking vs output tokens

| Task | LOC | Est. Time |
|------|-----|-----------|
| Update `token_counter.py` | 50 | 1 hour |
| Extend `LLMAttempt` dataclass | 20 | 15 min |
| Update `token_metrics_store.py` | 40 | 1 hour |
| **Subtotal** | **110** | **2.25 hours** |

### Phase 3: Structured Output Support (Week 2-3)
**Goal:** Leverage JSON schema enforcement in PlanExe tasks

| Task | LOC | Est. Time |
|------|-----|-----------|
| Create schema builder utilities | 80 | 1.5 hours |
| Refactor a sample task (e.g., `ReviewPlan`) | 100 | 2 hours |
| Integration tests | 100 | 2 hours |
| **Subtotal** | **280** | **5.5 hours** |

### **Total Estimate: ~600 LOC, 12 hours** (1.5 engineer-days)

---

## 7. llama_index Integration Status

### Current Reality
- **llama_index v0.9+:** Has `OpenAILike` base class for custom OpenAI-compatible endpoints
- **LMStudio class:** Only supports `/v1/chat/completions` (confirmed via docs)
- **No `/v1/responses`:** Not in llama_index as of March 2026

### Recommended Approach
**Inherit from `BaseLLM`, not `OpenAILike`**, because:
1. `/v1/responses` API is structurally different (input string vs messages array)
2. Response parsing is different (thinking + output, not just choice message)
3. Cleaner to implement from scratch (~150 LOC) than force-fit OpenAILike

### Future-Proofing
If/when llama_index adds Responses support, migration is trivial:
- Replace `LMStudioResponses` with llama_index's native class
- Backward-compatible interface (both extend BaseLLM)

---

## 8. Risk Assessment

### Low Risk
✅ No impact on existing `/v1/chat/completions` calls (backward compatible)  
✅ Minimal config changes (new variant, not replacement)  
✅ Wrapper is self-contained (no deep coupling)  

### Medium Risk
⚠️ **Conversation state management:** `/v1/responses` uses `previous_response_id` for context. If mismanaged, could lose conversation history. **Mitigation:** Explicit API design for tracking IDs.

⚠️ **Model compatibility:** Reasoning requires capable models. Smaller models may fail. **Mitigation:** Fallback to `/v1/chat/completions` if reasoning unsupported.

⚠️ **LM Studio version lock:** Requires LM Studio 0.3.9+. **Mitigation:** Version check on startup, warn if outdated.

### Low Downside
❌ No breaking changes to existing code  
❌ Completely additive (new config variant only)  
❌ Easy rollback (switch back to `ollama-llama3.1` in config)  

---

## 9. Deliverables & Success Criteria

### Deliverables
1. **`lmstudio_responses.py`** — Production-ready wrapper class
2. **Updated `llm_factory.py`** — Support for `LMStudioResponses` variant
3. **Config template** — Sample `llm_config/default.json` with Responses entry
4. **Unit & integration tests** — ~200 LOC test coverage
5. **Documentation** — Migration guide for PlanExe tasks
6. **Metrics instrumentation** — Track thinking vs output tokens

### Success Criteria
- ✅ `/v1/responses` calls work identically to `/v1/chat/completions` from PlanExe code
- ✅ Reasoning tokens tracked separately in metrics
- ✅ JSON schema enforcement reduces need for `json_repair_util`
- ✅ Zero regression in existing task success rates
- ✅ Config switch between old/new API is one-line change

---

## 10. Migration Roadmap (Post-Implementation)

### Phase A: Parallel Operation (Weeks 1-2)
- Both APIs available simultaneously
- Tasks can opt-in to Responses via config
- Measure performance delta

### Phase B: Gradual Rollout (Weeks 3-4)
- Priority-1 tasks (plan generation, validation) → Responses API
- Monitor token usage, reasoning quality
- A/B test output quality (reasoning-enabled vs disabled)

### Phase C: Full Transition (Weeks 5+)
- Remaining tasks → Responses API
- Deprecate `/v1/chat/completions` for new code
- Archive old fallback paths (keep for legacy safety)

---

## 11. Comparison Table: API Features

| Feature | `/v1/chat/completions` | `/v1/responses` | Impact |
|---------|------------------------|-----------------|--------|
| **Reasoning** | ❌ | ✅ (effort control) | Better interpretability, token transparency |
| **Structured output** | ❌ (post-hoc) | ✅ (token-level) | More reliable JSON, less repair logic |
| **Conversation state** | 🔄 (manual) | ✅ (prev_id) | Simpler multi-turn, less context leakage |
| **Streaming** | ✅ | ✅ (SSE events) | Feature parity |
| **Tool calling** | ✅ | ✅ | Feature parity |
| **Thinking tokens** | — | ✅ (separate count) | Cost accounting, reasoning transparency |
| **llama_index support** | ✅ | ❌ (custom needed) | +1 custom class, negligible complexity |

---

## 12. Recommendations

### Primary Recommendation: **Option A (Custom Wrapper)**
1. ✅ Minimal risk (additive, no breaking changes)
2. ✅ Maximum control (leverage all Responses API features)
3. ✅ Fastest path (12 hours vs 6+ months if waiting for llama_index)
4. ✅ Future-proof (when llama_index adds support, drop-in replacement)

### Implementation Priority
1. **Week 1:** Implement wrapper + factory integration (Phase 1)
2. **Week 2:** Token tracking, structured output utilities (Phases 2-3)
3. **Week 3+:** Gradual migration of PlanExe tasks

### Immediate Actions
- [ ] Confirm LM Studio 0.3.9+ deployment on target environments
- [ ] Prototype JSON schema builder for common PlanExe output types
- [ ] Plan token accounting changes (csv export format, dashboard updates)
- [ ] Brief task owners on API switch timeline

---

## 13. References

- **LM Studio docs:** https://lmstudio.ai/docs/advanced/structured-output
- **Responses endpoint:** https://lmstudio.ai/docs/developer/openai-compat/responses
- **OpenAI Responses API:** https://platform.openai.com/docs/api-reference/responses
- **llama_index LMStudio class:** Implements only `/v1/chat/completions` currently
- **PlanExe LLM factory:** `/Users/macmini/Documents/GitHub/PlanExe2026/worker_plan/worker_plan_internal/llm_factory.py`

---

## Questions & Next Steps

**Q: Will this break existing tasks?**  
**A:** No. New API is a config variant. Existing tasks use current LLM unchanged.

**Q: What if reasoning is unsupported on a model?**  
**A:** Wrapper can gracefully disable reasoning, fall back to standard generation.

**Q: How do we handle thinking token costs?**  
**A:** Track separately in metrics. For now, they're "free" overhead; charge if OpenAI parity needed.

**Q: Timeline to production?**  
**A:** Phase 1 done in 1 week, Phase 2-3 over next 2 weeks. Can go live Phase 1 immediately.

---

**Proposal prepared by:** Subagent Research Session  
**Contact:** Research findings compiled from LM Studio v0.3.9 API docs + PlanExe codebase analysis

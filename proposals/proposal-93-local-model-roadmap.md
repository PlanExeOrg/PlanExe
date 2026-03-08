# Proposal 93: Local Model Roadmap — March 2026

**Author:** PlanExe Core Team  
**Date:** March 7, 2026  
**Status:** Accepted  
**Target:** VoynichLabs/PlanExe2026 main branch

---

## Executive Summary

PlanExe achieved its first complete pipeline run on local hardware using Qwen 3.5-9B, executing all 63 tasks with zero failures. This milestone unlocks offline-first operation at zero API cost on consumer hardware (Mac Mini M4 Pro, 64GB). The root causes of all prior local model failures have been identified and remediated. This proposal documents the breakthrough, the technical fixes, and the roadmap for multi-model comparison and production hardening.

---

## Milestone Achieved: March 7, 2026

### The Run
- **Model:** Qwen 3.5-9B (quantized, via LM Studio)
- **Hardware:** Mac Mini M4 Pro (14-core CPU, 64GB unified memory)
- **Tasks:** 63-task PlanExe pipeline
- **Result:** 0 failures, 100% task completion rate
- **Cost:** $0.00 (fully offline)

### Significance
PlanExe can now execute its entire planning pipeline without cloud dependencies, internet connectivity, or API billing. This establishes a baseline for local-first AI planning workflows and opens the door to edge deployment, privacy-sensitive use cases, and cost-neutral scaling.

---

## Root Cause Analysis: Why Local Models Failed Until Now

All prior local model attempts failed silently or incompletely due to four specific issues:

### 1. Wrong Adapter Class (LMStudio vs OpenAILike)
The original integration used `class: LMStudio` from llama_index. This class sends JSON schemas as plain text in the user message, with no grammar enforcement. The local model would "see" the schema but had no mechanism to enforce it in structured output.

**Fix:** Switched to `class: OpenAILike` with `should_use_structured_outputs: true`, which sends a proper `response_format: json_schema` payload to LM Studio. LM Studio enforces this via Outlines grammar, guaranteeing JSON structure at inference time.

### 2. Silent 60-Second Timeout
The OpenAILike adapter configuration used `request_timeout` as a field name. OpenAILike ignores this field; the correct field is `timeout`. Without an explicit timeout, the adapter defaulted to 60 seconds. Long-running tasks would hang and fail silently.

**Fix:** Corrected field name from `request_timeout` to `timeout` in local.json adapter config.

### 3. Pydantic Enum Fields → $defs/$ref Issues
Pydantic's default JSON schema serialization for Enum fields generates `$defs` and `$ref` pointers. Outlines grammar cannot resolve these refs at runtime, causing schema validation to fail on tasks with Enum-typed fields.

**Fix:** Implemented FlatSchemaModel pattern across 9 core task files, converting Enum fields to Literal[...] unions which expand inline and require no refs.

### 4. GLM 4.7 Flash MLX Thinking Mode
When enabled, GLM 4.7 Flash puts all output in `reasoning_content` and leaves `content` empty. This broke response parsing, even when schema structure was correct.

**Fix:** Disable thinking mode via LM Studio preset configuration.

---

## Fixes Shipped (March 7, 2026)

1. **Adapter Config** (`config/local.json`)
   - Switched to OpenAILike class
   - Added `should_use_structured_outputs: true`
   - Corrected timeout field

2. **Literal/Enum Pattern** (9 task files)
   - Replaced Pydantic Enum with Literal[...] unions
   - Removed all $defs/$ref dependencies
   - Verified schema flattening

3. **CI Parity Test** (`tests/test_local_model_parity.py`)
   - 63-task end-to-end test using local model
   - Passes with zero failures (baseline established)

4. **LM Studio Preset** (`presets/planexe-agents.yml`)
   - Optimized for task completion
   - Disabled thinking mode
   - Structured output enforcement

---

## What Was Blocking Local Models (Technical Deep Dive)

| Issue | Impact | Root Cause | Solution |
|-------|--------|-----------|----------|
| LMStudio class | Schema not enforced | No grammar support | Switch to OpenAILike + json_schema format |
| request_timeout field | 60s silent timeout | OpenAILike ignores field | Rename to timeout |
| Pydantic Enum → $defs | Schema validation fails | Outlines can't resolve refs | Use Literal[...] unions instead |
| GLM thinking mode | Empty content field | Reasoning takes all output | Disable via preset |

---

## Roadmap: Next Steps

### Phase 1: Multi-Model Baseline Comparison (Weeks 1–2)

#### 1.1 GLM 4.7 Flash Full Pipeline Run
- Disable thinking mode via LM Studio preset
- Execute full 63-task pipeline
- Compare latency, accuracy, and token efficiency vs. Qwen 9B baseline

#### 1.2 Comparison Report
Likert-scale scoring matrix:
- **Models tested:** Qwen 3.5-9B, Qwen 35B, GLM 4.7 Flash
- **Baseline cloud:** OpenRouter Gemini 3.1 Flash Lite
- **Metrics:** Task completion rate, avg latency, reasoning quality, cost per run, memory footprint
- **Deliverable:** `reports/local-vs-cloud-comparison-march-2026.md`

---

### Phase 2: Schema Hardening (Weeks 2–3)

#### 2.1 FlatSchemaModel / $defs Pipeline-Wide Audit
Scan all 63 task definitions for:
- Remaining Enum fields (should be converted to Literal)
- Nested dataclasses with defaults that might trigger $defs
- Union types that don't flatten to Outlines grammar
- Document findings in audit report

#### 2.2 null Guard PR
- Branch: `fix/structured-response-null-guard`
- Utility: `require_raw()` — ensures structured output never returns None
- Target: 33 task files across pipeline
- Prevent silent null-reference errors in task output

---

### Phase 3: Local Model Infrastructure (Weeks 3–4)

#### 3.1 LM Studio /api/v1/chat Migration
- Current: `/v1/chat/completions` (OpenAI compatibility layer)
- Target: `/api/v1/chat` (native LM Studio endpoint)
- Rationale: Better latency, fewer abstraction layers, direct feature access
- Update OpenAILike adapter to support endpoint override

#### 3.2 Hub Preset Publishing
- Once PR #192 merges, publish preset to lmstudio.ai/82deutschmark/planexe-agents
- Community-discoverable, one-click install from LM Studio UI
- Includes: all task schema fixes, timeout tuning, thinking mode disable, Outlines grammar

---

### Phase 4: Production Readiness (Weeks 4–5)

#### 4.1 Multi-Model Fallback Strategy
- **Primary:** GLM 4.7 Flash (reasoning + speed)
- **Fallback 1:** Qwen 3.5-9B (proven stable, lower latency)
- **Fallback 2:** Cloud (OpenRouter Gemini 3.1 Flash Lite, if all local models unavailable)
- Implement graceful degradation with retry logic and telemetry

---

## Technical Debt & Known Risks

1. **Schema Refs in Non-OpenAILike Tasks** — Phase 2 audit will identify scope
2. **LM Studio Latency Under Load** — Compare sustained throughput with cloud baseline
3. **Quantization Quality** — 9B vs 35B vs GLM reasoning trade-offs not yet quantified
4. **Offline Inference Infrastructure** — LM Studio dependency; explore Ollama/vLLM alternatives later

---

## Success Criteria

- ✅ **Phase 1:** Qwen 9B baseline established (63/63 tasks, 0 failures)
- 🔄 **Phase 2:** GLM 4.7 Flash achieves 63/63 completion; comparison report published
- 🔄 **Phase 3:** FlatSchemaModel audit complete; null guards merged
- 🔄 **Phase 4:** Hub preset published; multi-model fallback operational
- 🔄 **Phase 5:** PlanExe deployable offline on M4 Pro baseline hardware; docs published

---

## Hardware Baseline

- **CPU:** Apple M4 Pro (14-core)
- **Memory:** 64GB unified
- **Storage:** 1TB SSD (model cache)
- **OS:** macOS 26.3
- **Runtime:** LM Studio 0.x + OpenAI-compatible adapter

---

## Timeline

| Phase | Weeks | Owner | Deliverables |
|-------|-------|-------|--------------|
| Phase 1 | 1–2 | Core team | GLM run, comparison report |
| Phase 2 | 2–3 | Core team | Audit report, null guard PR |
| Phase 3 | 3–4 | Core team | LM Studio native endpoint, preset publish |
| Phase 4 | 4–5 | Core team | Fallback logic, production docs |

---

## References

- **Qwen 3.5-9B:** https://huggingface.co/Qwen/Qwen2.5-9B
- **GLM 4.7 Flash:** https://huggingface.co/THUDM/glm-4-9b
- **LM Studio:** https://lmstudio.ai/
- **Outlines Grammar:** https://github.com/outlines-ai/outlines
- **OpenAI Chat Completions API:** https://platform.openai.com/docs/api-reference/chat/create

---

## Approval & Sign-Off

- **Date:** March 7, 2026
- **Status:** Ready for review by VoynichLabs/PlanExe2026 maintainers
- **Next Step:** Merge to main; begin Phase 1 execution


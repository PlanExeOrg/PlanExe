# Proposal 92 — Migrate Enum Fields to Literal in LLM Output Schemas

**Status:** Draft  
**Author:** EgonBot  
**Date:** 2026-03-08  
**Related:** PR #187 (establishes pattern), PR #184 (OpenAILike adapter fix)

---

## Problem

Python `str(Enum)` fields in Pydantic models produce `$defs` + `$ref` entries in the JSON schema emitted by `model_json_schema()`. Example:

```json
{
  "properties": {
    "purpose": { "$ref": "#/$defs/PlanPurpose" }
  },
  "$defs": {
    "PlanPurpose": {
      "enum": ["personal", "business", "other"],
      "type": "string"
    }
  }
}
```

Some LLM backends cannot resolve `$ref` references in JSON schemas:
- **LM Studio MLX (Outlines grammar compiler):** Returns `content: ""` when schema contains `$ref`.
- **Other grammar-based backends (llama.cpp GGUF, vLLM, etc.):** May or may not resolve `$ref` depending on version.

This is not an LM Studio-specific issue — it is a JSON Schema compliance gap that any grammar-constrained backend can exhibit. The root cause is in how Pydantic serializes Enum types, not in any specific backend.

Using `Literal["a", "b", "c"]` instead of the Enum class as the field type produces a flat, self-contained schema:

```json
{
  "properties": {
    "purpose": {
      "enum": ["personal", "business", "other"],
      "type": "string"
    }
  }
}
```

No `$defs`, no `$ref`. All backends handle this correctly.

---

## Solution

For every Pydantic model used as a structured LLM output schema that contains an Enum field:

1. **Keep the `str(Enum)` class** — it is used for downstream comparisons (`if x == MyEnum.value`) and provides documentation value.
2. **Change only the Pydantic field type** from `field: MyEnum` to `field: Literal["a", "b", "c"]`.
3. Add `from typing import Literal` to the file.
4. No other code changes needed — `str(Enum)` compares equal to plain strings, so all downstream logic continues to work unchanged.

This pattern was established in PR #187 (`identify_purpose.py`).

---

## Files to Migrate

| File | Enum Class | Field | Literal Values |
|------|-----------|-------|----------------|
| `assume/identify_purpose.py` | `PlanPurpose` | `PlanPurposeInfo.purpose` | `"personal"`, `"business"`, `"other"` | ✅ Done in PR #187 |
| `assume/identify_plan_type.py` | `PlanType` | `DocumentDetails.plan_type` | `"digital"`, `"physical"` |
| `assume/identify_risks.py` | `LowMediumHigh` | `RiskItem.likelihood`, `RiskItem.severity` | `"low"`, `"medium"`, `"high"` |
| `plan/data_collection.py` | `SensitivityScore` | `AssumptionItem.sensitivity_score` | `"low"`, `"medium"`, `"high"` |
| `team/enrich_team_members_with_contract_type.py` | `ContractType` | `TeamMember.contract_type` | `"full_time_employee"`, `"part_time_employee"`, `"independent_contractor"`, `"agency_temp"`, `"other"` |
| `lever/deduplicate_levers.py` | `LeverClassification` | `LeverDecision.classification` | `"keep"`, `"absorb"`, `"remove"` |
| `lever/focus_on_vital_few_levers.py` | `StrategicImportance` | `EnrichedLever.strategic_importance` / `LeverAssessment.strategic_importance` | `"Critical"`, `"High"`, `"Medium"`, `"Low"` |
| `document/filter_documents_to_create.py` | `DocumentImpact` | `DocumentItem.impact_rating` | `"Critical"`, `"High"`, `"Medium"`, `"Low"` |
| `document/filter_documents_to_find.py` | `DocumentImpact` | `DocumentItem.impact_rating` | `"Critical"`, `"High"`, `"Medium"`, `"Low"` |

---

## What Does NOT Change

- All `str(Enum)` class definitions remain in place.
- All downstream comparison code (`if x == MyEnum.foo`) continues to work — `str(Enum)` compares equal to plain strings.
- All `.value` accesses continue to work where they exist.
- No logic, no behaviour, no output format changes.
- No superclass, no Pydantic internals, no adapter-specific code.

---

## Risks

- **Low.** Only field type annotations change. All behaviour is preserved.
- **Pydantic/LlamaIndex upgrades:** A future Pydantic version that changes how `Literal` is serialized could affect this. However, `Literal` producing an inline `enum` array is the standard JSON Schema specification — this is far more stable than the `$defs`/`$ref` behaviour.
- **Missing files:** There may be additional Enum-in-schema fields introduced in future tasks. Recommend adding a CI lint rule: "Enum fields in Pydantic BaseModel classes used with `as_structured_llm` must use `Literal` type annotations."

---

## Implementation Plan

1. One PR per logical group (or one combined PR for all 8 remaining files).
2. Each commit: minimal diff, `from typing import Literal` added, one field annotation changed.
3. No squashing needed — individual commits are readable.

---

## References

- PR #187: First application of this pattern (`identify_purpose.py`)
- PR #184: Root cause fix (`OpenAILike` adapter sends `response_format: json_schema`)
- `VoynichLabs/PlanExe2026` branch: `fix/local-openai-like-structured-output`

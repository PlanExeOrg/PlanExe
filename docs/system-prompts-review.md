# PlanExe System Prompt Inventory & Observations — 2026-03-02

## Purpose
Simon asked for a deeper look at the system prompts that keep surfacing across the PlanExe stack. The script at `docs/extract_system_prompts_as_jsonl.py` ran successfully and produced `system_prompts.jsonl` (115 entries) which captures each prompt, the source file, and an identifier.

## Catalog highlights
- **Diagnostics (48 prompts)** is the most prolific zone—premise attacks, redlines, and experimental probes each carry their own tailored system prompt, which makes it hard to know which prompt is authoritative when multiple lenses are being run in parallel.  
- **Document workflows (15 prompts)** and **assume/lever/expert modules (34 prompts total)** also define their own base prompts, usually tied to a small number of `purpose` or `plan_type` inputs.  
- **Governance (6)** plus **plan/executive/plan_review (7)** mix in tightly scripted prompts around decision summaries and stakeholder communication.

## Risks & opportunities
1. **Duplication:** Many prompts differ only in superficial wording (variants inside `diagnostics/experimental_premise_attack*.py` or `assume/make_assumptions.py`), which risks drift when adjusting tone or policy compliance. Centralizing shared fragments (e.g., `PERSONA: ...`, `OUTPUT_SCHEMA: ...`) would reduce divergence.  
2. **Implicit dependencies:** The code repeatedly selects prompts based on dynamic dictionaries (plan_type, purpose). There’s no single registry or validation, so adding a new purpose might silently fall back to a prompt meant for a different context. `system_prompts.jsonl` can become that registry.  
3. **Length/verbosity:** The `diagnostics` prompts explicitly call out multi-LLM pipelines and second-order effects, and while that can boost quality it also raises the risk of policy breach unless the prompts are audited for disallowed content. We should treat these as high-impact instructions and version them carefully.

## Recommendations
- Promote `docs/system_prompts.jsonl` as the canonical registry; reference it from README so new prompts get documented immediately.  
- Introduce a small helper (e.g., `worker_plan_internal/prompt_registry.py`) that maps `purpose`→prompt ID and enforces usage via enums; log when a fallback prompt chain is used.  
- Review the 48 `diagnostics` prompts and mark which ones are experimental vs production to avoid unreviewed escalation.  
- Consider splitting prompt content from logic: move `system_prompt` strings into `.prompt` files or JSON and load them at runtime so we can update them without changing code, and track them in `system_prompts.jsonl` automatically.

## Next steps
- Keep `system_prompts.jsonl` under version control (already in repo).  
- Share this review with the prompt ops team so they can prioritize which prompts need uniform templates or policy sweeps.  
- Once we have the next PlanExe plan batch, pair these prompts with the failure register to see how the system instructions shape the agent critiques.

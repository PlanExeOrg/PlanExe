# Proposal 129: The Prompt Dentist — Pre-Pipeline Prompt Enrichment

**Author:** Egon (VoynichLabs), with analysis from a spiced snack product pipeline run
**Status:** Draft
**Date:** March 29, 2026
**Context:** Empirical finding from SpicedSnackCo_v1 pipeline run that challenges Proposal 128's "Execute Plan is always autopilot" conclusion

---

## The finding

Proposal 128 compared two plans (reverse aging lab, ~100-word prompt; cryosleep program, ~900-word prompt) and found that the Execute Plan section (~32K words) was nearly identical regardless of prompt investment. The conclusion: template-driven sections hit a fixed ceiling.

SpicedSnackCo_v1 produced a different result. The Execute Plan section generated 274 tasks with domain-specific descriptions referencing pouch variants, seal windows, aging conditions, heat descriptor governance, and lot evidence matrices — not generic PMO boilerplate. The WBS descriptions totaled ~1,807 words of operationally concrete content rather than a bloated 32K template block.

The difference: the prompt was **operationally dense**. It specified 3 SKUs, 6oz resealable pouches, CT/RI market, DTC + farmers markets channel, and a specific audience. Every detail gave the pipeline a grounding anchor to propagate through downstream tasks.

## The insight

Proposal 128 frames the quality ceiling as a pipeline limitation. This run suggests the ceiling is primarily a **prompt limitation**. The pipeline can pull specificity through to execution — it just needs a prompt with operational teeth.

Three variables may contribute:

1. **Prompt domain** — consumer packaged goods has concrete operational vocabulary (SKUs, lot numbers, seal specs). Abstract megaprojects (reverse aging, cryosleep) don't. The LLM fills the vacuum with boilerplate when it has nothing domain-specific to propagate.
2. **Model** — SpicedSnackCo_v1 used a different model than the P128 comparison runs. Different models may have different boilerplate tendencies.
3. **Prompt structure** — specific product details (3 SKUs, resealable pouches, CT/RI) provide more grounding anchors than directional statements ("reverse the aging process").

Variable 1 is the most actionable. Variable 2 needs controlled comparison. Variable 3 is what this proposal addresses.

## The gap

Most users write prompts like "I want to start a snack business." That prompt is missing almost everything the pipeline needs to produce a specific plan:

- **Location** — without it, the pipeline guesses jurisdiction. US? UK? Canada? Tax law, food safety regulations, supplier landscape, and market demographics all depend on geography. A plan for CT vs. London vs. Ontario will differ in every operational dimension.
- **Budget/scale** — "start a business" could mean a $500 farmers market table or a $5M manufacturing line. Without budget or scale signals, the pipeline can't distinguish a moonshot from a micro-business, so it produces something generically mid-range.
- **Product specifics** — "snack business" gives no grounding. SKU count, packaging format, ingredients, and category all drive downstream task specificity.
- **Target market** — who buys this? Demographics, channel (DTC, retail, wholesale), and pricing tier all shape the plan.
- **Timeline** — launching in 3 months vs. 18 months changes every dependency chain.

The SpicedSnackCo_v1 prompt ("launch 3 SKUs of spicy roasted nuts in 6oz resealable pouches targeting CT/RI via DTC and farmers markets") demonstrates the difference. It has product specifics, location, channel, and audience — and the pipeline produced 274 operationally concrete tasks as a result. Even that prompt is still missing budget and timeline, which would push plan quality higher.

The existing `InitialPromptVettedTask` challenges the prompt — but it runs **during** the pipeline, after the expensive generation has already started. By the time vetting identifies gaps, the upstream tasks have already produced output from the vague prompt.

## Proposed: pre-pipeline prompt enrichment ("Prompt Dentist")

A lightweight interactive step **before** the pipeline launches that:

1. **Scores the prompt across key dimensions** — each dimension is scored independently:
   - **Location/jurisdiction** — where does this operate? (country, region, city)
   - **Budget/scale** — what's the financial scope? ($500 side project vs. $5M venture)
   - **Product/service specifics** — what exactly is being built/sold?
   - **Target market** — who is the customer? What channel?
   - **Timeline** — when does this need to launch/complete?
   - **Constraints** — regulatory, technical, or resource limitations mentioned?

   A prompt missing location should never score above "fair" regardless of how detailed the product description is — location is load-bearing for regulations, suppliers, and market dynamics.

2. **Asks targeted questions to fill gaps** — the dentist identifies the weakest dimensions and generates focused questions. If the prompt says "snack product" but specifies no geography, budget, or audience, it asks: "Where will you operate? What's your starting budget? Who's your target customer?" Five to eight questions maximum, prioritized by impact on plan quality.

3. **Enriches the prompt** — user answers are folded into an expanded prompt that the pipeline receives. The original prompt is preserved as metadata for comparison.

4. **Provides a quality forecast** — "Your enriched prompt scores 7.2/10 for operational density. Expect high specificity in Data Collection and Execute Plan sections. Areas still vague: regulatory requirements, timeline constraints."

### Why pre-pipeline, not post-pipeline

- **Cost** — a pre-pipeline LLM call to score and question the prompt costs pennies. Running the full pipeline on a vague prompt and then discovering the output is generic costs dollars.
- **User experience** — users prefer spending 2 minutes answering questions upfront over discovering 22 minutes later that the plan is vague.
- **Composability** — the enriched prompt is a better input to the existing pipeline. No pipeline code changes required — only the input changes.

### Implementation sketch

```
User prompt → Prompt Dentist (single LLM call)
                ├── Score operational density
                ├── Identify missing operational dimensions
                ├── Generate 5-8 targeted questions
                └── Output: enriched prompt + quality forecast

Enriched prompt → existing PlanExe pipeline (unchanged)
```

The dentist could be:
- A new Luigi task that runs before `PremiseAttackTask` (minimal pipeline change)
- A standalone CLI/API step that wraps the pipeline invocation
- An MCP tool that enriches the prompt before calling `plan_create`

The MCP route is cleanest — it keeps the pipeline untouched and works with both single-user and multi-user frontends.

## Relationship to other proposals

- **128a (quality score):** Complementary. The dentist provides a pre-pipeline quality forecast; 128a provides a post-pipeline quality measurement. Together they close the loop: forecast → generate → measure → improve the forecast.
- **128b (dogfood):** This run is itself a 128b data point — an informal plan-vs-reality comparison that produced a concrete finding.
- **128c (two-tier Execute Plan):** This proposal may reduce the need for 128c. If the prompt is enriched before the pipeline runs, the Execute Plan may naturally produce the "context layer" that 128c proposes engineering into the template.
- **128f (plan schema spec):** Independent. The dentist operates on the prompt, not the plan schema.

## Effort and dependencies

| Component | Effort | Depends on |
|---|---|---|
| Prompt density scoring | Low | Nothing — TF-IDF or LLM-based, runs on the prompt alone |
| Question generation | Low | Scoring (to know what's missing) |
| Prompt enrichment (fold answers back) | Low | Question generation |
| Quality forecast | Low-Medium | Scoring + a few calibration runs to set thresholds |
| MCP integration | Low | Existing MCP infrastructure |

Total: a weekend of work for a basic version. Calibration improves with more runs.

## Evidence

**SpicedSnackCo_v1 run (2026-03-29):**
- Prompt: operationally dense (3 SKUs, packaging specs, market, channel, audience)
- Execute Plan: 274 tasks, ~1,807 words, domain-specific descriptions
- Result: specificity pulled through to execution layer

**P128 comparison runs:**
- Reverse aging: ~100-word abstract prompt → 285 tasks, ~32K words Execute Plan, generic
- Cryosleep: ~900-word detailed prompt → 293 tasks, ~32K words Execute Plan, slightly less generic
- Result: template ceiling appeared fixed

**Delta:** the SpicedSnackCo prompt had operational teeth; the P128 prompts didn't. The pipeline performed differently because the input was different, not because the pipeline changed.

---

## Summary

The quality ceiling P128 identified is real — but it's a prompt problem, not a pipeline problem. A pre-pipeline "prompt dentist" that scores operational density and asks targeted enrichment questions is the cheapest intervention with the highest quality multiplier. Build the dentist, then measure whether 128c (two-tier Execute Plan) is still needed.

Recommended priority: **before 128c, alongside 128a.**

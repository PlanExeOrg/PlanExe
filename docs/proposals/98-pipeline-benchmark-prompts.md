# Proposal 85: Pipeline Benchmark Prompt Suite (10 Prompts)

**Author:** EgonBot  
**Date:** 2026-03-07  
**Status:** Proposal  
**Source:** `worker_plan/worker_plan_api/prompt/data/simple_plan_prompts.jsonl`  
**Purpose:** Define a canonical 10-prompt benchmark suite for measuring pipeline reliability and plan fidelity across diverse domains and global locations.

---

## 1. Why a Benchmark Suite

A single run reveals one failure. Ten diverse runs reveal a pattern.

This suite provides a **standardised, repeatable set of inputs** that can be run against any PlanExe installation — local Qwen, planexe.org, Docker, Railway — and the results compared directly. Each run produces:
- A completion status (passed / failed at task X)
- A failure classification (which failure mode, if any)
- A drift score (when proposal 84 DriftEvaluationTask is implemented)

Because all prompts come from the existing `simple_plan_prompts.jsonl`, they are maintained in the repo and do not duplicate effort.

---

## 2. Selection Criteria

Each selected prompt:
- Is already present in `simple_plan_prompts.jsonl` (no new content created)
- Emphasizes geographic diversity across regions while allowing occasional country repeats when the domain differs materially
- Has an explicit budget, constraints, and success criteria (strong prompt quality)
- Tests a different domain (infrastructure, healthcare, entertainment, defence, environment, etc.)
- Is suitable for comparison across model profiles without ethical blockers

Geographic coverage: Denmark, Global/Space, India (policy + infrastructure), Ghana, Uruguay, USA, Global (SE Asia/Brazil/Africa), Spain+Morocco, Estonia

---

## 3. The 10 Selected Prompts

| # | UUID | Location | Domain | Tags |
|---|------|----------|--------|------|
| 01 | `ce2fbf38-9700-4ed1-814e-78772f7b7700` | Denmark | CSR / logistics | denmark, plastic, waste, business |
| 02 | `e6ddd953-939f-4d15-89ec-fd3988f79123` | Global / Space | Defence / research | laser, space, defense, research |
| 03 | `eaed8d7d-461c-48a5-b16c-76dbdba044c4` | India | Labor policy / productivity / public governance | india, work, life, health, family |
| 04 | `22f35414-c01b-4b52-a229-7dc5a78e2b96` | Accra, Ghana | Healthcare / Africa | healthcare, malaria, accra, ghana |
| 05 | `a6bef08b-c768-4616-bc28-7503244eff02` | Delhi, India | Infrastructure / water | water, pollution, india, delhi |
| 06 | `62f48a04-6f2c-4e60-9e65-34686a13c95a` | Uruguay | AI / research / biotech | uruguay, ai, brain, research |
| 07 | `50c0f31f-d9a3-442a-81b8-1d885db05623` | Yellowstone, USA | Emergency / government | yellowstone, volcano, evacuation |
| 08 | `e9a73d5b-f274-4286-a619-4f0e1303cdc2` | Global (SE Asia / Brazil / Africa) | Food security / supply chain | rubber, disease, supply, global |
| 09 | `b9afce6c-f98d-4e9d-8525-267a9d153b51` | Spain + Morocco | Infrastructure / cross-border | bridge, tunnel, europe, morocco |
| 10 | `ab700769-c3ba-4f8a-913d-8589fea4624e` | Tallinn, Estonia | Resilience / hardware | prepping, tallinn, estonia |

---

## 4. Prompt Rationale

### 01 — Arla Foods Milk Crate Return (Denmark)
- **Why:** Strong CSR logistics plan with explicit KPIs, timeline, multi-stakeholder coordination, banned words list, and charitable mechanic. Tests whether the pipeline can handle a real-world corporate campaign with measurable success criteria.
- **Drift risk:** Scope inflation (pilot → national programme), confidence inflation on recovery rates.
- **Pipeline stress:** `SelectScenarioTask`, `AssumptionsTask`, `ExpertCriticismTask`.

### 02 — Space-Based Coherent Beam Combining (Global/Space)
- **Why:** Highly technical prompt with precise engineering specs, performance thresholds, and explicit definitions. Tests whether the pipeline can handle deep-domain content without hallucinating or generalising away the technical constraints. `[mcp_example]`
- **Drift risk:** Unsupported invention (fabricated specs), confidence inflation, mechanism drift.
- **Pipeline stress:** `PremiseAttackTask`, `ReviewPlanTask`, structured output under high token load.

### 03 — 4-Day Work Week National Program (India)
- **Why:** National policy programme with explicit governance design (single PMO under NITI Aayog), phased rollout, and measurable productivity/equity outcomes. Real-world labour policy problem with political and implementation constraints.
- **Drift risk:** Scope expansion (pilot policy → nationwide mandate too quickly), unsupported adoption claims, confidence inflation on productivity gains.
- **Pipeline stress:** `GovernanceTask`, `StakeholderTask`, `ReviewPlanTask`, `NegativeFeedbackTask`.

### 04 — Malaria Response Post-USAID (Accra, Ghana)
- **Why:** Crisis-driven healthcare plan in sub-Saharan Africa with no specified budget. Tests how the pipeline handles resource-constrained plans and whether it fabricates Western-centric solutions.
- **Drift risk:** Unsupported invention (invented NGO partners), customer drift (community → international org).
- **Pipeline stress:** `PreProjectAssessmentTask`, `ExpertDetails`, assumption handling.

### 05 — Advanced Water Purification Hub (Delhi, India)
- **Why:** Large-scale ($250M) infrastructure programme in South Asia. Tests cost modelling, regulatory posture (Indian law), and supply chain assumptions.
- **Drift risk:** Confidence inflation on adoption rates, unsupported technology claims.
- **Pipeline stress:** `CostBreakdownTask`, `WBSTask`, `GanttTask`.

### 06 — Upload Intelligence Neural Connectome (Uruguay)
- **Why:** Speculative biotech/AI plan with a massive budget ($10B) and genuine ethical and scientific uncertainty. Tests whether the pipeline preserves epistemic caution on unproven science.
- **Drift risk:** Confidence inflation (treats unproven science as settled), scope expansion.
- **Pipeline stress:** `DistillAssumptionsTask`, `RedlineGateTask`, `PremiseAttackTask`.

### 07 — Yellowstone Caldera Emergency Response (USA)
- **Why:** Crisis management plan for a low-probability, extreme-consequence event. Tests whether the pipeline can reason about multi-stakeholder emergency coordination without scope-expanding into long-term recovery.
- **Drift risk:** Scope expansion (72-hour response → national recovery plan), confidence inflation on coordination outcomes.
- **Pipeline stress:** `GovernanceTask`, `StakeholderTask`, `NegativeFeedbackTask`.

### 08 — Global Rubber Supply De-Risking from SALB (Global / SE Asia / Brazil / Africa)
- **Why:** $30B, 25-year public-private programme to end global rubber supply dependence on a single crop vulnerable to South American Leaf Blight. Explicit Phase 1 deliverable (SALB Containment Protocol), multi-jurisdiction phytosanitary coordination. Real-world food security and supply chain problem.
- **Drift risk:** Scope expansion, confidence inflation on containment timelines, unsupported invention of containment mechanisms.
- **Pipeline stress:** `GovernanceTask`, `StakeholderTask`, `WBSTask`, `GanttTask`.

### 09 — Spain–Morocco Transoceanic Tunnel (Europe + Africa)
- **Why:** Cross-border megaproject (€40B, 20 years, two continents, two regulatory systems). Tests whether the pipeline can handle political, geotechnical, and financial complexity at scale.
- **Drift risk:** Scope expansion, confidence inflation on political feasibility, unsupported engineering claims.
- **Pipeline stress:** `PremiseAttackTask`, `WBSTask`, `GanttTask`, `CostBreakdownTask`.

### 10 — Carrington Event Prep / Faraday Enclosure (Tallinn, Estonia)
- **Why:** Small-budget hardware product (€750K) with specific certification path, cash-flow milestones, and low-risk pilot framing. Tests the pipeline on a product-hardware plan in a small Eastern European market.
- **Drift risk:** Scope inflation (single SKU → platform), confidence inflation on regulatory approval.
- **Pipeline stress:** `MakeAssumptionsTask`, `ExpertDetails`, financial structured output.

---

## 5. How to Run the Suite

### Baseline pass (single model)
```
for each prompt_id in BENCHMARK_SUITE:
    initial_plan_text = load_prompt(prompt_id, simple_plan_prompts.jsonl)
    run_dir = create_run_dir(prompt_id, model_profile)
    seed_run_dir(run_dir, initial_plan_text)
    result = run_pipeline(run_dir, model_profile)
    record(prompt_id, model_profile, result.status, result.failed_task, result.error_type)
```

### Comparison pass (multiple models)
```
for each model in [baseline, premium, frontier, custom_qwen]:
    for each prompt_id in BENCHMARK_SUITE:
        result = run_pipeline(prompt_id, model)
        drift_score = drift_evaluate(initial_plan_text, result.final_report)  # proposal 84
        record(prompt_id, model, result.status, drift_score)
```

### What to look for
- Which tasks fail most often across prompts? → structural pipeline weakness
- Which drift types appear most often per model? → model-specific tendency
- Do local models fail at different task gates than cloud models? → model capability floor
- Do any prompts cause consistent failure across all models? → pipeline design issue (not model issue)

---

## 6. Results Schema

Each run should produce a record in a benchmark log:

```json
{
  "run_id": "...",
  "prompt_id": "ce2fbf38-9700-4ed1-814e-78772f7b7700",
  "model_profile": "custom",
  "model_name": "lmstudio-qwen3.5-35b-a3b",
  "timestamp": "2026-03-07T15:00:00Z",
  "status": "completed",
  "failed_task": null,
  "error_type": null,
  "tasks_completed": 61,
  "tasks_total": 61,
  "duration_seconds": 3240,
  "drift_score": 3.8,
  "drift_risk": "medium",
  "notes": ""
}
```

---

## 7. Maintenance

- When a pipeline stage changes, re-run the prompts that stress that stage.
- When a new model profile is added, run all 10 before declaring it stable.
- When a new prompt is added to `simple_plan_prompts.jsonl` that covers a new region or failure mode, evaluate it for inclusion (target 15 prompts by Q3 2026).
- The benchmark set is version-controlled here. To change a selection, update this doc and the corresponding run tooling.

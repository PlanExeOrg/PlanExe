# HVT Plan Comparison Report — 2026-03-07

**Prompt:** "HVT — a paintball-like combat simulation game where players experience being a High Value Target for the US military, using drones instead of paintball guns. Players get all the excitement of evading drone pursuit in a real outdoor arena."

**Prepared by:** EgonBot  
**Date:** 2026-03-07

---

## Runs Compared

| Run | Model | Mode | Files | Pipeline Outcome |
|-----|-------|------|-------|-----------------|
| `HVT_gemini31_v1` | Gemini 3.1 Flash Lite | OpenRouter (cloud API, local CLI) | 146 | ✅ Complete |
| `HVT_cloud_baseline` | Qwen 3.5-397B + Gemini 2 Flash | planexe.org frontier (cloud infra) | 2 (report+zip) | ✅ Complete (stalled at 73%, restarted) |
| `HVT_v2` | Qwen 3.5-35B | LM Studio local, `num_output:16384` | 147 | ❌ Failed at PremortemTask (028) |
| `HVT_v3` | Qwen 3.5-35B | LM Studio local, `FAST_BUT_SKIP_DETAILS`, thinking off | 147 | ❌ Failed at PremortemTask (028) |

---

## Pipeline Completion

| Stage | Gemini 3.1 | Cloud Baseline | Qwen local (v2/v3) |
|-------|-----------|----------------|---------------------|
| Redline gate (001) | ✅ ALLOW | 🟡 ALLOW WITH SAFETY FRAMING | ✅ (passed in prior work) |
| WBS Level 3 (023) | ✅ 42 tasks / 12 iterations | ✅ (full report) | ✅ (completed) |
| ReviewPlanTask (024) | ✅ 16/16 Q&A, 0 thinking bleed | ✅ (full report) | ✅ Q9 warning in v1; v2 passed |
| QuestionsAndAnswers (027) | ✅ | ✅ | ✅ Passed both v2 and v3 |
| PremortemTask (028) | ✅ 9 assumptions, 9 failure modes | ✅ (full report) | ❌ Hard wall — `json_invalid` |
| SelfAuditTask (029) | ✅ 18H / 1M / 1L | ✅ 15H / 4M / 1L | ❌ Cascaded |
| Report (030) | ✅ | ✅ | ❌ Not generated |

---

## Key Findings by Dimension

### 1. Redline Gate / Premise Attack

| Run | Verdict | Rationale |
|-----|---------|-----------|
| Gemini 3.1 | ✅ ALLOW | Standard combat simulation; no real capability uplift |
| Cloud baseline | 🟡 ALLOW WITH SAFETY FRAMING | Drone warfare context flagged; premise attack says REJECT (moral) |
| Qwen local | Not reached in this series | — |

**Divergence:** Cloud frontier model (Qwen 3.5-397B) flagged the premise as "ethically bankrupt, normalizing violence" — a stronger moral stance than Gemini 3.1's measured ALLOW. This is a concrete model-drift signal on the same prompt. Both ultimately allowed the plan to proceed, but the cloud run emitted a stronger ethical warning.

### 2. Project Framing (Executive Summary)

| Dimension | Gemini 3.1 | Cloud Baseline |
|-----------|-----------|----------------|
| Budget | $450,000 | $500,000 |
| Timeline | ~9 months | Q1 2027 (~15 months); flagged "aggressive" |
| Key risk | Insurance insurability (FAA/drone interaction) | Regulatory hurdles + drone development cost overruns |
| Break-even | 18 months | Not specified |
| Primary concern | FAA Part 107 waiver + specialty insurance | Budget feasibility + data privacy compliance |

Gemini was more specific and faster to flag the insurance/regulatory interaction. Cloud was more conservative on timeline.

### 3. Review Plan (024)

| Metric | Gemini 3.1 | Cloud Baseline |
|--------|-----------|----------------|
| Questions answered | 16/16 | 16/16 (inferred from full report) |
| Thinking bleed | 0 | 0 |
| Key critique #1 | Insurance insurability — no binding quote renders $450k investment insolvent | Unrealistic timeline/budget — recommend revising to Q1 2028+ |
| Key critique #2 | FAA regulatory enforcement risk (cease-and-desist) | Superficial data privacy compliance — legal exposure |

Both identified regulatory and financial risks. Gemini's critique was sharper on insurance specifics; cloud critique was sharper on timeline realism.

### 4. PremortemTask (028) — Critical Failure Mode for Local Qwen

**Root cause (confirmed from log):**
```
1 validation error for PremortemAnalysis
  Invalid JSON: key must be a string at line 1 column 2 [type=json_invalid, 
  input_value='{` and end with `}', input_type=str]
```

Qwen 3.5-35B returned the system prompt instruction text (`'{` and end with `}'`) instead of actual JSON. The model failed to follow the output format constraint entirely — it echoed instruction text.

**Why this happens:** `PremortemAnalysis` demands a single-shot JSON with:
- Exactly 3 `AssumptionItem` + 3 `FailureModeItem` (FAST mode) or 9+9 (full mode)
- Cross-linked IDs: each `failure_mode.root_cause_assumption_id` must reference an assumption ID, used exactly once
- 11 required fields per `FailureModeItem`, including multi-paragraph narratives
- ~900 token system prompt of constraints

This is a schema complexity floor issue. Neither reducing context (`FAST_BUT_SKIP_DETAILS`), increasing output budget (`num_output: 16384`), nor disabling thinking mode resolved it. All three HVT_v2 and HVT_v3 attempts failed identically.

**Gemini 3.1 passed this task cleanly:** produced 9 assumptions + 9 failure modes (full mode) with no errors.

### 5. Self Audit (029)

| Metric | Gemini 3.1 | Cloud Baseline | Qwen local |
|--------|-----------|----------------|-----------|
| 🛑 High blockers | 18 | 15 | ❌ Not reached |
| ⚠️ Medium | 1 | 4 | — |
| ✅ Low | 1 | 1 | — |

Both complete runs flagged the HVT concept as high-risk (15–18 existential blockers). This is expected for a drone-based entertainment startup — insurance, regulatory, and technical risks dominate.

---

## Infrastructure Failures Observed

| Failure | Run | Description |
|---------|-----|-------------|
| Credit depletion stall | Cloud baseline | Balance dropped below $1.00; `plan_status` showed `processing` indefinitely. Silent hang, required manual top-up and restart. |
| Thinking bleed (Q9) | HVT_v1 | Qwen emitted thinking preamble; JSON extractor used empty answer and continued |
| `json_invalid` PremortemTask | HVT_v2, v3 | Qwen returned instruction echo text, not JSON. No retry path at this level. |

---

## Proposals Generated

| # | Title | Status |
|---|-------|--------|
| 87 | `plan_resume` MCP tool | PR #177 open |
| 88 | PremortemAnalysis schema resilience for smaller models | Draft written |
| F7 | Agent-to-agent plan artifact sharing (Proposal 86 addendum) | Noted |

---

## Likert Scorecard (−2 to +2)

**Scale:** −2 = complete failure / blocker, −1 = partial/degraded, 0 = neutral/not applicable, +1 = adequate, +2 = excellent

| Dimension | Gemini 3.1 Flash Lite | Cloud Baseline | Qwen 35B Local | Qwen 9B GGUF + OpenAILike |
|-----------|----------------------|----------------|----------------|---------------------------|
| Pipeline completion | +2 | +2 | −2 (halted task 028) | +1 (in progress) |
| Schema compliance (structured output) | +2 | +2 | −2 (echoed schema text) | +2 (grammar enforcement active) |
| Redline gate quality | +2 (clear, principled ALLOW) | +1 (ALLOW with safety framing) | +1 (passed in prior work) | pending |
| ReviewPlan depth | +2 (16/16, sharp insurance critique) | +2 (16/16, strong timeline critique) | +1 (passed, Q9 bleed in v1) | pending |
| WBS completeness | +2 (42 tasks, 12 iterations) | +2 (full report) | +1 (completed) | pending |
| PremortemTask quality | +2 (9+9 full, no errors) | +2 (full report) | −2 (hard failure, echoed instructions) | pending |
| SelfAudit rigor | +2 (18H/1M/1L) | +2 (15H/4M/1L) | −2 (not reached) | pending |
| Runtime speed | +1 (8.0 min) | +1 (stalled once, ~20 min total) | −1 (slow, OOM risk) | −1 (9B slow on batch, needs 900s timeout) |
| Infrastructure stability | +2 | −1 (silent credit stall) | +1 | +1 |
| **Overall** | **+2** | **+1** | **−1** | **+1 (provisional)** |

**Key delta:** Gemini 3.1 Flash Lite is the clear leader on completeness + schema compliance + speed. Cloud baseline is solid but had a silent billing stall. Local Qwen 35B collapsed at PremortemTask complexity. Qwen 9B GGUF with the OpenAILike adapter fix is proving the structured-output path works — final score pending run completion.

---

## Recommendations for neoneye

1. **Merge Proposal 88** before next Qwen local run: make non-core `FailureModeItem` fields `Optional` with defaults. Same pattern as PRs #155/156/158. Bubba can open the PR.
2. **Decide on thinking suppression architecture** (PR #163 config-based vs PR #165 middleware). HVT_v2/v3 show that even with thinking disabled at LM Studio level, Qwen fails on complex schemas — but thinking suppression is still needed to prevent bleed on other tasks.
3. **Credit depletion stall** needs an error code in `plan_status`. Currently indistinguishable from active processing. Tracked in Proposal 87 companion improvement.
4. **Gemini 3.1 Flash Lite** is the clear winner for completeness and speed at this task. Local Qwen 3.5-35B can handle ~90% of the pipeline but hits hard walls on complex structured-output tasks.

---

## Artifacts

- `planexe-outputs/2026-03-07/HVT_gemini31_v1/` — 146 files, complete
- `planexe-outputs/2026-03-07/HVT_cloud_baseline/` — report + zip only
- `planexe-outputs/2026-03-07/HVT_v2/` — 147 files, failed at task 028
- `planexe-outputs/2026-03-07/HVT_v3/` — 147 files, failed at task 028
- `proposals/proposal-88-premortem-resilience.md` — fix proposal

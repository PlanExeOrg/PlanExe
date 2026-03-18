# System prompt optimizer

My overall plan is to optimize all system prompts across PlanExe.
By first starting optimizing the earliest system prompt, when that consistently performs better than
older system prompts, then it's a keeper. Then move on to the next system prompt in the luigi pipeline.
By the end of this prompt optimization, the overall plan quality should have been improved.

I want to track metrics for how much improvement have happened.


## Status (2026-03-18)

### Infrastructure

- **Data repo**: [PlanExe-prompt-lab](https://github.com/PlanExeOrg/PlanExe-prompt-lab) — holds baseline data, run history, and analysis artifacts.
- **Baseline data**: 5 train plans × 9 verify plans, extracted from zips into `baseline/train/` and `baseline/verify/`.
- **Runner** (`self_improve/runner.py`): re-executes a single pipeline step against baseline plans using the system prompt constant from PlanExe's code. Auto-incrementing history in `history/{bucket}/{counter}_{step}/`. Supports `--model`, `--prompt-lab-dir`, resume, and parallel workers.
- **Analysis pipeline** (5 phases in `analysis/`):
  - Phase 0: `prepare_iteration.py` — verifies PR state, pre-creates history dirs, and writes analysis `meta.json` with PR info. Replaces the former `create_analysis_dir.py` + `update_meta_pr.py`.
  - Phase 1: `run_insight.py` — Claude Code + Codex in parallel produce independent `insight_*.md` files with quantitative metrics and PR impact verdicts.
  - Phase 2: `run_code_review.py` — both agents review PlanExe source code, producing `code_*.md` with file:line references.
  - Phase 3: `run_synthesis.py` — single agent reconciles all findings into `synthesis.md` with top 5 ranked directions.
  - Phase 4: `run_assessment.py` — before/after comparison, metric table, keeper verdict (YES/NO/CONDITIONAL), and evaluation of the synthesis recommendation for the next iteration.
  - `run_analysis.py` — orchestrates phases 1–4 sequentially with retry support (default 3 attempts per phase, 20 min timeout). Cleans up error marker files before retrying.
- **`run_optimization_iteration.py`** — orchestrates a full loop: implement recommendation → create PR → run experiments → run analysis pipeline. Supports `--skip-implement`, `--skip-runner`, `--skip-analysis`, `--models`.
- **Conventions**: [`analysis/AGENTS.md`](https://github.com/PlanExeOrg/PlanExe-prompt-lab/blob/main/analysis/AGENTS.md).

### Models

7 models tested per iteration (5 plans each = 35 runs):

| Model | Alias | Status |
|-------|-------|--------|
| ollama-llama3.1 | llama | Working (3-5/5) |
| openrouter-openai-gpt-oss-20b | gpt-oss | Working (4-5/5) |
| openai-gpt-5-nano | gpt5-nano | Working (5/5) |
| openrouter-qwen3-30b-a3b | qwen | Working (5/5) |
| openrouter-openai-gpt-4o-mini | gpt4o-mini | Working (5/5) |
| openrouter-gemini-2.0-flash-001 | gemini-flash | Working (5/5) — added iter 13 |
| anthropic-claude-haiku-4-5-pinned | haiku | Working (4-5/5) |

Removed: GLM (PR #266, schema-echoing), StepFun (removed from config), nemotron (0/5 all iterations, structural incompatibility).

### Iteration History

Currently optimizing: `identify_potential_levers` (the first step after plan intake).

30 analysis rounds (0–29).

| Iter | PR | Change | Verdict | Runs | Analysis |
|------|-----|--------|---------|------|----------|
| 0 | — | Baseline | — | 00–08 | [analysis/0](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/0_identify_potential_levers) |
| 1 | [#268](https://github.com/PlanExeOrg/PlanExe/pull/268) | Fix doubled user prompt | YES | 09–16 | [analysis/1](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/1_identify_potential_levers) |
| 2 | [#270](https://github.com/PlanExeOrg/PlanExe/pull/270) | Fix assistant turn serialization | CONDITIONAL | 17–23 | [analysis/2](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/2_identify_potential_levers) |
| 3 | [#272](https://github.com/PlanExeOrg/PlanExe/pull/272) | Novelty-aware follow-up prompts | YES | 24–31 | [analysis/3](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/3_identify_potential_levers) |
| 4 | [#273](https://github.com/PlanExeOrg/PlanExe/pull/273) | Remove exemplar strings + make wrapper fields optional | CONDITIONAL | 32–38 | [analysis/4](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/4_identify_potential_levers) |
| 5 | [#274](https://github.com/PlanExeOrg/PlanExe/pull/274) | Align Pydantic field descriptions with system prompt | YES | 39–45 | [analysis/5](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/5_identify_potential_levers) |
| 6 | [#275](https://github.com/PlanExeOrg/PlanExe/pull/275) | Fix consequences length + trade-off requirement + review format | YES | 46–52 | [analysis/6](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/6_identify_potential_levers) |
| 7 | [#276](https://github.com/PlanExeOrg/PlanExe/pull/276) | Enforce schema contract: levers min/max 5, summary required | CONDITIONAL | 53–59 | [analysis/7](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/7_identify_potential_levers) |
| 8 | [#278](https://github.com/PlanExeOrg/PlanExe/pull/278) | Fresh context per call + relax lever count to 5–7 | CONDITIONAL | 60–66 | [analysis/8](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/8_identify_potential_levers) |
| 9 | [#279](https://github.com/PlanExeOrg/PlanExe/pull/279) | Remove naming template | YES | 67–73 | [analysis/9](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/9_identify_potential_levers) |
| 10 | [#281](https://github.com/PlanExeOrg/PlanExe/pull/281) | Keyword quality gate (reverted in [#282](https://github.com/PlanExeOrg/PlanExe/pull/282)) | NO | 74–80 | [analysis/10](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/10_identify_potential_levers) |
| 11 | [#283](https://github.com/PlanExeOrg/PlanExe/pull/283) | RetryConfig in runner (reverted in [#284](https://github.com/PlanExeOrg/PlanExe/pull/284)) | NO | 81–87 | [analysis/11](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/11_identify_potential_levers) |
| 12 | [#286](https://github.com/PlanExeOrg/PlanExe/pull/286) | Remove max_length=7 hard constraint on levers | YES | 88–94 | [analysis/12](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/12_identify_potential_levers) |
| 13 | [#289](https://github.com/PlanExeOrg/PlanExe/pull/289) | Add options count and review format validators | CONDITIONAL | 95–101 | [analysis/13](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/13_identify_potential_levers) |
| 14 | [#292](https://github.com/PlanExeOrg/PlanExe/pull/292) | Recover partial results when a call fails | YES | 102–108 | [analysis/14](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/14_identify_potential_levers) |
| 15 | [#294](https://github.com/PlanExeOrg/PlanExe/pull/294) | Consolidate review_lever prompt to prevent format alternation | YES | 09–16 | [analysis/15](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/15_identify_potential_levers) |
| 16 | [#295](https://github.com/PlanExeOrg/PlanExe/pull/295) | Continue loop after call failure instead of breaking | YES | 17–23 | [analysis/16](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/16_identify_potential_levers) |
| 17 | [#296](https://github.com/PlanExeOrg/PlanExe/pull/296) | Auto-correct review_lever before hard-rejecting | CONDITIONAL | 24–30 | [analysis/17](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/17_identify_potential_levers) |
| 18 | [#297](https://github.com/PlanExeOrg/PlanExe/pull/297) | Simplify lever prompt to restore content quality | YES | 31–37 | [analysis/18](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/18_identify_potential_levers) |
| 19 | [#299](https://github.com/PlanExeOrg/PlanExe/pull/299) | Remove bracket placeholders and fragile English-only validator | CONDITIONAL | 38–45 | [analysis/19](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/19_identify_potential_levers) |
| 20 | [#309](https://github.com/PlanExeOrg/PlanExe/pull/309) | Add option-quality reminder to call-2/3 prompts | YES | 46–52 | [analysis/20](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/20_identify_potential_levers) |
| 21 | [#313](https://github.com/PlanExeOrg/PlanExe/pull/313) | Add anti-fabrication reminder to call-2/3 prompts | CONDITIONAL | 53–59 | [analysis/21](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/21_identify_potential_levers) |
| 22 | [#316](https://github.com/PlanExeOrg/PlanExe/pull/316) | Replace two-bullet review_lever prompt with single flowing example | CONDITIONAL | 60–66 | [analysis/22](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/22_identify_potential_levers) |
| 23 | [#326](https://github.com/PlanExeOrg/PlanExe/pull/326) | Add second review_lever example to break template lock | KEEP | 67–73 | [analysis/23](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/23_identify_potential_levers) |
| 24 | [#334](https://github.com/PlanExeOrg/PlanExe/pull/334) | (INVALID — runner ran against main, not PR branch) | INVALID | 75–81 | [analysis/24](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/24_identify_potential_levers) |
| 25 | [#334](https://github.com/PlanExeOrg/PlanExe/pull/334) | Remove unused summary field, slim call-2/3 prefix | YES | 82–88 | [analysis/25](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/25_identify_potential_levers) |
| 26 | [#337](https://github.com/PlanExeOrg/PlanExe/pull/337) | Replace generic review_lever examples with domain-specific ones (agriculture, urban planning, insurance) | YES | 89–95 | [analysis/26](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/26_identify_potential_levers) |
| 27 | [#339](https://github.com/PlanExeOrg/PlanExe/pull/339) | Relax option count validator, raise review min_length to 50, clean up dead code | YES | 96–02 | [analysis/27](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/27_identify_potential_levers) |
| 28 | [#340](https://github.com/PlanExeOrg/PlanExe/pull/340) | Remove template-lock phrase "the options neglect" from example 3, deduplicate field description examples | CONDITIONAL | 03–09 | [analysis/28](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/28_identify_potential_levers) |
| 29 | [#288](https://github.com/PlanExeOrg/PlanExe/pull/288) | Fix zero cost for OpenAI/Anthropic models via fallback pricing + Anthropic token capture | YES | 10–16 | [analysis/29](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/29_identify_potential_levers) |

### Key Improvements So Far

Comparing iteration 0 (baseline) to iteration 29 (current state):

- **Review format violations**: 67 → ~4 (enforced by validator since iter 13)
- **Consequence chain violations**: 35 → 7
- **Bracket placeholder leakage**: ~17 → 0
- **Template leakage** (naming suffix): 83–100% → 2–10%
- **review_lever template lock**: llama3.1 100% → 0%, qwen3 ~100% → 6% (broken by domain-specific examples in iter 26)
- **"The options [verb]" secondary lock**: llama3.1 gta_game 100% → 62.5% (partial fix in iter 28; examples 1 & 2 still use options-as-subject)
- **Cross-call duplication**: eliminated via novelty-aware follow-ups
- **Chat-history contamination**: eliminated via fresh context per call
- **Validation failures discarding valid levers**: eliminated — `max_length=7` removed (iter 12), partial result recovery added (iter 14)
- **Option count violations**: caught by `check_option_count` validator (iter 13), relaxed to `< 3` in iter 27
- **review_lever min_length**: raised from 20 to 50 chars (iter 27) — catches stub reviews
- **Unused summary field**: removed (iter 25) — eliminated 105 wasted LLM generations per batch
- **Duplicate example injection**: field description examples removed (iter 28) — ~150-200 token savings per call
- **Consequences contamination**: "A weakness in this approach is that…" in consequences field eliminated (iter 28, downstream of B2 fix)
- **Cost tracking**: OpenAI and Anthropic models now report token counts and estimated cost (iter 29)
- **Overall success rate**: 88.6% → 97.1% (iter 27)
- **Option word count (llama3.1)**: ~7 → ~12 avg words (iter 25)
- **Gemini-flash baseline comparison**: name uniqueness 71%→100%, cross-call duplication 15→0, consequence richness +40%, option prefix leakage 16→0 ([comparison](https://github.com/PlanExeOrg/PlanExe-prompt-lab/blob/main/analysis/13_identify_potential_levers/comparison.md))

### Key Lessons Learned

1. **No hardcoded English keywords in validators.** PlanExe users create plans in many languages. Keyword-based quality gates (checking for "Controls", "Weakness:") break for non-English plans. All validation must be language-agnostic — structural checks, field length ratios, duplicate detection. Iteration 10 was a disaster because of this.
2. **Don't merge PRs before the verdict.** The correct order is: create PR → run experiments on the branch → run analysis → read verdict → merge only if verdict confirms improvement. Iteration 11 was merged prematurely.
3. **The runner always uses the code constant.** There is no external prompt file or CLI override — the prompt is whatever is committed in PlanExe's code at run time.
4. **Auto-implementing synthesis recommendations can conflict with user intent.** The `run_optimization_iteration.py` script auto-applies the top recommendation, but this can conflict with explicit user preferences (e.g., reverting to "exactly 5 levers" when the user wanted 5–7). Use `--skip-implement` when needed.
5. **Dual-agent analysis catches real errors.** Codex corrected Claude's success rate miscount in iteration 0. Independent analysis is worth the extra cost.
6. **Prefer soft prompt guidance over hard Pydantic constraints.** `max_length=7` on the levers field discarded entire LLM responses when a model returned 8 levers. The downstream dedup step already handles extras. Hard caps waste tokens on retries; soft guidance in the prompt is cheaper and more fault-tolerant.
7. **Recover partial results rather than failing completely.** When a 3-call loop has call 2 or 3 fail, keeping levers from prior successful calls is better than discarding everything. A single validator rejection should not wipe out 10+ valid levers.
8. **Domain-specific examples break template lock.** Generic examples ("centralization and local autonomy") are portable — models copy them verbatim regardless of plan topic. Domain-specific examples (agriculture, insurance) force models to generate original text. Iteration 26 broke llama3.1 template lock from 100% to 0% with this approach.
9. **Verify the runner is on the PR branch, not main.** Iteration 24 was invalidated because the runner imported code from main instead of the PR branch. `run_optimization_iteration.py` now verifies this automatically.
10. **Fixing one of N copyable examples shifts the lock, not breaks it.** Iteration 28 replaced one of three examples using "the options" as grammatical subject. The lock rate dropped from 100% to 62.5% for call 1, but calls 2-3 retained the pattern at ~100%. The remaining examples are still active lock sources. All must be fixed simultaneously.
11. **Replacement examples can introduce new regressions.** Iteration 28's replacement example 3 introduced fabricated % claims in llama3.1 consequences ("by at least 20%"). Replacement text must be reviewed for quantitative language that models might copy.

### Current State of `identify_potential_levers.py`

After 29 iterations, the step has these characteristics:

- System prompt says "5 to 7 levers per response", schema has `min_length=5` (no `max_length` — downstream dedup handles extras)
- Follow-up calls use novelty-aware prompts (exclude already-generated lever names)
- Each LLM call gets a fresh context (no chat history accumulation)
- Naming guidance says "avoid formulaic patterns or repeated prefixes" (no template)
- Pydantic validators: `check_option_count` (exactly 3 options), `check_review_format` (structural only — min length + no bracket placeholders, language-agnostic)
- Partial result recovery: if call 2 or 3 fails, keep levers from prior successful calls instead of discarding everything
- Quality gate: duplicate name filter (language-agnostic)
- `review_lever` prompt uses three domain-specific non-portable examples (agriculture, urban planning, insurance) to prevent template lock; example 3 rewritten to remove "the options neglect" (iter 28)
- `review_lever` field description no longer duplicates system prompt examples (iter 28) — ~150-200 token savings per call
- `DocumentDetails.summary` field removed (unused, wasted tokens)
- Call-2/3 prefix slimmed (no duplicate quality/anti-fabrication reminders)
- `OPTIMIZE_INSTRUCTIONS` documents 6 known problems (including template lock)
- System prompt: `IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT` constant in code
- **Cost tracking**: fallback pricing registry estimates cost from token counts for OpenAI/Anthropic models; Anthropic httpx hook captures token usage that LlamaIndex discards (iter 29)

### Not Started

- **Numeric evaluator** — no deterministic scoring script. Metrics are computed by LLM agents, not reproducible code.
- **Candidate generator** — prompt variants are proposed by synthesis agents and implemented manually, not auto-generated.
- **Train/verify scoring loop** — no automated optimizer that tests candidates against `baseline/verify/`.
- **Other pipeline steps** — only `identify_potential_levers` has been optimized so far.

### Next Steps

1. **Replace all three review_lever examples** — examples 1 and 2 still use "the options" / "none of the options" as grammatical subject, confirmed as active lock sources (gpt-5-nano run 05: "the options neglect"; llama3.1 parasomnia 100% lock rate). Replace all three with critiques whose subject is never options-centric. Ensure no quantitative language to avoid fabricated % regression.
2. **Fix OPTIMIZE_INSTRUCTIONS self-contradiction** — lines 77-79 praise example 1 as "the correct structural template" while it contains a copyable opener. Fix in same PR as example replacement.
3. **Build a deterministic evaluator** — compute metrics (uniqueness, option count, template leakage, format compliance) without LLM calls. Makes assessment reproducible and fast.
4. **Move to the next pipeline step** once `identify_potential_levers` is stable.


## Stage 1 - one improvement iteration

Here is how one iteration is going to be:
Optimize one system prompt for a single step in the luigi pipeline in run_plan_pipeline.py
Take the "dataset train" and unzip the files to a work dir.
Alter the system prompt of a named task in the pipeline.
For MODEL_LIST, loop through the models, (these are LLMs without reasoning):
- It's very expensive to run the full pipeline. So I only want to rerun the luigi pipeline for that just step, see if the new system prompt does a better job than the reference plan.
- Capture the output.

Have a reasoning model do a comparison of the captured outputs.
If it does better than the original prompt then it's a candidate for keeping.
If it does does worse than the original prompt then write it to a failed attempt log.
Find prompts that consistently yield better outputs than the original prompt.
Store the feedback.

At this point there should be a ranked list of candidate system prompts that are better than the original.
Go through the "dataset verify" and compute score of the best candidate prompts.
Pick the best one and commit it to git.

For all plans both train and verify, commit the new generated files for each plan.
Commit improvement stats.

## Stage 2 - multiple improvement iteration

Run "one improvement iteration" multiple times, until there score isn't improving significantly.

## Stage 3 - next item in the luigi pipeline

As more upstream tasks gets optimized, then move on to further downstream tasks.

From time to time regenerate full plan, and have a reasoning model compare how
much better or worse is the new full plan is over various KPIs.


# Dataset

Config json file that contains the "dataset train" and "dataset verify".

The dataset lives in the [PlanExe-prompt-lab](https://github.com/PlanExeOrg/PlanExe-prompt-lab) repository.

These zip files are what PlanExe currently outputs, so these are the "baseline" to compare with.


## Dataset train

20260310_hong_kong_game.zip
20250329_gta_game.zip
20250321_silo.zip
20260311_parasomnia_research_unit.zip
20260308_sovereign_identity.zip

## Dataset verify

20260303_crate_recovery_campaign.zip
20260215_nuuk_clay_workshop.zip
20260202_heatwave_resilience.zip
20260201_media_rescue.zip
20260201_yellowstone_evacuation.zip
20260131_clear_english.zip
20260129_euro_adoption.zip
20260114_cbc_validation.zip
20251101_e_bus_security.zip


# Feedback

## Strengths

- Incremental approach is smart — optimizing one pipeline step at a time and proving it before moving on avoids compounding errors.
- Train/verify split — using separate datasets to prevent overfitting to the evaluation set is the right call.
- Failed attempt log — tracking what didn't work is often overlooked but very valuable.

## Questions & Gaps

1. What are the KPIs?
The document mentions "score," "better," "worse," and "various KPIs" but never defines them. This is the most critical gap. Without concrete metrics, the reasoning model's comparison will be subjective and inconsistent. What makes one output "better"? Completeness? Specificity? Actionability? Structure?

2. How is the reasoning model prompted for comparison?
"Have a reasoning model do a comparison" is underspecified. Will it use a rubric? Pairwise ranking? A numeric score? The evaluation prompt itself will heavily influence which candidate "wins." This prompt may need its own optimization.

3. What does "consistently" mean?
"Find prompts that consistently yield better outputs" — across all train plans? A majority? With what confidence threshold? A candidate that's great for 3 plans but terrible for 1 might still be worse overall.

4. MODEL_LIST is undefined.
Which models are being tested? Are you running each candidate prompt across multiple LLMs to find prompts that are model-robust, or optimizing per-model?

5. How are candidate prompts generated?
The doc describes the evaluation loop but not how new prompt variants are created. Manual authoring? LLM-generated rewrites? A mix?

6. "Rerun just that step" — dependency handling.
If you rerun step N with a new prompt, its output feeds into step N+1 in the real pipeline. Are you only evaluating step N's output in isolation, or propagating through downstream steps? Evaluating in isolation is faster but may miss regressions that only show up downstream.

7. Budget/cost controls.
You mention "it's very expensive to run the full pipeline." What about the cost of this optimization loop itself? With multiple models x multiple candidate prompts x multiple train plans x multiple iterations, this can get expensive fast. Consider setting a budget cap or max iterations.

8. The envvar for the dataset path — good instinct, but also specify a fallback or error message if it's unset.

## Suggestions

- Define a scoring rubric (even a simple 1-5 scale across 3-4 dimensions) before starting.
- Version-control the evaluation prompts alongside the system prompts.
- Consider running a baseline score for the current prompts first so you have a numeric starting point, not just pairwise comparisons.
- Add a "no regression" gate: the new prompt must not score worse on any plan by more than X%, even if average improves.


# Repo Structure

Two repos: the optimization engine lives in PlanExe, the data/artifacts live in a separate repo.

## Repo 1: `PlanExe` (existing — add optimization tooling)

```
worker_plan/
  worker_plan_internal/
    assume/
      identify_purpose.py          # contains SYSTEM_PROMPT (current)
      ...
    ...

self_improve/                   # NEW — the optimization engine
  __init__.py
  config.py                        # loads dataset config, env vars
  runner.py                        # reruns a single pipeline step with a candidate prompt
  evaluator.py                     # calls reasoning model to score/compare outputs
  scorer.py                        # rubric definition, numeric scoring
  candidate_generator.py           # generates prompt variants
  optimizer.py                     # orchestrates the train/verify loop (Stage 1-2)
  cli.py                           # CLI entry point

  rubrics/                          # evaluation rubrics per pipeline step
    identify_purpose.yaml
    make_assumptions.yaml
    ...

  tests/
    ...
```

This keeps the optimizer close to the code it's modifying — it needs to import pipeline tasks,
swap prompts, and run individual steps.

## Repo 2: [`PlanExe-prompt-lab`](https://github.com/PlanExeOrg/PlanExe-prompt-lab) (data repo)

```
README.md
dataset.json                        # train/verify split definition
populate_baseline.py                # script to populate baseline from zip files

baseline/                           # current outputs (extracted from dataset zips)
  train/
    20260310_hong_kong_game/
      001-1-start_time.json
      001-2-plan.txt
      ...
      030-report.html
    20250329_gta_game/
      ...
    20250321_silo/
      ...
    20260311_parasomnia_research_unit/
      ...
    20260308_sovereign_identity/
      ...
  verify/
    20260303_crate_recovery_campaign/
      ...
    20260215_nuuk_clay_workshop/
      ...
    (7 more plans)

history/                                      # captured output, global run counter
  # Path: history/{counter // 100}/{counter % 100:02d}_{step_name}/
  # Counter is auto-incremented: scan history/ for the highest existing
  # run number and add 1. No counter file needed.
  # Runs for different steps are interleaved chronologically.
  0/                                          # runs 0-99
  1/                                          # runs 100-199
  2/                                          # runs 200-299
    00_identify_purpose/                      # run 200
    01_identify_potential_levers/              # run 201
    02_identify_potential_levers/              # run 202
      meta.json                                # which step, which system prompt, what model used
      events.jsonl
      outputs.jsonl
      outputs/
        20250321_silo/
          002-9-potential_levers_raw.json
          002-10-potential_levers.json
          activity_overview.json
          usage_metrics.jsonl
        20260310_hong_kong_game/
        20260201_media_rescue/
    03_identify_potential_levers/
    ...
    98_identify_potential_levers/
    99_identify_potential_levers/
  3/
analysis/
  AGENTS.md                         # conventions for analysis artifacts
  prepare_iteration.py              # phase 0: verify PR, pre-create history dirs
  run_analysis.py                   # orchestrates phases 1-4 sequentially
  run_insight.py                    # phase 1: Claude + Codex insight in parallel
  run_code_review.py                # phase 2: Claude + Codex code review in parallel
  run_synthesis.py                  # phase 3: Claude synthesis (single agent)
  run_assessment.py                 # phase 4: before/after comparison + verdict
  0_identify_potential_levers/      # pre-fix analysis (runs 00-08)
    meta.json
    insight_claude.md
    insight_codex.md
    code_claude.md
    code_codex.md
    synthesis.md
  1_identify_potential_levers/      # post-fix analysis (runs 09-16, PR #268)
    meta.json                       # includes pr_url, pr_title, pr_description
    insight_claude.md
    insight_codex.md
    code_claude.md
    code_codex.md
    synthesis.md
    assessment.md                   # verdict: YES/NO/CONDITIONAL

run_optimization_iteration.py       # orchestrates full iteration

scores/                             # longitudinal tracking
  scoreboard.csv                    # step, date, baseline_score, best_score, delta
  history.json                      # full history for charting

full_plan_comparisons/              # Stage 3 periodic full-plan regenerations
  2026-03-20/
    hong_kong_game/
      030-report.html
    kpi_comparison.json
```

## Connecting the two repos

`dataset.json` defines the train/verify split. The `populate_baseline.py` script reads this file and extracts the zips into `baseline/`.

```json
{
  "comment": "Replace the zip files with your own. Run the populate_baseline.py script to populate the baseline/ directory.",
  "train": [
    "20260310_hong_kong_game.zip",
    "20250329_gta_game.zip",
    "20250321_silo.zip",
    "20260311_parasomnia_research_unit.zip",
    "20260308_sovereign_identity.zip"
  ],
  "verify": [
    "20260303_crate_recovery_campaign.zip",
    "20260215_nuuk_clay_workshop.zip",
    "20260202_heatwave_resilience.zip",
    "20260201_media_rescue.zip",
    "20260201_yellowstone_evacuation.zip",
    "20260131_clear_english.zip",
    "20260129_euro_adoption.zip",
    "20260114_cbc_validation.zip",
    "20251101_e_bus_security.zip"
  ]
}
```

## Why this split

- `PlanExe` stays clean — no multi-GB plan artifacts in the code repo.
- `PlanExe-prompt-lab` can use Git LFS for the zip files and large outputs.
- The data repo is a full audit trail — every candidate, evaluation, and score is committed.
- `scoreboard.csv` gives the metrics tracking at a glance.

## Feedback from Bubba (2026-03-13)

### Additional Questions & Gaps

**Evaluation prompt is itself a prompt — it needs the same rigor.**
The reasoning model used for comparison will be guided by an evaluation prompt. That prompt encodes the scoring rubric. If the evaluation prompt is poorly specified, the optimizer will find prompts that score well on the evaluator but don't actually produce better plans. The evaluation prompt should be versioned, reviewed, and held constant across a run (or explicitly varied as a separate experiment).

**Step isolation vs. cascade quality.**
Optimizing step N in isolation assumes that step N's output quality is a meaningful proxy for end-to-end plan quality. This may not hold: a step N output that scores 5/5 in isolation might feed step N+1 in a way that degrades the downstream result. Recommendation: after each isolated step optimization, run at least one full-pipeline spot check on the best candidate to confirm no downstream regressions before committing.

**`FAST_BUT_SKIP_DETAILS` vs. `ALL_DETAILS_BUT_SLOW` — which mode for the optimizer?**
The optimizer reruns individual pipeline steps. Those steps behave differently depending on `SPEED_VS_DETAIL`. Which mode should the optimizer use? Using `FAST_BUT_SKIP_DETAILS` is cheaper but may not reflect production output quality. This should be explicit in `meta.json` and the scoring rubric should account for it.

**Candidate prompt generation is the missing piece.**
The doc describes the evaluation loop thoroughly but says nothing about how candidate prompts are created. This is the creative/generative step that drives the whole system. Options to consider:
- LLM-generated rewrites (give the current prompt + failure examples to a reasoning model, ask for 3-5 variants)
- Manual authoring with structured variation (change one constraint at a time)
- Mutation from failed-attempts log (learn what patterns correlate with failure)
The choice here will determine how fast the optimizer converges.

### Implementation Suggestion: Prompt diff in `failed_attempts.log`

The `failed_attempts.log` currently captures that a candidate failed. It should also capture:
- The diff between the candidate prompt and the base prompt
- Which specific train plans it failed on (not just aggregate)
- The evaluator's reasoning (not just the score)

This turns the failure log into a learning dataset that can feed the next round of candidate generation.

### Implementation Suggestion: Freeze the evaluation model

The reasoning model used for comparison should be pinned to a specific model version (not a rolling alias) for the duration of an optimization run. Model updates can shift evaluation scores without any change to the system prompts being tested. `meta.json` should record the exact evaluator model used.

## Feedback from Claude Code (2026-03-14)

After building and running the first complete optimization iteration (B1 fix, PR #268), here are observations on the loop's strengths and weaknesses.

### What works well

**The audit trail is the real win.** Every change has a PR, every PR has an assessment with a comparison table and verdict. You can trace any decision back to evidence. Most optimization systems are black boxes; this one is readable.

**Dual-agent analysis reduces blind spots.** Claude and Codex independently analyze the same data, then synthesis resolves disagreements by checking source code. This caught real issues — codex corrected claude's success rate miscount in iteration 0 (claude counted raw `outputs.jsonl` rows including pre-retry errors; codex counted final artifacts on disk).

**It found a real bug on the first iteration.** B1 was a genuine code defect, not a prompt tweak. The loop surfaced it, fixed it, verified the fix, and moved on to deeper issues. That's the loop working as intended.

### Concerns

**It's slow and expensive.** Each iteration runs 7 models × 5 plans, then 4 phases of LLM analysis (2 insight + 2 code review + 1 synthesis + 1 assessment = 6 LLM sessions). That's a lot of API calls for one recommendation. The low-hanging fruit (B1) justified it, but as fixes get more marginal the cost-per-insight ratio will worsen.

**There's no numeric scoring yet.** The assessment says "review violations 67→4" but those numbers come from agents counting by hand, not a deterministic evaluator. Two runs of the same analysis could produce different counts. Until `evaluator.py` exists with a pinned rubric, the loop is making keep/reject decisions on qualitative agent impressions, not reproducible scores.

**The synthesis bottleneck.** Everything funnels through one synthesis agent that picks THE recommendation. If it picks wrong (e.g., a prompt tweak when a code fix would help more), you burn a full iteration discovering that. The ranking is only as good as that single agent's judgment.

**It optimizes in series, not parallel.** One recommendation per iteration. You could test multiple fixes simultaneously on separate branches and compare, but the current flow is strictly sequential.

### Suggested priorities

1. **Build the numeric evaluator** — a deterministic script that computes the same metrics the insight agents compute (uniqueness, option count violations, template leakage, format compliance) without LLM calls. This makes assessment reproducible and fast.
2. **Use the evaluator to gate the assessment** — run it before the expensive 4-phase analysis. If the numbers clearly improve, fast-track. If they clearly regress, skip analysis entirely.
3. **Consider parallelizing fixes** — the synthesis already produces a top-5 list. You could implement 2-3 in parallel on separate branches and let the evaluator pick the winner.

The loop is a solid foundation. The main risk is that it stays in "expensive qualitative analysis" mode when it should graduate to "cheap quantitative scoring with qualitative analysis reserved for ambiguous cases."

# System prompt optimizer

My overall plan is to optimize all system prompts across PlanExe.
By first starting optimizing the earliest system prompt, when that consistently performs better than
older system prompts, then it's a keeper. Then move on to the next system prompt in the luigi pipeline.
By the end of this prompt optimization, the overall plan quality should have been improved.

I want to track metrics for how much improvement have happened.


## Status (2026-03-14)

### Done

- **Data repo created**: [PlanExe-prompt-lab](https://github.com/PlanExeOrg/PlanExe-prompt-lab) — holds baseline data, optimization run artifacts, and scores.
- **Baseline data populated**: 5 train plans and 9 verify plans extracted from zips into `baseline/train/` and `baseline/verify/`. Each plan contains ~175-189 files covering the full pipeline output.
- **`dataset.json`** defines the train/verify split with zip filenames.
- **`populate_baseline.py`** script automates ingesting baseline data from local paths or URLs.
- **Directory structure** for `runs/`, `scores/`, and `full_plan_comparisons/` is in place (empty, awaiting optimizer).
- **`prompt_optimizer/` runner** ([PR #263](https://github.com/PlanExeOrg/PlanExe/pull/263)) — re-executes the `IdentifyPotentialLevers` step with a candidate system prompt against baseline plans. Features:
  - Added optional `system_prompt` parameter to `IdentifyPotentialLevers.execute()` (backward-compatible).
  - CLI with `--system-prompt-file`, `--baseline-dir`/`--plan-dir`, `--model`.
  - `--prompt-lab-dir` auto-creates runs in `history/{counter//100}/{counter%100:02d}_{step}/` with auto-incrementing counter.
  - `--output-dir` for manual placement (alternative to `--prompt-lab-dir`).
  - Streaming progress: `meta.json` written at start, `events.jsonl` for real-time monitoring, `outputs.jsonl` for per-plan results.
  - Per-plan `activity_overview.json` and `usage_metrics.jsonl` for token/cost tracking.
  - System info (OS, CPU, memory, GPU) captured in `meta.json`.
  - Resume support: re-run the same command to skip completed plans and retry errors.
  - Automatic parallelism based on `luigi_workers` from `llm_config/*.json` (cloud models get 4 workers, local models get 1).
  - Thread-safe: per-thread `LLMExecutor`, thread-local usage metrics via `threading.local()`, `TrackActivity` duplicate-write guard, locked file writes.
  - Tested against all 5 training plans with local `ollama-llama3.1` (sequential) and `openrouter-openai-gpt-oss-20b` (parallel, 4 workers).
- **`register_prompt.py`** — extracts the current system prompt for a step and saves it to `prompts/{step}/prompt_{index}_{sha256}.txt` in the prompt-lab repo. Auto-increments index, skips duplicates by SHA256.
- **Thread-safety fixes to `worker_plan_internal`**:
  - `usage_metrics.py`: replaced module-level global with `threading.local()` so each thread gets its own metrics path.
  - `track_activity.py`: `_record_file_usage_metric` guards against duplicate writes when multiple handlers are registered on the shared dispatcher.
- **Anthropic instrumentation fix** ([PR #264](https://github.com/PlanExeOrg/PlanExe/pull/264)):
  - **Root cause**: `llama-index-llms-anthropic` overrides `structured_predict()` and calls `self._client.beta.messages.parse()` directly, bypassing `self.chat()`. LlamaIndex instrumentation events (`LLMChatEndEvent`) never fire, so `TrackActivity` never writes `usage_metrics.jsonl` or `activity_overview.json`.
  - **Fix**: `LLMExecutor._record_attempt_token_metrics()` now records basic metrics (model, duration, success) for all calls, not just failures. For Anthropic, this is the only record; for other backends, TrackActivity also writes richer data (token counts, cost).
  - Added `ANTHROPIC_API_KEY` to `.env` example files and documented Anthropic usage in `prompt_optimizer/AGENTS.md`.
- **GLM models removed** ([PR #266](https://github.com/PlanExeOrg/PlanExe/pull/266)) — `openrouter-z-ai-glm-4-7-flash` returned the schema instead of data (schema-echoing failure). Removed from `llm_config`.
- **Runner tested across 8 models** (history runs 00–08, pre-fix):
  - 5/5: `ollama-llama3.1`, `openrouter-openai-gpt-oss-20b`, `openai-gpt-5-nano`, `openrouter-qwen3-30b-a3b`, `openrouter-openai-gpt-4o-mini`, `anthropic-claude-haiku-4-5-pinned`
  - 0/5: `openrouter-z-ai-glm-4-7-flash` (schema-echoing), `openrouter-nvidia-nemotron-3-nano-30b-a3b` (empty output), `openrouter-stepfun-step-3-5-flash` (removed from config)
- **Analysis automation** — Python scripts in `PlanExe-prompt-lab/analysis/` that run Claude Code and Codex:
  - **`create_analysis_dir.py`** (phase 0): scans existing analysis directories and history runs, creates a new auto-incremented analysis directory with `meta.json` containing only unanalyzed runs. Ensures no runs are accidentally skipped.
  - **`run_insight.py`** (phase 1): reads `meta.json`, builds a prompt referencing `AGENTS.md` conventions and history runs, runs Claude Code and Codex in parallel. Produces independent `insight_claude.md` and `insight_codex.md` with quantitative metrics, tiered rankings, and labeled hypotheses (H1–H4, C1–C4).
  - **`run_code_review.py`** (phase 2): reads the insight files from phase 1 as context, asks both agents to review PlanExe source code (`identify_potential_levers.py`, `runner.py`) for bugs and improvements. Produces `code_claude.md` and `code_codex.md` with file:line references traced back to insight findings.
  - **`run_synthesis.py`** (phase 3): reads all insight and code review files, cross-references across agents, resolves disagreements by reading source code, and produces a single `synthesis.md` with top 5 ranked directions and 1 recommendation.
  - **`run_assessment.py`** (phase 4): compares the current analysis against the previous one, reads actual output samples from history, and produces `assessment.md` with a metric-by-metric comparison table and a keeper verdict (YES/NO/CONDITIONAL).
  - **`update_meta_pr.py`**: fetches PR info from GitHub via `gh` and writes `pr_url`, `pr_title`, `pr_description` into `meta.json` for provenance.
  - Conventions documented in [`analysis/AGENTS.md`](https://github.com/PlanExeOrg/PlanExe-prompt-lab/blob/main/analysis/AGENTS.md).
- **`run_optimization_iteration.py`** — orchestrates a full optimization loop: reads the latest `synthesis.md`, extracts the top recommendation, invokes Claude Code to implement it (branch + PR), runs experiments across models, and runs the full analysis pipeline (phases 0–4). Supports `--skip-implement`, `--skip-runner`, `--skip-analysis`, and `--models`.
- **First analysis complete** ([analysis 0](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis/0_identify_potential_levers)):
  - Insight files identify run 09 (claude-haiku) as best content quality, run 02 (gpt-5-nano) as best structural regularity. ~70% artifact-level success rate. Three models failed completely (schema mismatch, empty output, config error).
  - Code review files confirmed bugs: B1 (double user-prompt), B7 (field description mismatch), thread-safety issues, no post-call validation.
  - Synthesis ranked B1 as the top recommendation.
- **First optimization iteration complete** — B1 fix ([PR #268](https://github.com/PlanExeOrg/PlanExe/pull/268)):
  - **Fix**: removed the initial `USER(user_prompt)` from `chat_message_list` initialization in `identify_potential_levers.py`. All three LLM turns now handled uniformly by the loop.
  - **Post-fix runs** (09–16): 7 models re-tested, 29/30 plans succeeded (96.7% excluding infrastructure failures).
  - **[Assessment verdict: YES](https://github.com/PlanExeOrg/PlanExe-prompt-lab/blob/main/analysis/1_identify_potential_levers/assessment.md)** — PR merged. Key improvements:
    - Review format violations: 67 → ~4
    - Consequence chain violations: 35 → 7
    - Bracket placeholder leakage: ~17 → ~1
    - No regressions in success rate, content depth, or cross-call duplication.
  - New issues surfaced (not caused by B1): assistant turn stored as `dict` instead of `str` (corrupts turns 2–3 context), metric exemplar leakage ("25% faster scaling through").

### Not Started

- **Evaluator prompt** — no scoring rubric or comparison prompt written.
- **Candidate generator** — no mechanism for producing prompt variants.
- **Scorer / Optimizer loop** — no automated train/verify loop.

### Next Steps

1. **Fix assistant turn serialization** in `identify_potential_levers.py:196`. `result["chat_response"].raw.model_dump()` passes a Python dict as `ChatMessage.content`; it should be `result["chat_response"].message.content` (the raw JSON string). This corrupts conversation context for turns 2 and 3 on 100% of runs and is the top-priority fix per synthesis 1.
2. **Replace metric exemplar** in `identify_potential_levers.py:95`. The literal `"Systemic: 25% faster scaling through..."` is copied verbatim by 5/6 models. Replace with `"Systemic: [N]% [measurable outcome] through [mechanism]"`.
3. **Add stateful "more" turns** — inject already-generated lever names into turns 2 and 3 to reduce cross-call thematic redundancy.
4. **Add per-call lever count validation** — `levers_raw.extend()` has no count guard; llama3.1 produced 20 levers instead of 15.
5. **Add model preflight check** — move `LLMModelFromName.from_names()` before the plan loop to catch bad model aliases early.
6. **Fix `Lever.options` field description** from `"2-5 options"` to `"Exactly 3 options"` (B7 from synthesis 0).
7. **Design the evaluator prompt and scoring rubric.** Define concrete dimensions (completeness, specificity, actionability, structure) and a numeric scale. Version-control it alongside the system prompts.
8. **Build `evaluator.py`** — calls a pinned reasoning model to score/compare runner outputs against baseline.
9. **Run baseline scoring** — score the current default prompt outputs to establish a numeric starting point.
10. **Build `candidate_generator.py`** — produce prompt variants (LLM rewrites, structured mutations), informed by synthesis conclusions and failed-attempt logs.
11. **Build `optimizer.py`** — orchestrate the train/verify loop (Stage 1-2).
12. **Add regression detection** — after optimizing one step, re-run downstream steps to check for cascade regressions before committing.
13. **Extend runner to other pipeline steps** — the runner is designed for this; each step needs an adapter for input assembly, execute call, output filenames, and step name.


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

prompt_optimizer/                   # NEW — the optimization engine
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
prompts/
  identify_potential_levers/
    prompt_0_e51751c30bc0c48402ecf759afdb996d8067cd8c5f057d0e242a9d93a856151e.txt      # prompt_index_sha.txt
    prompt_1_long-sha-here.txt
    prompt_2_long-sha-here.txt
analysis/
  AGENTS.md                         # conventions for analysis artifacts
  create_analysis_dir.py            # phase 0: create analysis dir with unanalyzed runs
  run_insight.py                    # phase 1: Claude + Codex insight in parallel
  run_code_review.py                # phase 2: Claude + Codex code review in parallel
  run_synthesis.py                  # phase 3: Claude synthesis (single agent)
  run_assessment.py                 # phase 4: before/after comparison + verdict
  update_meta_pr.py                 # register PR info in meta.json
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

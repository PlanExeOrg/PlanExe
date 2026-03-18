# Prompt Optimizer

Optimizes system prompts for pipeline steps in `run_plan_pipeline.py` through
an iterative loop: implement a fix, test across models, analyze results, and
decide whether to keep or revert. Each iteration produces an auditable trail
with a PR, quantitative comparison, and a keeper verdict.

Currently optimizing the `IdentifyPotentialLevers` step (26 iterations
completed). The runner infrastructure (progress tracking, CLI, output
structure) is shared across steps; each new step requires only a custom
adapter.

See [proposal 117](../docs/proposals/117-system-prompt-optimizer.md) for the
full design and current status.

## Optimization Loop

```
Read latest synthesis.md (top recommendation)
  → Implement fix (Claude Code → branch + PR)
    → Run experiments (runner.py × N models × 5 plans)
      → Analyze (insight → code review → synthesis → assessment)
        → Verdict: YES → merge, NO → close, CONDITIONAL → fix + re-test
          → Loop back
```

Each iteration produces a numbered analysis directory in PlanExe-prompt-lab:

```
analysis/1_identify_potential_levers/
  meta.json           ← provenance: history runs, PR info
  insight_claude.md   ← quality analysis (Claude Code)
  insight_codex.md    ← quality analysis (Codex)
  code_claude.md      ← code review (Claude Code)
  code_codex.md       ← code review (Codex)
  synthesis.md        ← cross-agent reconciliation, top 5 directions
  assessment.md       ← before/after comparison, keeper verdict
```

### Running a full iteration

```bash
# From PlanExe-prompt-lab repo:
python run_optimization_iteration.py

# Or step-by-step:
python run_optimization_iteration.py --skip-implement --models haiku,llama
python run_optimization_iteration.py --skip-implement --skip-runner
```

See `run_optimization_iteration.py --help` for all options.

**Important:** Do NOT merge the PR before the verdict. The correct order is:
create PR → run experiments → run analysis → read verdict → merge only if
verdict confirms improvement.

### Running analysis phases

```bash
# From PlanExe-prompt-lab repo:
python analysis/prepare_iteration.py identify_potential_levers 326     # Phase 0: verify PR, pre-create history dirs
python analysis/run_analysis.py analysis/12_identify_potential_levers   # Phases 1-4: insight → code review → synthesis → assessment
```

`prepare_iteration.py` replaces the former `create_analysis_dir.py` and
`update_meta_pr.py` — it handles PR verification, history dir pre-creation,
and meta.json population in a single step.

`run_analysis.py` orchestrates the four analysis phases sequentially. Each
phase is resumable (skipped if output files already exist). Individual phases
can still be run directly if needed:

```bash
python analysis/run_insight.py analysis/12_identify_potential_levers     # Phase 1
python analysis/run_code_review.py analysis/12_identify_potential_levers # Phase 2
python analysis/run_synthesis.py analysis/12_identify_potential_levers   # Phase 3
python analysis/run_assessment.py analysis/12_identify_potential_levers  # Phase 4
```

## Runner Usage

The runner must be invoked with Python 3.11 (`/opt/homebrew/bin/python3.11`)
which has the required llama_index dependencies.

```bash
# Auto-increment into prompt-lab history/
/opt/homebrew/bin/python3.11 -m self_improve.runner \
    --baseline-dir /path/to/baseline/train \
    --prompt-lab-dir /path/to/PlanExe-prompt-lab \
    --model ollama-llama3.1

# Or manual output directory
/opt/homebrew/bin/python3.11 -m self_improve.runner \
    --baseline-dir /path/to/baseline/train \
    --output-dir /path/to/my_run/outputs \
    --model ollama-llama3.1
```

The runner always uses the `IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT` constant
from PlanExe's code — there is no external prompt file or CLI override.

### Options

| Flag | Required | Description |
|------|----------|-------------|
| `--baseline-dir` | No | Directory containing plan subdirectories (process all) |
| `--plan-dir` | No | Single plan directory to process (overrides `--baseline-dir`) |
| `--prompt-lab-dir` | No | Path to PlanExe-prompt-lab repo (auto-creates history run dir) |
| `--output-dir` | No | Manual output directory (alternative to `--prompt-lab-dir`) |
| `--model` | Yes | LLM model name. First is primary; additional are fallbacks |

Either `--baseline-dir` or `--plan-dir` must be provided.
Either `--prompt-lab-dir` or `--output-dir` must be provided.

### Anthropic models

Anthropic models need custom profile env vars. The `run_optimization_iteration.py`
script handles this automatically for models in its `CUSTOM_PROFILE_MODELS` dict.

```bash
PLANEXE_MODEL_PROFILE=custom \
PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=anthropic_claude.json \
/opt/homebrew/bin/python3.11 -m self_improve.runner \
    --baseline-dir /path/to/baseline/train \
    --prompt-lab-dir /path/to/PlanExe-prompt-lab \
    --model anthropic-claude-haiku-4-5-pinned
```

## Output Structure

With `--prompt-lab-dir`, outputs go to `history/{counter // 100}/{counter % 100:02d}_{step}/`. The counter auto-increments by scanning existing history directories.

```
<run-dir>/
  meta.json          # written at start: step name, model, workers, system
  outputs.jsonl      # one row per completed plan: {name, status, duration_seconds, error}
  events.jsonl       # timestamped events: run_single_plan_start, _complete, _error
  outputs/
    <plan_name>/
      002-9-potential_levers_raw.json
      002-10-potential_levers.json
      activity_overview.json
      usage_metrics.jsonl
```

## Quick Start

Run against a single plan:

```bash
/opt/homebrew/bin/python3.11 -m self_improve.runner \
    --plan-dir /path/to/baseline/train/20250321_silo \
    --prompt-lab-dir /path/to/PlanExe-prompt-lab \
    --model ollama-llama3.1
```

## Two-Repo Architecture

```
PlanExe/                              PlanExe-prompt-lab/
  self_improve/                     baseline/train/       ← gold-standard outputs
    runner.py                           history/              ← runner output per model
  worker_plan/.../                      analysis/             ← insight/review/synthesis/assessment
    identify_potential_levers.py          prepare_iteration.py  ← Phase 0: PR + history dirs
                                          run_analysis.py       ← Phases 1-4 orchestrator
  llm_config/                           run_optimization_iteration.py
    baseline.json
    anthropic_claude.json
```

PRs are created in PlanExe (the code being optimized). Data and analysis
artifacts are committed directly to PlanExe-prompt-lab.

## Parallelism

The runner automatically parallelizes based on the `luigi_workers` value in `llm_config/*.json` for the model being used. Cloud models typically use 4 workers; local models use 1. The worker count is recorded in `meta.json`.

Each plan runs in its own thread with an independent `LLMExecutor`. Usage metrics use thread-local storage to avoid cross-thread interference.

## Models

7 models tested per iteration (5 plans each = 35 runs per batch):

| Alias | Model | Status |
|-------|-------|--------|
| llama | ollama-llama3.1 | Working (3-5/5) |
| gpt-oss | openrouter-openai-gpt-oss-20b | Working (4-5/5) |
| gpt5-nano | openai-gpt-5-nano | Working (5/5) |
| qwen | openrouter-qwen3-30b-a3b | Working (5/5) |
| gpt4o-mini | openrouter-openai-gpt-4o-mini | Working (5/5) |
| gemini-flash | openrouter-gemini-2.0-flash-001 | Working (5/5) — added iter 13 |
| haiku | anthropic-claude-haiku-4-5-pinned | Working (4-5/5) |

Removed models:
- `nemotron` (openrouter-nvidia-nemotron-3-nano-30b-a3b) — 0/5 all iterations, structural incompatibility

## Iteration History

| Iter | PR | Change | Verdict |
|------|-----|--------|---------|
| 0 | — | Baseline | — |
| 1 | [#268](https://github.com/PlanExeOrg/PlanExe/pull/268) | Fix doubled user prompt | YES |
| 2 | [#270](https://github.com/PlanExeOrg/PlanExe/pull/270) | Fix assistant turn serialization | CONDITIONAL |
| 3 | [#272](https://github.com/PlanExeOrg/PlanExe/pull/272) | Novelty-aware follow-up prompts | YES |
| 4 | [#273](https://github.com/PlanExeOrg/PlanExe/pull/273) | Remove exemplar strings + optional wrapper fields | CONDITIONAL |
| 5 | [#274](https://github.com/PlanExeOrg/PlanExe/pull/274) | Align Pydantic field descriptions with system prompt | YES |
| 6 | [#275](https://github.com/PlanExeOrg/PlanExe/pull/275) | Fix consequences length + trade-off + review format | YES |
| 7 | [#276](https://github.com/PlanExeOrg/PlanExe/pull/276) | Enforce schema contract: levers min/max 5, summary required | CONDITIONAL |
| 8 | [#278](https://github.com/PlanExeOrg/PlanExe/pull/278) | Fresh context per call + relax lever count to 5–7 | CONDITIONAL |
| 9 | [#279](https://github.com/PlanExeOrg/PlanExe/pull/279) | Remove naming template | YES |
| 10 | [#281](https://github.com/PlanExeOrg/PlanExe/pull/281) | Keyword quality gate (reverted in [#282](https://github.com/PlanExeOrg/PlanExe/pull/282)) | NO |
| 11 | [#283](https://github.com/PlanExeOrg/PlanExe/pull/283) | RetryConfig in runner (reverted in [#284](https://github.com/PlanExeOrg/PlanExe/pull/284)) | NO |
| 12 | [#286](https://github.com/PlanExeOrg/PlanExe/pull/286) | Remove max_length=7 hard constraint on levers | YES |
| 13 | [#289](https://github.com/PlanExeOrg/PlanExe/pull/289) | Add options count and review format validators | CONDITIONAL |
| 14 | [#292](https://github.com/PlanExeOrg/PlanExe/pull/292) | Recover partial results when a call fails | YES |
| 15 | [#294](https://github.com/PlanExeOrg/PlanExe/pull/294) | Consolidate review_lever prompt to prevent format alternation | YES |
| 16 | [#295](https://github.com/PlanExeOrg/PlanExe/pull/295) | Continue loop after call failure instead of breaking | YES |
| 17 | [#296](https://github.com/PlanExeOrg/PlanExe/pull/296) | Auto-correct review_lever before hard-rejecting | CONDITIONAL |
| 18 | [#297](https://github.com/PlanExeOrg/PlanExe/pull/297) | Simplify lever prompt to restore content quality | YES |
| 19 | [#299](https://github.com/PlanExeOrg/PlanExe/pull/299) | Remove bracket placeholders and fragile English-only validator | CONDITIONAL |
| 20 | [#309](https://github.com/PlanExeOrg/PlanExe/pull/309) | Add option-quality reminder to call-2/3 prompts | YES |
| 21 | [#313](https://github.com/PlanExeOrg/PlanExe/pull/313) | Add anti-fabrication reminder to call-2/3 prompts | CONDITIONAL |
| 22 | [#316](https://github.com/PlanExeOrg/PlanExe/pull/316) | Replace two-bullet review_lever prompt with single flowing example | CONDITIONAL |
| 23 | [#326](https://github.com/PlanExeOrg/PlanExe/pull/326) | Add second review_lever example to break template lock | KEEP |
| 24 | [#334](https://github.com/PlanExeOrg/PlanExe/pull/334) | (INVALID — runner ran against main, not PR branch) | INVALID |
| 25 | [#334](https://github.com/PlanExeOrg/PlanExe/pull/334) | Remove unused summary field, slim call-2/3 prefix | YES |
| 26 | [#337](https://github.com/PlanExeOrg/PlanExe/pull/337) | Replace generic review_lever examples with domain-specific ones | YES |

Full analysis artifacts for each iteration are in
[PlanExe-prompt-lab/analysis/](https://github.com/PlanExeOrg/PlanExe-prompt-lab/tree/main/analysis).

## Critical Rules

1. **Do NOT merge PRs before the verdict.** Create PR → run experiments → run
   analysis → read verdict → merge only if confirmed.
2. **No hardcoded English keywords in validators.** PlanExe users create plans
   in many languages. All validation must be language-agnostic.
3. **Never delete from the history directory.** Runs are permanent records.
4. **The runner always uses the code constant.** There is no external prompt file
   or CLI override. To change the prompt, modify `identify_potential_levers.py`
   on the PR branch before running experiments.
5. **Verify the runner is on the PR branch, not main.** The runner imports code
   from PlanExe, so it must be on the PR branch to test the PR's changes.
   `run_optimization_iteration.py` verifies this automatically. Iteration 24
   was invalidated because the runner ran against main.

## Architecture

The runner is designed to extend to other pipeline steps. Each step needs four things:

1. **Input files** — which files to read and how to assemble the user prompt
2. **Execute call** — which class/method to invoke
3. **Output filenames** — which files to save
4. **Step name** — identifier for meta.json

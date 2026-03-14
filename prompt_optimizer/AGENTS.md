# Agent instructions for prompt_optimizer

## What this does

Optimizes system prompts for pipeline steps in `run_plan_pipeline.py`. Re-executes a step with a candidate system prompt against baseline training data and captures the output. Currently supports `IdentifyPotentialLevers`; will extend to other pipeline tasks.

## Optimization Flow

The optimizer runs in a loop. Each iteration reads the previous analysis,
implements the top recommendation, tests it across models, and evaluates the
result. The loop produces an auditable trail: every change has a PR, every PR
has an assessment verdict (YES/NO/CONDITIONAL).

```
                    ┌──────────────────────────────────────┐
                    │        Read latest synthesis.md       │
                    │     (extract top recommendation)      │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │     Step 1: Implement recommendation  │
                    │  Claude Code creates branch + PR in   │
                    │  PlanExe repo (code fix or prompt)    │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │     Step 2: Run experiments           │
                    │  runner.py × N models × 5 plans      │
                    │  outputs land in prompt-lab history/  │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │     Step 3: Analysis pipeline         │
                    │                                       │
                    │  Phase 0: create_analysis_dir.py      │
                    │    (diff history vs already-analyzed)  │
                    │                                       │
                    │  Phase 1: run_insight.py               │
                    │    (Claude + Codex in parallel)        │
                    │                                       │
                    │  Phase 2: run_code_review.py           │
                    │    (Claude + Codex in parallel)        │
                    │                                       │
                    │  Phase 3: run_synthesis.py              │
                    │    (Claude only — reconciles analyses) │
                    │                                       │
                    │  Phase 4: run_assessment.py             │
                    │    (compare before vs after, verdict)  │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │     Register PR in meta.json          │
                    │  update_meta_pr.py <dir> <PR#>        │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │     Verdict: is the PR a keeper?       │
                    │  YES → merge PR, loop back to top     │
                    │  NO  → close PR, adjust approach      │
                    │  CONDITIONAL → fix issues, re-test    │
                    └──────────────────────────────────────┘
```

### Running a full iteration

```bash
# From the prompt-lab repo:
python run_optimization_iteration.py
```

This reads the latest `synthesis.md`, implements the recommendation, runs
experiments, and runs the full analysis pipeline. Supports `--skip-implement`,
`--skip-runner`, `--skip-analysis`, and `--models` for partial runs.

### Running steps individually

```bash
# Step 1: implement (or do it manually, then --skip-implement)
# Step 2: run experiments for specific models
python run_optimization_iteration.py --skip-implement --models haiku,llama

# Step 3: analysis phases individually
python analysis/create_analysis_dir.py identify_potential_levers      # Phase 0
python analysis/run_insight.py analysis/1_identify_potential_levers    # Phase 1
python analysis/run_code_review.py analysis/1_identify_potential_levers # Phase 2
python analysis/run_synthesis.py analysis/1_identify_potential_levers   # Phase 3
python analysis/update_meta_pr.py analysis/1_identify_potential_levers 268  # Register PR
python analysis/run_assessment.py analysis/1_identify_potential_levers  # Phase 4
```

## Two-Repo Architecture

The optimizer spans two repositories:

- **PlanExe** (`prompt_optimizer/runner.py`, pipeline step source code) —
  the code being optimized. PRs are created here.
- **PlanExe-prompt-lab** (data repo) — baseline training data, history
  outputs, registered prompts, and analysis artifacts. No PRs; commits
  directly to main.

```
PlanExe/                              PlanExe-prompt-lab/
  prompt_optimizer/                     baseline/train/          ← gold-standard outputs
    runner.py                           prompts/                 ← registered system prompts
    register_prompt.py                  history/                 ← runner output per model
  worker_plan/.../                      analysis/                ← insight/review/synthesis/assessment
    identify_potential_levers.py          AGENTS.md
  llm_config/                             create_analysis_dir.py
    baseline.json                         run_insight.py
    anthropic_claude.json                 run_code_review.py
                                          run_synthesis.py
                                          run_assessment.py
                                          update_meta_pr.py
                                        run_optimization_iteration.py
```

## Analysis Artifacts Per Iteration

Each iteration produces a numbered directory in `analysis/`:

```
analysis/1_identify_potential_levers/
  meta.json           ← provenance: prompt, history runs, PR info
  insight_claude.md   ← independent quality analysis (Claude Code)
  insight_codex.md    ← independent quality analysis (Codex)
  code_claude.md      ← code review informed by insights (Claude Code)
  code_codex.md       ← code review informed by insights (Codex)
  synthesis.md        ← cross-agent reconciliation, top 5 directions, recommendation
  assessment.md       ← before/after comparison, metric table, keeper verdict
```

The `meta.json` links to the PR being evaluated:

```json
{
  "prompt": "identify_potential_levers/prompt_0_fa5dfb88...txt",
  "pr_url": "https://github.com/PlanExeOrg/PlanExe/pull/268",
  "pr_title": "fix: remove doubled user prompt (B1)",
  "pr_description": "...",
  "history": ["0/09_identify_potential_levers", "0/10_identify_potential_levers", ...]
}
```

## Assessment Verdicts

Each iteration ends with an assessment that compares the before and after
analyses and produces a verdict:

- **YES** — the PR improved quality, merge it.
- **NO** — the PR made things worse or did not help, close it.
- **CONDITIONAL** — the PR helps but needs additional changes before merging.

The assessment compares metrics only for models that appear in both batches:
success rate, bracket placeholder leakage, option count violations, lever
name uniqueness, template leakage, review format compliance, consequence chain
format, content depth, and cross-call duplication.

## Prerequisites

- Python venv: `worker_plan/.venv/bin/python` (has llama_index and dependencies)
- Ollama running locally with a model (e.g. `ollama-llama3.1`)
- Baseline training data at: `/Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/baseline/train/`

## Extract the default system prompt

```bash
worker_plan/.venv/bin/python -c "
import sys; sys.path.insert(0, 'worker_plan')
from worker_plan_internal.lever.identify_potential_levers import IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT
print(IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT.strip())
" > /tmp/baseline_prompt.txt
```

## Run against a single plan (auto-increment history)

```bash
worker_plan/.venv/bin/python -m prompt_optimizer.runner \
    --system-prompt-file /tmp/baseline_prompt.txt \
    --plan-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/baseline/train/20250321_silo \
    --prompt-lab-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab \
    --model ollama-llama3.1
```

## Run against all baseline plans (auto-increment history)

```bash
worker_plan/.venv/bin/python -m prompt_optimizer.runner \
    --system-prompt-file /tmp/baseline_prompt.txt \
    --baseline-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/baseline/train \
    --prompt-lab-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab \
    --model ollama-llama3.1
```

## Run with Anthropic models

Anthropic models live in `llm_config/anthropic_claude.json`, which is not loaded by default. Set `PLANEXE_MODEL_PROFILE=custom` and `PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=anthropic_claude.json` to make them available:

```bash
PLANEXE_MODEL_PROFILE=custom \
PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=anthropic_claude.json \
worker_plan/.venv/bin/python -m prompt_optimizer.runner \
    --system-prompt-file /tmp/baseline_prompt.txt \
    --baseline-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/baseline/train \
    --prompt-lab-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab \
    --model anthropic-claude-haiku-4-5-pinned
```

Available Anthropic model names are defined in `llm_config/anthropic_claude.json`. Requires `ANTHROPIC_API_KEY` in `.env`.

**Note:** Anthropic's LlamaIndex integration overrides `structured_predict()` and bypasses `self.chat()`, so LlamaIndex instrumentation events don't fire. `usage_metrics.jsonl` will contain basic entries (model, duration, success) but no token counts. `activity_overview.json` is not generated.

## Run with manual output dir (no history)

```bash
worker_plan/.venv/bin/python -m prompt_optimizer.runner \
    --system-prompt-file /tmp/baseline_prompt.txt \
    --plan-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/baseline/train/20250321_silo \
    --output-dir /tmp/prompt_opt_run/outputs \
    --model ollama-llama3.1
```

## Monitor progress during a run

```bash
tail -f /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/history/0/00_identify_potential_levers/events.jsonl
```

## Resume an interrupted run

Re-run the same command. Plans with status `ok` in `outputs.jsonl` are skipped. Errored plans are retried.

```bash
# Same command as before — skips already-completed plans
worker_plan/.venv/bin/python -m prompt_optimizer.runner \
    --system-prompt-file /tmp/baseline_prompt.txt \
    --baseline-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/baseline/train \
    --output-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab/history/0/00_identify_potential_levers/outputs \
    --model ollama-llama3.1
```

## Output structure

With `--prompt-lab-dir`, outputs go to `history/{counter // 100}/{counter % 100:02d}_{step}/`:

```
history/0/00_identify_potential_levers/
  meta.json          # written at start: step name, system_prompt SHA256, model, workers, system
  outputs.jsonl      # one row per completed plan: {name, status, duration_seconds, error}
  events.jsonl       # timestamped events: run_single_plan_start, _complete, _error
  outputs/
    <plan_name>/
      002-9-potential_levers_raw.json
      002-10-potential_levers.json
      activity_overview.json
      usage_metrics.jsonl
```

The counter auto-increments by scanning existing history directories.

## Parallelism

The runner reads `luigi_workers` from `llm_config/*.json` for the given model. Cloud models (e.g. `openrouter-openai-gpt-oss-20b`) typically have `luigi_workers: 4`; local models (e.g. `ollama-llama3.1`) have `luigi_workers: 1`. The worker count is recorded in `meta.json`.

With `workers > 1`, plans run in parallel using a thread pool. Thread safety:

- Each plan gets its own `LLMExecutor` instance.
- `set_usage_metrics_path` uses thread-local storage — each thread writes to its own `usage_metrics.jsonl`.
- `TrackActivity` event handlers guard against duplicate writes by checking the thread-local path matches the handler's output directory.
- `outputs.jsonl` and `events.jsonl` writes are protected by a lock.

## Timing

- **Local** (`ollama-llama3.1`, 1 worker): ~60-80s per plan, ~5-7 min for 5 plans.
- **Cloud** (`openrouter-openai-gpt-oss-20b`, 4 workers): ~30-180s per plan, ~3 min for 5 plans in parallel.

## Available baseline plans

- `20250321_silo`
- `20250329_gta_game`
- `20260308_sovereign_identity`
- `20260310_hong_kong_game`
- `20260311_parasomnia_research_unit`

## Register a system prompt

Saves the current system prompt for a step into the prompt-lab repo. Auto-increments the index and skips duplicates (by SHA256).

```bash
worker_plan/.venv/bin/python -m prompt_optimizer.register_prompt \
    --step identify_potential_levers \
    --prompt-lab-dir /Users/neoneye/git/PlanExeGroup/PlanExe-prompt-lab
```

Output: `prompts/identify_potential_levers/prompt_{index}_{sha256}.txt`

Available steps: `identify_potential_levers`

## Architecture notes

The runner is designed to extend to other pipeline steps. Each step adapter needs:

1. **Input files** — which files to read and how to assemble the user prompt
2. **Execute call** — which class/method to invoke
3. **Output filenames** — which files to save
4. **Step name** — identifier for meta.json

The outer infrastructure (CLI, progress tracking via events.jsonl/outputs.jsonl, meta.json) is shared across all steps.

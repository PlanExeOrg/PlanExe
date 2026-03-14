# Prompt Optimizer

Optimizes system prompts for pipeline steps in `run_plan_pipeline.py` through
an iterative loop: implement a fix, test across models, analyze results, and
decide whether to keep or revert. Each iteration produces an auditable trail
with a PR, quantitative comparison, and a keeper verdict.

Currently optimizing the `IdentifyPotentialLevers` step. The runner
infrastructure (progress tracking, CLI, output structure) is shared across
steps; each new step requires only a custom adapter.

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
  meta.json           ← provenance: prompt, history runs, PR info
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

### Running analysis phases individually

```bash
# From PlanExe-prompt-lab repo:
python analysis/create_analysis_dir.py identify_potential_levers       # Phase 0
python analysis/run_insight.py analysis/1_identify_potential_levers     # Phase 1
python analysis/run_code_review.py analysis/1_identify_potential_levers # Phase 2
python analysis/run_synthesis.py analysis/1_identify_potential_levers   # Phase 3
python analysis/update_meta_pr.py analysis/1_identify_potential_levers 268
python analysis/run_assessment.py analysis/1_identify_potential_levers  # Phase 4
```

## Runner Usage

```bash
# Auto-increment into prompt-lab history/
python -m prompt_optimizer.runner \
    --system-prompt-file candidate.txt \
    --baseline-dir /path/to/baseline/train \
    --prompt-lab-dir /path/to/PlanExe-prompt-lab \
    --model ollama-llama3.1

# Or manual output directory
python -m prompt_optimizer.runner \
    --system-prompt-file candidate.txt \
    --baseline-dir /path/to/baseline/train \
    --output-dir /path/to/my_run/outputs \
    --model ollama-llama3.1
```

### Options

| Flag | Required | Description |
|------|----------|-------------|
| `--system-prompt-file` | Yes | Path to a text file containing the candidate system prompt |
| `--baseline-dir` | No | Directory containing plan subdirectories (process all) |
| `--plan-dir` | No | Single plan directory to process (overrides `--baseline-dir`) |
| `--prompt-lab-dir` | No | Path to PlanExe-prompt-lab repo (auto-creates history run dir) |
| `--output-dir` | No | Manual output directory (alternative to `--prompt-lab-dir`) |
| `--model` | Yes | LLM model name. First is primary; additional are fallbacks |

Either `--baseline-dir` or `--plan-dir` must be provided.
Either `--prompt-lab-dir` or `--output-dir` must be provided.

### Anthropic models

Anthropic models need custom profile env vars:

```bash
PLANEXE_MODEL_PROFILE=custom \
PLANEXE_LLM_CONFIG_CUSTOM_FILENAME=anthropic_claude.json \
python -m prompt_optimizer.runner \
    --system-prompt-file candidate.txt \
    --baseline-dir /path/to/baseline/train \
    --prompt-lab-dir /path/to/PlanExe-prompt-lab \
    --model anthropic-claude-haiku-4-5-pinned
```

## Output Structure

With `--prompt-lab-dir`, outputs go to `history/{counter // 100}/{counter % 100:02d}_{step}/`. The counter auto-increments by scanning existing history directories.

```
<run-dir>/
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

## Quick Start

Extract the current default system prompt to a file, then run against a single plan:

```bash
python -c "
import sys; sys.path.insert(0, 'worker_plan')
from worker_plan_internal.lever.identify_potential_levers import IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT
print(IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT.strip())
" > baseline_prompt.txt

python -m prompt_optimizer.runner \
    --system-prompt-file baseline_prompt.txt \
    --plan-dir /path/to/baseline/train/20250321_silo \
    --prompt-lab-dir /path/to/PlanExe-prompt-lab \
    --model ollama-llama3.1
```

## Two-Repo Architecture

```
PlanExe/                              PlanExe-prompt-lab/
  prompt_optimizer/                     baseline/train/       ← gold-standard outputs
    runner.py                           prompts/              ← registered system prompts
    register_prompt.py                  history/              ← runner output per model
  worker_plan/.../                      analysis/             ← insight/review/synthesis/assessment
    identify_potential_levers.py        run_optimization_iteration.py
  llm_config/
    baseline.json
    anthropic_claude.json
```

PRs are created in PlanExe (the code being optimized). Data and analysis
artifacts are committed directly to PlanExe-prompt-lab.

## Parallelism

The runner automatically parallelizes based on the `luigi_workers` value in `llm_config/*.json` for the model being used. Cloud models typically use 4 workers; local models use 1. The worker count is recorded in `meta.json`.

Each plan runs in its own thread with an independent `LLMExecutor`. Usage metrics use thread-local storage to avoid cross-thread interference.

## Iteration History

| Iteration | PR | Change | Verdict | Key Result |
|-----------|-----|--------|---------|------------|
| 0 | — | Baseline (no change) | — | 8 models tested, 3 failed completely |
| 1 | [#268](https://github.com/PlanExeOrg/PlanExe/pull/268) | Fix doubled user prompt (B1) | **YES** | Review violations 67→4, bracket leakage 17→1 |

## Architecture

The runner is designed to extend to other pipeline steps. Each step needs four things:

1. **Input files** — which files to read and how to assemble the user prompt
2. **Execute call** — which class/method to invoke
3. **Output filenames** — which files to save
4. **Step name** — identifier for meta.json

---
name: run-napkin-math-pipeline
description: Use when the user wants to run the napkin-math pipeline end-to-end on a PlanExe report, or resume a partially populated output directory by filling in only the missing stages. Orchestrates digest preparation, parameter extraction, validation, bounds, calculations, scenarios, Monte Carlo, and assessment rendering. Never copies artifacts forward from prior runs, and never re-runs a stage whose output is already on disk.
---

# Run the Napkin-Math Pipeline

## Overview

End-to-end orchestrator for the pipeline documented in
`experiments/napkin_math/README.md`. Decides which stages to run based on
what is already present in the target output directory, then walks
forward through the remaining stages one at a time.

PlanExe source reports live under `/Users/neoneye/git/PlanExe-web/`
(e.g. `/Users/neoneye/git/PlanExe-web/20251114_paperclip_automation/`).
Outputs land under
`experiments/napkin_math/output/<version>/<plan-slug>/`.

## What the orchestrator does and does not do

The orchestrator is **dispatch-only**. It looks at filenames in the target
directory, decides which stage is missing, and delegates the work to the
appropriate sibling skill (via a subagent) or Python runner. It does **not**
read the content of large input files into its own context.

**Do not read into orchestrator context:**

- `extract_parameters_input.md` (the digest — typically ~25K tokens)
- `compress_*.md` or `compress_*_raw.json` files
- The raw PlanExe HTML report
- The full `parameters.json`, `bounds.json`, `scenarios.json`, `montecarlo.json` (except quick `jq`/Python extraction of a single field)

**OK to read into orchestrator context:**

- Directory listings (filenames only)
- `validation.json` exit status (via Python one-liner, not full file)
- `montecarlo.json` per-threshold `probability` values (via Python one-liner)
- Sibling skill SKILL.md / system-prompt.txt **only when updating the skill itself**

Pulling a digest into orchestrator context defeats the architecture. Each
stage's sibling skill knows how to read its own inputs; the orchestrator's
job is to invoke that skill, not to do its work.

## Per-stage delegation

For each missing stage, dispatch as follows. Pass the agent the file
paths and a one-line task; do not paste file contents into the prompt.

| Stage | How to dispatch |
|---|---|
| 0. Digest | `Bash` → `prepare_extract_input.py --planexe-dir <PlanExe-web/...> --output-dir <target> [--prior <prior-version>/<plan-slug>/parameters.json]` (see "Prior-baseline context for reruns" below) |
| 1. Parameters | `Agent` with the sibling skill name `extract-parameters-from-digest`; prompt: "Read `<target>/extract_parameters_input.md` per `system-prompt.txt`, write the result to `<target>/parameters.json`." |
| 2. Validation | `Bash` → `validate_parameters.py --parameters … --output …` |
| 3. Bounds | `Agent` with `generate-bounds`; prompt: "Read `<target>/parameters.json` per the generate-bounds rules, write `<target>/bounds.json`." |
| 4. Calculations | `Agent` with `generate-calculations`; prompt: "Read `<target>/parameters.json`, write `<target>/calculations.py` per the skill rules." |
| 5. Scenarios | `Agent` with `run-scenarios`; prompt: "Read parameters.json, bounds.json, calculations.py, write `<target>/scenarios.json`." |
| 6. MC settings | Small hand-written JSON file based on `parameters.json` — extract `output_name`s via Python one-liner, write thresholds with `>= 0`. No content reading needed. |
| 7. Monte Carlo | `Bash` → `run_monte_carlo.py …` |
| 8. Assessment | `Bash` → `summarize_assessment.py …` |

After each agent returns, verify the output file exists and is non-empty,
then move on. If validation fails after Stage 1, dispatch a repair agent
with the validation report — do not pull the parameters into context yourself.

## Why resume mode exists

Resume is not a convenience. It is the experimental control for this
pipeline.

Several stages — digest preparation, parameter extraction, bounds,
calculations, scenarios — are LLM-driven and therefore
non-deterministic. If every run regenerated them from scratch, every
downstream change would be confounded with upstream sampling noise,
and the user could not tell whether a different `assessment.md` came
from a skill/prompt tweak or just from a different roll of the LLM.

The intended workflow is:

1. On the first run for a plan, populate the directory once. The
   `extract_parameters_input.md` digest (Stage 0) becomes the pinned
   starting point.
2. On subsequent runs — typically after editing a skill prompt, a
   validator rule, or a runner — the user deletes only the artifacts
   downstream of the change, and resumes from the highest-numbered
   stage whose output is still present.
3. Because the input to the first re-run stage is byte-identical to
   the prior run, any downstream difference can be attributed to the
   tweak under test.

The corollary is strict: **never re-run a stage whose output is
already on disk.** Doing so silently changes the input that every
later stage sees, and destroys the comparability the user is trying
to preserve. If the user wants a stage re-run, they will delete its
output file first; absence of the file is the signal to run, presence
is the signal to leave alone.

## Prior-baseline context for reruns

When a pipeline rerun produces a new version v(N) for a plan slug that
already has at least one earlier version on disk, Stage 0 MUST pass
`--prior <prior-version>/<plan-slug>/parameters.json` to
`prepare_extract_input.py`. The script appends a `# Prior Signal Ledger`
section to `extract_parameters_input.md` listing the prior iteration's
named signals (entry ids and output_names with their section,
formula_hint, and depends_on), and stamps a `Prior source:
\`<path>\`` line so the Stage 1 preflight (below) can confirm the
digest was built against the intended accepted baseline rather than an
arbitrary or probe prior. The extract LLM uses the ledger to decide
which prior signals to preserve and which to record in
`dropped_signals` with `origin: "prior_baseline"`. Without the ledger,
the LLM cannot emit prior-baseline-origin drops, and the
source-preservation audit cannot classify v(N-1) → v(N) signal loss.

**Selecting the prior.** The prior is an *accepted baseline* the user
names — typically the immediately-preceding integer-numbered version
(for a rerun at v59, the prior is `v58/<plan-slug>/parameters.json`;
for a comparison against v49, the prior is
`v49/<plan-slug>/parameters.json`). Letter-suffixed dirs (`v52a`,
`v53b`, `v57a_<topic>`, …) are experimental probes, not accepted
baselines; never auto-select one. If the user has not named the
baseline, ask — do not derive it from mtime or "most-recent earlier
version on disk."

**When to omit `--prior`.** First-iteration extractions only — the plan
slug has no earlier `parameters.json` on disk under any accepted
baseline. The script's `build_prior_signal_ledger` returns a
"first-iteration baseline" stub when handed an empty prior, which is
not the same as omitting `--prior` entirely; pass `--prior` only when a
real prior exists.

**Stage 1 preflight (mandatory).** Before invoking the extract skill,
the orchestrator MUST verify that the prior-baseline context is wired
correctly — regardless of whether Stage 0 just ran or the digest is
being resumed from a previous run:

1. Establish whether a prior accepted baseline exists for this slug
   under `output/`. The user names it (or confirms a suggested
   immediately-preceding integer version); if no accepted prior
   exists, this is a first-iteration extraction.
2. If a prior exists, run **two** preflight checks on the digest.
   Presence is necessary but not sufficient — a ledger built from the
   wrong prior (e.g. a probe dir) still passes the presence check, so
   also verify the source stamp:

   ```sh
   # (a) Ledger present?
   grep -q '# Prior Signal Ledger' $D/extract_parameters_input.md

   # (b) Ledger built from the named accepted baseline?
   #     The ledger is stamped with `Prior source:
   #     output/<baseline>/<slug>/parameters.json` (relative to
   #     experiments/napkin_math/) when prepare_extract_input.py was
   #     given a prior under that tree, or with an absolute path
   #     otherwise. Match the baseline+slug suffix:
   grep -q "Prior source:.*$BASELINE/$SLUG/parameters.json" \
     $D/extract_parameters_input.md
   ```

   - Both exit 0 → ledger present and built from the intended baseline
     → proceed to Stage 1.
   - (a) non-zero (ledger absent) **or** (b) non-zero (ledger built
     from a different prior) → STOP. Two acceptable resolutions
     (offer both to the user, do not pick silently):
     - **Regenerate Stage 0 with `--prior`.** Delete
       `extract_parameters_input.md` (and the four `compress_*.md`
       files if a fresh compression is also desired) and re-run Stage 0
       with `--prior <baseline>/<plan-slug>/parameters.json`. The
       pinned-digest cardinal rule does not apply here: the digest is
       being repaired, not re-rolled for convenience.
     - **Record an explicit waiver.** Write a one-line file
       `$D/.no_prior_waiver` naming the reason the rerun is proceeding
       without the prior ledger (e.g. "v(N-1) parameters.json was
       lost; rebuilding without comparability"). The waiver file is
       the *only* path that lets Stage 1 proceed with absent ledger
       when a prior exists.

   This closes the v58 failure mode: a digest produced without
   `--prior` cannot enter Stage 1 unnoticed.

3. If no prior exists for the slug (first-iteration extraction),
   record that explicitly: a missing `# Prior Signal Ledger` is
   expected. No waiver file needed.

## The two cardinal rules

**1. Never copy a pipeline artifact from a different output directory
into the target directory.** Skill prompts, validator rules, and runner
logic evolve between versions; copying `output/<vN-1>/<plan>/<file>` into
`output/<vN>/<plan>/` silently embeds the old rules into the new run.
If at any point you are about to type `cp /…/v{N-1}/<plan>/<file> $D/`,
stop and run the stage that produces `<file>` instead.

**2. Never re-run a stage whose output is already on disk.** The user's
A/B comparability depends on the input to each stage being byte-identical
to the prior run, up to the first deleted artifact. Re-running a present
stage (especially Stage 0, the LLM-driven digest) re-rolls the dice on
its input distribution and confounds every downstream comparison. The
only file presence the orchestrator changes is by *adding* missing files,
never by *replacing* present ones.

Together: file present ⇒ leave alone; file absent ⇒ produce it from
this stage's runner (never by copy).

## Pipeline file map

Each stage produces exactly one or two files in the target directory.
A stage is "done" iff its output file(s) exist in the target dir.

| Order | Stage | Producer | Output file(s) |
|---|---|---|---|
| 0 | Digest preparation | `prepare_extract_input.py` (Python) | `compress_*.md`, `compress_*_raw.json` (4 sections), `extract_parameters_input.md` |
| 1 | Parameter extraction | `extract-parameters-from-digest` skill (LLM) | `parameters.json` |
| 2 | Validation | `validate_parameters.py` (Python) | `validation.json` |
| 3 | Bounds | `generate-bounds` skill (LLM) | `bounds.json` |
| 4 | Calculations | `generate-calculations` skill (LLM) | `calculations.py` |
| 5 | Scenarios | `run-scenarios` skill (LLM, deterministic) | `scenarios.json` |
| 6 | Monte Carlo settings | hand-written | `montecarlo_settings.json` |
| 7 | Monte Carlo | `run_monte_carlo.py` (Python) | `montecarlo.json` |
| 8 | Assessment | `summarize_assessment.py` (Python) | `assessment.md` |

`montecarlo_settings.json` is a tiny JSON file (n_runs, seed,
distribution_default, thresholds) — write it from the threshold list
in `recommended_first_calculations`, with `>= 0` operator for every
margin/surplus/buffer output. Standard values: `n_runs: 10000`,
`seed: 12345`, `distribution_default: "triangular"`.

## Inputs the orchestrator needs

Two scenarios:

**(a) Fresh start from a PlanExe-web report.** User points at
`/Users/neoneye/git/PlanExe-web/<date>_<slug>/` and a target version.
Create `output/<version>/<slug>/` if it doesn't exist. Run stage 0 first.
If a prior accepted baseline exists for this slug under `output/`, ask
the user to name (or confirm) the accepted baseline — typically the
immediately-preceding integer-numbered version — and pass it as
`--prior` to Stage 0 per "Prior-baseline context for reruns". Never
auto-select a letter-suffixed dir (`v52a`, `v53b`, `v57a_<topic>`, …)
as the prior; those are experimental probes.

**(b) Resume from a partially populated output directory.** User points
at `output/<version>/<slug>/`. List the dir, classify which stages are
done, run forward from the first missing stage.

If the user did not name a version, ask. Do not guess from sibling
directories (last-mtime, etc.).

## Workflow

1. **Identify the target directory.** Absolute path under
   `experiments/napkin_math/output/<version>/<plan-slug>/`.
2. **List its contents.** Decide per stage whether the output file is
   present.
3. **Find the first missing stage.** Run it. After it completes,
   re-check and continue with the next missing stage. Do not batch
   multiple stages together until you have verified the previous one
   landed.
4. **For LLM-driven stages, invoke the corresponding sibling skill.**
   Each sibling skill's `system-prompt.txt` is authoritative — read it
   and apply its rules. The orchestrator does not paraphrase or short-circuit those rules.
5. **For Python stages, invoke the script.** See the commands below.
6. **Re-validate after each non-deterministic stage.** Specifically:
   after parameter extraction, run validate-parameters and confirm
   `valid: true` before moving on. If `valid: false`, hand-fix the
   reported violations and re-validate; do not proceed with invalid
   parameters.
7. **Final summary.** When `assessment.md` exists and Monte Carlo
   completed with zero warnings, report the worst-gate verdict, the
   five pass probabilities, and the overall band.

## Commands (Python stages)

Always invoke Python 3.11 explicitly (the runner uses NumPy + llama_index
that resolve under that interpreter):

```sh
PY=/opt/homebrew/bin/python3.11
NM=/Users/neoneye/git/PlanExeGroup/PlanExe/experiments/napkin_math
```

Stage 0 — digest preparation (creates the 8 compress files + the digest):

```sh
$PY $NM/prepare_extract_input.py \
  --planexe-dir /Users/neoneye/git/PlanExe-web/<date>_<slug> \
  --output-dir  $NM/output/<version>/<plan-slug> \
  --prior       $NM/output/<prior-version>/<plan-slug>/parameters.json
```

Omit `--prior` only on first-iteration extractions for a plan slug that
has no earlier `parameters.json` on disk. Otherwise the rerun produces a
digest without the `# Prior Signal Ledger` and the extract LLM cannot
record prior-baseline-origin drops. See "Prior-baseline context for
reruns" above.

Stage 2 — validation:

```sh
$PY $NM/validate_parameters.py \
  --parameters $D/parameters.json \
  --output     $D/validation.json
```

Stage 7 — Monte Carlo:

```sh
$PY $NM/run_monte_carlo.py \
  --parameters   $D/parameters.json \
  --bounds       $D/bounds.json \
  --calculations $D/calculations.py \
  --settings     $D/montecarlo_settings.json \
  --output       $D/montecarlo.json
```

Stage 8 — assessment rendering:

```sh
$PY $NM/summarize_assessment.py \
  --parameters   $D/parameters.json \
  --bounds       $D/bounds.json \
  --scenarios    $D/scenarios.json \
  --montecarlo   $D/montecarlo.json \
  --validation   $D/validation.json \
  --settings     $D/montecarlo_settings.json \
  --extract-input $D/extract_parameters_input.md \
  --calculations $D/calculations.py \
  --output       $D/assessment.md
```

After Stage 7, delete the `__pycache__/` that the runner creates next
to `calculations.py`:

```sh
rm -rf $D/__pycache__
```

## LLM-driven stages — how to actually run them

For stages 1, 3, 4, 5: do not paste the upstream artifacts and improvise.
Walk through the sibling skill's workflow:

- Stage 1 — `extract-parameters-from-digest`
  - Read `$D/extract_parameters_input.md`.
  - Read `../extract-parameters-from-digest/system-prompt.txt`.
  - Produce `parameters.json` per the schema at the end of the system
    prompt. Apply every "Important", "Additional modelling rules", and
    "Formula and dependency rules" section.
- Stage 3 — `generate-bounds`
  - Read `$D/parameters.json` (must be valid).
  - Read `../generate-bounds/system-prompt.txt` and SKILL.md.
  - Apply the actual-vs-commitment default: base centers at the
    source-named committed value unless a named Risk / Issue /
    Decision / premortem / expert-criticism passage forecasts a gap.
  - Skip threshold/target variables (ids ending in `_threshold`,
    `_target`, `_ceiling`, `_floor`, `_limit`) — they enter the
    simulation as their stated single value.
- Stage 4 — `generate-calculations`
  - One Python function per entry in `recommended_first_calculations`.
    Function name = `output_name`; arguments = `depends_on`. No hidden
    state; add divide-by-zero guards only where the formula divides.
- Stage 5 — `run-scenarios`
  - Build three input pools (low / base / high) by selecting the
    matching value from `bounds.json` for bounded variables, and the
    declared `key_values[*].value` for unbounded variables.
  - Run each `recommended_first_calculations` function against each
    pool.
  - Emit `scenarios.json` with `inputs`, `outputs`, `comparison`
    (low/base/high + spread_ratio + spread_absolute), and any
    `warnings`. Compute spread arithmetically; do not copy the
    structure from a prior run.

## Resume-mode decision table

When the user points at a partially populated dir, decide by file
presence only — never by sibling-directory comparison.

| Present | First missing | Action |
|---|---|---|
| nothing | digest | If user gave a PlanExe-web dir, run Stage 0. If they only gave the output dir, ask for the source dir. When a prior version exists for this slug under `output/`, pass `--prior` per "Prior-baseline context for reruns". |
| 8 compress files + digest | `parameters.json` | Run the Stage 1 preflight from "Prior-baseline context for reruns" (ledger present AND `Prior source:` matches the named baseline), then Stage 1. If either check fails, do not proceed — repair the digest or record an explicit waiver. |
| + `parameters.json` | `validation.json` | Run Stage 2. If validation reports errors, fix the parameters and re-validate before continuing. |
| + `validation.json` (valid) | `bounds.json` | Run Stage 3. |
| + `bounds.json` | `calculations.py` | Run Stage 4. |
| + `calculations.py` | `scenarios.json` | Run Stage 5. |
| + `scenarios.json` | `montecarlo_settings.json` | Write settings (n_runs 10000, seed 12345, triangular, `>= 0` thresholds per recommended_first_calculations). |
| + `montecarlo_settings.json` | `montecarlo.json` | Run Stage 7. Expect 0 warnings; if not, diagnose. |
| + `montecarlo.json` | `assessment.md` | Run Stage 8. |

If an intermediate file is present but a later one is also present
with stale numbers (e.g. `bounds.json` updated but `montecarlo.json`
not regenerated), ask the user whether to keep or regenerate the
later file. Do not silently overwrite.

## Anti-patterns

| Pattern | What's wrong | Do this instead |
|---|---|---|
| `cp $D47/parameters.json $D49/` | Bypasses Stage 1 entirely; embeds stale extractor rules | Read the digest and produce a fresh `parameters.json` per the current extract-parameters-from-digest system prompt |
| `cp $D48/bounds.json $D49/` and claiming a "fresh run" | The generate-bounds skill's actual-vs-commitment rule and threshold-skip rule have changed multiple times; copying defeats the point of the update | Read `parameters.json`, apply the current generate-bounds rules from scratch, justify each `low/base/high` against a named source passage |
| `diff -q ...` finds digests are identical, so copy everything forward | Identical digest means Stage 0 doesn't need to re-run. It does NOT mean Stages 1–8 can be copied — those use the current LLM-driven skills, which may have evolved | Re-run Stages 1–8 freshly. If the bounds genuinely land in the same place, that's a valid outcome — but you have to do the work to know |
| Re-running Stage 0 (digest) when `extract_parameters_input.md` is already present, "to be safe" | Destroys the experimental control. The whole point of resuming is to pin the digest so downstream changes are attributable to the tweak under test. Re-rolling it confounds the comparison silently | If the user wants the digest regenerated, they will delete it first. Presence of the file is an explicit instruction to leave it alone |
| Re-running any present LLM-driven stage to "freshen" it | Same as above — re-roll changes the input that the next stage sees, defeating A/B comparability | Only run stages whose output file is absent |
| Skip validation because "parameters look fine" | Validator catches comment word-cap, dead-end variables, formula RHS not declared, fraction range, and more | Always run `validate_parameters.py` after extraction and confirm `valid: true` |
| Write `montecarlo_settings.json` with thresholds copied from another plan | Threshold ids must match this plan's `recommended_first_calculations` `output_name`s | Build the thresholds map from the current `parameters.json` |
| Report the verdict after a stage emits warnings | Warnings often signal threshold-stripping or missing dependencies that change the meaning of pass probabilities | Resolve warnings first, then report |
| Hand-compute Monte Carlo pass probabilities | The LLM can't sample 10k draws correctly in-prompt | Always invoke `run_monte_carlo.py` |

## Self-audit before final report

Before declaring the run complete, verify:

1. `validation.json` shows `valid: true` and 0 errors.
2. `montecarlo.json` has 0 warnings, or the warnings are understood and
   acknowledged in the summary.
3. Every threshold id in `montecarlo_settings.json` matches an
   `output_name` in `parameters.json`'s `recommended_first_calculations`.
4. `assessment.md` exists and its `## Machine summary` block lists
   `overall_risk_band` consistent with the worst per-gate pass
   probability under the band table:

   | Pass rate | Band |
   |---|---|
   | ≥ 0.80 | Robust |
   | 0.50 – 0.80 | Marginal |
   | 0.20 – 0.50 | Fragile |
   | < 0.20 | Critical |

5. No `__pycache__/` directory remains in the output dir.

If any check fails, fix it before reporting back. Reporting "done"
with a known issue is worse than asking the user how to proceed.

## Reference

- Pipeline overview: `../../../README.md` (Stages 1–7 + design principles)
- Sibling skill prompts (authoritative for each stage): `../extract-parameters-from-digest/`, `../generate-bounds/`, `../generate-calculations/`, `../run-scenarios/`, `../monte-carlo/`, `../summarize-assessment/`, `../validate-parameters/`
- Python runners: `../../../prepare_extract_input.py`, `../../../validate_parameters.py`, `../../../run_monte_carlo.py`, `../../../summarize_assessment.py`
- PlanExe source reports: `/Users/neoneye/git/PlanExe-web/<date>_<slug>/`
- Output convention: `experiments/napkin_math/output/v<N>/<plan-slug>/`

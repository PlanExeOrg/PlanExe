---
name: extract-parameters-from-digest
description: Use when the user wants to extract parameters from a PlanExe extraction-input digest (the markdown produced by experiments/napkin_math/prepare_extract_input.py — the 137-recommended section bundle, with the four "Keep or compress" sections compressed) instead of the full PlanExe HTML report
---

# Extract Parameters from a PlanExe Extraction-Input Digest

## Overview

A drop-in alternative to `extract-parameters-from-full` that reads the digest
produced by `prepare_extract_input.py` (see
`experiments/napkin_math/prepare_extract_input.py`) rather than the full
PlanExe HTML report.

The digest is the 137-recommended extraction bundle in 137's order:
Executive Summary, Project Plan, Selected Scenario, Assumptions, Review
Plan, Premortem, Expert Criticism, Data Collection. Strategic Decisions is
replaced by Selected Scenario per proposal 139.

It mixes two formats:

- **Compressed sections** (Selected Scenario, Review Plan, Premortem,
  Expert Criticism) — produced by `compress_report_section`. Bullets carry
  inline epistemic tags of the form
  `[<source_status> | e=N r=N | quote: verified|unverified]`.
- **Raw sections** (Executive Summary, Project Plan, Assumptions, Data
  Collection) — passed through unchanged from the PlanExe source. No
  inline tags.

The system prompt at `system-prompt.txt` explains how to read both
formats.

The digest may also end with a `# Prior Signal Ledger (advisory)`
section listing the previous iteration's named signals. When present,
preserve still-supported prior signals and record drops in
`dropped_signals` with `origin: "prior_baseline"` per the "Prior Signal
Ledger" rules in `system-prompt.txt`. Absence of the ledger is
ambiguous on its own (first-iteration vs. a rerun whose Stage 0 was
called without `--prior`); the orchestrator (`run-napkin-math-pipeline`)
disambiguates it via the Stage 1 preflight before invoking this skill.
When invoked standalone, treat absent-ledger as first-iteration only
when the caller has confirmed there is no prior accepted baseline for
the slug on disk; otherwise stop and ask.

Output schema and hard limits are identical to `extract-parameters-from-full`, so the
two skills can be compared head-to-head on the same plan.

## When to Use

- The user has run `prepare_extract_input.py` against a PlanExe sample and
  wants parameters extracted from the resulting digest
- The user is comparing whether this pipeline produces better parameters
  than feeding the full HTML report

For plain PlanExe HTML/text reports, use `extract-parameters-from-full` instead.

## Workflow

1. **Get the digest path.** Usually
   `experiments/napkin_math/output/<plan-name>/extract_parameters_input.md`.
   If the user did not provide one, ask. Do not guess.
2. **Read `system-prompt.txt`** (sibling of this SKILL.md). Treat it as the
   authoritative extraction instructions.
3. **Read the digest file.** Mid-sized — much smaller than a raw PlanExe
   HTML report. Compressed sections (Selected Scenario, Review Plan,
   Premortem, Expert Criticism) carry inline tags; raw sections (Executive
   Summary, Project Plan, Assumptions, Data Collection) do not.
4. **Canonicalize across sections.** The four compressed sections often
   surface the same real-world quantity under different phrasings
   ("minimum viable rental rate" / "off-peak hourly price" / "speculative
   high hourly rate" all name one rate). Merge near-duplicates into a
   single canonical snake_case id before writing the JSON. Prefer
   framings closest to a modelling primitive (rate, count, fraction,
   amount-per-period). Two ids for the same quantity will silently
   fragment downstream bounds and Monte Carlo correlations.
5. **Produce the JSON** following the schema at the end of `system-prompt.txt`.
   For compressed sections, map the inline `source_status` tags to the JSON
   `value_type` field: `[explicit]` → `explicit`, `[derived]` → `derived`,
   `[inferred]` → `inferred`, `[missing]` items belong in
   `missing_values_to_estimate`, `[stress_test]` items are
   scenario-stress inputs (not baseline `key_values`). For raw sections,
   apply general parameter-extraction triage.
6. **Output destination.** Default: print JSON to the chat. If the user
   asks for a file, write to the path they specify. Default suggestion:
   `<digest-basename>.parameters.json` next to the digest.

## Hard Rules (re-stated for emphasis)

- **JSON only.** No markdown fences, no prose, no explanation.
- **Use the digest's tags where they exist.** For compressed sections,
  prefer `[explicit] + quote: verified` items for baseline `key_values`.
  Treat `quote: unverified` items with extra scepticism. `[missing]`
  items belong in `missing_values_to_estimate`. `[stress_test]` items are
  downside-scenario inputs, not plan facts.
- **For raw sections** (Executive Summary, Project Plan, Assumptions,
  Data Collection), apply general triage: prefer numeric anchors,
  deadlines, denominators, and explicit gate criteria.
- **Canonicalize across sections.** The compressor over-produces on
  purpose; collapsing cross-section near-duplicates into one canonical
  id is your job at this stage, not its job. Never preserve two ids for
  the same real-world quantity.
- **Percentages as fractions** between 0 and 1 with `unit: "fraction"`.
- **No invented ids in `formula_hint`** — every variable must be declared
  in `key_values`, `missing_values_to_estimate`, or the object's own
  `depends_on`.
- **Every entry with a non-null `formula_hint` MUST also declare `output_name`
  (snake_case id of the computed value) and `output_unit`** (e.g. `"DKK"`,
  `"people"`, `"fraction"`). Downstream consumers — generate-calculations,
  run-scenarios, monte-carlo — read these directly and do not parse
  `formula_hint` or pattern-match on tokens. The LLM is the single authority
  for both fields.

## Reference

- System prompt (authoritative): `system-prompt.txt`
- Producer of the input digest: `experiments/napkin_math/prepare_extract_input.py`
- Parallel skill for full HTML reports: `../extract-parameters-from-full/SKILL.md`
- Background: `docs/proposals/137-section_filtering_for_parameter_extraction.md`,
  `docs/proposals/139-compress-for-monte-carlo.md`

---
name: extract-parameters-from-full
description: Use when the user wants to extract parameters, modelling values, or key variables from a PlanExe report (HTML or text) for napkin math, triage, or Monte Carlo simulation
---

# Extract Parameters from a PlanExe Report

## Overview

Wraps the quantitative-triage system prompt at `system-prompt.txt` (next to this file) and applies it to a PlanExe report the user supplies. Output is strict JSON matching the schema in the system prompt — no markdown, no commentary.

## When to Use

- User says "extract parameters", "extract modelling values", "pull key variables", or similar from a PlanExe report
- User points at a PlanExe report file (typically HTML, may be 100KB–1MB+) and wants structured input for downstream simulation
- User wants a triage list of values that would matter for Monte Carlo or sensitivity analysis

Not for: full report summarisation, narrative analysis, code generation. The system prompt explicitly forbids those.

The input may also end with a `# Prior Signal Ledger (advisory)` section
listing the previous iteration's named signals. When present, preserve
still-supported prior signals and record drops in `dropped_signals` with
`origin: "prior_baseline"` per the "Prior Signal Ledger" rules in
`system-prompt.txt`. Absence of the ledger is ambiguous on its own
(first-iteration vs. a rerun whose Stage 0 was called without
`--prior`); the orchestrator (`run-napkin-math-pipeline`) disambiguates
it via the Stage 1 preflight before invoking this skill. When invoked
standalone, treat absent-ledger as first-iteration only when the caller
has confirmed there is no prior accepted baseline for the slug on disk;
otherwise stop and ask.

## Workflow

1. **Get the report path.** If the user did not provide one, ask. Do not guess.
2. **Read `system-prompt.txt`** (sibling of this SKILL.md). Treat its contents as the authoritative extraction instructions — every rule, hard limit, and schema constraint applies.
3. **Read the report file.** For large HTML reports, read the whole file; the system prompt's hard limits (≤8 key_values, ≤5 of each list, ≤25-word comments) keep output bounded regardless of input size.
4. **Produce the JSON** following the exact schema at the end of `system-prompt.txt`. Apply every "Important", "Additional modelling rules", and "Formula and dependency rules" section as you generate each field.
5. **Output destination.** Default: print JSON to the chat. If the user asks for a file, write to the path they specify. If they want a default file path, suggest `<report-basename>.parameters.json` next to the report.

## Hard Rules (from system-prompt.txt — re-stated for emphasis)

- **JSON only.** No markdown fences, no prose, no explanation before or after.
- **Percentages as fractions** between 0 and 1 with `unit: "fraction"`. Never `value: 60` for 60%.
- **No invented ids in `formula_hint`** — every variable must be declared in `key_values`, `missing_values_to_estimate`, or the object's own `depends_on`.
- **Every entry with a non-null `formula_hint` MUST also declare `output_name` (snake_case id of the computed value) and `output_unit`** (e.g. `"DKK"`, `"people"`, `"fraction"`). Downstream consumers — generate-calculations, run-scenarios, monte-carlo — read these directly and do not parse `formula_hint` or pattern-match on tokens. The LLM is the single authority for both fields.
- **Prefer missing-but-needed values over minor explicit values.** Don't dump every budget line.
- **Clean `source_text`** — strip citations, footnote markers, replacement chars, UI artifacts.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Wrapping JSON in ```` ```json ```` fences | Raw JSON only — the system prompt forbids markdown |
| Returning >8 key_values "because the report has many" | Hard cap. Triage. The point is to surface the few that matter |
| Including `suggested_low/base/high` by default | Only include when essential to the value's meaning |
| Using `value: 60` for "60%" | Use `value: 0.6` and `unit: "fraction"` |
| Citing variable in `formula_hint` that isn't declared anywhere | Either add it to `missing_values_to_estimate` or rewrite the formula |
| Picking a descriptive timeline value over a funding gate | Prefer the gate — it determines pass/fail |

## Reference

- System prompt (authoritative): `system-prompt.txt`
- Example report for testing: `/Users/neoneye/git/PlanExe-web/20250720_faraday_enclosure_report.html`

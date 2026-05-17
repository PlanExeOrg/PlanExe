#!/usr/bin/env python3
"""Summarize napkin_math pipeline outputs into a plan assessment.

The output (`assessment.md`) is a thin interpretation layer — a navigation
and judgment file over the intermediary artifacts (`parameters.json`,
`bounds.json`, `calculations.py`, `scenarios.json`,
`montecarlo_settings.json`, `montecarlo.json`, `validation.json`). It does
not reproduce the raw simulation tables — those live in the JSON files and
are referenced via the provenance map.

It declares:
- what this artifact is and isn't (artifact contract)
- a compact machine-readable manifest (JSON block) at the top
- which intermediary file holds what (provenance map)
- critical findings (gates that fail, scenarios that already break, missing
  inputs, model-collapse blanks)
- gate verdicts (DOOM / FRAGILE / MARGINAL / ROBUST) with an aggregation
  warning when units are incompatible
- failure drivers per failing gate (one row, not a full quartile table)
- missing inputs ranked by simulation impact
- per-output confidence and trust boundaries
- a short scenario sanity check
- suggested next actions for whatever consumes this file

Doom verdicts are driven entirely by the user's own threshold definitions
(operator + value). The script never inspects identifier strings, output
names, or units to decide whether a number is good or bad — that would be
domain-bias. It only knows: "the user said this should pass; here is how
often it does in 10k Monte Carlo runs."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ASSESSMENT_SCHEMA_VERSION = 6

VERDICT_BANDS = [
    (0.80, "ROBUST",   "passes in the strong majority of runs"),
    (0.50, "MARGINAL", "passes more often than not but uncomfortably close"),
    (0.20, "FRAGILE",  "fails in the majority of runs"),
    (0.00, "DOOM",     "rarely passes under current bounds"),
]

VERDICT_SEVERITY = {"DOOM": 0, "FRAGILE": 1, "MARGINAL": 2, "ROBUST": 3, "UNKNOWN": 4}

PRIMARY_MODEL_RESULT_FROM_WORST = {
    "DOOM": "doom",
    "FRAGILE": "fragile",
    "MARGINAL": "marginal",
    "ROBUST": "viable",
    "UNKNOWN": "unknown",
}

PRIMARY_RESULT_REASON = {
    "doom": "at least one declared gate has pass rate < 20% (DOOM band)",
    "fragile": "at least one declared gate has pass rate in 20–50% (FRAGILE band)",
    "marginal": "at least one declared gate has pass rate in 50–80% (MARGINAL band)",
    "viable": "every declared gate has pass rate ≥ 80% (ROBUST band)",
    "unknown": "no threshold pass rates available",
}

# Translate the bounds.json `source` label into a less misleading "basis" value.
# `source: data` in this pipeline means "anchored in the source report's
# narrative", NOT externally observed real-world data. Renaming the value
# avoids that confusion downstream.
BASIS_FROM_SOURCE = {
    "data": "report_derived",
    "assumption": "model_assumption",
}

# Translate the `value_type` on a key_value into a `threshold_basis` label.
# Mirrors the bounds.json `source` translation, but for the threshold side.
VALUE_TYPE_TO_THRESHOLD_BASIS = {
    "explicit": "report_explicit",
    "inferred": "report_inferred",
}

SCHEMA_NOTES = {
    "overall_risk_band_enum": ["viable", "marginal", "fragile", "doom", "unknown"],
    "verdict_enum": ["ROBUST", "MARGINAL", "FRAGILE", "DOOM", "UNKNOWN"],
    # basis_enum is the union of values that may appear in the `Basis` column
    # of "Missing inputs ranked by impact" once the pipeline grows. The current
    # pipeline only emits `report_derived` and `model_assumption`; the rest
    # are reserved for future provenance types (external research, manual
    # overrides, etc.).
    "basis_enum": [
        "report_explicit",
        "report_inferred",
        "report_derived",
        "model_assumption",
        "external_reference",
        "manual_override",
        "unknown",
    ],
    "threshold_basis_enum": ["report_explicit", "report_inferred", "model_defined", "unknown"],
    "primary_model_result_semantics": (
        "overall_risk_band reflects the worst declared gate's pass-rate band; "
        "it is not a calibrated whole-plan probability and does not mean the plan is impossible."
    ),
}

OPEN_QUESTIONS = [
    "Are the current input bounds too narrow, too wide, or directionally biased?",
    "Which failed gates are truly independent, and which are correlated?",
    "Which gates are hard stop/go gates versus soft optimisation targets?",
    "Which missing inputs can be replaced by external research or user-supplied facts?",
    "Does the source report contain unmodelled gates that should be added?",
]

DO_NOT_TREAT_AS = [
    "an external feasibility proof for the plan's claims",
    "a real-world probability calibration",
    "a replacement for the source report",
    "evidence that the input bounds match reality",
]

TRUST_NOT_VALIDATED = [
    "real-world accuracy of the input bounds",
    "independence or correlation assumptions across inputs",
    "external feasibility of regulatory, grid, supply, or market dependencies",
    "factual truth of the source plan's narrative claims",
]


def load_json(path: Path) -> dict | None:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"warning: cannot parse {path}: {exc}", file=sys.stderr)
        return None


def fmt_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        abs_v = abs(value)
        if abs_v >= 1_000:
            return f"{value:,.0f}"
        if abs_v >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def verdict_for(probability: float | None) -> tuple[str, str]:
    if probability is None:
        return "UNKNOWN", "no valid runs"
    for cutoff, label, note in VERDICT_BANDS:
        if probability >= cutoff:
            return label, note
    return "DOOM", "rarely passes under current bounds"


def aggregate_output_ids(params: dict | None) -> set[str]:
    """Return output_ids whose formula uses min() over other gates.

    Aggregate magnitudes pick the worst of several independent gates. The
    verdict pass/fail is meaningful, the raw number is not. Surface them
    after individual gates.
    """
    if not params:
        return set()
    ids: set[str] = set()
    for src in ("recommended_first_calculations", "derived_questions"):
        for entry in params.get(src, []):
            hint = entry.get("formula_hint") or ""
            if "min(" in hint and entry.get("output_name"):
                ids.add(entry["output_name"])
    return ids


def threshold_entries(mc: dict | None, params: dict | None) -> list[dict]:
    """One entry per threshold, sorted worst-first; aggregates last within band."""
    if not mc or not mc.get("thresholds"):
        return []
    aggregates = aggregate_output_ids(params)
    outputs = mc.get("outputs") or {}
    rows = []
    for output_id, t in mc["thresholds"].items():
        verdict, note = verdict_for(t.get("probability"))
        unit = (outputs.get(output_id) or {}).get("unit")
        rows.append({
            "id": output_id,
            "operator": t["operator"],
            "value": t["value"],
            "probability": t.get("probability"),
            "verdict": verdict,
            "note": note,
            "is_aggregate": output_id in aggregates,
            "unit": unit,
        })
    rows.sort(key=lambda r: (
        VERDICT_SEVERITY.get(r["verdict"], 99),
        1 if r["is_aggregate"] else 0,
    ))
    return rows


# ─── plan summary frontmatter ──────────────────────────────────────────────

def render_title_and_frontmatter(params: dict | None) -> list[str]:
    if not params:
        return ["# Assessment", "", "_(parameters.json not available)_", ""]
    summary = params.get("plan_summary", {})
    name = summary.get("plan_name") or "unnamed"
    return [
        f"# Assessment: {name}",
        "",
        f"**Type:** {summary.get('plan_type', 'unknown')}  ",
        f"**Primary goal:** {summary.get('primary_goal', '—')}",
        "",
    ]


# ─── artifact contract ─────────────────────────────────────────────────────

def render_artifact_contract() -> list[str]:
    return [
        "## Artifact contract",
        "",
        "This assessment is a derived interpretation layer over the simulation artifacts listed in the provenance map below. It summarises what the model tested, which gates fail or pass, which inputs drive the result, which assumptions remain unvalidated, and what to inspect next. It is **not** a copy of the raw simulation data: for exact distributions, bounds rationales, formulas, and run settings, open the referenced artifacts directly.",
        "",
    ]


# ─── machine summary (JSON) ────────────────────────────────────────────────

def derive_primary_model_result(thresholds: list[dict]) -> dict:
    """Structured headline result. The field name is `overall_risk_band` rather
    than `label` so downstream consumers don't read it as a whole-plan
    feasibility verdict — it is the worst declared gate's pass-rate band.
    """
    base = {
        "basis": "worst declared gate's pass-rate band; not a calibrated whole-plan probability",
    }
    if not thresholds:
        return {
            **base,
            "overall_risk_band": "unknown",
            "reason": PRIMARY_RESULT_REASON["unknown"],
            "worst_gate": None,
            "worst_gate_pass_rate": None,
        }
    worst_severity = min(VERDICT_SEVERITY.get(r["verdict"], 99) for r in thresholds)
    worst_verdict = next(
        (k for k, v in VERDICT_SEVERITY.items() if v == worst_severity),
        "UNKNOWN",
    )
    band = PRIMARY_MODEL_RESULT_FROM_WORST.get(worst_verdict, "unknown")
    worst_band_rows = [r for r in thresholds if r["verdict"] == worst_verdict]
    worst_band_rows.sort(key=lambda r: r["probability"] if r["probability"] is not None else 1.0)
    worst_gate = worst_band_rows[0] if worst_band_rows else None
    return {
        **base,
        "overall_risk_band": band,
        "reason": PRIMARY_RESULT_REASON.get(band, PRIMARY_RESULT_REASON["unknown"]),
        "worst_gate": worst_gate["id"] if worst_gate else None,
        "worst_gate_pass_rate": worst_gate["probability"] if worst_gate else None,
    }


def lookup_gate_metadata(params: dict | None, gate_id: str) -> dict:
    """Find the gate's own rationale and its threshold parameter, from parameters.json.

    The recommended_first_calculations / derived_questions entries carry a
    `why_first` / `why_it_matters` line written by the extractor for THIS plan;
    surfacing it lets the insights file include plan-specific framing without
    inventing tactical advice. The threshold parameter (the key_value the
    formula tests against) carries a `value_type` (explicit/inferred) — that
    distinguishes thresholds the plan states directly from those the
    extractor inferred.
    """
    if not params:
        return {
            "why": None, "threshold_param_id": None,
            "threshold_value_type": None, "gate_found": False,
        }
    key_values_by_id = {kv["id"]: kv for kv in params.get("key_values", [])}
    for src in ("recommended_first_calculations", "derived_questions"):
        for entry in params.get(src, []):
            if entry.get("output_name") != gate_id:
                continue
            why = entry.get("why_first") or entry.get("why_it_matters")
            threshold_param = None
            for d in entry.get("depends_on", []):
                if d in key_values_by_id:
                    threshold_param = key_values_by_id[d]
                    break
            return {
                "why": why,
                "threshold_param_id": threshold_param["id"] if threshold_param else None,
                "threshold_value_type": threshold_param.get("value_type") if threshold_param else None,
                "gate_found": True,
            }
    return {
        "why": None, "threshold_param_id": None,
        "threshold_value_type": None, "gate_found": False,
    }


def threshold_basis_for(value_type: str | None, gate_found: bool = False) -> str:
    """Categorise where the threshold's value came from.

    - report_explicit / report_inferred when the gate's formula tests against
      a declared key_value with a matching value_type.
    - model_defined when the gate exists in parameters.json but its threshold
      is a bare numerical comparison (typically `>= 0`) defined in
      montecarlo_settings.json with no anchoring key_value — that is a
      modelling default, not derived from the source narrative.
    - unknown only when the gate isn't in parameters.json at all (so the
      validator/summarizer can't reason about its origin).
    """
    if value_type:
        return VALUE_TYPE_TO_THRESHOLD_BASIS.get(value_type, "model_defined")
    if gate_found:
        return "model_defined"
    return "unknown"


def derive_artifact_set(params_path: Path | None) -> dict:
    """Best-effort portable identifier parsed from the source directory.

    Looks for an `output/<version>/<plan_slug>/` suffix on the resolved path.
    If the layout doesn't match, version and plan_slug are null and the
    relative_dir falls back to the immediate parent directory name.
    """
    if params_path is None:
        return {"version": None, "plan_slug": None, "relative_dir": None}
    parts = params_path.parent.resolve().parts
    try:
        idx = len(parts) - 1 - parts[::-1].index("output")
    except ValueError:
        return {
            "version": None,
            "plan_slug": params_path.parent.name or None,
            "relative_dir": params_path.parent.name or None,
        }
    relative_parts = parts[idx:]
    return {
        "version": relative_parts[1] if len(relative_parts) > 1 else None,
        "plan_slug": relative_parts[2] if len(relative_parts) > 2 else None,
        "relative_dir": "/".join(relative_parts),
    }


def render_machine_summary(params: dict | None, mc: dict | None,
                           validation: dict | None,
                           params_path: Path | None) -> list[str]:
    plan_summary = (params or {}).get("plan_summary", {}) or {}
    thresholds = threshold_entries(mc, params)
    settings = (mc or {}).get("settings") or {}
    failed = [r["id"] for r in thresholds if r["verdict"] in ("DOOM", "FRAGILE")]
    drivers = [
        e["id"] for e in (mc or {}).get("missing_value_priority", [])[:3]
    ]
    validation_status = "unknown"
    if validation is not None:
        validation_status = "valid" if validation.get("valid") else "invalid"

    unmodelled = (params or {}).get("unmodelled_gates") or []
    unmodelled_ids = [
        g.get("id") for g in unmodelled
        if isinstance(g, dict) and isinstance(g.get("id"), str)
    ]
    scope_warning = (
        "Declared simulation gates cover financial / operational viability only; "
        "the gates listed in `known_unmodelled_existential_gates` are not simulated "
        "and may dominate the modelled financial result."
        if unmodelled_ids else None
    )
    manifest = {
        "assessment_schema_version": ASSESSMENT_SCHEMA_VERSION,
        "artifact_type": "interpretation_layer",
        "plan_name": plan_summary.get("plan_name"),
        "artifact_set": derive_artifact_set(params_path),
        "source_plan_dir": str(params_path.parent.resolve()) if params_path else None,
        "primary_model_result": derive_primary_model_result(thresholds),
        "validation_status": validation_status,
        "simulation": {
            "n_runs": settings.get("n_runs"),
            "seed": settings.get("seed"),
            "distribution_default": settings.get("distribution_default"),
        },
        "primary_failed_gates": failed,
        "primary_uncertainty_drivers": drivers,
        "known_unmodelled_existential_gates": unmodelled_ids,
        "assessment_scope_warning": scope_warning,
        "do_not_treat_as": DO_NOT_TREAT_AS,
        "schema_notes": SCHEMA_NOTES,
    }
    block = json.dumps(manifest, indent=2)
    return [
        "## Machine summary",
        "",
        "Compact manifest for programmatic consumers. The fields below are the structured form of the prose verdicts that follow. `artifact_set.relative_dir` is the portable identifier; `source_plan_dir` is the absolute path on the generating machine.",
        "",
        "```json",
        block,
        "```",
        "",
    ]


# ─── provenance map ────────────────────────────────────────────────────────

PROVENANCE_ROWS = [
    ("extract_parameters_input.md",
     "Source-text digest fed to the parameter extractor.",
     "Auditing what the extractor actually saw."),
    ("parameters.json",
     "Extracted constants, missing inputs, derived questions, formula hints.",
     "Need model structure or to confirm a formula's wording."),
    ("bounds.json",
     "low/base/high ranges with rationales, sampling discipline, source labels.",
     "Auditing the source of an uncertainty range."),
    ("calculations.py",
     "Executable Python formula implementation, one function per gate.",
     "Need the exact arithmetic."),
    ("scenarios.json",
     "Three deterministic scenarios (low/base/high) with outputs and warnings.",
     "Sanity-checking the model against deterministic anchor points."),
    ("scenario_outputs.json",
     "Auxiliary per-scenario output file.",
     "Cross-referencing the scenario outputs."),
    ("montecarlo_settings.json",
     "Run settings: n_runs, seed, distribution_default, threshold definitions.",
     "Reproducing the run or changing the threshold definitions."),
    ("montecarlo.json",
     "Full simulation results: distributions, pass rates, sensitivity, quartile_analysis, binding_gate_analysis, required_input_thresholds, missing_value_priority, model_confidence.",
     "Need raw simulation data, distributions, or per-driver analysis."),
    ("validation.json",
     "Structural validation report (which checks passed, which failed).",
     "Confirming the pipeline ran without structural problems."),
]


def render_provenance_map(present_files: set[str]) -> list[str]:
    rows = [
        "## Provenance map",
        "",
        "Each intermediary file and the question it answers.",
        "",
        "| File | Role | Open when |",
        "|---|---|---|",
    ]
    for name, role, when in PROVENANCE_ROWS:
        marker = "" if name in present_files else " _(not in this run)_"
        rows.append(f"| `{name}`{marker} | {role} | {when} |")
    rows.append("")
    return rows


# ─── modelling frame ───────────────────────────────────────────────────────

def render_modelling_frame(params: dict | None) -> list[str]:
    if not params:
        return []
    frame = (params.get("plan_summary") or {}).get("modelling_frame")
    if not frame:
        return []
    rows = ["## Modelling frame", "", frame, ""]
    unmodelled = (params or {}).get("unmodelled_gates") or []
    if unmodelled:
        n = len(unmodelled)
        rows.append(
            f"**Note:** This assessment is a financial / operational stress test. "
            f"{n} unmodelled existential gate{'s' if n != 1 else ''} "
            f"(legal, political, compliance, or external-actor commitments) "
            f"{'are' if n != 1 else 'is'} listed below but not evaluated by the simulation. "
            f"Treat the gate-verdict pass rates as conditional on those gates holding."
        )
        rows.append("")
    return rows


def render_unmodelled_gates(params: dict | None) -> list[str]:
    """Render the gates the deterministic model cannot evaluate, with source
    anchors. Section omitted entirely when the array is empty or absent.
    """
    if not params:
        return []
    gates = params.get("unmodelled_gates") or []
    if not gates:
        return []
    rows = [
        "## Known unmodelled existential gates",
        "",
        "Gates whose failure would end the plan independently of any financial or operational threshold the model tests. Sourced from the extractor's flag in `parameters.unmodelled_gates`. The simulation does not evaluate these; the source report is the authoritative reference for each.",
        "",
        "| Gate | Why it matters | Source anchor | Consequence if false |",
        "|---|---|---|---|",
    ]
    for g in gates:
        if not isinstance(g, dict):
            continue
        gid = g.get("id", "—")
        why = g.get("why_it_matters", "—")
        raw_anchor = g.get("source_anchor", "—")
        # Prefix with the canonical source artifact so downstream agents can
        # trace the claim back to a specific section of the PlanExe report.
        anchor = f"report.html / {raw_anchor}" if raw_anchor and raw_anchor != "—" else "—"
        consequence = g.get("consequence_if_false", "—")
        rows.append(f"| `{gid}` | {why} | {anchor} | {consequence} |")
    rows.append("")
    return rows


# ─── simulation settings ───────────────────────────────────────────────────

def render_simulation_settings(mc: dict | None, validation: dict | None) -> list[str]:
    if not mc:
        return []
    settings = mc.get("settings") or {}
    if not settings:
        return []
    rows = ["## Simulation settings", ""]
    rows.append(f"- n_runs: {settings.get('n_runs', '—'):,}"
                if isinstance(settings.get("n_runs"), int)
                else f"- n_runs: {settings.get('n_runs', '—')}")
    rows.append(f"- seed: {settings.get('seed', '—')}")
    rows.append(f"- distribution_default: {settings.get('distribution_default', '—')}")
    if validation is not None:
        status = "valid" if validation.get("valid") else "invalid"
        err = validation.get("error_count", 0)
        warn = validation.get("warn_count", 0)
        rows.append(f"- validation: {status} ({err} errors, {warn} warnings)")
    rows.append("")
    return rows


# ─── critical findings ─────────────────────────────────────────────────────

def render_critical_findings(mc: dict | None, scenarios: dict | None,
                             params: dict | None, bounds: dict | None) -> list[str]:
    """Severity-ordered bullets that signal the plan does not survive its own assumptions."""
    thresholds = threshold_entries(mc, params)
    doom = [r for r in thresholds if r["verdict"] == "DOOM"]
    fragile = [r for r in thresholds if r["verdict"] == "FRAGILE"]
    scenario_warns = (scenarios or {}).get("warnings", []) if scenarios else []

    collapses: list[tuple[str, float]] = []
    if mc:
        for name, o in (mc.get("outputs") or {}).items():
            total = o["missing_count"] + o["count"]
            if total == 0:
                continue
            rate = o["missing_count"] / total
            if rate >= 0.05:
                collapses.append((name, rate))

    still_missing: list[dict] = []
    if params:
        bound_ids = set(bounds or {})
        for mv in params.get("missing_values_to_estimate", []):
            if mv["id"] not in bound_ids:
                still_missing.append(mv)

    unmodelled_gates = (params or {}).get("unmodelled_gates") or [] if params else []

    if not (doom or fragile or scenario_warns or collapses or still_missing or unmodelled_gates):
        return []

    rows = ["## Critical findings", ""]
    if unmodelled_gates:
        labels = [
            (g.get("label") or g.get("id") or "") for g in unmodelled_gates
            if isinstance(g, dict)
        ]
        # Lower-case the first letter of each label when emitting in mid-sentence
        # prose. Labels are stored title-cased for the table; in a flowing
        # sentence ("the simulation does not evaluate ...") the title casing
        # reads as proper nouns. Preserve labels that start with an acronym:
        # if the second character is also uppercase, the first word is an
        # acronym (e.g. "AML/KYC compliant ...") and the casing stays.
        def _lower_first(s: str) -> str:
            if len(s) >= 2 and s[1].isupper():
                return s
            return s[:1].lower() + s[1:]
        labels = [_lower_first(l) for l in labels if l]
        if len(labels) > 1:
            labels_str = ", ".join(labels[:-1]) + ", or " + labels[-1]
        elif labels:
            labels_str = labels[0]
        else:
            labels_str = "the listed gates"
        rows.append(
            f"- **SCOPE WARNING** — The simulation does not evaluate {labels_str}. "
            f"These are existential gates and may dominate the modelled financial result. "
            f"See `## Known unmodelled existential gates` for details."
        )
    for r in doom:
        fail_pct = (1 - (r["probability"] or 0)) * 100
        rows.append(
            f"- **DOOM** — `{r['id']}` fails {r['operator']} {fmt_number(r['value'])} "
            f"in {fail_pct:.1f}% of simulated runs under the current input bounds."
        )
    for r in fragile:
        pass_pct = (r["probability"] or 0) * 100
        rows.append(
            f"- **FRAGILE** — `{r['id']}` holds in only {pass_pct:.1f}% of simulated runs "
            f"under the current input bounds (fails in the majority)."
        )
    for w in scenario_warns:
        scope = w.get("scenario") or "all"
        calc = w.get("calculation") or "—"
        rows.append(f"- **Scenario warn ({scope})** — `{calc}`: {w['message']}")
    for name, rate in sorted(collapses, key=lambda x: -x[1]):
        rows.append(
            f"- **Model collapse** — `{name}` came back blank in {rate * 100:.1f}% of runs. "
            f"An input is missing or a denominator can legitimately land at zero. Fix the inputs and re-run."
        )
    for mv in still_missing:
        rows.append(f"- **Still missing input** — `{mv['id']}`: {mv.get('why_needed', '')}")
    rows.append("")
    return rows


# ─── gate verdicts ─────────────────────────────────────────────────────────

def render_gate_verdicts(mc: dict | None, params: dict | None) -> list[str]:
    thresholds = threshold_entries(mc, params)
    if not thresholds:
        return []
    n_runs = (mc.get("settings") or {}).get("n_runs")
    runs_phrase = f"{n_runs:,} simulated runs" if isinstance(n_runs, int) else "the simulated runs"
    rows = [
        "## Gate verdicts",
        "",
        f"Pass rate for every declared threshold over {runs_phrase}. Bands: ≥80% **ROBUST**, 50–80% **MARGINAL**, 20–50% **FRAGILE**, <20% **DOOM**. The `Threshold basis` column reports whether the threshold value came from the source report explicitly (`report_explicit`), was inferred from the report (`report_inferred`), or has no narrative anchor (`model_defined` / `unknown`). Rows with a leading `min` marker are aggregate gates computed via `min()` over independent pools — their verdict is meaningful, their raw magnitude is not.",
        "",
        "| | Output | Condition | Threshold basis | Pass rate | Verdict | Meaning |",
        "|---|---|---|---|---:|---|---|",
    ]
    for r in thresholds:
        prob = r["probability"]
        prob_str = f"{prob * 100:.1f}%" if prob is not None else "n/a"
        marker = "min" if r["is_aggregate"] else ""
        meta = lookup_gate_metadata(params, r["id"])
        tbasis = threshold_basis_for(meta["threshold_value_type"], meta.get("gate_found", False))
        rows.append(
            f"| {marker} | `{r['id']}` | {r['operator']} {fmt_number(r['value'])} | "
            f"{tbasis} | {prob_str} | **{r['verdict']}** | {r['note']} |"
        )
    rows.append("")

    units = {r["unit"] for r in thresholds if r["unit"]}
    has_aggregate = any(r["is_aggregate"] for r in thresholds)
    if len(units) > 1 and not has_aggregate:
        rows.append("### Aggregation warning")
        rows.append("")
        unit_list = ", ".join(sorted(u for u in units if u))
        rows.append(
            f"The thresholds above use incompatible units ({unit_list}) and the source plan does not declare a combined-gate aggregate. **Do not average or otherwise combine these pass rates into a single scalar.** Use the categorical verdicts per gate, or read the per-output distributions in `montecarlo.json`."
        )
        rows.append("")
    return rows


# ─── decision implications ─────────────────────────────────────────────────

# Pass-rate floor below which we treat a gate as a "saturated failure": the
# quartile-analysis Δ-pp values are too small to be informative because no
# input quartile movement changes the outcome — the gate fails (or passes)
# structurally across the whole sampled space. Pretending a top-driver row
# is a useful "structural lever" in that regime gives a misleading
# optimisation hint.
SATURATED_PASS_RATE_THRESHOLD = 0.005   # < 0.5% pass = effectively zero
SATURATED_QUARTILE_DELTA_FLOOR = 0.5    # |Δ-pp| < 0.5pp = no meaningful sensitivity


def is_saturated_failure(probability: float | None, top_driver: dict | None) -> bool:
    """A gate is saturated when its pass rate is effectively zero AND no
    single input quartile movement meaningfully shifts that. Both conditions
    must hold: a low pass rate with a strong driver is just hard, not
    saturated.
    """
    if probability is None or probability >= SATURATED_PASS_RATE_THRESHOLD:
        return False
    if top_driver is None:
        return True
    return abs(top_driver.get("delta_pp", 0)) < SATURATED_QUARTILE_DELTA_FLOOR


def render_decision_implications(mc: dict | None, params: dict | None) -> list[str]:
    """Bridge: gate result → planning consequence → revision direction.

    Templates are deliberately generic. Plan-specific tactical revisions
    require human or LLM interpretation; what we can emit deterministically
    is the *type* of consequence and the *direction* of repair given the top
    driver from quartile_analysis.
    """
    thresholds = threshold_entries(mc, params)
    interpreted = [r for r in thresholds if r["verdict"] in ("DOOM", "FRAGILE", "MARGINAL")]
    if not interpreted:
        return []
    quartile = (mc or {}).get("quartile_analysis") or {}

    rows = [
        "## Decision implications",
        "",
        "Bridge from gate result to planning consequence. **Structural lever** names the input whose quartile movement has the largest effect on this gate (from `quartile_analysis` in `montecarlo.json`). **Gate meaning** surfaces the gate's own rationale lifted verbatim from `parameters.recommended_first_calculations[].why_first` (or `derived_questions[].why_it_matters`) plus the threshold parameter the formula tests against. This section identifies the affected planning lever; concrete revisions should be derived by reading the source report and the relevant intermediary artifacts.",
        "",
        "| Gate | Verdict | Planning consequence | Structural lever | Gate meaning |",
        "|---|---|---|---|---|",
    ]
    for r in interpreted:
        prob = r["probability"] or 0
        cond = f"{r['operator']} {fmt_number(r['value'])}"
        if r["verdict"] == "DOOM":
            consequence = (
                f"The `{cond}` requirement is not credible under current bounds: "
                f"only {prob * 100:.1f}% of runs hold. Commitments that depend on "
                f"this should not be made without revision."
            )
        elif r["verdict"] == "FRAGILE":
            consequence = (
                f"The `{cond}` requirement fails in the majority of runs "
                f"({(1 - prob) * 100:.1f}%). External commitments built on this "
                f"gate are exposed."
            )
        else:  # MARGINAL (50–80% pass rate)
            # The MARGINAL band spans a large range. A 51% pass is genuinely
            # coin-flip; a 79% pass is one slip from ROBUST. Bucket the
            # wording so the language tracks the position within the band.
            if prob >= 0.70:
                consequence = (
                    f"The `{cond}` requirement passes {prob * 100:.1f}% of runs — "
                    f"just below the ROBUST band. The gate passes in most runs, "
                    f"but downstream commitments should not treat it as secure."
                )
            else:
                consequence = (
                    f"The `{cond}` requirement passes {prob * 100:.1f}% of runs — "
                    f"close to coin-flip. Downstream commitments should not "
                    f"assume it holds."
                )
        drivers = quartile.get(r["id"]) or []
        top = max(drivers, key=lambda d: abs(d.get("delta_pp", 0)), default=None)
        if is_saturated_failure(r["probability"], top):
            lever = (
                f"Saturated failure: pass rate is {prob * 100:.1f}% and no single "
                f"input quartile movement changes that. Quartile sensitivity is "
                f"not informative; audit the input bounds and the threshold definition."
            )
        elif top is None:
            lever = "No single driver dominates — audit the cluster of inputs feeding this gate."
        else:
            delta = top.get("delta_pp", 0)
            direction = "improve toward its high bound" if delta >= 0 else "reduce toward its low bound"
            lever = f"`{top['id']}` ({direction}; quartile Δ-pp {delta:+.1f})"

        meta = lookup_gate_metadata(params, r["id"])
        why = meta["why"] or ""
        if meta["threshold_param_id"]:
            tref = f"Threshold parameter: `{meta['threshold_param_id']}`."
            hint = (why + " " + tref).strip() if why else tref
        elif why:
            hint = why
        else:
            hint = "No gate rationale in `parameters.json`; revisit the threshold definition."

        rows.append(
            f"| `{r['id']}` | **{r['verdict']}** | {consequence} | {lever} | {hint} |"
        )
    rows.append("")
    return rows


# ─── failure drivers (one row per failing gate) ────────────────────────────

def render_failure_drivers(mc: dict | None, params: dict | None) -> list[str]:
    if not mc:
        return []
    thresholds = threshold_entries(mc, params)
    failing = [r for r in thresholds if r["verdict"] in ("DOOM", "FRAGILE")]
    if not failing:
        return []
    quartile = mc.get("quartile_analysis") or {}
    required = mc.get("required_input_thresholds") or {}
    binding = mc.get("binding_gate_analysis") or {}

    rows = [
        "## Failure drivers",
        "",
        "For each failing gate (DOOM or FRAGILE): the single input with the largest pass-rate swing between its bottom and top quartile, and the conditional input restriction that would lift the gate to an 80% pass rate (when one exists). Full per-driver tables and binding-gate frequencies are in `montecarlo.json`.",
        "",
        "| Failing gate | Top driver | Δ-pp (bottom→top quartile) | 80% pass requires |",
        "|---|---|---:|---|",
    ]
    for r in failing:
        gate = r["id"]
        drivers = quartile.get(gate) or []
        top = max(drivers, key=lambda d: abs(d.get("delta_pp", 0)), default=None)
        saturated = is_saturated_failure(r["probability"], top)
        if saturated:
            driver_cell, delta_cell = "(saturated failure)", "n/a"
        elif top is None:
            driver_cell, delta_cell = "—", "—"
        else:
            driver_cell = f"`{top['id']}`"
            delta_cell = f"{top['delta_pp']:+.1f}"

        req_entries = required.get(gate) or []
        if saturated:
            req_cell = "saturated failure: no single input restriction can lift the pass rate; revisit bounds and threshold"
        elif req_entries:
            req = req_entries[0]
            direction = "above" if req["direction"] == "above" else "below"
            pct = req["input_percentile_cutoff"]
            req_cell = f"`{req['id']}` {direction} p{pct}"
        else:
            req_cell = "no single input restriction sufficient"
        rows.append(f"| `{gate}` | {driver_cell} | {delta_cell} | {req_cell} |")
    rows.append("")

    if binding:
        rows.append("Binding-gate notes (aggregates only):")
        rows.append("")
        for agg_name, info in binding.items():
            top_binder = max(
                info.get("binding_when_aggregate_fails", []),
                key=lambda x: x.get("frequency", 0),
                default=None,
            )
            if top_binder is None:
                continue
            rows.append(
                f"- `{agg_name}` (fails {info['fail_count']:,} runs): "
                f"`{top_binder['dependency']}` is the binder in "
                f"{top_binder['frequency'] * 100:.1f}% of failed runs."
            )
        rows.append("")
    return rows


# ─── missing inputs ranked by impact ───────────────────────────────────────

def render_missing_inputs_ranked(mc: dict | None) -> list[str]:
    if not mc or not mc.get("missing_value_priority"):
        return []
    rows = [
        "## Missing inputs ranked by impact",
        "",
        "The plan does not state these values; the model assumed bounds. Rank by `|Δ-pp on the worst-affected gate| × (1 − that gate's pass rate) × bound-width-ratio` — the higher, the more decision-value in pinning this input down.",
        "",
        "| Rank | Input | Worst-affected gate | Score | Bound width / base | Basis |",
        "|---:|---|---|---:|---:|---|",
    ]
    for i, e in enumerate(mc["missing_value_priority"], 1):
        basis = BASIS_FROM_SOURCE.get(e["source"], e["source"])
        rows.append(
            f"| {i} | `{e['id']}` | `{e['worst_gate']}` | {e['score']:.2f} | "
            f"{e['bound_width_ratio']:.2f} | {basis} |"
        )
    rows.append("")
    rows.append(
        "`Basis` values: `report_derived` = bound anchored in the source report's narrative (not externally verified); "
        "`model_assumption` = bound is a modelling assumption with no narrative anchor. Neither value is empirical ground truth."
    )
    rows.append("")
    return rows


# ─── confidence and trust boundaries ───────────────────────────────────────

def render_confidence_and_trust(mc: dict | None, validation: dict | None) -> list[str]:
    rows = ["## Confidence and trust boundaries", ""]
    rows.append("### Validated")
    rows.append("")
    if validation is not None:
        checks = (validation.get("summary") or {}).get("checks_performed", [])
        if checks:
            rows.append(
                "Structural checks that passed (`validation.json`): "
                + ", ".join(f"`{c}`" for c in checks) + "."
            )
        else:
            rows.append("_(validation.json does not list checks)_")
    else:
        rows.append("_(validation.json not in this run)_")
    rows.append("")
    rows.append("### Not validated")
    rows.append("")
    for item in TRUST_NOT_VALIDATED:
        rows.append(f"- {item}")
    rows.append("")

    if mc and mc.get("model_confidence"):
        rows.append("### Per-output confidence")
        rows.append("")
        rows.append("| Output | Grade | Declared-source inputs | Bound-width / base |")
        rows.append("|---|:---:|---:|---:|")
        for output_id, info in mc["model_confidence"].items():
            if "data_source_fraction" not in info:
                continue
            data_pct = info["data_source_fraction"] * 100
            width = info["average_bound_width_ratio"]
            rows.append(
                f"| `{output_id}` | **{info['grade']}** | {data_pct:.0f}% | {width:.2f} |"
            )
        rows.append("")
        rows.append(
            "Per-output reasons are in `montecarlo.json` under `model_confidence`. "
            "`Declared-source inputs` is the share of input bounds anchored in the source report's narrative (`bounds.source == data`); the remainder are modelling assumptions. **Neither is empirically observed real-world data.**"
        )
        rows.append("")
    return rows


# ─── scenario sanity check ─────────────────────────────────────────────────

def render_scenario_sanity_check(scenarios: dict | None) -> list[str]:
    if not scenarios or not scenarios.get("comparison"):
        return []
    rows = [
        "## Scenario sanity check",
        "",
        "Three deterministic scenarios: every uncertain input at the **low** end of its range, every input at its **base** value, every input at the **high** end. The column labels refer to **inputs**, not to whether the outcome is good or bad. Column names match the `low`/`base`/`high` keys in `scenarios.json`. A negative `base inputs` column on a gate the plan needs to pass means the plan is already in trouble at its own central assumptions.",
        "",
        "| Output | Unit | Low inputs | Base inputs | High inputs |",
        "|---|---|---:|---:|---:|",
    ]
    for name, o in scenarios["comparison"]["outputs"].items():
        rows.append(
            f"| `{name}` | {o.get('unit', '—')} | "
            f"{fmt_number(o['low'])} | {fmt_number(o['base'])} | {fmt_number(o['high'])} |"
        )
    rows.append("")
    return rows


# ─── suggested next actions ────────────────────────────────────────────────

def render_suggested_next_actions(mc: dict | None, params: dict | None) -> list[str]:
    thresholds = threshold_entries(mc, params)
    failing = [r for r in thresholds if r["verdict"] in ("DOOM", "FRAGILE")]
    rows = [
        "## Suggested next actions",
        "",
        "Imperatives for whatever consumes this file next.",
        "",
    ]
    if failing:
        # Distinguish DOOM (<20%) from FRAGILE (20–50%) and name the worst
        # gate by id + pass rate. "N gates fail" by itself understates a
        # 0.0% DOOM the way "1 gate fails" understates total structural
        # failure.
        doom_sorted = sorted(
            (r for r in thresholds if r["verdict"] == "DOOM"),
            key=lambda r: r["probability"] if r["probability"] is not None else 1.0,
        )
        fragile_sorted = sorted(
            (r for r in thresholds if r["verdict"] == "FRAGILE"),
            key=lambda r: r["probability"] if r["probability"] is not None else 1.0,
        )
        worst = doom_sorted[0] if doom_sorted else fragile_sorted[0]
        worst_pct = (worst["probability"] or 0) * 100
        band_parts = []
        if doom_sorted:
            band_parts.append(
                f"{len(doom_sorted)} declared gate{'s' if len(doom_sorted) != 1 else ''} in the DOOM band"
            )
        if fragile_sorted:
            band_parts.append(
                f"{len(fragile_sorted)} in the FRAGILE band"
            )
        band_summary = "; ".join(band_parts)
        rows.append(
            f"1. To answer whether the plan is viable, lead with the gate verdicts above — not the source plan's narrative. "
            f"{band_summary}. Worst: `{worst['id']}` at {worst_pct:.1f}% pass rate under current bounds."
        )
    else:
        rows.append(
            "1. To answer whether the plan is viable, lead with the gate verdicts above. No declared gate is in the DOOM or FRAGILE band — but read the bounds and trust boundaries before treating that as a green light."
        )
    rows.append(
        "2. To prioritise data-gathering, inspect `missing_value_priority` in `montecarlo.json`. The top-scored entries are the cheapest improvements to the simulation's predictive value."
    )
    rows.append(
        "3. To audit whether the simulation is trustworthy, open `bounds.json` (range rationales and the `source` label per bound, surfaced here as the `Basis` column in `Missing inputs ranked by impact`), `montecarlo_settings.json` (n_runs, seed, distribution_default, threshold definitions), and `validation.json` (which structural checks passed). Neither `report_derived` nor `model_assumption` constitutes externally observed ground truth."
    )
    rows.append(
        "4. To improve the plan, target the gates with the lowest pass rates first. Failure-driver rows above name the single input whose quartile movement has the largest effect on each failing gate."
    )
    rows.append(
        "5. For exact formulas and the executable model, use `parameters.json` (declared formula hints) and `calculations.py` (implementation). Do not infer formulas from prose elsewhere in this file."
    )
    rows.append("")
    return rows


# ─── open questions for next analysis pass ────────────────────────────────

def render_open_questions() -> list[str]:
    rows = [
        "## Open questions for next analysis pass",
        "",
        "Standing questions for whoever picks this directory up next. These are not gate verdicts; they are the audit checks the simulation can't answer on its own.",
        "",
    ]
    for i, q in enumerate(OPEN_QUESTIONS, 1):
        rows.append(f"{i}. {q}")
    rows.append("")
    return rows


# ─── build ─────────────────────────────────────────────────────────────────

def build_assessment(params: dict | None, bounds: dict | None,
                   scenarios: dict | None, mc: dict | None,
                   validation: dict | None,
                   params_path: Path | None, bounds_path: Path | None,
                   scenarios_path: Path | None, mc_path: Path | None,
                   validation_path: Path | None,
                   settings_path: Path | None,
                   extract_input_path: Path | None,
                   calculations_path: Path | None,
                   scenario_outputs_path: Path | None) -> str:
    present_files: set[str] = set()
    for p in (params_path, bounds_path, scenarios_path, mc_path, validation_path,
              settings_path, extract_input_path, calculations_path,
              scenario_outputs_path):
        if p and p.exists():
            present_files.add(p.name)

    sections: list[list[str]] = [
        render_title_and_frontmatter(params),
        render_artifact_contract(),
        render_machine_summary(params, mc, validation, params_path),
        render_provenance_map(present_files),
        render_modelling_frame(params),
        render_unmodelled_gates(params),
        render_simulation_settings(mc, validation),
        render_critical_findings(mc, scenarios, params, bounds),
        render_gate_verdicts(mc, params),
        render_decision_implications(mc, params),
        render_failure_drivers(mc, params),
        render_missing_inputs_ranked(mc),
        render_confidence_and_trust(mc, validation),
        render_scenario_sanity_check(scenarios),
        render_suggested_next_actions(mc, params),
        render_open_questions(),
    ]
    out: list[str] = []
    for section in sections:
        if section:
            out.extend(section)
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--bounds", type=Path)
    p.add_argument("--scenarios", type=Path)
    p.add_argument("--montecarlo", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--settings", type=Path)
    p.add_argument("--extract-input", type=Path)
    p.add_argument("--calculations", type=Path)
    p.add_argument("--scenario-outputs", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    base = args.parameters.parent
    validation_path = args.validation or (base / "validation.json")
    settings_path = args.settings or (base / "montecarlo_settings.json")
    extract_input_path = args.extract_input or (base / "extract_parameters_input.md")
    calculations_path = args.calculations or (base / "calculations.py")
    scenario_outputs_path = args.scenario_outputs or (base / "scenario_outputs.json")

    output = args.output or (base / "assessment.md")
    md = build_assessment(
        params=load_json(args.parameters),
        bounds=load_json(args.bounds) if args.bounds else None,
        scenarios=load_json(args.scenarios) if args.scenarios else None,
        mc=load_json(args.montecarlo) if args.montecarlo else None,
        validation=load_json(validation_path) if validation_path.exists() else None,
        params_path=args.parameters,
        bounds_path=args.bounds,
        scenarios_path=args.scenarios,
        mc_path=args.montecarlo,
        validation_path=validation_path,
        settings_path=settings_path,
        extract_input_path=extract_input_path,
        calculations_path=calculations_path,
        scenario_outputs_path=scenario_outputs_path,
    )
    output.write_text(md)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

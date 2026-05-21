"""
Prepare the parameter-extraction input bundle for the
extract-parameters-from-digest skill.

Given a PlanExe sample directory (e.g.
``/Users/neoneye/git/PlanExe-web/20260215_nuuk_clay_workshop``), this script:

1. Runs ``compress_report_section`` on the four sections worth compressing
   (selected_scenario, review_plan, premortem, expert_criticism), feeding each
   one the same multi-file blob the corresponding Luigi node ingests, plus
   that node's own output appended. See proposals 137 and 139 for the
   rationale.
2. Writes the four per-section markdown digests and raw JSON to the output
   directory.
3. Concatenates the 137-recommended extraction bundle into
   ``extract_parameters_input.md`` — Executive Summary, Project Plan,
   Selected Scenario, Assumptions, Review Plan, Premortem, Expert Criticism,
   Data Collection — in that order. The four sections marked "Keep or
   compress" in 137 use the compressed digests above; the four sections
   marked plain "Keep" (Executive Summary, Project Plan, Assumptions, Data
   Collection) are passed through raw. Strategic Decisions is replaced by
   Selected Scenario per proposal 139 to avoid feeding rejected alternatives
   into the parameter extractor.

That combined file is the input you point the extract-parameters-from-digest
skill at, to compare head-to-head with the extract-parameters-from-full skill (which
reads the full PlanExe HTML report).

Defaults to a sibling ``output/<planexe-dir-name>/`` directory next to this
script. Override with ``--output-dir`` or ``--llm``.

When rerunning the pipeline at version v(N) for a plan slug that already
has an earlier version on disk, pass ``--prior <prior parameters.json>``
to append a compact ``# Prior Signal Ledger`` to the combined digest.
That ledger is the wiring point for the source-preservation audit and
the extract LLM's ``dropped_signals`` rules; without it, a rerun cannot
emit prior-baseline-origin drops. See
``.claude/skills/run-napkin-math-pipeline/SKILL.md`` for orchestration.

PROMPT> python experiments/napkin_math/prepare_extract_input.py \\
            --planexe-dir /Users/neoneye/git/PlanExe-web/20260215_nuuk_clay_workshop
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PLAN_DIR = REPO_ROOT / "worker_plan"
NAPKIN_MATH_DIR = Path(__file__).resolve().parent

# Section bundle and order per docs/proposals/137-section_filtering_for_parameter_extraction.md,
# with the substitution from docs/proposals/139-compress-for-monte-carlo.md: Strategic
# Decisions is replaced by Selected Scenario to avoid feeding rejected alternatives into
# parameter extraction. Each entry is (display_title, kind, source) where kind is either
# "compressed" (source is the name of a compress_<name>.md file produced by
# run_compress_full) or "raw" (source is the filename inside the PlanExe sample dir to
# pass through unchanged).
BUNDLE: tuple[tuple[str, str, str], ...] = (
    ("Executive Summary",  "raw",        "executive_summary.md"),
    ("Project Plan",       "raw",        "project_plan.md"),
    ("Selected Scenario",  "compressed", "selected_scenario"),
    ("Assumptions",        "raw",        "consolidate_assumptions_short.md"),
    ("Review Plan",        "compressed", "review_plan"),
    ("Premortem",          "compressed", "premortem"),
    ("Expert Criticism",   "compressed", "expert_criticism"),
    ("Data Collection",    "raw",        "data_collection.md"),
)

LEGEND = """Sections come in two forms.

Compressed sections (Selected Scenario, Review Plan, Premortem, Expert
Criticism). Each bullet carries an inline tag of the form
``[<source_status> | e=N r=N | quote: verified|unverified]``:

- ``explicit``    — the plan commits directly to this value
- ``derived``     — calculable from one or more ``explicit`` values
- ``inferred``    — source-stated non-binding claim, or a model-added
                    plausible guess
- ``stress_test`` — a downside-scenario magnitude, not a plan fact
- ``missing``     — a primitive input the source does not supply

- ``e=N`` — source evidence (1-5)
- ``r=N`` — modelling relevance (1-5)
- ``quote: verified|unverified`` — independent substring check of the
  source_quote against the original section text

Raw sections (Executive Summary, Project Plan, Assumptions, Data
Collection). These carry no inline tags. Apply general parameter-extraction
triage to them: prefer plan commitments, numeric anchors, denominators,
and missing inputs over narrative framing.

When extracting parameters from the compressed sections, prefer
``explicit`` and ``derived`` items with ``quote: verified``. Treat
``stress_test`` items as downside-scenario inputs rather than baseline
plan facts. Items in ``Missing data to estimate`` are the ones to surface
as ``missing_values_to_estimate`` in the output.
"""


def run_compress(planexe_dir: Path, output_dir: Path, llm: str | None) -> None:
    """Invoke the existing run_compress_full driver as a module. Sample and
    output directories are passed via the env vars the driver already
    supports.
    """
    env = os.environ.copy()
    env["COMPRESS_FULL_SAMPLE_DIR"] = str(planexe_dir)
    env["COMPRESS_FULL_OUTPUT_DIR"] = str(output_dir)
    if llm:
        env["COMPRESS_FULL_LLM"] = llm
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{WORKER_PLAN_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(WORKER_PLAN_DIR)
    )
    cmd = [
        sys.executable,
        "-m",
        "worker_plan_internal.parameter_extraction.run_compress_full",
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"  COMPRESS_FULL_SAMPLE_DIR = {planexe_dir}")
    print(f"  COMPRESS_FULL_OUTPUT_DIR = {output_dir}")
    if llm:
        print(f"  COMPRESS_FULL_LLM        = {llm}")
    subprocess.run(cmd, env=env, cwd=str(WORKER_PLAN_DIR), check=True)


SECTIONS_WITH_IDS_FOR_LEDGER: tuple[str, ...] = (
    "key_values",
    "missing_values_to_estimate",
    "derived_questions",
    "recommended_first_calculations",
    "unmodelled_gates",
)

SECTIONS_WITH_OUTPUT_NAMES_FOR_LEDGER: tuple[str, ...] = (
    "key_values",
    "derived_questions",
    "recommended_first_calculations",
)

PRIOR_LEDGER_HEADER: str = """\
# Prior Signal Ledger (advisory)

This ledger lists the prior iteration's named signals — entry ids and
output_names from a previous `parameters.json`. Treat it as a
preservation budget, NOT a target to copy. The source digest above is
the authoritative input.

If a prior signal is still source-supported and load-bearing, preserve
it: keep the same id, the same output_name, or use it as a formula
dependency. When a prior signal must be replaced, made redundant, moved
to `unmodelled_gates`, dropped under cap pressure, or excluded as out
of scope, record the explanation in `dropped_signals` per the schema —
with `origin: "prior_baseline"` and a reference that resolves to a
current id / output_name / unmodelled_gates id.

Do not invent `dropped_signals` entries for signals that are not in
this ledger and are not stated in the source digest above. The ledger
defines the universe of prior-baseline signals the audit will check.
"""


def render_prior_source(prior_path: Path) -> str:
    """Render the prior path stably across worktrees. When the path is
    under ``NAPKIN_MATH_DIR``, return the suffix (e.g.
    ``output/v58/<slug>/parameters.json``); otherwise return the
    absolute path. The result is what gets stamped into the ledger and
    what the orchestrator's Stage 1 preflight greps for to confirm the
    digest was generated against the intended accepted baseline.
    """
    resolved = prior_path.resolve()
    try:
        return str(resolved.relative_to(NAPKIN_MATH_DIR))
    except ValueError:
        return str(resolved)


def build_prior_signal_ledger(
    prior_params: dict, prior_path: Path | None = None,
) -> str:
    """Build a compact markdown ledger listing the prior baseline's
    named signals — entry ids and output_names across the five sections
    that carry them. Intentionally narrow: no source_text, no labels,
    no comments. The LLM gets a preservation budget to compare against
    the source digest, not a phrasing target to anchor on.

    Each signal is listed with its section, kind (id or output_name),
    formula_hint (when present), and depends_on (when non-empty) so
    structural relationships survive. Signals are deduplicated: a name
    that appears as both an id and an output_name is listed once with
    kind = ``id`` (the more authoritative reading).

    When ``prior_path`` is supplied, a ``Prior source: `<path>``` line
    is stamped between the header and the signal list. This lets the
    orchestrator's Stage 1 preflight verify the ledger was built
    against the named accepted baseline, not against an arbitrary or
    probe prior.
    """
    seen: dict[str, dict[str, Any]] = {}
    for section in SECTIONS_WITH_IDS_FOR_LEDGER:
        for entry in prior_params.get(section, []) or []:
            if not isinstance(entry, dict):
                continue
            eid = entry.get("id")
            if not isinstance(eid, str) or not eid:
                continue
            seen.setdefault(eid, {
                "kind": "id",
                "section": section,
                "formula_hint": entry.get("formula_hint"),
                "depends_on": entry.get("depends_on") or [],
            })
    for section in SECTIONS_WITH_OUTPUT_NAMES_FOR_LEDGER:
        for entry in prior_params.get(section, []) or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("output_name")
            if not isinstance(name, str) or not name:
                continue
            seen.setdefault(name, {
                "kind": "output_name",
                "section": section,
                "formula_hint": entry.get("formula_hint"),
                "depends_on": entry.get("depends_on") or [],
            })
    lines: list[str] = [PRIOR_LEDGER_HEADER.rstrip(), ""]
    if prior_path is not None:
        lines.append(f"Prior source: `{render_prior_source(prior_path)}`")
        lines.append("")
    lines.extend(["## Signals", ""])
    for name in sorted(seen):
        meta = seen[name]
        lines.append(f"- `{name}` [{meta['section']}/{meta['kind']}]")
        formula = meta.get("formula_hint")
        if isinstance(formula, str) and formula.strip():
            lines.append(f"  - formula_hint: `{formula.strip()}`")
        depends = meta.get("depends_on") or []
        depends = [d for d in depends if isinstance(d, str) and d]
        if depends:
            lines.append(f"  - depends_on: {', '.join('`' + d + '`' for d in depends)}")
    if not seen:
        lines.append("(no prior signals — first-iteration baseline)")
    return "\n".join(lines) + "\n"


def build_combined_digest(
    planexe_dir: Path, output_dir: Path,
    prior_params: dict | None = None,
    prior_path: Path | None = None,
) -> Path:
    """Concatenate the 137-recommended extraction bundle, in 137's order, with
    a legend at the top. Compressed sections come from ``output_dir/compress_*.md``;
    raw sections come straight from ``planexe_dir``. Missing sections are
    skipped with a warning rather than aborting, so partial output is still
    usable.

    When ``prior_params`` is provided (the parsed contents of a prior
    ``parameters.json``), a compact prior-signal ledger is appended after
    the regular bundle so the extract LLM can decide which prior signals
    to preserve and which to record in ``dropped_signals``. The ledger is
    intentionally narrow — names, sections, formula_hints and depends_on
    only — so it acts as a preservation budget rather than a phrasing
    target.

    When ``prior_path`` is also supplied, the ledger is stamped with a
    ``Prior source: `<path>``` line so the orchestrator's Stage 1
    preflight can verify the digest was built against the intended
    accepted baseline.
    """
    parts: list[str] = [LEGEND.rstrip(), "", "---", ""]
    found_any = False
    for title, kind, source in BUNDLE:
        if kind == "compressed":
            path = output_dir / f"compress_{source}.md"
            if not path.exists():
                print(f"WARNING: compressed section {source!r} not produced ({path.name}); skipping")
                continue
            body = path.read_text(encoding="utf-8").strip()
            # The compressed .md files already carry their own H1 title; use them as-is.
        else:
            path = planexe_dir / source
            if not path.exists():
                print(f"WARNING: raw section file {source!r} not found in {planexe_dir}; skipping")
                continue
            raw = path.read_text(encoding="utf-8").strip()
            body = f"# {title}\n\n{raw}"
        parts.append(body)
        parts.append("")
        parts.append("---")
        parts.append("")
        found_any = True
    if not found_any:
        raise SystemExit(
            f"No sections were produced for the combined digest. Check the "
            f"run_compress_full output above for compression errors, and verify "
            f"that the raw section files exist under {planexe_dir}."
        )
    if prior_params is not None:
        parts.append(
            build_prior_signal_ledger(prior_params, prior_path=prior_path).rstrip()
        )
        parts.append("")
    combined = output_dir / "extract_parameters_input.md"
    combined.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return combined


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--planexe-dir",
        required=True,
        type=Path,
        help="PlanExe sample directory containing plan.txt, premortem.md, etc.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the four section digests and combined file. "
        "Default: experiments/napkin_math/output/<planexe-dir-name>/",
    )
    ap.add_argument(
        "--llm",
        default=None,
        help="LLM name passed through to run_compress_full via "
        "COMPRESS_FULL_LLM. Default: run_compress_full's own default model.",
    )
    ap.add_argument(
        "--prior",
        type=Path,
        default=None,
        help="Optional path to a prior iteration's parameters.json. When "
        "provided, a compact Prior Signal Ledger is appended to the "
        "combined digest so the extract LLM can decide which prior "
        "signals to preserve and which to record in dropped_signals "
        "(proposal 141 PR 3). The ledger contains only signal names, "
        "sections, formula_hints, and depends_on — no source_text or "
        "labels — so it acts as a preservation budget rather than a "
        "phrasing target. Omit for first-iteration extractions.",
    )
    args = ap.parse_args()

    planexe_dir: Path = args.planexe_dir.resolve()
    if not planexe_dir.is_dir():
        raise SystemExit(f"--planexe-dir not found or not a directory: {planexe_dir}")

    output_dir: Path = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else NAPKIN_MATH_DIR / "output" / planexe_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Planexe dir : {planexe_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"LLM         : {args.llm or '(run_compress_full default)'}\n")

    prior_params: dict | None = None
    prior_path: Path | None = None
    if args.prior is not None:
        resolved_prior: Path = args.prior.resolve()
        if not resolved_prior.is_file():
            raise SystemExit(f"--prior not found or not a file: {resolved_prior}")
        try:
            prior_params = json.loads(resolved_prior.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--prior is not valid JSON: {exc}") from exc
        print(f"Prior        : {resolved_prior}\n")
        prior_path = resolved_prior

    run_compress(planexe_dir, output_dir, args.llm)
    combined = build_combined_digest(
        planexe_dir, output_dir,
        prior_params=prior_params, prior_path=prior_path,
    )

    print(f"\nWrote combined digest: {combined}")
    print("Feed this file to the extract-parameters-from-digest skill.")
    if prior_params is not None:
        print("Includes a Prior Signal Ledger appended after the bundle.")


if __name__ == "__main__":
    main()

"""Focused unit tests for the Prior Signal Ledger builder added in
proposal 141 PR 3. Tests use synthetic in-memory parameters dicts and
hermetic tmpdirs for the end-to-end ``build_combined_digest`` path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


NAPKIN_DIR = Path(__file__).resolve().parent.parent
PREPARE_PATH = NAPKIN_DIR / "prepare_extract_input.py"

spec = importlib.util.spec_from_file_location("prepare_extract_input", PREPARE_PATH)
prepare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules.setdefault("prepare_extract_input", prepare)
spec.loader.exec_module(prepare)


def _kv(eid: str, **extra: object) -> dict[str, object]:
    return {
        "id": eid,
        "label": extra.get("label", eid),
        "category": extra.get("category", "test"),
        "value_type": extra.get("value_type", "explicit"),
        "unit": extra.get("unit", "test_unit"),
        "value": extra.get("value", 1.0),
        "comment": extra.get("comment", ""),
        "formula_hint": extra.get("formula_hint"),
        "output_name": extra.get("output_name"),
        "output_unit": extra.get("output_unit"),
        "depends_on": extra.get("depends_on", []),
        "modelling_priority": extra.get("modelling_priority", "medium"),
        "uncertainty": extra.get("uncertainty", "low"),
        "source_text": extra.get("source_text", ""),
    }


def _calc(eid: str, *, output_name: str, formula: str,
          depends_on: list[str] | None = None) -> dict[str, object]:
    return {
        "id": eid,
        "label": eid,
        "formula_hint": formula,
        "output_name": output_name,
        "output_unit": "test_unit",
        "depends_on": depends_on or [],
        "why_first": "test",
    }


def _build_params(**sections: object) -> dict[str, object]:
    base: dict[str, object] = {
        "plan_summary": {
            "plan_name": "synthetic",
            "plan_type": "synthetic",
            "primary_goal": "test",
            "modelling_frame": "test",
        },
        "key_values": [],
        "derived_questions": [],
        "missing_values_to_estimate": [],
        "recommended_first_calculations": [],
    }
    base.update(sections)
    return base


# ─── build_prior_signal_ledger ────────────────────────────────────────────

def test_ledger_lists_key_value_ids_with_section_and_kind() -> None:
    params = _build_params(key_values=[_kv("alpha"), _kv("beta")])
    ledger = prepare.build_prior_signal_ledger(params)
    assert "# Prior Signal Ledger" in ledger
    assert "`alpha` [key_values/id]" in ledger
    assert "`beta` [key_values/id]" in ledger


def test_ledger_lists_output_name_with_section_and_kind() -> None:
    """A calc entry's output_name is tracked separately when it differs
    from its entry id (typical for q_* derived_questions producing a
    named margin)."""
    params = _build_params(
        derived_questions=[{
            "id": "q_margin",
            "question": "What is the margin?",
            "why_it_matters": "test",
            "formula_hint": "buildable_area_surplus = actual - required",
            "output_name": "buildable_area_surplus",
            "output_unit": "km2",
            "depends_on": ["actual", "required"],
        }],
    )
    ledger = prepare.build_prior_signal_ledger(params)
    assert "`q_margin` [derived_questions/id]" in ledger
    assert "`buildable_area_surplus` [derived_questions/output_name]" in ledger


def test_ledger_includes_formula_hint_when_present() -> None:
    params = _build_params(
        recommended_first_calculations=[
            _calc("calc_total", output_name="total_budget",
                  formula="total_budget = a + b", depends_on=["a", "b"]),
        ],
    )
    ledger = prepare.build_prior_signal_ledger(params)
    assert "formula_hint: `total_budget = a + b`" in ledger


def test_ledger_includes_depends_on_when_non_empty() -> None:
    params = _build_params(
        recommended_first_calculations=[
            _calc("calc_total", output_name="total_budget",
                  formula="total_budget = a + b", depends_on=["a", "b"]),
        ],
    )
    ledger = prepare.build_prior_signal_ledger(params)
    assert "depends_on: `a`, `b`" in ledger


def test_ledger_omits_formula_hint_when_null() -> None:
    """key_value entries usually have formula_hint=None; the ledger
    should not emit a 'formula_hint: None' line for them."""
    params = _build_params(key_values=[_kv("alpha")])
    ledger = prepare.build_prior_signal_ledger(params)
    assert "formula_hint" not in ledger


def test_ledger_dedupes_when_id_equals_output_name() -> None:
    """A calc whose entry id equals its output_name is listed once with
    kind=id (the more authoritative reading)."""
    params = _build_params(
        recommended_first_calculations=[
            _calc("burn_rate", output_name="burn_rate",
                  formula="burn_rate = annual / 12", depends_on=["annual"]),
        ],
    )
    ledger = prepare.build_prior_signal_ledger(params)
    assert ledger.count("`burn_rate`") == 1
    assert "[recommended_first_calculations/id]" in ledger
    assert "[recommended_first_calculations/output_name]" not in ledger


def test_ledger_includes_unmodelled_gate_ids() -> None:
    params = _build_params(
        unmodelled_gates=[{
            "id": "regulatory_floor_gate",
            "label": "Regulatory floor",
            "why_it_matters": "test",
            "source_anchor": "assumptions",
            "consequence_if_false": "test",
        }],
    )
    ledger = prepare.build_prior_signal_ledger(params)
    assert "`regulatory_floor_gate` [unmodelled_gates/id]" in ledger


def test_ledger_lists_no_corpus_literals_for_empty_prior() -> None:
    """Empty prior → first-iteration baseline message. No invented
    signals."""
    ledger = prepare.build_prior_signal_ledger(_build_params())
    assert "first-iteration baseline" in ledger
    assert "`alpha`" not in ledger


def test_ledger_does_not_include_source_text_or_label() -> None:
    """The ledger is a preservation budget, NOT a phrasing target.
    label, source_text, comment, and value are intentionally excluded
    so the LLM cannot copy old wording or anchor on old framings."""
    params = _build_params(
        key_values=[_kv(
            "alpha",
            label="Some descriptive label",
            source_text="A long source quote about alpha and its meaning",
            comment="A long comment about alpha's role",
        )],
    )
    ledger = prepare.build_prior_signal_ledger(params)
    assert "Some descriptive label" not in ledger
    assert "long source quote" not in ledger
    assert "long comment" not in ledger
    # value is also excluded.
    assert "1.0" not in ledger


# ─── build_combined_digest end-to-end ─────────────────────────────────────

def _make_minimal_planexe_dir(planexe_dir: Path) -> None:
    """Write the raw .md / .csv files build_combined_digest expects, so
    the BUNDLE iteration finds at least one section."""
    (planexe_dir / "executive_summary.md").write_text("Executive summary text.\n")
    (planexe_dir / "project_plan.md").write_text("Project plan text.\n")
    (planexe_dir / "consolidate_assumptions_short.md").write_text("Assumptions text.\n")
    (planexe_dir / "data_collection.md").write_text("Data collection text.\n")


def test_build_combined_digest_omits_ledger_when_no_prior_provided() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        planexe = tmpdir / "planexe"
        planexe.mkdir()
        _make_minimal_planexe_dir(planexe)
        outdir = tmpdir / "out"
        outdir.mkdir()
        combined = prepare.build_combined_digest(planexe, outdir)
        text = combined.read_text(encoding="utf-8")
    assert "Prior Signal Ledger" not in text
    assert "Executive summary text" in text


def test_build_combined_digest_appends_ledger_when_prior_provided() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        planexe = tmpdir / "planexe"
        planexe.mkdir()
        _make_minimal_planexe_dir(planexe)
        outdir = tmpdir / "out"
        outdir.mkdir()
        prior = _build_params(key_values=[_kv("alpha")])
        combined = prepare.build_combined_digest(planexe, outdir, prior_params=prior)
        text = combined.read_text(encoding="utf-8")
    assert "Prior Signal Ledger" in text
    assert "`alpha` [key_values/id]" in text
    # Ledger comes AFTER the bundle so the source remains authoritative.
    ledger_idx = text.index("Prior Signal Ledger")
    summary_idx = text.index("Executive summary text")
    assert summary_idx < ledger_idx


def _write_minimal_compressed_sections(outdir: Path) -> None:
    """Write the four ``compress_<name>.md`` files build_combined_digest
    expects under ``outdir``. Used to stand in for run_compress_full
    output when exercising ``main()`` without invoking the real
    compressor."""
    for name in ("selected_scenario", "review_plan", "premortem",
                 "expert_criticism"):
        (outdir / f"compress_{name}.md").write_text(
            f"# {name.replace('_', ' ').title()}\n\nCompressed body.\n",
            encoding="utf-8",
        )


def test_main_with_prior_writes_ledger_into_combined_digest(monkeypatch) -> None:
    """End-to-end CLI contract: invoking ``main()`` with ``--prior``
    appends ``# Prior Signal Ledger`` to ``extract_parameters_input.md``.
    This locks the wiring on the orchestrator side — a refactor that
    drops ``--prior`` from ``main()`` or stops feeding it into
    ``build_combined_digest`` will fail this test."""
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        planexe = tmpdir / "planexe"
        planexe.mkdir()
        _make_minimal_planexe_dir(planexe)
        outdir = tmpdir / "out"
        outdir.mkdir()
        _write_minimal_compressed_sections(outdir)
        prior = _build_params(key_values=[_kv("alpha")])
        prior_path = tmpdir / "prior_parameters.json"
        prior_path.write_text(json.dumps(prior), encoding="utf-8")

        monkeypatch.setattr(prepare, "run_compress",
                            lambda planexe_dir, output_dir, llm: None)
        monkeypatch.setattr(sys, "argv", [
            "prepare_extract_input.py",
            "--planexe-dir", str(planexe),
            "--output-dir", str(outdir),
            "--prior", str(prior_path),
        ])
        prepare.main()

        text = (outdir / "extract_parameters_input.md").read_text(encoding="utf-8")
    assert "# Prior Signal Ledger" in text
    assert "`alpha` [key_values/id]" in text


def test_main_without_prior_omits_ledger_from_combined_digest(monkeypatch) -> None:
    """Mirror of the with-prior test. The no-prior path stays clean —
    no ledger is appended, no first-iteration stub leaks in unless
    ``--prior`` was supplied."""
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        planexe = tmpdir / "planexe"
        planexe.mkdir()
        _make_minimal_planexe_dir(planexe)
        outdir = tmpdir / "out"
        outdir.mkdir()
        _write_minimal_compressed_sections(outdir)

        monkeypatch.setattr(prepare, "run_compress",
                            lambda planexe_dir, output_dir, llm: None)
        monkeypatch.setattr(sys, "argv", [
            "prepare_extract_input.py",
            "--planexe-dir", str(planexe),
            "--output-dir", str(outdir),
        ])
        prepare.main()

        text = (outdir / "extract_parameters_input.md").read_text(encoding="utf-8")
    assert "Prior Signal Ledger" not in text


def test_main_with_missing_prior_path_fails_loud(monkeypatch) -> None:
    """``--prior`` pointing at a non-existent path must exit non-zero
    rather than silently skip the ledger. This is the contract the
    orchestrator relies on: a typoed prior path is a loud failure, not
    a silent first-iteration extraction."""
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        planexe = tmpdir / "planexe"
        planexe.mkdir()
        _make_minimal_planexe_dir(planexe)
        outdir = tmpdir / "out"
        outdir.mkdir()

        monkeypatch.setattr(prepare, "run_compress",
                            lambda planexe_dir, output_dir, llm: None)
        monkeypatch.setattr(sys, "argv", [
            "prepare_extract_input.py",
            "--planexe-dir", str(planexe),
            "--output-dir", str(outdir),
            "--prior", str(tmpdir / "does_not_exist.json"),
        ])
        import pytest
        with pytest.raises(SystemExit):
            prepare.main()


def test_build_combined_digest_ledger_section_uses_authoritative_framing() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        planexe = tmpdir / "planexe"
        planexe.mkdir()
        _make_minimal_planexe_dir(planexe)
        outdir = tmpdir / "out"
        outdir.mkdir()
        prior = _build_params(key_values=[_kv("alpha")])
        combined = prepare.build_combined_digest(planexe, outdir, prior_params=prior)
        text = combined.read_text(encoding="utf-8")
    # Required posture: ledger is advisory, source is authoritative.
    assert "advisory" in text.lower()
    assert "preservation budget" in text.lower()
    assert "not a target to copy" in text.lower()

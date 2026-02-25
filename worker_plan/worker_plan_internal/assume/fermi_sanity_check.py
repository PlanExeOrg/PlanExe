"""Validation helpers for QuantifiedAssumption data."""
from __future__ import annotations

from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from worker_plan_internal.assume.quantified_assumptions import ConfidenceLevel, QuantifiedAssumption

MAX_SPAN_RATIO = 100.0
MIN_EVIDENCE_LENGTH = 40
BUDGET_LOWER_THRESHOLD = 1_000.0
BUDGET_UPPER_THRESHOLD = 100_000_000.0
TIMELINE_MAX_DAYS = 3650
TIMELINE_MIN_DAYS = 1
TEAM_MIN = 1
TEAM_MAX = 1000

CURRENCY_UNITS = {
    "usd",
    "eur",
    "dkk",
    "gbp",
    "cad",
    "aud",
    "sek",
    "nzd",
    "mxn",
    "chf"
}

TIME_UNIT_TO_DAYS = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365
}

TEAM_KEYWORDS = {
    "team",
    "people",
    "engineer",
    "engineers",
    "staff",
    "headcount",
    "crew",
    "members",
    "contractors",
    "workers"
}

BUDGET_KEYWORDS = {
    "budget",
    "cost",
    "funding",
    "investment",
    "price",
    "capex",
    "spend",
    "expense",
    "capital"
}

TIMELINE_KEYWORDS = {
    "timeline",
    "duration",
    "schedule",
    "milestone",
    "delivery",
    "months",
    "years",
    "weeks",
    "days"
}


class ValidationEntry(BaseModel):
    assumption_id: str = Field(description="Stable identifier for the assumption")
    question: str = Field(description="Source question for context")
    passed: bool = Field(description="Whether the assumption passed validation")
    reasons: List[str] = Field(description="List of validation failures")


class ValidationReport(BaseModel):
    entries: List[ValidationEntry] = Field(description="Detailed result per assumption")
    total_assumptions: int = Field(description="Total number of assumptions processed")
    passed: int = Field(description="Count of assumptions that passed")
    failed: int = Field(description="Count of assumptions that failed")
    pass_rate_pct: float = Field(description="Percentage of assumptions that passed")


def validate_quantified_assumptions(
    assumptions: Sequence[QuantifiedAssumption]
) -> ValidationReport:
    entries: List[ValidationEntry] = []
    passed = 0

    for assumption in assumptions:
        reasons: List[str] = []
        lower = assumption.lower_bound
        upper = assumption.upper_bound

        if lower is None or upper is None:
            reasons.append("Missing lower or upper bound.")
        elif lower > upper:
            reasons.append("Lower bound is greater than upper bound.")
        else:
            if ratio := assumption.span_ratio:
                if ratio > MAX_SPAN_RATIO:
                    reasons.append("Range spans more than 100×; too wide.")

        if assumption.confidence == ConfidenceLevel.low:
            evidence = assumption.evidence or ""
            if len(evidence.strip()) < MIN_EVIDENCE_LENGTH:
                reasons.append("Low confidence claim lacks sufficient evidence.")

        if _should_check_budget(assumption):
            _apply_budget_constraints(lower, upper, reasons)

        if _should_check_timeline(assumption):
            _apply_timeline_constraints(lower, upper, assumption.unit, reasons)

        if _should_check_team(assumption):
            _apply_team_constraints(lower, upper, reasons)

        passed_flag = not reasons
        if passed_flag:
            passed += 1

        entry = ValidationEntry(
            assumption_id=assumption.assumption_id,
            question=assumption.question,
            passed=passed_flag,
            reasons=reasons
        )
        entries.append(entry)

    total = len(entries)
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0
    return ValidationReport(
        entries=entries,
        total_assumptions=total,
        passed=passed,
        failed=failed,
        pass_rate_pct=round(pass_rate, 2)
    )


def render_validation_summary(report: ValidationReport) -> str:
    lines = [
        "# Fermi Sanity Check",
        "",
        f"- Total assumptions: {report.total_assumptions}",
        f"- Passed: {report.passed}",
        f"- Failed: {report.failed}",
        f"- Pass rate: {report.pass_rate_pct:.1f}%",
        ""
    ]

    if report.failed:
        lines.append("## Failed assumptions")
        for entry in report.entries:
            if not entry.passed:
                reasons = ", ".join(entry.reasons) if entry.reasons else "No details provided."
                lines.append(f"- `{entry.assumption_id}` ({entry.question or 'question missing'}): {reasons}")

    return "\n".join(lines)


def _should_check_budget(assumption: QuantifiedAssumption) -> bool:
    text = (assumption.question or "").lower()
    return any(keyword in text for keyword in BUDGET_KEYWORDS) or (assumption.unit or "") in CURRENCY_UNITS


def _should_check_timeline(assumption: QuantifiedAssumption) -> bool:
    text = (assumption.question or "").lower()
    return any(keyword in text for keyword in TIMELINE_KEYWORDS)


def _should_check_team(assumption: QuantifiedAssumption) -> bool:
    text = (assumption.question or "").lower()
    return any(keyword in text for keyword in TEAM_KEYWORDS)


def _apply_budget_constraints(lower: Optional[float], upper: Optional[float], reasons: List[str]) -> None:
    if lower is not None and lower < BUDGET_LOWER_THRESHOLD:
        reasons.append(f"Budget below ${BUDGET_LOWER_THRESHOLD:,.0f}.")
    if upper is not None and upper > BUDGET_UPPER_THRESHOLD:
        reasons.append(f"Budget above ${BUDGET_UPPER_THRESHOLD:,.0f}.")


def _apply_timeline_constraints(
    lower: Optional[float], upper: Optional[float], unit: Optional[str], reasons: List[str]
) -> None:
    lower_days = _normalize_to_days(lower, unit)
    upper_days = _normalize_to_days(upper, unit)

    if lower_days is not None and lower_days < TIMELINE_MIN_DAYS:
        reasons.append("Timeline below 1 day.")
    if upper_days is not None and upper_days > TIMELINE_MAX_DAYS:
        reasons.append("Timeline exceeds ten years (3,650 days).")


def _normalize_to_days(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if not unit:
        return value
    normalized = TIME_UNIT_TO_DAYS.get(unit.lower())
    if normalized is None:
        return value
    return value * normalized


def _apply_team_constraints(lower: Optional[float], upper: Optional[float], reasons: List[str]) -> None:
    if lower is not None and lower < TEAM_MIN:
        reasons.append("Team size below 1 person.")
    if upper is not None and upper > TEAM_MAX:
        reasons.append("Team size above 1,000 people.")

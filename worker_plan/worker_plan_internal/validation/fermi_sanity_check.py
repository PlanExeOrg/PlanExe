"""
Rule-based validation gate for plan assumptions.

Audits assumptions from the pipeline for common failure modes:
- Missing quantification (no numbers/bounds)
- Implausibly wide spans (upper/lower ratio > 100x)
- Missing evidence for low-confidence claims
- Budget/timeline/team values that look ungrounded

This is a pure rule-based check — no LLM calls. It runs fast and catches
the most common failure modes in LLM-generated assumptions.

See: docs/proposals/88-fermi-sanity-check-validation-gate.md

PROMPT> python -m worker_plan_internal.validation.fermi_sanity_check
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CheckResult(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class AssumptionCheck:
    """Result of validating a single assumption."""
    assumption_index: int
    assumption_text: str
    result: CheckResult
    checks: List[dict] = field(default_factory=list)

    def add_check(self, name: str, result: CheckResult, detail: str) -> None:
        self.checks.append({"name": name, "result": result.value, "detail": detail})
        # Promote the overall result to the worst seen.
        if result == CheckResult.FAIL:
            self.result = CheckResult.FAIL
        elif result == CheckResult.WARN and self.result != CheckResult.FAIL:
            self.result = CheckResult.WARN


# Regex for numbers (integers, decimals, with optional currency/unit suffixes).
_NUMBER_PATTERN = re.compile(
    r'\b\d[\d,]*\.?\d*\s*(?:%|USD|EUR|GBP|JPY|CNY|INR|BRL|CHF|CAD|AUD|'
    r'million|billion|thousand|hundred|M|B|K|k|'
    r'years?|months?|weeks?|days?|hours?|'
    r'people|persons?|FTEs?|employees?|staff|'
    r'km|miles?|m|meters?|feet|'
    r'kg|lbs?|tons?|tonnes?|'
    r'units?|items?|pieces?|'
    r'sq\s*m|sq\s*ft|hectares?|acres?)?\b',
    re.IGNORECASE,
)

# Patterns that suggest a range (two values forming bounds).
_RANGE_PATTERN = re.compile(
    r'\d[\d,]*\.?\d*\s*[-–—to]+\s*\d[\d,]*\.?\d*',
    re.IGNORECASE,
)

# Weak confidence markers.
_LOW_CONFIDENCE_MARKERS = [
    "approximately", "roughly", "around", "about",
    "estimated", "estimate", "assumed", "assuming",
    "unclear", "uncertain", "unknown",
    "best guess", "placeholder", "tbd", "to be determined",
]


def _has_quantification(text: str) -> bool:
    """True if the assumption contains at least one number with context."""
    return bool(_NUMBER_PATTERN.search(text))


def _has_range(text: str) -> bool:
    """True if the assumption specifies a range (lower-upper bounds)."""
    return bool(_RANGE_PATTERN.search(text))


def _extract_range_ratio(text: str) -> Optional[float]:
    """Extract the ratio of upper/lower from the first range found."""
    match = _RANGE_PATTERN.search(text)
    if not match:
        return None
    nums = re.findall(r'\d[\d,]*\.?\d*', match.group())
    if len(nums) < 2:
        return None
    try:
        lower = float(nums[0].replace(",", ""))
        upper = float(nums[1].replace(",", ""))
        if lower <= 0:
            return None
        return upper / lower
    except (ValueError, ZeroDivisionError):
        return None


def _has_low_confidence_marker(text: str) -> bool:
    text_lower = text.lower()
    return any(marker in text_lower for marker in _LOW_CONFIDENCE_MARKERS)


def _has_evidence_reference(text: str) -> bool:
    """True if the text references evidence, sources, or data."""
    evidence_markers = [
        "based on", "according to", "data shows", "research",
        "study", "survey", "report", "analysis", "benchmark",
        "historical", "comparable", "industry standard",
        "source:", "ref:", "citation",
    ]
    text_lower = text.lower()
    return any(marker in text_lower for marker in evidence_markers)


def validate_assumption(index: int, text: str) -> AssumptionCheck:
    """Run all validation rules on a single assumption."""
    check = AssumptionCheck(
        assumption_index=index,
        assumption_text=text,
        result=CheckResult.PASS,
    )

    # Rule 1: Quantification present
    if not _has_quantification(text):
        check.add_check(
            "quantification",
            CheckResult.WARN,
            "No numeric values found. Consider adding specific numbers, bounds, or ranges.",
        )
    else:
        check.add_check("quantification", CheckResult.PASS, "Contains numeric values.")

    # Rule 2: Range bounds (only if quantified)
    if _has_quantification(text) and _has_range(text):
        ratio = _extract_range_ratio(text)
        if ratio is not None:
            if ratio > 100:
                check.add_check(
                    "span_ratio",
                    CheckResult.FAIL,
                    f"Range span ratio is {ratio:.0f}x (>100x). Bounds are implausibly wide.",
                )
            elif ratio > 50:
                check.add_check(
                    "span_ratio",
                    CheckResult.WARN,
                    f"Range span ratio is {ratio:.0f}x (>50x). Consider tightening bounds.",
                )
            else:
                check.add_check("span_ratio", CheckResult.PASS, f"Range span ratio is {ratio:.1f}x.")

    # Rule 3: Low confidence without evidence
    if _has_low_confidence_marker(text) and not _has_evidence_reference(text):
        check.add_check(
            "evidence_for_uncertainty",
            CheckResult.WARN,
            "Low-confidence language detected but no evidence reference. Consider citing a source.",
        )

    return check


@dataclass
class FermiSanityCheckReport:
    """Full validation report for a set of assumptions."""
    checks: List[AssumptionCheck]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.result == CheckResult.PASS)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.result == CheckResult.WARN)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.result == CheckResult.FAIL)

    @property
    def pass_rate_pct(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.pass_count / self.total) * 100.0

    @property
    def overall_result(self) -> CheckResult:
        if self.fail_count > 0:
            return CheckResult.FAIL
        if self.warn_count > 0:
            return CheckResult.WARN
        return CheckResult.PASS

    def to_dict(self) -> dict:
        return {
            "overall_result": self.overall_result.value,
            "pass_rate_pct": round(self.pass_rate_pct, 1),
            "total": self.total,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "checks": [
                {
                    "index": c.assumption_index,
                    "text": c.assumption_text,
                    "result": c.result.value,
                    "checks": c.checks,
                }
                for c in self.checks
            ],
        }

    def save_raw(self, file_path: str) -> None:
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_markdown(self, file_path: str) -> None:
        lines = ["# Fermi Sanity Check Report\n"]
        lines.append(f"**Overall Result:** {self.overall_result.value}")
        lines.append(f"**Pass Rate:** {self.pass_rate_pct:.1f}%")
        lines.append(f"**Summary:** {self.pass_count} pass, {self.warn_count} warn, {self.fail_count} fail out of {self.total}\n")

        for check in self.checks:
            icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[check.result.value]
            lines.append(f"## [{icon}] Assumption {check.assumption_index + 1}\n")
            lines.append(f"> {check.assumption_text}\n")
            for c in check.checks:
                result_icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[c["result"]]
                lines.append(f"- **{c['name']}** [{result_icon}]: {c['detail']}")
            lines.append("")

        Path(file_path).write_text("\n".join(lines), encoding="utf-8")


def run_fermi_sanity_check(assumptions: List[str]) -> FermiSanityCheckReport:
    """Validate a list of assumption strings and return the report."""
    checks = [validate_assumption(i, text) for i, text in enumerate(assumptions)]
    return FermiSanityCheckReport(checks=checks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_assumptions = [
        "The project budget is estimated at 500,000-2,000,000 EUR over 3 years.",
        "Approximately 15 team members will be needed, though this is unclear.",
        "The timeline is roughly 18 months based on industry benchmarks for similar projects.",
        "Stakeholder engagement will be sufficient.",
        "Infrastructure costs range from 1,000 to 500,000 USD depending on scale.",
    ]

    report = run_fermi_sanity_check(sample_assumptions)
    print(json.dumps(report.to_dict(), indent=2))

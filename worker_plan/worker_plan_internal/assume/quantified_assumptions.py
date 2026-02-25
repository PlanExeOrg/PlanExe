"""Structured helpers for extracting numerical assumptions from MakeAssumptions outputs."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from worker_plan_internal.assume.make_assumptions import MakeAssumptions

LOGGER = logging.getLogger(__name__)

RANGE_PATTERN = re.compile(
    r"(?P<low>-?\d+(?:[\.,]\d+)?)(?:\s*(?:-|–|—|to|and)\s*(?P<high>-?\d+(?:[\.,]\d+)?))?",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"-?\d+(?:[\.,]\d+)?")
UNIT_WORD_PATTERN = re.compile(r"\b([A-Za-z%°µΩ]+)\b")

LOW_CONFIDENCE_WORDS = {
    "estimate",
    "approx",
    "approximately",
    "around",
    "roughly",
    "maybe",
    "could",
    "likely",
    "tends",
    "suggest",
}
HIGH_CONFIDENCE_WORDS = {
    "will",
    "must",
    "guarantee",
    "ensure",
    "ensures",
    "ensuring",
    "required",
    "definitely",
    "strongly",
    "committed",
}

ASSUMPTION_PREFIX = "Assumption:"


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class QuantifiedAssumption(BaseModel):
    assumption_id: str = Field(description="Unique identifier for this assumption")
    question: str = Field(description="Source question that elicited the assumption")
    claim: str = Field(description="Normalized assumption text without the 'Assumption:' label")
    lower_bound: Optional[float] = Field(None, description="Lower bound extracted from the claim")
    upper_bound: Optional[float] = Field(None, description="Upper bound extracted from the claim")
    unit: Optional[str] = Field(None, description="Unit associated with the bounds")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.medium,
        description="Estimated confidence level for this claim",
    )
    evidence: str = Field(description="Evidence excerpt or justification for the numeric claim")
    extracted_numbers: List[float] = Field(default_factory=list)
    raw_assumption: str = Field(description="Original assumption text from MakeAssumptions")

    class Config:
        allow_mutation = False
        frozen = True

    @property
    def span_ratio(self) -> Optional[float]:
        if self.lower_bound is None or self.upper_bound is None:
            return None
        if self.lower_bound <= 0:
            return None
        ratio = self.upper_bound / self.lower_bound
        LOGGER.debug("Computed span_ratio=%.2f for %s", ratio, self.assumption_id)
        return ratio


@dataclass
class QuantifiedAssumptionSummary:
    assumptions: List[QuantifiedAssumption]

    @property
    def average_span(self) -> Optional[float]:
        spans = [assumption.span_ratio for assumption in self.assumptions if assumption.span_ratio is not None]
        if not spans:
            return None
        return sum(spans) / len(spans)


class QuantifiedAssumptionExtractor:
    """Extract structured numeric assumptions from MakeAssumptions outputs."""

    def extract(self, assumption_entries: Sequence[dict]) -> List[QuantifiedAssumption]:
        results: list[QuantifiedAssumption] = []
        for idx, entry in enumerate(assumption_entries, start=1):
            question = (entry.get("question") or "").strip()
            raw_assumption = (entry.get("assumptions") or "").strip()
            if not raw_assumption:
                LOGGER.debug("Skipping empty assumption entry at index %s", idx)
                continue
            claim = self._normalize_claim(raw_assumption)
            lower, upper, unit = self._parse_bounds(claim)
            extracted = self._extract_numbers(claim)
            confidence = self._guess_confidence(claim)
            assumption_id = entry.get("assumption_id") or f"assumption-{idx}"
            results.append(
                QuantifiedAssumption(
                    assumption_id=assumption_id,
                    question=question,
                    claim=claim,
                    lower_bound=lower,
                    upper_bound=upper,
                    unit=unit,
                    confidence=confidence,
                    evidence=claim,
                    extracted_numbers=extracted,
                    raw_assumption=raw_assumption,
                )
            )
        return results

    def extract_from_make_assumptions(self, result: MakeAssumptions) -> List[QuantifiedAssumption]:
        return self.extract(result.assumptions)

    def _guess_confidence(self, claim: str) -> ConfidenceLevel:
        lowered = claim.lower()
        if any(word in lowered for word in LOW_CONFIDENCE_WORDS):
            return ConfidenceLevel.low
        if any(word in lowered for word in HIGH_CONFIDENCE_WORDS):
            return ConfidenceLevel.high
        return ConfidenceLevel.medium

    def _normalize_claim(self, raw_assumption: str) -> str:
        trimmed = raw_assumption.strip()
        if trimmed.lower().startswith(ASSUMPTION_PREFIX.lower()):
            trimmed = trimmed[len(ASSUMPTION_PREFIX) :].strip()
        trimmed = re.sub(r"^[\-:]+", "", trimmed).strip()
        trimmed = re.sub(r"\s{2,}", " ", trimmed)
        return trimmed

    def _parse_bounds(self, claim: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
        sanitized = claim.replace("—", "-").replace("–", "-")
        match = RANGE_PATTERN.search(sanitized)
        if not match:
            return None, None, self._extract_unit(claim)
        lower = self._coerce_number(match.group("low"))
        upper = self._coerce_number(match.group("high")) if match.group("high") else lower
        unit = self._extract_unit(claim, match.end())
        return lower, upper, unit

    def _extract_unit(self, claim: str, position: Optional[int] = None) -> Optional[str]:
        target = claim
        if position is not None:
            target = claim[position : position + 20]
        match = UNIT_WORD_PATTERN.search(target)
        if match:
            return match.group(1).lower()
        return None

    def _extract_numbers(self, claim: str) -> List[float]:
        numbers: List[float] = []
        for value in NUMBER_PATTERN.findall(claim):
            coerced = self._coerce_number(value)
            if coerced is not None:
                numbers.append(coerced)
        return numbers

    def _coerce_number(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            LOGGER.debug("Failed to coerce %s to float", value)
            return None


if __name__ == "__main__":
    extractor = QuantifiedAssumptionExtractor()
    with open("worker_plan/worker_plan_internal/assume/test_data/assumptions_solar_farm_in_denmark.json", "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    for assumption in extractor.extract(entries):
        print(assumption.json(indent=2))

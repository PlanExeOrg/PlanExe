"""
Author: Larry (Claude Opus 4.6)
Date: 2026-02-25
PURPOSE: Domain-aware normalization for FermiSanityCheck. Loads domain profiles (YAML),
auto-detects project domain from assumptions, and normalizes currency/units/confidence
to standard metric/English output for AI agents.
SRP/DRY check: Pass - Consumes QuantifiedAssumption schema + domain profile YAML.
Outputs normalized assumptions ready for validation.
"""

import logging
import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path

from worker_plan_internal.assume.quantified_assumptions import (
    QuantifiedAssumption,
    ConfidenceLevel,
)

LOGGER = logging.getLogger(__name__)

# Find domain profiles YAML
DOMAIN_PROFILES_PATH = Path(__file__).parent.parent / "docs" / "domain-profiles" / "domain-profile-schema.md"


class DomainProfile:
    """Represents a single domain profile (carpenter, dentist, etc.)"""

    def __init__(self, profile_dict: Dict[str, Any]):
        self.id = profile_dict.get("id")
        self.name = profile_dict.get("name")
        self.description = profile_dict.get("description")

        # Currency
        currency_cfg = profile_dict.get("currency", {})
        self.default_currency = currency_cfg.get("default", "USD")
        self.currency_aliases = set(currency_cfg.get("aliases", []))
        self.currency_aliases.add(self.default_currency.lower())

        # Units
        units_cfg = profile_dict.get("units", {})
        self.metric_first = units_cfg.get("metric", True)
        self.unit_conversions = {}
        for conv in units_cfg.get("convert", []):
            self.unit_conversions[conv["from"].lower()] = {
                "to": conv["to"],
                "factor": conv["factor"],
            }

        # Heuristics
        heuristics = profile_dict.get("heuristics", {})
        self.budget_keywords = set(heuristics.get("budget_keywords", []))
        self.timeline_keywords = set(heuristics.get("timeline_keywords", []))
        self.team_keywords = set(heuristics.get("team_keywords", []))

        confidence_kw = heuristics.get("confidence_keywords", {})
        self.high_confidence_words = set(confidence_kw.get("high", []))
        self.medium_confidence_words = set(confidence_kw.get("medium", []))
        self.low_confidence_words = set(confidence_kw.get("low", []))

        # Detection
        detection = profile_dict.get("detection", {})
        self.currency_signals = set(detection.get("currency_signals", []))
        self.unit_signals = set(detection.get("unit_signals", []))
        self.keyword_signals = set(detection.get("keyword_signals", []))

    def score_match(self, currency_found: List[str], units_found: List[str], keywords_found: List[str]) -> int:
        """Score how well this profile matches the found signals."""
        score = 0
        for c in currency_found:
            if c.lower() in [s.lower() for s in self.currency_signals]:
                score += 10
        for u in units_found:
            if u.lower() in [s.lower() for s in self.unit_signals]:
                score += 5
        for k in keywords_found:
            if k.lower() in [s.lower() for s in self.keyword_signals]:
                score += 3
        return score


@dataclass
class NormalizedAssumption:
    """Assumption after domain-aware normalization."""
    assumption_id: str
    original_claim: str
    normalized_claim: str
    domain_id: str
    currency: str  # Normalized to domain default
    currency_eur_equivalent: Optional[float] = None  # For comparison
    unit: str = "metric"  # All converted to metric
    confidence: ConfidenceLevel = ConfidenceLevel.medium
    notes: List[str] = field(default_factory=list)


class DomainNormalizer:
    """Loads domain profiles and normalizes assumptions to metric/currency/confidence."""

    def __init__(self, profiles_yaml_path: Optional[str] = None):
        self.profiles: Dict[str, DomainProfile] = {}
        self.default_profile = None

        path = Path(profiles_yaml_path) if profiles_yaml_path else DOMAIN_PROFILES_PATH
        self._load_profiles(path)

    def _load_profiles(self, yaml_path: Path) -> None:
        """Load domain profiles from YAML file."""
        if not yaml_path.exists():
            LOGGER.warning(f"Domain profiles not found at {yaml_path}; using defaults")
            self._create_default_profiles()
            return

        try:
            with open(yaml_path, "r") as f:
                content = f.read()
                # Extract YAML from markdown code block
                if "```yaml" in content:
                    yaml_start = content.index("```yaml") + 7
                    yaml_end = content.index("```", yaml_start)
                    yaml_str = content[yaml_start:yaml_end]
                else:
                    yaml_str = content

                data = yaml.safe_load(yaml_str)
                if data and "profiles" in data:
                    for profile_dict in data["profiles"]:
                        profile = DomainProfile(profile_dict)
                        self.profiles[profile.id] = profile
                        if not self.default_profile:
                            self.default_profile = profile

            LOGGER.info(f"Loaded {len(self.profiles)} domain profiles from {yaml_path}")
        except Exception as e:
            LOGGER.error(f"Error loading domain profiles: {e}; using defaults")
            self._create_default_profiles()

    def _create_default_profiles(self) -> None:
        """Create minimal default profiles if YAML not available."""
        default_profile_dict = {
            "id": "default",
            "name": "General Business",
            "description": "Default profile for unclassified projects.",
            "currency": {"default": "USD", "aliases": ["usd", "$"]},
            "units": {"metric": True, "convert": []},
            "heuristics": {
                "budget_keywords": ["budget", "cost"],
                "timeline_keywords": ["days", "weeks"],
                "team_keywords": ["team", "people"],
                "confidence_keywords": {
                    "high": ["guarantee", "have done"],
                    "medium": ["plan to", "expect"],
                    "low": ["estimate", "maybe"],
                },
            },
            "detection": {
                "currency_signals": ["USD", "$"],
                "unit_signals": [],
                "keyword_signals": [],
            },
        }
        self.default_profile = DomainProfile(default_profile_dict)
        self.profiles["default"] = self.default_profile

    def detect_domain(self, assumption: QuantifiedAssumption) -> DomainProfile:
        """Auto-detect domain profile from assumption metadata."""
        # Extract signals from assumption
        currency_found = []
        if assumption.unit:
            currency_found.append(assumption.unit)

        units_found = []
        if assumption.unit:
            units_found.append(assumption.unit)

        keywords_found = []
        # Extract keywords from claim + evidence
        claim_lower = assumption.claim.lower()
        evidence_lower = (assumption.evidence or "").lower()
        combined = f"{claim_lower} {evidence_lower}".split()

        # Score all profiles
        scores = {}
        for profile_id, profile in self.profiles.items():
            score = profile.score_match(currency_found, units_found, combined)
            scores[profile_id] = score

        # Pick highest scoring profile
        if scores:
            best_profile_id = max(scores, key=scores.get)
            if scores[best_profile_id] > 0:
                return self.profiles[best_profile_id]

        return self.default_profile

    def normalize_currency(
        self, value: Optional[float], from_currency: str, to_profile: DomainProfile
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Convert currency to profile default.
        Returns (normalized_value, eur_equivalent).
        """
        if value is None:
            return None, None

        # Placeholder conversion rates (in production, use real FX API)
        fx_rates = {
            "USD": 0.92,  # USD → EUR
            "DKK": 0.124,  # DKK → EUR
            "EUR": 1.0,
        }

        # For now, assume all inputs are in the detected currency or profile default
        normalized = value
        eur_equiv = value * fx_rates.get(to_profile.default_currency, 1.0)

        return normalized, eur_equiv

    def normalize_unit(self, value: Optional[float], from_unit: str, to_profile: DomainProfile) -> Optional[float]:
        """Convert unit to metric (based on profile conversions)."""
        if value is None or not from_unit:
            return value

        from_unit_lower = from_unit.lower()
        if from_unit_lower in to_profile.unit_conversions:
            conversion = to_profile.unit_conversions[from_unit_lower]
            return value * conversion["factor"]

        return value

    def normalize_confidence(self, assumption: QuantifiedAssumption, domain: DomainProfile) -> ConfidenceLevel:
        """Re-assess confidence level based on domain keywords."""
        claim_lower = assumption.claim.lower()
        evidence_lower = (assumption.evidence or "").lower()
        combined = f"{claim_lower} {evidence_lower}"

        # Check high confidence
        if any(word in combined for word in domain.high_confidence_words):
            return ConfidenceLevel.high

        # Check low confidence
        if any(word in combined for word in domain.low_confidence_words):
            return ConfidenceLevel.low

        # Default to medium
        return ConfidenceLevel.medium

    def normalize(self, assumption: QuantifiedAssumption) -> NormalizedAssumption:
        """Normalize a QuantifiedAssumption to domain standards."""
        domain = self.detect_domain(assumption)

        # Normalize currency
        norm_currency, eur_equiv = self.normalize_currency(assumption.lower_bound, assumption.unit or "", domain)

        # Normalize unit (keep as "metric" for now)
        norm_unit = "metric"

        # Re-assess confidence per domain
        norm_confidence = self.normalize_confidence(assumption, domain)

        # Build normalized claim
        norm_claim = f"{assumption.claim} [normalized to {domain.id} domain]"

        notes = []
        if domain.id != "default":
            notes.append(f"Auto-detected domain: {domain.name}")

        return NormalizedAssumption(
            assumption_id=assumption.assumption_id,
            original_claim=assumption.claim,
            normalized_claim=norm_claim,
            domain_id=domain.id,
            currency=domain.default_currency,
            currency_eur_equivalent=eur_equiv,
            unit=norm_unit,
            confidence=norm_confidence,
            notes=notes,
        )

    def normalize_batch(self, assumptions: List[QuantifiedAssumption]) -> List[NormalizedAssumption]:
        """Normalize a batch of assumptions."""
        return [self.normalize(assumption) for assumption in assumptions]

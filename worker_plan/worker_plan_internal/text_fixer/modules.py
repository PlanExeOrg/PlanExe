"""
TextFixer — Post-Processing Modules for LLM Outputs

Author: Egon (VoynichLabs), 2026-03-29

PURPOSE: Deterministic, zero-cost post-processing that removes hedging language,
preambles, disclaimers, and overly formal vocabulary from LLM-generated plan text.
Each module is a composable transformer: text in, cleaned text out.

These run AFTER the LLM generates output, catching patterns that persist
regardless of prompt quality.

Patterns are defined in rules.json — editable without touching Python code.
Multiple rule files can be loaded (e.g. per-language: rules_zh.json, rules_ja.json).

SRP: Each module handles one class of text cleanup.
DRY: Shared TextFixer.apply() chains any combination of modules.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# PatternBuilder — fluent API for constructing regex pattern lists
# =============================================================================

class PatternBuilder:
    """Build a list of (compiled_regex, replacement) pairs with a fluent API."""

    def __init__(self):
        self._patterns: List[tuple] = []

    def regex_advanced(self, pattern: str, replacement: Union[str, Callable], flags: int) -> None:
        """Add a regex pattern with its replacement (string or callable) and flags. All params required."""
        self._patterns.append((re.compile(pattern, flags), replacement))

    def regex(self, pattern: str, replacement: str = '', flags: int = re.IGNORECASE) -> None:
        """Add a regex pattern with its replacement and flags."""
        self.regex_advanced(pattern, replacement, flags)

    def regex_m(self, pattern: str, replacement: str = '') -> None:
        """Add a regex pattern with IGNORECASE | MULTILINE flags."""
        self.regex(pattern, replacement, flags=re.IGNORECASE | re.MULTILINE)

    def regex_s(self, pattern: str, replacement: str = '') -> None:
        """Add a regex pattern with IGNORECASE | DOTALL flags."""
        self.regex(pattern, replacement, flags=re.IGNORECASE | re.DOTALL)

    def word(self, text: str, replacement: str = '') -> None:
        """Add a word-boundary pattern: 'prior to' → r'\\bprior to\\b'."""
        self.regex(rf'\b{re.escape(text)}\b', replacement)

    def phrase(self, text: str, replacement: str = '') -> None:
        """Word-boundary match with optional trailing comma and whitespace consumed."""
        self.regex(rf'\b{re.escape(text)}\b,?\s*', replacement)

    def load_rule(self, rule: dict) -> None:
        """Load a rule dict from JSON into this builder. Handles nested groups."""
        rule_type = rule['type']
        pattern = rule.get('pattern', '')
        replacement = rule.get('replacement', '')

        if rule_type == 'group':
            for sub_rule in rule.get('rules', []):
                self.load_rule(sub_rule)
        elif rule_type == 'regex':
            self.regex(pattern, replacement)
        elif rule_type == 'regex_m':
            self.regex_m(pattern, replacement)
        elif rule_type == 'regex_s':
            self.regex_s(pattern, replacement)
        elif rule_type == 'word':
            self.word(pattern, replacement)
        elif rule_type == 'phrase':
            self.phrase(pattern, replacement)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    def load_rules(self, rules: list) -> None:
        """Load a list of rule dicts from JSON."""
        for rule in rules:
            self.load_rule(rule)

    @property
    def patterns(self) -> List[tuple]:
        """Return the built pattern list."""
        return list(self._patterns)


# =============================================================================
# TextFixerModule — a single composable text cleanup unit
# =============================================================================

@dataclass
class TextFixerModule:
    """A single text fixer module — one class of regex-based cleanup."""
    id: str
    name: str
    description: str
    patterns: List[tuple] = field(default_factory=list)
    enabled: bool = True
    hit_counts: dict = field(default_factory=dict, repr=False)

    def transform(self, text: str) -> str:
        """Apply all patterns in this module to the input text.
        Tracks how many substitutions each pattern made in hit_counts."""
        result = text
        for i, (pattern, replacement) in enumerate(self.patterns):
            result, count = pattern.subn(replacement, result)
            if count > 0:
                self.hit_counts[i] = self.hit_counts.get(i, 0) + count
        return result

    def reset_counts(self) -> None:
        """Reset all hit counters to zero."""
        self.hit_counts.clear()

    @property
    def total_hits(self) -> int:
        """Total number of substitutions across all patterns."""
        return sum(self.hit_counts.values())


# =============================================================================
# Post-processing patterns (appended to every module)
# =============================================================================

def _cleanup_spaces(b: PatternBuilder) -> None:
    """Add space-cleanup patterns to the builder."""
    b.regex(r'  +', ' ')
    b.regex_m(r'^\s+', '')


def _fix_capitalization(b: PatternBuilder) -> None:
    """Add capitalization-fix patterns to the builder."""
    b.regex_advanced(r'^([a-z])', lambda m: m.group(1).upper(), re.IGNORECASE)
    b.regex_advanced(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), re.IGNORECASE)
    b.regex_advanced(r'^([*\-•]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), re.IGNORECASE | re.MULTILINE)


# =============================================================================
# TextFixer — catalog of modules, loaded on demand
# =============================================================================

class TextFixer:
    """
    A catalog of TextFixerModules, loaded from JSON rule files.

    Usage:
        fixer = TextFixer()
        fixer.load(Path('rules.json'))
        result = fixer.apply(text)
        result = fixer.apply(text, module_ids=['hedge_reducer'])
    """

    def __init__(self):
        self._modules: Dict[str, TextFixerModule] = {}

    def load(self, filepath: Path) -> None:
        """Load modules from a JSON rules file. Can be called multiple times
        to load additional rule files (e.g. per-language)."""
        logger.debug(f"TextFixer.load. filepath: {filepath!r}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for module_data in data.get('modules', []):
            module_id = module_data.get('id')
            if not module_id:
                logger.error(f"Missing 'id' in module definition in {filepath}. Skipping.")
                continue

            if module_id in self._modules:
                logger.warning(f"Duplicate module id '{module_id}' in {filepath}. Overwriting.")

            b = PatternBuilder()
            b.load_rules(module_data.get('rules', []))
            _cleanup_spaces(b)
            _fix_capitalization(b)

            module = TextFixerModule(
                id=module_id,
                name=module_data.get('name', module_id),
                description=module_data.get('comment', ''),
                patterns=b.patterns,
                enabled=module_data.get('enabled', True),
            )
            self._modules[module_id] = module

    @classmethod
    def default_rules_path(cls) -> Path:
        """Return the path to the default rules.json file."""
        return Path(__file__).parent / 'rules.json'

    def get_module(self, module_id: str) -> Optional[TextFixerModule]:
        """Get a module by its ID. Returns None if not found."""
        return self._modules.get(module_id)

    def module_ids(self) -> List[str]:
        """Return all loaded module IDs in insertion order."""
        return list(self._modules.keys())

    def enabled_module_ids(self) -> List[str]:
        """Return IDs of all enabled modules in insertion order."""
        return [m.id for m in self._modules.values() if m.enabled]

    def all_modules(self) -> List[TextFixerModule]:
        """Return all loaded modules in insertion order."""
        return list(self._modules.values())

    def apply(self, text: str, module_ids: Optional[List[str]] = None) -> str:
        """
        Apply modules to text in sequence.

        Args:
            text: The LLM output text to clean up.
            module_ids: List of module IDs to apply, in order.
                        If None, applies all enabled modules.

        Returns:
            Cleaned text with all specified modules applied.
        """
        if module_ids is None:
            module_ids = self.enabled_module_ids()

        result = text
        for module_id in module_ids:
            module = self._modules.get(module_id)
            if module is None:
                logger.warning(f"Unknown module '{module_id}', skipping.")
                continue
            if module.enabled:
                result = module.transform(result)

        return result

    def reset_all_counts(self) -> None:
        """Reset hit counters on all loaded modules."""
        for module in self._modules.values():
            module.reset_counts()


# =============================================================================
# Convenience functions (backward compatible)
# =============================================================================

def apply_modules(text: str, module_ids: Optional[List[str]] = None, rules_path: Optional[Path] = None) -> str:
    """Convenience function: load default rules and apply modules."""
    fixer = TextFixer()
    fixer.load(rules_path or TextFixer.default_rules_path())
    return fixer.apply(text, module_ids)


def apply_all_enabled(text: str, rules_path: Optional[Path] = None) -> str:
    """Convenience function: load default rules and apply all enabled modules."""
    fixer = TextFixer()
    fixer.load(rules_path or TextFixer.default_rules_path())
    return fixer.apply(text)

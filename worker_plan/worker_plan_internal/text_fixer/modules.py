"""
TextFixer — Post-Processing Modules for LLM Outputs

Author: Egon (VoynichLabs), 2026-03-29

PURPOSE: Deterministic, zero-cost post-processing that removes hedging language,
preambles, disclaimers, and overly formal vocabulary from LLM-generated plan text.
Each module is a composable transformer: text in, cleaned text out.

These run AFTER the LLM generates output, catching patterns that persist
regardless of prompt quality.

Patterns are defined in rules.json — editable without touching Python code.

SRP: Each module handles one class of text cleanup.
DRY: Shared apply_modules() chains any combination of modules.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union


# =============================================================================
# PatternBuilder — fluent API for constructing regex pattern lists
# =============================================================================

class PatternBuilder:
    """Build a list of (compiled_regex, replacement) pairs with a fluent API."""

    def __init__(self):
        self._patterns: List[tuple] = []

    def regex(self, pattern: str, replacement: Union[str, Callable] = '', flags: int = re.IGNORECASE) -> None:
        """Add a regex pattern with its replacement (string or callable) and flags."""
        self._patterns.append((re.compile(pattern, flags), replacement))

    def regex_m(self, pattern: str, replacement: str = '') -> None:
        """Add a regex pattern with IGNORECASE | MULTILINE flags."""
        return self.regex(pattern, replacement, flags=re.IGNORECASE | re.MULTILINE)

    def regex_s(self, pattern: str, replacement: str = '') -> None:
        """Add a regex pattern with IGNORECASE | DOTALL flags."""
        return self.regex(pattern, replacement, flags=re.IGNORECASE | re.DOTALL)

    def word(self, text: str, replacement: str = '') -> None:
        """Add a word-boundary pattern: 'prior to' → r'\\bprior to\\b'."""
        return self.regex(rf'\b{re.escape(text)}\b', replacement)

    def phrase(self, text: str, replacement: str = '') -> None:
        """Word-boundary match with optional trailing comma and whitespace consumed."""
        return self.regex(rf'\b{re.escape(text)}\b,?\s*', replacement)

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


def _cleanup_spaces(b: PatternBuilder) -> None:
    """Add space-cleanup patterns to the builder."""
    b.regex(r'  +', ' ')
    b.regex_m(r'^\s+', '')


def _fix_capitalization(b: PatternBuilder) -> None:
    """Add capitalization-fix patterns to the builder."""
    # Start of text
    b.regex(r'^([a-z])', lambda m: m.group(1).upper())
    # After sentence-ending punctuation
    b.regex(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper())
    # Start of markdown list items
    b.regex_m(r'^([*\-•]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper())


# =============================================================================
# JSON Rule Loader
# =============================================================================

def _load_module(module_data: dict) -> TextFixerModule:
    """Load a TextFixerModule from a JSON module definition."""
    b = PatternBuilder()
    b.load_rules(module_data.get('rules', []))
    _cleanup_spaces(b)
    _fix_capitalization(b)

    return TextFixerModule(
        id=module_data['id'],
        name=module_data['name'],
        description=module_data.get('comment', ''),
        patterns=b.patterns,
        enabled=module_data.get('enabled', True),
    )


def load_rules(path: Optional[Path] = None) -> List[TextFixerModule]:
    """Load all modules from a rules JSON file."""
    if path is None:
        path = Path(__file__).parent / 'rules.json'

    with open(path, 'r') as f:
        data = json.load(f)

    return [_load_module(m) for m in data.get('modules', [])]


# =============================================================================
# MODULE REGISTRY & PUBLIC API
# =============================================================================

# Load modules from rules.json at import time
ALL_MODULES: List[TextFixerModule] = load_rules()

# Default module set — all enabled modules
DEFAULT_MODULES: List[str] = [m.id for m in ALL_MODULES if m.enabled]

# Registry for lookup by ID
MODULE_REGISTRY: dict = {m.id: m for m in ALL_MODULES}

# Convenience accessors for individual modules
preamble_stripper = MODULE_REGISTRY.get('preamble_stripper')
hedge_reducer = MODULE_REGISTRY.get('hedge_reducer')
disclaimer_stripper = MODULE_REGISTRY.get('disclaimer_stripper')
formal_reducer = MODULE_REGISTRY.get('formal_reducer')


def get_module(module_id: str) -> Optional[TextFixerModule]:
    """Get a module by its ID."""
    return MODULE_REGISTRY.get(module_id)


def apply_modules(text: str, module_ids: Optional[List[str]] = None) -> str:
    """
    Apply TextFixer modules to text in sequence.

    Args:
        text: The LLM output text to clean up.
        module_ids: List of module IDs to apply, in order.
                    If None, applies all enabled modules.

    Returns:
        Cleaned text with all specified modules applied.
    """
    if module_ids is None:
        module_ids = DEFAULT_MODULES

    result = text
    for module_id in module_ids:
        module = MODULE_REGISTRY.get(module_id)
        if module and module.enabled:
            result = module.transform(result)

    return result


def apply_all_enabled(text: str) -> str:
    """Apply all enabled modules in recommended order."""
    result = text
    for module in ALL_MODULES:
        if module.enabled:
            result = module.transform(result)
    return result

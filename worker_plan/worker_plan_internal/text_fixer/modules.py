"""
TextFixer — Post-Processing Modules for LLM Outputs

Author: Egon (VoynichLabs), 2026-03-29

PURPOSE: Deterministic, zero-cost post-processing that removes hedging language,
preambles, disclaimers, and overly formal vocabulary from LLM-generated plan text.
Each module is a composable transformer: text in, cleaned text out.

These run AFTER the LLM generates output, catching patterns that persist
regardless of prompt quality.

SRP: Each module handles one class of text cleanup.
DRY: Shared apply_modules() chains any combination of modules.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# PatternBuilder — fluent API for constructing regex pattern lists
# =============================================================================

class PatternBuilder:
    """Build a list of (compiled_regex, replacement) pairs with a fluent API."""

    def __init__(self):
        self._patterns: List[tuple] = []

    def regex(self, pattern: str, replacement: str = '', flags: int = re.IGNORECASE) -> 'PatternBuilder':
        """Add a regex pattern with its replacement and flags."""
        self._patterns.append((re.compile(pattern, flags), replacement))
        return self

    def regex_m(self, pattern: str, replacement: str = '') -> 'PatternBuilder':
        """Add a regex pattern with IGNORECASE | MULTILINE flags."""
        return self.regex(pattern, replacement, flags=re.IGNORECASE | re.MULTILINE)

    def regex_s(self, pattern: str, replacement: str = '') -> 'PatternBuilder':
        """Add a regex pattern with IGNORECASE | DOTALL flags."""
        return self.regex(pattern, replacement, flags=re.IGNORECASE | re.DOTALL)

    def word(self, text: str, replacement: str = '') -> 'PatternBuilder':
        """Add a word-boundary pattern: 'prior to' → r'\\bprior to\\b'."""
        return self.regex(rf'\b{re.escape(text)}\b', replacement)

    def phrase(self, text: str, replacement: str = '') -> 'PatternBuilder':
        """Word-boundary match with optional trailing comma and whitespace consumed."""
        return self.regex(rf'\b{re.escape(text)}\b,?\s*', replacement)

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

    def transform(self, text: str) -> str:
        """Apply all patterns in this module to the input text."""
        result = text
        for pattern, replacement in self.patterns:
            result = pattern.sub(replacement, result)
        # Clean up artifacts from removals
        result = re.sub(r'  +', ' ', result)
        result = re.sub(r'^\s+', '', result, flags=re.MULTILINE)
        result = _fix_capitalization(result)
        return result


def _fix_capitalization(text: str) -> str:
    """Capitalize sentence starts that were left lowercase after pattern removal."""
    text = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), text)
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'^([*\-•]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text, flags=re.MULTILINE)
    return text


# =============================================================================
# MODULE: Hedge Reducer
# Strips hedging and uncertainty language that weakens plan documents.
# =============================================================================

b = PatternBuilder()
# Epistemic hedges — "I think X" → "X"
b.phrase('I think that')
b.phrase('I think')
b.phrase('I believe that')
b.phrase('I believe')
b.phrase('I would say that')
b.phrase('I would say')
b.phrase('I would suggest that')
b.phrase('I would argue that')
b.phrase('in my opinion')
b.phrase('from my perspective')
b.phrase('from my understanding')
# Probability hedges
b.phrase('perhaps')
b.phrase('maybe')
b.phrase('probably')
b.phrase('possibly')
b.phrase('conceivably')
# Meta-commentary — "it's worth noting that X" → "X"
b.regex(r"\bIt'?s important to note that\s+")
b.regex(r"\bIt'?s worth noting that\s+")
b.regex(r"\bIt'?s worth mentioning that\s+")
b.phrase('it should be noted that')
b.phrase('it is important to consider that')
b.phrase('I should mention that')
b.phrase('I need to point out that')
b.phrase('I must emphasize that')
b.phrase('it bears mentioning that')

hedge_reducer = TextFixerModule(
    id='hedge_reducer',
    name='Hedge Reducer',
    description='Strips hedging and uncertainty language from plan text',
    patterns=b.patterns,
    enabled=True,
)


# =============================================================================
# MODULE: Preamble Stripper
# Removes LLM conversational openers and filler that add no information.
# =============================================================================

b = PatternBuilder()
# Conversational openers at start of text/paragraph
b.regex_m(r'^Sure[,!.]?\s*')
b.regex_m(r'^Of course[,!.]?\s*')
b.regex_m(r'^Certainly[,!.]?\s*')
b.regex_m(r'^Absolutely[,!.]?\s*')
b.regex_m(r'^Great question[!.]?\s*')
b.regex_m(r'^Excellent question[!.]?\s*')
b.regex_m(r"^That'?s a (?:great|good|excellent|interesting) question[!.]?\s*")
b.regex_m(r"^I'?d be happy to help(?: you)?(?: with that)?[.!]?\s*")
b.regex_m(r'^Let me help you with that[.!]?\s*')
b.regex_m(r'^Thanks for (?:asking|sharing|your question)[.!]?\s*')
b.regex_m(r'^I understand (?:your|the) (?:question|concern|request)[.!]?\s*')
# Plan-document preambles (can appear mid-text)
b.regex(r'\bHere is a comprehensive\s+')
b.regex(r'\bBelow is a detailed\s+')
b.regex(r'\bThe following provides?\s+')
b.regex_m(r'^Based on (?:the|my|our) (?:analysis|review|assessment|evaluation),?\s*')
b.regex_m(r'^After (?:careful|thorough|detailed) (?:analysis|review|consideration|evaluation),?\s*')
b.regex_m(r'^Upon (?:careful|thorough)? ?(?:review|analysis|examination),?\s*')

preamble_stripper = TextFixerModule(
    id='preamble_stripper',
    name='Preamble Stripper',
    description='Removes conversational openers and filler phrases',
    patterns=b.patterns,
    enabled=True,
)


# =============================================================================
# MODULE: Disclaimer Stripper
# Removes safety disclaimers and "consult a professional" boilerplate.
# PlanExe generates plans, not advice — these disclaimers add noise.
# =============================================================================

b = PatternBuilder()
# Parenthesized disclaimer blocks
b.regex_s(r'\s*\(Note: This (?:analysis|plan|review|assessment|document) (?:is |should ).*?\)\s*')
b.regex(r'\s*\(Disclaimer:[^)]*\)\s*')
# Standalone disclaimer blocks
b.regex_s(r'\s*Disclaimer:.*?(?:\n\n|\Z)')
# "Consult a professional" variants
b.regex(r'\s*Please consult (?:a |with )?(?:qualified |professional |licensed )?[^.]*professional[^.]*\.')
b.regex(r'\s*(?:You should |We recommend )consult(?:ing)? (?:a |with )?[^.]*\.')
# "Not advice" variants
b.regex(r'\s*This (?:is not|should not be (?:considered|taken as)|does not constitute) (?:legal|financial|medical|professional|investment|tax) advice[^.]*\.')
# "Note: This analysis..." (non-parenthesized)
b.regex_m(r'\s*Note: This (?:analysis|plan|review|assessment) (?:is |should ).*?(?:\.|$)')

disclaimer_stripper = TextFixerModule(
    id='disclaimer_stripper',
    name='Disclaimer Stripper',
    description='Removes safety disclaimers and "consult a professional" boilerplate',
    patterns=b.patterns,
    enabled=True,
)


# =============================================================================
# MODULE: Formal Reducer
# Replaces overly formal vocabulary with plain language equivalents.
# Off by default — some formality is appropriate for plan documents.
# =============================================================================

b = PatternBuilder()
# Transition words
b.word('furthermore', 'also')
b.word('moreover', 'also')
b.word('additionally', 'also')
b.word('nevertheless', 'still')
b.word('consequently', 'so')
b.word('nonetheless', 'still')
# Verbose verbs
b.word('utilize', 'use')
b.word('utilization', 'use')
b.word('facilitate', 'help')
b.word('leverage', 'use')
b.word('commence', 'start')
b.word('implement', 'set up')
# Verbose phrases → concise equivalents
b.word('prior to', 'before')
b.word('subsequent to', 'after')
b.word('in order to', 'to')
b.word('due to the fact that', 'because')
b.word('at this point in time', 'now')
b.word('in the event that', 'if')
b.word('for the purpose of', 'to')
b.word('with regard to', 'about')
b.word('in light of', 'given')
# Note: "leverage" as a NOUN (as in P128 "lever identification") is domain-specific
# and should NOT be replaced. The pattern above only matches the standalone word,
# which is almost always the verb form in LLM output.

formal_reducer = TextFixerModule(
    id='formal_reducer',
    name='Formal Reducer',
    description='Replaces overly formal vocabulary with plain language',
    patterns=b.patterns,
    enabled=False,  # Off by default — some formality is appropriate for plan documents
)


# =============================================================================
# MODULE REGISTRY & PUBLIC API
# =============================================================================

# All available modules, in recommended application order
ALL_MODULES: List[TextFixerModule] = [
    preamble_stripper,    # Strip openers first (they're at the start of text)
    hedge_reducer,        # Then strip hedging throughout
    disclaimer_stripper,  # Strip disclaimers (usually at the end)
    formal_reducer,       # Vocabulary cleanup last (least aggressive)
]

# Default module set for PlanExe pipeline tasks
DEFAULT_MODULES: List[str] = ['preamble_stripper', 'hedge_reducer', 'disclaimer_stripper']

# Registry for lookup by ID
MODULE_REGISTRY: dict = {m.id: m for m in ALL_MODULES}


def get_module(module_id: str) -> Optional[TextFixerModule]:
    """Get a module by its ID."""
    return MODULE_REGISTRY.get(module_id)


def apply_modules(text: str, module_ids: Optional[List[str]] = None) -> str:
    """
    Apply TextFixer modules to text in sequence.

    Args:
        text: The LLM output text to clean up.
        module_ids: List of module IDs to apply, in order.
                    If None, applies DEFAULT_MODULES.

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

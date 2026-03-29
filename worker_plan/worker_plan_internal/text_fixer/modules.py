"""
TextFixer — Post-Processing Modules for LLM Outputs

Author: Egon (VoynichLabs), 2026-03-29
Rewritten: 2026-03-29 — clean-room implementation, no third-party code

PURPOSE: Deterministic, zero-cost post-processing that removes hedging language,
preambles, disclaimers, and overly formal vocabulary from LLM-generated plan text.
Each module is a composable transformer: text in, cleaned text out.

These run AFTER the LLM generates output, catching patterns that persist
regardless of prompt quality.

SRP: Each module handles one class of text cleanup.
DRY: Shared apply_modules() chains any combination of modules.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TextFixerModule:
    """A single text fixer module — one class of regex-based cleanup."""
    id: str
    name: str
    description: str
    patterns: List[tuple]  # List of (compiled_regex, replacement) pairs
    enabled: bool = True

    def transform(self, text: str) -> str:
        """Apply all patterns in this module to the input text."""
        result = text
        for pattern, replacement in self.patterns:
            result = pattern.sub(replacement, result)
        # Clean up artifacts from removals
        result = re.sub(r'  +', ' ', result)  # collapse double spaces
        result = re.sub(r'^\s+', '', result, flags=re.MULTILINE)  # strip leading whitespace
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
# These patterns are common across all LLM outputs and well-documented in
# prompt engineering literature. Written independently for PlanExe.
# =============================================================================

def _ci(pattern: str) -> re.Pattern:
    """Compile a case-insensitive pattern with word boundary."""
    return re.compile(pattern, re.IGNORECASE)

_HEDGE_PATTERNS = [
    # Epistemic hedges — "I think X" → "X"
    (_ci(r'\bI think\s+that\s+'), ''),
    (_ci(r'\bI think\s+'), ''),
    (_ci(r'\bI believe\s+that\s+'), ''),
    (_ci(r'\bI believe\s+'), ''),
    (_ci(r'\bI would say\s+that\s+'), ''),
    (_ci(r'\bI would say\s+'), ''),
    (_ci(r'\bI would suggest\s+that\s+'), ''),
    (_ci(r'\bI would argue\s+that\s+'), ''),
    (_ci(r'\bIn my opinion,?\s*'), ''),
    (_ci(r'\bFrom my perspective,?\s*'), ''),
    (_ci(r'\bFrom my understanding,?\s*'), ''),
    # Probability hedges
    (_ci(r'\bperhaps\s+'), ''),
    (_ci(r'\bmaybe\s+'), ''),
    (_ci(r'\bprobably\s+'), ''),
    (_ci(r'\bpossibly\s+'), ''),
    (_ci(r'\bconceivably\s+'), ''),
    # Meta-commentary — "it's worth noting that X" → "X"
    (_ci(r"\bIt'?s important to note that\s+"), ''),
    (_ci(r"\bIt'?s worth noting that\s+"), ''),
    (_ci(r"\bIt'?s worth mentioning that\s+"), ''),
    (_ci(r'\bIt should be noted that\s+'), ''),
    (_ci(r'\bIt is important to consider that\s+'), ''),
    (_ci(r'\bI should mention that\s+'), ''),
    (_ci(r'\bI need to point out that\s+'), ''),
    (_ci(r'\bI must emphasize that\s+'), ''),
    (_ci(r'\bIt bears mentioning that\s+'), ''),
]

hedge_reducer = TextFixerModule(
    id='hedge_reducer',
    name='Hedge Reducer',
    description='Strips hedging and uncertainty language from plan text',
    patterns=_HEDGE_PATTERNS,
    enabled=True,
)


# =============================================================================
# MODULE: Preamble Stripper
# Removes LLM conversational openers and filler that add no information.
# =============================================================================

_PREAMBLE_PATTERNS = [
    # Conversational openers at start of text/paragraph
    (re.compile(r'^Sure[,!.]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Of course[,!.]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Certainly[,!.]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Absolutely[,!.]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Great question[!.]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Excellent question[!.]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r"^That'?s a (?:great|good|excellent|interesting) question[!.]?\s*", re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r"^I'?d be happy to help(?: you)?(?: with that)?[.!]?\s*", re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Let me help you with that[.!]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Thanks for (?:asking|sharing|your question)[.!]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^I understand (?:your|the) (?:question|concern|request)[.!]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    # Plan-document preambles (can appear mid-text)
    (_ci(r'\bHere is a comprehensive\s+'), ''),
    (_ci(r'\bBelow is a detailed\s+'), ''),
    (_ci(r'\bThe following provides?\s+'), ''),
    (re.compile(r'^Based on (?:the|my|our) (?:analysis|review|assessment|evaluation),?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^After (?:careful|thorough|detailed) (?:analysis|review|consideration|evaluation),?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Upon (?:careful|thorough)? ?(?:review|analysis|examination),?\s*', re.IGNORECASE | re.MULTILINE), ''),
]

preamble_stripper = TextFixerModule(
    id='preamble_stripper',
    name='Preamble Stripper',
    description='Removes conversational openers and filler phrases',
    patterns=_PREAMBLE_PATTERNS,
    enabled=True,
)


# =============================================================================
# MODULE: Disclaimer Stripper
# Removes safety disclaimers and "consult a professional" boilerplate.
# PlanExe generates plans, not advice — these disclaimers add noise.
# =============================================================================

_DISCLAIMER_PATTERNS = [
    # Parenthesized disclaimer blocks
    (re.compile(r'\s*\(Note: This (?:analysis|plan|review|assessment|document) (?:is |should ).*?\)\s*', re.IGNORECASE | re.DOTALL), ''),
    (re.compile(r'\s*\(Disclaimer:[^)]*\)\s*', re.IGNORECASE), ''),
    # Standalone disclaimer blocks
    (re.compile(r'\s*Disclaimer:.*?(?:\n\n|\Z)', re.IGNORECASE | re.DOTALL), ''),
    # "Consult a professional" variants
    (re.compile(r'\s*Please consult (?:a |with )?(?:qualified |professional |licensed )?[^.]*professional[^.]*\.', re.IGNORECASE), ''),
    (re.compile(r'\s*(?:You should |We recommend )consult(?:ing)? (?:a |with )?[^.]*\.', re.IGNORECASE), ''),
    # "Not advice" variants
    (re.compile(r'\s*This (?:is not|should not be (?:considered|taken as)|does not constitute) (?:legal|financial|medical|professional|investment|tax) advice[^.]*\.', re.IGNORECASE), ''),
    # "Note: This analysis..." (non-parenthesized)
    (re.compile(r'\s*Note: This (?:analysis|plan|review|assessment) (?:is |should ).*?(?:\.|$)', re.IGNORECASE | re.MULTILINE), ''),
]

disclaimer_stripper = TextFixerModule(
    id='disclaimer_stripper',
    name='Disclaimer Stripper',
    description='Removes safety disclaimers and "consult a professional" boilerplate',
    patterns=_DISCLAIMER_PATTERNS,
    enabled=True,
)


# =============================================================================
# MODULE: Formal Reducer
# Replaces overly formal vocabulary with plain language equivalents.
# Off by default — some formality is appropriate for plan documents.
# =============================================================================

_FORMAL_PATTERNS = [
    # Transition words
    (re.compile(r'\bFurthermore\b'), 'Also'),
    (re.compile(r'\bMoreover\b'), 'Also'),
    (re.compile(r'\bAdditionally\b'), 'Also'),
    (re.compile(r'\bNevertheless\b'), 'Still'),
    (re.compile(r'\bConsequently\b'), 'So'),
    (re.compile(r'\bNonetheless\b'), 'Still'),
    # Verbose verbs
    (re.compile(r'\bUtilize\b'), 'Use'),
    (re.compile(r'\butilize\b'), 'use'),
    (re.compile(r'\bUtilization\b'), 'Use'),
    (re.compile(r'\butilization\b'), 'use'),
    (re.compile(r'\bfacilitate\b'), 'help'),
    (re.compile(r'\bFacilitate\b'), 'Help'),
    (re.compile(r'\bleverage\b'), 'use'),
    (re.compile(r'\bLeverage\b'), 'Use'),
    (re.compile(r'\bCommence\b'), 'Start'),
    (re.compile(r'\bcommence\b'), 'start'),
    (re.compile(r'\bImplement\b'), 'Set up'),
    (re.compile(r'\bimplement\b'), 'set up'),
    # Verbose phrases → concise equivalents
    (_ci(r'\bPrior to\b'), 'Before'),
    (_ci(r'\bSubsequent to\b'), 'After'),
    (_ci(r'\bIn order to\b'), 'To'),
    (_ci(r'\bDue to the fact that\b'), 'Because'),
    (_ci(r'\bAt this point in time\b'), 'Now'),
    (_ci(r'\bIn the event that\b'), 'If'),
    (_ci(r'\bFor the purpose of\b'), 'To'),
    (_ci(r'\bWith regard to\b'), 'About'),
    (_ci(r'\bIn light of\b'), 'Given'),
    # Note: "leverage" as a NOUN (as in P128 "lever identification") is domain-specific
    # and should NOT be replaced. The pattern above only matches the standalone word,
    # which is almost always the verb form in LLM output.
]

formal_reducer = TextFixerModule(
    id='formal_reducer',
    name='Formal Reducer',
    description='Replaces overly formal vocabulary with plain language',
    patterns=_FORMAL_PATTERNS,
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

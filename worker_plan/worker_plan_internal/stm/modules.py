"""
Semantic Transformation Modules (STM) — Post-Processing for LLM Outputs

Ported from G0DM0D3 (elder-plinius/G0DM0D3) src/stm/modules.ts
Original author: Elder Plinius
Python port: Egon (VoynichLabs), 2026-03-29

PURPOSE: Deterministic, zero-cost post-processing that removes hedging language,
preambles, and overly formal vocabulary from LLM-generated text. Each module is
a composable transformer that takes text in and returns cleaned text out.

These are NOT prompt engineering. They run AFTER the LLM generates output,
catching patterns that persist regardless of prompt quality.

SRP: Each module handles one class of text cleanup.
DRY: Shared apply_modules() chains any combination of modules.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class STMModule:
    """A single semantic transformation module."""
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
        # Clean up: fix double spaces, leading whitespace on lines, capitalize after removal
        result = re.sub(r'  +', ' ', result)
        result = re.sub(r'^\s+', '', result, flags=re.MULTILINE)
        result = _capitalize_sentence_starts(result)
        return result


def _capitalize_sentence_starts(text: str) -> str:
    """Capitalize the first letter of each sentence after pattern removal."""
    # Handle start of text
    text = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), text)
    # Handle after sentence-ending punctuation
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    # Handle start of lines (for markdown lists, paragraphs)
    text = re.sub(r'^([*\-•]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text, flags=re.MULTILINE)
    return text


# =============================================================================
# MODULE: Hedge Reducer
# Removes hedging language for more confident, direct outputs.
# Source: G0DM0D3 src/stm/modules.ts hedgeReducer
# =============================================================================

_HEDGE_PATTERNS = [
    (re.compile(r'\bI think\s+', re.IGNORECASE), ''),
    (re.compile(r'\bI believe\s+', re.IGNORECASE), ''),
    (re.compile(r'\bperhaps\s+', re.IGNORECASE), ''),
    (re.compile(r'\bmaybe\s+', re.IGNORECASE), ''),
    (re.compile(r'\bIt seems like\s+', re.IGNORECASE), ''),
    (re.compile(r'\bIt appears that\s+', re.IGNORECASE), ''),
    (re.compile(r'\bprobably\s+', re.IGNORECASE), ''),
    (re.compile(r'\bpossibly\s+', re.IGNORECASE), ''),
    (re.compile(r'\bI would say\s+', re.IGNORECASE), ''),
    (re.compile(r'\bIn my opinion,?\s*', re.IGNORECASE), ''),
    (re.compile(r'\bFrom my perspective,?\s*', re.IGNORECASE), ''),
    # PlanExe-specific hedge patterns (not in G0DM0D3)
    (re.compile(r"\bIt'?s important to note that\s+", re.IGNORECASE), ''),
    (re.compile(r"\bIt'?s worth noting that\s+", re.IGNORECASE), ''),
    (re.compile(r'\bIt should be noted that\s+', re.IGNORECASE), ''),
    (re.compile(r'\bIt is important to consider that\s+', re.IGNORECASE), ''),
    (re.compile(r'\bI should mention that\s+', re.IGNORECASE), ''),
    (re.compile(r'\bI need to point out that\s+', re.IGNORECASE), ''),
    (re.compile(r'\bI must emphasize that\s+', re.IGNORECASE), ''),
]

hedge_reducer = STMModule(
    id='hedge_reducer',
    name='Hedge Reducer',
    description='Removes hedging language for more confident, direct outputs',
    patterns=_HEDGE_PATTERNS,
    enabled=True,
)


# =============================================================================
# MODULE: Direct Mode
# Removes preambles and filler phrases that add no information.
# Source: G0DM0D3 src/stm/modules.ts directMode
# =============================================================================

_PREAMBLE_PATTERNS = [
    (re.compile(r'^Sure,?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Of course,?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Certainly,?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Absolutely,?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Great question!?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r"^That'?s a great question!?\s*", re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r"^I'?d be happy to help( you)?( with that)?[.!]?\s*", re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Let me help you with that[.!]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^I understand[.!]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^Thanks for asking[.!]?\s*', re.IGNORECASE | re.MULTILINE), ''),
    # PlanExe-specific preamble patterns (no ^ anchor — these can appear after other preamble stripping)
    (re.compile(r'\bHere is a comprehensive\s+', re.IGNORECASE), ''),
    (re.compile(r'\bBelow is a detailed\s+', re.IGNORECASE), ''),
    (re.compile(r'\bThe following provides?\s+', re.IGNORECASE), ''),
    (re.compile(r'^Based on (?:the|my) (?:analysis|review|assessment),?\s*', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'^After careful (?:analysis|review|consideration),?\s*', re.IGNORECASE | re.MULTILINE), ''),
]

direct_mode = STMModule(
    id='direct_mode',
    name='Direct Mode',
    description='Removes preambles and filler phrases',
    patterns=_PREAMBLE_PATTERNS,
    enabled=True,
)


# =============================================================================
# MODULE: Formal Reducer
# Replaces overly formal vocabulary with plain language.
# Source: G0DM0D3 src/stm/modules.ts casualMode (adapted — less aggressive)
# =============================================================================

_FORMAL_PATTERNS = [
    (re.compile(r'\bFurthermore\b'), 'Also'),
    (re.compile(r'\bMoreover\b'), 'Also'),
    (re.compile(r'\bAdditionally\b'), 'Also'),
    (re.compile(r'\bNevertheless\b'), 'Still'),
    (re.compile(r'\bConsequently\b'), 'So'),
    (re.compile(r'\bUtilize\b'), 'Use'),
    (re.compile(r'\butilize\b'), 'use'),
    (re.compile(r'\bUtilization\b'), 'Use'),
    (re.compile(r'\butilization\b'), 'use'),
    (re.compile(r'\bCommence\b'), 'Start'),
    (re.compile(r'\bcommence\b'), 'start'),
    (re.compile(r'\bPrior to\b', re.IGNORECASE), 'Before'),
    (re.compile(r'\bSubsequent to\b', re.IGNORECASE), 'After'),
    (re.compile(r'\bIn order to\b', re.IGNORECASE), 'To'),
    (re.compile(r'\bDue to the fact that\b', re.IGNORECASE), 'Because'),
    (re.compile(r'\bAt this point in time\b', re.IGNORECASE), 'Now'),
    (re.compile(r'\bIn the event that\b', re.IGNORECASE), 'If'),
    (re.compile(r'\bfacilitate\b'), 'help'),
    (re.compile(r'\bFacilitate\b'), 'Help'),
    (re.compile(r'\bleverage\b'), 'use'),
    (re.compile(r'\bLeverage\b'), 'Use'),
    # Note: "leverage" as a NOUN (as in P128 "lever identification") is domain-specific
    # and should NOT be replaced. This pattern only matches the verb form, which is
    # almost always corporate filler. The noun "leverage" in financial contexts is fine.
]

formal_reducer = STMModule(
    id='formal_reducer',
    name='Formal Reducer',
    description='Replaces overly formal vocabulary with plain language',
    patterns=_FORMAL_PATTERNS,
    enabled=False,  # Off by default — some formality is appropriate for plan documents
)


# =============================================================================
# MODULE: Disclaimer Stripper
# Removes safety disclaimers and "consult a professional" language.
# PlanExe-specific — not from G0DM0D3.
# =============================================================================

_DISCLAIMER_PATTERNS = [
    # Match parenthesized disclaimer blocks (greedy within parens)
    (re.compile(r'\s*\(Note: This (?:analysis|plan|review|assessment) (?:is |should ).*?\)\s*', re.IGNORECASE | re.DOTALL), ''),
    # Match non-parenthesized "Note: This analysis..." to end of line/paragraph
    (re.compile(r'\s*Note: This (?:analysis|plan|review|assessment) (?:is |should ).*?(?:\.|$)', re.IGNORECASE | re.MULTILINE), ''),
    (re.compile(r'\s*Please consult (?:a |with )?(?:qualified |professional |licensed )?[^.]*\.', re.IGNORECASE), ''),
    (re.compile(r'\s*This (?:is not|should not be considered) (?:legal|financial|medical|professional) advice[^.]*\.', re.IGNORECASE), ''),
    (re.compile(r'\s*\(Disclaimer:[^)]*\)\s*', re.IGNORECASE), ''),
    (re.compile(r'\s*Disclaimer:.*?(?:\n\n|\Z)', re.IGNORECASE | re.DOTALL), ''),
]

disclaimer_stripper = STMModule(
    id='disclaimer_stripper',
    name='Disclaimer Stripper',
    description='Removes safety disclaimers and "consult a professional" boilerplate',
    patterns=_DISCLAIMER_PATTERNS,
    enabled=True,
)


# =============================================================================
# MODULE REGISTRY & PUBLIC API
# =============================================================================

# All available modules, in recommended application order
ALL_MODULES: List[STMModule] = [
    direct_mode,        # Strip preambles first (they're at the start of text)
    hedge_reducer,      # Then strip hedging throughout
    disclaimer_stripper,  # Strip disclaimers (usually at the end)
    formal_reducer,     # Vocabulary cleanup last (least aggressive)
]

# Default module set for PlanExe pipeline tasks
DEFAULT_MODULES: List[str] = ['direct_mode', 'hedge_reducer', 'disclaimer_stripper']

# Registry for lookup by ID
MODULE_REGISTRY: dict = {m.id: m for m in ALL_MODULES}


def get_module(module_id: str) -> Optional[STMModule]:
    """Get a module by its ID."""
    return MODULE_REGISTRY.get(module_id)


def apply_modules(text: str, module_ids: Optional[List[str]] = None) -> str:
    """
    Apply STM modules to text in sequence.

    Args:
        text: The LLM output text to clean up.
        module_ids: List of module IDs to apply, in order.
                    If None, applies DEFAULT_MODULES.

    Returns:
        Cleaned text with all enabled modules applied.
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

"""
Fallback cost estimation from token counts when the provider doesn't report cost.

Some providers (e.g. OpenRouter) include a ``cost`` field in the usage response.
Others (e.g. direct OpenAI, Anthropic) do not.  This module maintains a
pricing registry keyed by model-id so that ``track_activity`` can estimate
cost from token counts when the provider-reported cost is unavailable.

The registry is populated from the ``pricing`` field in llm_config entries
when the config is loaded (see ``load_pricing_from_llm_config``).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "register_model_pricing",
    "estimate_cost",
    "load_pricing_from_llm_config",
]

# {model_id: (input_per_million_tokens, output_per_million_tokens)}
_PRICING_REGISTRY: dict[str, tuple[float, float]] = {}


def register_model_pricing(
    model_id: str,
    input_per_million_tokens: float,
    output_per_million_tokens: float,
) -> None:
    """Register pricing rates for a model identifier."""
    _PRICING_REGISTRY[model_id] = (input_per_million_tokens, output_per_million_tokens)


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> Optional[float]:
    """Estimate cost from token counts using the pricing registry.

    Returns ``None`` if the model is not found in the registry.
    Thinking/reasoning tokens are billed at the output rate.
    """
    pricing = _find_pricing(model_name)
    if pricing is None:
        return None
    input_rate, output_rate = pricing
    cost = (
        input_tokens * input_rate + (output_tokens + thinking_tokens) * output_rate
    ) / 1_000_000
    return round(cost, 6)


def load_pricing_from_llm_config(llm_config_dict: dict) -> None:
    """Extract ``pricing`` from llm_config entries and populate the registry.

    Each config entry may contain::

        "pricing": {
            "input_per_million_tokens": 0.15,
            "output_per_million_tokens": 0.60
        }

    The model id used as the registry key comes from
    ``entry["arguments"]["model"]``.
    """
    for _key, entry in llm_config_dict.items():
        if not isinstance(entry, dict):
            continue
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            continue
        arguments = entry.get("arguments")
        if not isinstance(arguments, dict):
            continue
        model_id = arguments.get("model")
        if not model_id:
            continue
        input_rate = pricing.get("input_per_million_tokens")
        output_rate = pricing.get("output_per_million_tokens")
        if input_rate is None and output_rate is None:
            continue
        try:
            register_model_pricing(
                model_id=str(model_id),
                input_per_million_tokens=float(input_rate or 0.0),
                output_per_million_tokens=float(output_rate or 0.0),
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping pricing for %s: %s", model_id, exc)


def _find_pricing(model_name: str) -> Optional[tuple[float, float]]:
    """Look up pricing for *model_name*.

    Tries, in order:
    1. Exact match
    2. Longest-prefix match (handles version suffixes like
       ``gpt-5-nano-2025-08-07`` matching ``gpt-5-nano``)
    """
    if model_name in _PRICING_REGISTRY:
        return _PRICING_REGISTRY[model_name]

    best_match: Optional[str] = None
    best_len = 0
    for registered_id in _PRICING_REGISTRY:
        if model_name.startswith(registered_id) and len(registered_id) > best_len:
            best_match = registered_id
            best_len = len(registered_id)

    if best_match is not None:
        return _PRICING_REGISTRY[best_match]

    return None

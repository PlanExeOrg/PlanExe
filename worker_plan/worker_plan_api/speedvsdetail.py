from enum import Enum

"""
Speed-vs-detail modes for the PlanExe pipeline.

- PING_LLM: Loads the pipeline entrypoint and sends a single LLM request to confirm
  the model responds. Produces a small report quickly.
- FAST_BUT_SKIP_DETAILS: Runs the full Luigi pipeline but reduces looped/detail
  stages to the smallest set. Typically completes in ~5-10 minutes.
- ALL_DETAILS_BUT_SLOW: Runs the full pipeline with all details. Typically
  completes in ~10-20 minutes.
"""

class SpeedVsDetailEnum(str, Enum):
    """Options for the pipeline runtime/detail tradeoff."""
    ALL_DETAILS_BUT_SLOW = "all_details_but_slow"
    FAST_BUT_SKIP_DETAILS = "fast_but_skip_details"
    PING_LLM = "ping_llm"

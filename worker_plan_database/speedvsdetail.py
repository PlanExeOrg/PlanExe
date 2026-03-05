import logging
from typing import Any, Optional

from worker_plan_internal.plan.speedvsdetail import SpeedVsDetailEnum

logger = logging.getLogger(__name__)


def resolve_speedvsdetail(parameters: Optional[dict[str, Any]]) -> SpeedVsDetailEnum:
    speed_vs_detail_value: Optional[str] = None
    if isinstance(parameters, dict):
        speed_vs_detail_value = parameters.get("speed_vs_detail") or parameters.get("speedvsdetail")

    if isinstance(speed_vs_detail_value, str) and speed_vs_detail_value:
        for enum_value in SpeedVsDetailEnum:
            if enum_value.value.lower() == speed_vs_detail_value.lower():
                return enum_value
        logger.warning("Invalid speed_vs_detail value %r. Falling back to legacy flags.", speed_vs_detail_value)

    fast = isinstance(parameters, dict) and "fast" in parameters
    return SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS if fast else SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW

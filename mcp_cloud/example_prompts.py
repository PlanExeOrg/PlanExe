"""PlanExe MCP Cloud – example prompt loading."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_mcp_example_prompts() -> list[str]:
    """Load prompts from the catalog that are marked as MCP examples (mcp_example or mcp-example-prompt true).

    Uses worker_plan_api.PromptCatalog the same way as frontend_single_user and frontend_multi_user
    (no env var). Tries repo-root import first, then adds worker_plan to sys.path so worker_plan_api
    is top-level (same as frontends). Falls back to built-in examples if the catalog is unavailable.
    """
    catalog = None
    try:
        from worker_plan.worker_plan_api.prompt_catalog import PromptCatalog

        catalog = PromptCatalog()
        catalog.load_simple_plan_prompts()
    except Exception:
        try:
            # Same as frontends when worker_plan exists; when not (e.g. Docker), repo_root has worker_plan_api
            import sys

            repo_root = Path(__file__).resolve().parent.parent
            worker_plan_dir = repo_root / "worker_plan"
            path_to_add = str(worker_plan_dir if worker_plan_dir.exists() else repo_root)
            if path_to_add not in sys.path:
                sys.path.insert(0, path_to_add)
            from worker_plan_api.prompt_catalog import PromptCatalog

            catalog = PromptCatalog()
            catalog.load_simple_plan_prompts()
        except Exception as e:
            logger.warning(
                "Prompt catalog unavailable (%s); using built-in examples.",
                e,
            )
            return _builtin_mcp_example_prompts()

    if catalog is None:
        return _builtin_mcp_example_prompts()

    samples: list[str] = []
    for item in catalog.all():
        if item.extras.get("mcp_example") is True or item.extras.get("mcp-example-prompt") is True:
            samples.append(item.prompt)
    if not samples:
        return _builtin_mcp_example_prompts()
    return samples


def _builtin_mcp_example_prompts() -> list[str]:
    """Fallback example prompts when the catalog file is missing or has no mcp_example entries."""
    return [
        (
            "Vegan Butcher Shop. That sells artificial meat (Plant-Based). Location Kødbyen, Copenhagen. "
            "Sell sandwiches and sausages. Provocative marketing. Budget: 10 million DKK. Grand Opening in month 3. "
            "Profitability Goal: month 12. Create a signature item that is a social media hit. "
            "Pick a realistic scenario. I already have negotiated a 2 year lease inside Kødbyen. "
            "Banned words: blockchain, VR, AR, AI, Robots."
        ),
        (
            "Start a dental clinic in Copenhagen with 3 treatment rooms, targeting families and children. "
            "Budget 2.5M DKK. Open within 12 months. Include equipment, staffing, permits, and marketing. "
            "Pick a realistic scenario; avoid overly ambitious timelines."
        ),
    ]

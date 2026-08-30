"""
CLI helper: score a single task output from a completed run directory.

Loads a task output JSON file from a run_id_dir, reads the plan prompt
from 001-2-plan.txt, and scores it using TaskOutputScorer.

PROMPT> python -m worker_plan_internal.scoring.score_run_task --run-id-dir /path/to/run --task-filename 014-1-swot_analysis_raw.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from worker_plan_internal.scoring.task_output_scorer import TaskOutputScorer
from worker_plan_internal.llm_factory import get_llm

logger = logging.getLogger(__name__)


def load_plan_prompt(run_id_dir: Path) -> str:
    """Load the plan prompt from the run directory."""
    plan_file = run_id_dir / "001-2-plan.txt"
    if not plan_file.exists():
        raise FileNotFoundError(f"Plan prompt file not found: {plan_file}")
    return plan_file.read_text(encoding="utf-8").strip()


def load_task_output(run_id_dir: Path, task_filename: str) -> dict:
    """Load a task output JSON file from the run directory."""
    task_file = run_id_dir / task_filename
    if not task_file.exists():
        raise FileNotFoundError(f"Task output file not found: {task_file}")
    return json.loads(task_file.read_text(encoding="utf-8"))


def derive_task_name(task_filename: str) -> str:
    """Derive a human-readable task name from the filename.

    Example: '014-1-swot_analysis_raw.json' -> 'swot_analysis_raw'
    """
    stem = Path(task_filename).stem
    # Strip leading NNN-N- prefix
    parts = stem.split("-", 2)
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a pipeline task output using an LLM judge."
    )
    parser.add_argument(
        "--run-id-dir",
        type=str,
        required=True,
        help="Path to the completed run directory.",
    )
    parser.add_argument(
        "--task-filename",
        type=str,
        required=True,
        help="Filename of the task output JSON within the run directory.",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default=None,
        help="LLM model name to use as judge. Uses default if not specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path for the score JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("worker_plan_internal.scoring").setLevel(logging.INFO)

    run_id_dir = Path(args.run_id_dir)
    if not run_id_dir.is_dir():
        print(f"Error: run-id-dir does not exist: {run_id_dir}", file=sys.stderr)
        sys.exit(1)

    plan_prompt = load_plan_prompt(run_id_dir)
    task_output = load_task_output(run_id_dir, args.task_filename)
    task_name = derive_task_name(args.task_filename)

    logger.info(f"Scoring task '{task_name}' from {run_id_dir}")

    if args.llm_name:
        llm = get_llm(args.llm_name)
    else:
        llm = get_llm()

    result = TaskOutputScorer.score(
        llm=llm,
        task_output_json=task_output,
        plan_prompt=plan_prompt,
        task_name=task_name,
    )

    result_dict = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    result_json = json.dumps(result_dict, indent=2)

    if args.output:
        Path(args.output).write_text(result_json, encoding="utf-8")
        print(f"Score saved to: {args.output}")
    else:
        print(result_json)


if __name__ == "__main__":
    main()

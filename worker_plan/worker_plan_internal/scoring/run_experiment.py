"""
CLI for running a single A/B experiment comparing baseline vs candidate system prompts.

PROMPT> python -m worker_plan_internal.scoring.run_experiment \
    --task swot_business \
    --candidate-prompt "You are an expert strategic consultant with 20 years of experience." \
    --candidate-description "Added experience qualifier" \
    --results-dir ./experiments
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from worker_plan_internal.scoring.experiment_config import ExperimentConfig
from worker_plan_internal.scoring.experiment_runner import ExperimentRunner, _TASK_REGISTRY
from worker_plan_internal.scoring.results_tracker import ResultsEntry, ResultsTracker

logger = logging.getLogger(__name__)


# Default baseline system prompts per task
_DEFAULT_BASELINES: dict[str, str] = {
    "swot_business": (
        "You are a universal strategic consultant with expertise in project management, "
        "business analysis, and innovation across various industries."
    ),
}


def _load_reference_prompts(prompts_file: str | None, inline_prompt: str | None) -> list[str]:
    """Load reference plan prompts from file or inline."""
    if prompts_file:
        path = Path(prompts_file)
        if not path.exists():
            print(f"Error: prompts file not found: {path}", file=sys.stderr)
            sys.exit(1)
        # One prompt per line, skip empty lines
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif inline_prompt:
        return [inline_prompt]
    else:
        # Default reference prompt for testing
        return ["Build a SaaS platform for project management targeting small teams with limited budgets."]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an A/B experiment comparing baseline vs candidate system prompts."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(_TASK_REGISTRY.keys()),
        help="Task to run the experiment on.",
    )
    parser.add_argument(
        "--candidate-prompt",
        type=str,
        required=True,
        help="Candidate system prompt to test.",
    )
    parser.add_argument(
        "--candidate-description",
        type=str,
        default="",
        help="Short description of what changed in the candidate prompt.",
    )
    parser.add_argument(
        "--baseline-prompt",
        type=str,
        default=None,
        help="Baseline system prompt. Uses default for the task if not specified.",
    )
    parser.add_argument(
        "--reference-prompts-file",
        type=str,
        default=None,
        help="File with reference plan prompts (one per line).",
    )
    parser.add_argument(
        "--reference-prompt",
        type=str,
        default=None,
        help="Single inline reference plan prompt.",
    )
    parser.add_argument(
        "--generation-llm",
        type=str,
        default=None,
        help="LLM model name for task generation.",
    )
    parser.add_argument(
        "--judge-llm",
        type=str,
        default=None,
        help="LLM model name for scoring/judging.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiments",
        help="Directory for results JSONL file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum score delta to declare keep/discard (default: 0.5).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from worker_plan_internal.llm_factory import get_llm

    baseline_prompt = args.baseline_prompt or _DEFAULT_BASELINES.get(args.task)
    if not baseline_prompt:
        print(f"Error: no default baseline for task {args.task!r}, use --baseline-prompt", file=sys.stderr)
        sys.exit(1)

    reference_prompts = _load_reference_prompts(args.reference_prompts_file, args.reference_prompt)

    config = ExperimentConfig(
        task_name=args.task,
        baseline_system_prompt=baseline_prompt,
        candidate_system_prompt=args.candidate_prompt,
        candidate_description=args.candidate_description or "No description",
        reference_plan_prompts=reference_prompts,
        judge_llm_name=args.judge_llm,
        generation_llm_name=args.generation_llm,
    )

    generation_llm = get_llm(args.generation_llm) if args.generation_llm else get_llm()
    judge_llm = get_llm(args.judge_llm) if args.judge_llm else get_llm()

    print(f"Running experiment: {config.experiment_id}")
    print(f"Task: {config.task_name}")
    print(f"Reference prompts: {len(reference_prompts)}")
    print(f"Candidate: {config.candidate_description}")
    print()

    result = ExperimentRunner.run(
        config=config,
        generation_llm=generation_llm,
        judge_llm=judge_llm,
        threshold=args.threshold,
    )

    # Save full result
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    full_result_path = results_dir / f"experiment_{config.experiment_id}.json"
    full_result_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\nFull result saved to: {full_result_path}")

    # Append to JSONL tracker
    tracker = ResultsTracker(results_dir / "experiment_results.jsonl")
    entry = ResultsEntry(
        experiment_id=config.experiment_id,
        timestamp=result.timestamp,
        task_name=config.task_name,
        baseline_avg_score=result.summary.baseline_avg,
        candidate_avg_score=result.summary.candidate_avg,
        delta=result.summary.delta,
        status=result.summary.status,
        candidate_description=config.candidate_description,
        duration_seconds=result.duration_seconds,
    )
    tracker.append(entry)

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULT: {result.summary.status.upper()}")
    print(f"Baseline avg: {result.summary.baseline_avg:.1f}")
    print(f"Candidate avg: {result.summary.candidate_avg:.1f}")
    print(f"Delta: {result.summary.delta:+.1f}")
    print(f"Duration: {result.duration_seconds}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

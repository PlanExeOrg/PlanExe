"""
Minimal A/B experiment runner: runs baseline + candidate system prompts on a task,
scores both outputs, and returns a comparison result.

For Phase 0 (#94) / Phase A (#59): calls task functions directly (no Luigi pipeline),
using the plan prompt as user_prompt.

PROMPT> python -m worker_plan_internal.scoring.experiment_runner
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil
from typing import Optional

from llama_index.core.llms.llm import LLM

from worker_plan_internal.scoring.experiment_config import ExperimentConfig
from worker_plan_internal.scoring.task_output_scorer import TaskOutputScore, TaskOutputScorer

logger = logging.getLogger(__name__)


# Registry of known task functions. Each entry maps a task_name to a callable
# with signature: (llm, user_prompt, system_prompt) -> dict
# Populated lazily to avoid import overhead.
_TASK_REGISTRY: dict[str, tuple[str, str]] = {
    "swot_business": (
        "worker_plan_internal.swot.swot_phase2_conduct_analysis",
        "swot_phase2_conduct_analysis",
    ),
}


def _get_task_function(task_name: str):
    """Dynamically import and return the task function."""
    if task_name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_name: {task_name!r}. "
            f"Available: {list(_TASK_REGISTRY.keys())}"
        )
    module_path, func_name = _TASK_REGISTRY[task_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


@dataclass
class ExperimentArm:
    """Result from one arm (baseline or candidate) of an experiment."""
    label: str  # "baseline" or "candidate"
    system_prompt: str
    task_output: dict
    score: TaskOutputScore


@dataclass
class ExperimentSummary:
    """Aggregate comparison between baseline and candidate."""
    baseline_avg: float
    candidate_avg: float
    delta: float
    status: str  # "keep", "discard", or "inconclusive"

    def to_dict(self) -> dict:
        return {
            "baseline_avg": round(self.baseline_avg, 2),
            "candidate_avg": round(self.candidate_avg, 2),
            "delta": round(self.delta, 2),
            "status": self.status,
        }


@dataclass
class ExperimentResult:
    """Full result from an A/B experiment."""
    config: ExperimentConfig
    arms: list[ExperimentArm]
    timestamp: str
    summary: ExperimentSummary
    duration_seconds: int = 0

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "arms": [
                {
                    "label": arm.label,
                    "score": arm.score.model_dump(),
                }
                for arm in self.arms
            ],
            "timestamp": self.timestamp,
            "summary": self.summary.to_dict(),
            "duration_seconds": self.duration_seconds,
        }


def _compute_summary(
    baseline_scores: list[float],
    candidate_scores: list[float],
    threshold: float = 0.5,
) -> ExperimentSummary:
    """Compute summary from score lists. Threshold is minimum delta to 'keep'."""
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    candidate_avg = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0
    delta = candidate_avg - baseline_avg

    if delta >= threshold:
        status = "keep"
    elif delta <= -threshold:
        status = "discard"
    else:
        status = "inconclusive"

    return ExperimentSummary(
        baseline_avg=baseline_avg,
        candidate_avg=candidate_avg,
        delta=delta,
        status=status,
    )


class ExperimentRunner:
    """Runs an A/B experiment comparing baseline vs candidate system prompts."""

    @staticmethod
    def run(
        config: ExperimentConfig,
        generation_llm: LLM,
        judge_llm: LLM,
        threshold: float = 0.5,
    ) -> ExperimentResult:
        """
        Run the experiment: for each reference plan prompt, call the task function
        with both baseline and candidate system prompts, score both, compare.

        Args:
            config: Experiment configuration.
            generation_llm: LLM used to generate task outputs.
            judge_llm: LLM used to score task outputs.
            threshold: Minimum delta to declare "keep" or "discard".
        """
        task_fn = _get_task_function(config.task_name)

        arms: list[ExperimentArm] = []
        baseline_scores: list[float] = []
        candidate_scores: list[float] = []

        start_time = time.perf_counter()

        for i, plan_prompt in enumerate(config.reference_plan_prompts):
            logger.info(
                f"Reference prompt {i+1}/{len(config.reference_plan_prompts)}"
            )

            # Run baseline
            logger.info("Running baseline arm...")
            baseline_output = task_fn(
                generation_llm, plan_prompt, config.baseline_system_prompt
            )
            baseline_score_result = TaskOutputScorer.score(
                llm=judge_llm,
                task_output_json=baseline_output,
                plan_prompt=plan_prompt,
                task_name=config.task_name,
            )
            baseline_arm = ExperimentArm(
                label="baseline",
                system_prompt=config.baseline_system_prompt,
                task_output=baseline_output,
                score=baseline_score_result.response,
            )
            arms.append(baseline_arm)
            baseline_scores.append(baseline_score_result.response.composite_score)

            # Run candidate
            logger.info("Running candidate arm...")
            candidate_output = task_fn(
                generation_llm, plan_prompt, config.candidate_system_prompt
            )
            candidate_score_result = TaskOutputScorer.score(
                llm=judge_llm,
                task_output_json=candidate_output,
                plan_prompt=plan_prompt,
                task_name=config.task_name,
            )
            candidate_arm = ExperimentArm(
                label="candidate",
                system_prompt=config.candidate_system_prompt,
                task_output=candidate_output,
                score=candidate_score_result.response,
            )
            arms.append(candidate_arm)
            candidate_scores.append(candidate_score_result.response.composite_score)

            logger.info(
                f"Prompt {i+1}: baseline={baseline_score_result.response.composite_score:.1f}, "
                f"candidate={candidate_score_result.response.composite_score:.1f}"
            )

        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))

        summary = _compute_summary(baseline_scores, candidate_scores, threshold)

        return ExperimentResult(
            config=config,
            arms=arms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            duration_seconds=duration,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demonstrate structure only (requires LLM to actually run)
    config = ExperimentConfig(
        task_name="swot_business",
        baseline_system_prompt="You are a strategic consultant.",
        candidate_system_prompt="You are an expert strategic consultant with 20 years of experience.",
        candidate_description="Added experience qualifier",
        reference_plan_prompts=["Build a SaaS platform for small teams."],
    )
    print(json.dumps(config.to_dict(), indent=2))
    print("\nTo run a full experiment, use: python -m worker_plan_internal.scoring.run_experiment")

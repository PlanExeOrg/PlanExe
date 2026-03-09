"""
Configuration for a single A/B experiment comparing baseline vs candidate system prompts.

PROMPT> python -m worker_plan_internal.scoring.experiment_config
"""
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single prompt optimization experiment."""
    task_name: str
    baseline_system_prompt: str
    candidate_system_prompt: str
    candidate_description: str
    reference_plan_prompts: list[str]
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    judge_llm_name: Optional[str] = None
    generation_llm_name: Optional[str] = None

    def __post_init__(self):
        if not self.task_name:
            raise ValueError("task_name must be non-empty.")
        if not self.baseline_system_prompt:
            raise ValueError("baseline_system_prompt must be non-empty.")
        if not self.candidate_system_prompt:
            raise ValueError("candidate_system_prompt must be non-empty.")
        if not self.reference_plan_prompts:
            raise ValueError("reference_plan_prompts must have at least one entry.")

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "task_name": self.task_name,
            "baseline_system_prompt": self.baseline_system_prompt,
            "candidate_system_prompt": self.candidate_system_prompt,
            "candidate_description": self.candidate_description,
            "reference_plan_prompts": self.reference_plan_prompts,
            "judge_llm_name": self.judge_llm_name,
            "generation_llm_name": self.generation_llm_name,
        }


if __name__ == "__main__":
    config = ExperimentConfig(
        task_name="swot_business",
        baseline_system_prompt="You are a strategic consultant.",
        candidate_system_prompt="You are an expert strategic consultant with 20 years of experience.",
        candidate_description="Added experience qualifier to system prompt",
        reference_plan_prompts=["Build a SaaS platform for small teams."],
    )
    import json
    print(json.dumps(config.to_dict(), indent=2))

"""
LLM-as-judge that scores a single pipeline task output against a 5-dimension rubric.

This is the core building block for autonomous prompt optimization (#94)
and A/B testing promotion (#59).

PROMPT> python -m worker_plan_internal.scoring.task_output_scorer
"""
import json
import time
import logging
from math import ceil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.format_json_for_use_in_query import format_json_for_use_in_query

logger = logging.getLogger(__name__)


class ScoreDimension(BaseModel):
    """A single scoring dimension with its score and justification."""
    dimension: str = Field(description="Name of the scoring dimension.")
    score: int = Field(description="Score from 1 to 10.", ge=1, le=10)
    justification: str = Field(
        description="Why this score was given. ~30 words."
    )


class TaskOutputScore(BaseModel):
    """Complete score for a task output across all dimensions."""
    dimensions: list[ScoreDimension] = Field(
        description="Scores for each evaluation dimension."
    )
    composite_score: float = Field(
        description="Weighted average score across all dimensions (1.0-10.0)."
    )
    overall_assessment: str = Field(
        description="Brief overall assessment of the task output quality. 2-3 sentences."
    )


DEFAULT_WEIGHTS: dict[str, float] = {
    "Specificity": 0.25,
    "Actionability": 0.25,
    "Completeness": 0.20,
    "Internal Consistency": 0.15,
    "Conciseness": 0.15,
}

SCORER_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated project planning outputs. \
Your task is to score a single task output on exactly 5 dimensions using a 1-10 scale.

DIMENSIONS AND WEIGHTS:
1. Specificity (weight 0.25) — Is the output concrete and grounded in the project context? \
Does it reference specific details from the plan prompt rather than generic advice? \
Score 1 = entirely generic, 10 = deeply tailored to this project.

2. Actionability (weight 0.25) — Can someone act on this output directly? \
Are there clear next steps, owners, or deliverables? \
Score 1 = vague platitudes, 10 = ready to execute.

3. Completeness (weight 0.20) — Are the obvious aspects covered? \
Are there glaring omissions given the task type? \
Score 1 = major gaps, 10 = thorough coverage.

4. Internal Consistency (weight 0.15) — Does the output align with the upstream plan context? \
Are there contradictions or unsupported leaps? \
Score 1 = contradictory, 10 = fully coherent.

5. Conciseness (weight 0.15) — Is the output free of filler, redundancy, and padding? \
Score 1 = extremely bloated, 10 = every word earns its place.

SCORING RULES:
- Score each dimension independently on a 1-10 integer scale.
- Provide a ~30 word justification for each score.
- Compute composite_score as the weighted average: \
sum(score_i * weight_i) for all dimensions.
- Write an overall_assessment of 2-3 sentences.
- Be calibrated: reserve 9-10 for genuinely excellent output, 1-3 for poor output. \
Most competent outputs should land in the 5-7 range.
"""


@dataclass
class TaskOutputScorer:
    """Scores a single pipeline task output using an LLM judge."""
    system_prompt: str
    user_prompt: str
    response: TaskOutputScore
    metadata: dict
    markdown: str

    @classmethod
    def score(
        cls,
        llm: LLM,
        task_output_json: dict,
        plan_prompt: str,
        task_name: str,
        weights: Optional[dict[str, float]] = None,
    ) -> 'TaskOutputScorer':
        """
        Score a task output against the 5-dimension rubric.

        Args:
            llm: LLM instance to use as judge.
            task_output_json: The task output to score (dict from JSON file).
            plan_prompt: The original plan prompt for context.
            task_name: Name of the pipeline task that produced this output.
            weights: Optional custom weights. Defaults to DEFAULT_WEIGHTS.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(task_output_json, dict):
            raise ValueError("task_output_json must be a dict.")
        if not isinstance(plan_prompt, str) or not plan_prompt.strip():
            raise ValueError("plan_prompt must be a non-empty string.")

        effective_weights = weights or DEFAULT_WEIGHTS

        # Strip metadata before presenting to judge
        cleaned_output = format_json_for_use_in_query(task_output_json)

        system_prompt = SCORER_SYSTEM_PROMPT

        user_prompt = f"""Score the following task output.

TASK NAME: {task_name}

PLAN PROMPT (the original project description):
{plan_prompt}

TASK OUTPUT TO SCORE:
{cleaned_output}

WEIGHTS TO USE:
{json.dumps(effective_weights, indent=2)}

Score each dimension, compute the weighted composite_score, and provide an overall_assessment."""

        chat_message_list = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        sllm = llm.as_structured_llm(TaskOutputScore)
        start_time = time.perf_counter()
        try:
            chat_response = sllm.chat(chat_message_list)
        except Exception as e:
            logger.debug(f"LLM scoring interaction failed: {e}")
            logger.error("LLM scoring interaction failed.", exc_info=True)
            raise ValueError("LLM scoring interaction failed.") from e

        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))

        task_output_score: TaskOutputScore = chat_response.raw

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["task_name"] = task_name

        markdown = cls._to_markdown(task_output_score, task_name)

        return cls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=task_output_score,
            metadata=metadata,
            markdown=markdown,
        )

    def to_dict(
        self,
        include_metadata: bool = True,
        include_system_prompt: bool = True,
        include_user_prompt: bool = True,
    ) -> dict:
        d: dict = {
            "response": self.response.model_dump(),
        }
        if include_metadata:
            d["metadata"] = self.metadata
        if include_system_prompt:
            d["system_prompt"] = self.system_prompt
        if include_user_prompt:
            d["user_prompt"] = self.user_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2))

    def save_markdown(self, output_file_path: str) -> None:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(self.markdown)

    @staticmethod
    def _to_markdown(score: TaskOutputScore, task_name: str) -> str:
        rows = []
        rows.append(f"# Task Output Score: {task_name}\n")
        rows.append(f"**Composite Score: {score.composite_score:.1f}/10**\n")
        rows.append("## Dimension Scores\n")
        rows.append("| Dimension | Score | Justification |")
        rows.append("|---|---|---|")
        for dim in score.dimensions:
            rows.append(f"| {dim.dimension} | {dim.score}/10 | {dim.justification} |")
        rows.append(f"\n## Overall Assessment\n")
        rows.append(score.overall_assessment)
        return "\n".join(rows)


if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Example: score a sample task output
    sample_output = {
        "strengths": ["Strong team", "Clear vision"],
        "weaknesses": ["Limited budget"],
        "opportunities": ["Growing market"],
        "threats": ["Competition"],
        "metadata": {"duration": 42},
    }
    sample_plan_prompt = "Build a SaaS platform for project management targeting small teams."

    llm = get_llm("ollama-llama3.1")
    result = TaskOutputScorer.score(
        llm=llm,
        task_output_json=sample_output,
        plan_prompt=sample_plan_prompt,
        task_name="swot_analysis",
    )

    print(json.dumps(result.to_dict(include_system_prompt=False, include_user_prompt=False), indent=2))
    print("\nMarkdown:")
    print(result.markdown)

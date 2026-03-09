"""
Pre-planning LLM stage that scores prompt quality and rewrites weak prompts.

Runs before the main pipeline to ensure the initial prompt has enough detail
for PlanExe to produce a high-quality plan.  If the prompt scores below a
configurable threshold, the LLM rewrites it as flowing prose with the missing
dimensions filled in from reasonable defaults.

PROMPT> python -m worker_plan_internal.plan.boost_initial_prompt
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal

from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_executor import LLMExecutor

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 6  # out of 10; prompts scoring below this get rewritten


class PromptDimensionScore(BaseModel):
    dimension: str = Field(description="Name of the dimension being scored.")
    score: int = Field(description="Score from 1 (absent) to 10 (excellent).", ge=1, le=10)
    note: str = Field(description="Brief note explaining the score.")


class PromptQualityAssessment(BaseModel):
    overall_score: int = Field(
        description="Overall prompt quality score from 1 to 10.", ge=1, le=10
    )
    dimensions: List[PromptDimensionScore] = Field(
        description="Scores for each quality dimension."
    )
    missing_elements: List[str] = Field(
        description="List of elements that are absent or underspecified."
    )
    verdict: Literal["sufficient", "needs_boost"] = Field(
        description="Whether the prompt is sufficient or needs boosting."
    )
    boosted_prompt: Optional[str] = Field(
        None,
        description=(
            "Rewritten prompt as flowing prose (~300-800 words) with missing "
            "dimensions filled in. Only present when verdict is needs_boost."
        ),
    )


SYSTEM_PROMPT = """You are a prompt quality assessor for PlanExe, a strategic project-planning system.

Your job is to evaluate the user's project prompt and decide if it has enough detail for a high-quality 20+ section strategic plan. Score the prompt on these dimensions (1-10 each):

1. **Objective clarity** — Is the goal specific and unambiguous?
2. **Scope definition** — Are boundaries clear (what's included/excluded)?
3. **Constraints** — Are budget, timeline, regulatory, or technical constraints mentioned?
4. **Stakeholders** — Are key stakeholders, beneficiaries, or team roles identified?
5. **Success criteria** — Are measurable outcomes or KPIs defined?
6. **Context/background** — Is enough context given to understand the domain?

Rules:
- Score each dimension 1-10. Compute an overall score as the average rounded to nearest integer.
- If overall_score < 6, set verdict to "needs_boost" and provide a boosted_prompt.
- If overall_score >= 6, set verdict to "sufficient" and set boosted_prompt to null.
- The boosted_prompt must be flowing prose (not markdown with headers or bullets), ~300-800 words.
- Preserve the original intent. Add reasonable defaults for missing dimensions.
- Do NOT change the fundamental project goal; only enrich with specifics.
- Do NOT fabricate domain-specific facts the user didn't mention.
"""


@dataclass
class BoostInitialPrompt:
    original_prompt: str
    assessment: PromptQualityAssessment
    metadata: dict

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, prompt: str) -> 'BoostInitialPrompt':
        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT.strip()),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        def execute_function(llm: LLM) -> dict:
            sllm = llm.as_structured_llm(PromptQualityAssessment)
            response = sllm.chat(chat_messages)
            return {"response": response, "metadata": dict(llm.metadata)}

        result = llm_executor.run(execute_function)
        assessment: PromptQualityAssessment = result["response"].raw

        return cls(
            original_prompt=prompt,
            assessment=assessment,
            metadata=result.get("metadata", {}),
        )

    @property
    def effective_prompt(self) -> str:
        """Return the boosted prompt if available, otherwise the original."""
        if self.assessment.verdict == "needs_boost" and self.assessment.boosted_prompt:
            return self.assessment.boosted_prompt
        return self.original_prompt

    def save_raw(self, file_path: str) -> None:
        data = {
            "original_prompt": self.original_prompt,
            "assessment": self.assessment.model_dump(),
            "metadata": self.metadata,
        }
        Path(file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_markdown(self, file_path: str) -> None:
        lines = ["# Prompt Quality Assessment\n"]
        lines.append(f"**Overall Score:** {self.assessment.overall_score}/10\n")
        lines.append(f"**Verdict:** {self.assessment.verdict}\n")

        lines.append("\n## Dimension Scores\n")
        lines.append("| Dimension | Score | Note |")
        lines.append("|-----------|-------|------|")
        for dim in self.assessment.dimensions:
            lines.append(f"| {dim.dimension} | {dim.score}/10 | {dim.note} |")

        if self.assessment.missing_elements:
            lines.append("\n## Missing Elements\n")
            for elem in self.assessment.missing_elements:
                lines.append(f"- {elem}")

        if self.assessment.boosted_prompt:
            lines.append("\n## Boosted Prompt\n")
            lines.append(self.assessment.boosted_prompt)

        Path(file_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    test_prompt = "I want to open a coffee shop."
    llm_models = LLMModelFromName.from_names(["ollama-llama3.1"])
    executor = LLMExecutor(llm_models=llm_models)

    result = BoostInitialPrompt.execute(executor, test_prompt)
    print(f"Score: {result.assessment.overall_score}/10")
    print(f"Verdict: {result.assessment.verdict}")
    print(f"Effective prompt: {result.effective_prompt[:200]}...")

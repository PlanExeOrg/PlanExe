"""
Ask questions about the almost finished plan.

PROMPT> python -m worker_plan_internal.plan.review_plan

IDEA: Executive Summary: Briefly summarizing critical insights from the review.
"""
import os
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole, ChatResponse
from llama_index.core.llms.llm import LLM
from worker_plan_api.speedvsdetail import SpeedVsDetailEnum
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelFromName, PipelineStopRequested

logger = logging.getLogger(__name__)

class DocumentDetails(BaseModel):
    bullet_points: list[str] = Field(
        description="Answers to the questions in bullet points."
    )

REVIEW_PLAN_SYSTEM_PROMPT = """
You are an expert in reviewing plans for projects of all scales. Your goal is to identify the most critical issues that could impact the project's success and provide actionable recommendations to address them.

A good plan is specific, measurable, achievable, relevant, and time-bound (SMART). It addresses potential risks with concrete mitigation strategies, has clear roles and responsibilities, and considers relevant constraints. A strong plan has a detailed financial model, addresses grid connection complexities, and a solid operations and maintenance strategy.

For each question, you MUST provide exactly three concise and distinct bullet points as your answer. Each bullet point must combine all required details for that question into one sentence or clause. **The *very first phrase or a short sentence of each bullet point should act as a concise summary/title for that bullet point, and it MUST BE BOLDED. This first phrase helps the user understand quickly. Make sure the phrase is informative.  Then the rest of the bullet point will contain details, explanation, and actionable recommendations.** Prioritize the *most* critical issues and provide *specific, actionable* recommendations.  For each recommendation, explain *why* it's important and what the potential impact of *not* addressing it would be.**

For example:
- If a question asks for key dependencies along with their likelihood (e.g., Medium, High, Low) and control (internal or external), then each bullet point must include the dependency name, its likelihood, and whether it is controlled internally or externally—all combined into one sentence.  **Indicate which dependency is the *most* critical and why.**
- If a question asks for regulatory requirements, each bullet point must state the requirement and briefly explain how it will be met. Do not include any extra header lines or additional bullet points.

If additional details are needed, merge or summarize them so that your final answer always consists of exactly three bullet points.

Your final output must be a JSON object in the following format:
{
  "bullet_points": [
    "Bullet point 1 (including all required details)",
    "Bullet point 2 (including all required details)",
    "Bullet point 3 (including all required details)"
  ]
}

Do not include any extra bullet points, header lines, or any additional text outside of this JSON structure.

Do not duplicate issues already identified in previous questions of this review.

Do not include any thinking process, reasoning steps, or analysis before the JSON. Output the JSON object directly and immediately.
"""

@dataclass
class ReviewPlanRunResult:
    chat_response: ChatResponse
    metadata: dict

@dataclass
class ReviewPlan:
    """
    Take a look at the proposed plan and provide feedback.
    """
    system_prompt: str
    question_answers_list: list[dict]
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, speed_vs_detail: SpeedVsDetailEnum, document: str) -> 'ReviewPlan':
        """
        Invoke LLM with the data to be reviewed.
        """
        if not isinstance(llm_executor, LLMExecutor):
            raise ValueError("Invalid LLMExecutor instance.")
        if not isinstance(speed_vs_detail, SpeedVsDetailEnum):
            raise ValueError("Invalid SpeedVsDetailEnum instance.")
        if not isinstance(document, str):
            raise ValueError("Invalid document.")

        logger.debug(f"Document:\n{document}")

        system_prompt = REVIEW_PLAN_SYSTEM_PROMPT.strip()
        system_prompt += "\n\nDocument for review:\n"
        system_prompt += document

        title_question_list = [
            ("Critical Issues", "Identify exactly three critical or urgent issues highlighted in the report. For each issue, provide a brief explanation of its quantified significance (e.g., impact in terms of cost, risk, or timing) on the immediate priorities, desired outcomes, or overall effectiveness. Also, explain how these issues might interact with or influence each other, along with a brief, actionable recommendation to address it. Please present your answer in exactly three bullet points."),
            ("Implementation Consequences", "Identify exactly three significant consequences—both positive and negative—that may result from implementing the plan. For each consequence, provide a brief explanation of its *quantified* impact (e.g., in terms of cost, time, or ROI) on the plan's overall feasibility, outcomes, or long-term success. *Also, explain how these consequences might interact with or influence each other*, *along with a brief, actionable recommendation to address it*. Please present your answer in exactly three bullet points."),
            ("Recommended Actions", "Identify exactly three specific actions recommended by the report. For each action, briefly quantify its expected impact (e.g., cost savings, risk reduction, time improvements), clearly state its priority level, and provide a brief, actionable recommendation on how it should be implemented. Present your answer in exactly three bullet points. Actions listed here should complement, extend, or provide additional details to recommendations mentioned previously, rather than repeat them directly."),
            ("Showstopper Risks", "Identify exactly three 'showstopper' risks to the project's success that have not yet been addressed.\nFor each risk:\n- Quantify its potential impact (e.g., in terms of budget increase, timeline delays, ROI reduction).\n- State explicitly its likelihood (High, Medium, Low).\n- Clearly explain how these risks might interact or compound each other.\n- Provide a brief, actionable recommendation to address it.\nPresent your answer in exactly three bullet points, avoiding repetition of previously covered issues or actions. Additionally, for each risk, briefly suggest a contingency measure to be activated if the initial mitigation action proves insufficient."),
            ("Critical Assumptions", "List exactly three critical assumptions underlying the plan that must hold true for it to succeed. For each assumption, quantify the impact if proven incorrect (e.g., cost increase, ROI decrease, timeline delays), explain clearly how each assumption interacts or compounds with previously identified risks or consequences, and provide a brief, actionable recommendation for validating or adjusting each assumption. Present your answer in exactly three bullet points without repeating previously mentioned risks or issues."),
            ("Key Performance Indicators", "Identify exactly three Key Performance Indicators (KPIs) essential for measuring the project's long-term success. For each KPI, quantify specific target values or ranges that indicate success or require corrective action. Clearly explain how each KPI interacts with previously identified risks, assumptions, or recommended actions, and provide a brief, actionable recommendation on how to regularly monitor or achieve each KPI. Present your answer in exactly three bullet points without repeating previously mentioned details."),
            ("Report Objectives", "What exactly are the primary objectives and deliverables of this report? Clearly state the intended audience, the key decisions this report aims to inform, and how Version 2 should differ from Version 1. Present your answer concisely in three bullet points."),
            ("Data Quality Concerns", "Identify exactly three key areas in the current draft where data accuracy or completeness may be uncertain or insufficient. For each area, briefly explain why the data is critical, quantify the potential consequences of relying on incorrect or incomplete data, and recommend a clear approach for validating or improving data quality before Version 2. Present your answer in exactly three bullet points."),
            ("Stakeholder Feedback", "List exactly three important pieces of stakeholder feedback or clarification needed before finalizing Version 2 of the report. For each piece of feedback, explain why it's critical, how unresolved stakeholder concerns might impact the project (quantify potential impact), and provide a brief recommendation for obtaining and incorporating this feedback effectively. Present your answer in exactly three bullet points."),
            ("Changed Assumptions", "Identify exactly three assumptions or conditions from the initial planning stage that might have changed or require re-evaluation since Version 1 was drafted. For each assumption, briefly quantify how changes might affect the report's outcomes (cost, ROI, timeline), explain how these revised assumptions could influence previously identified risks or recommendations, and suggest an actionable approach to reviewing or updating each assumption. Present your answer in exactly three bullet points without repeating previously discussed points."),
            ("Budget Clarifications", "Provide exactly three critical budget clarifications necessary before finalizing Version 2. For each clarification, quantify its impact on the project's financial planning (e.g., cost adjustments, budget reserves, ROI impact), clearly explain why it's needed, and recommend actionable steps to resolve the uncertainty."),
            ("Role Definitions", "Identify exactly three key roles or responsibilities that must be explicitly defined or clarified in Version 2. For each role, briefly explain why clarification is essential, quantify potential impacts (e.g., timeline delays, accountability risks) if roles remain unclear, and recommend actionable steps to ensure clear assignment and accountability."),
            ("Timeline Dependencies", "Identify exactly three timeline dependencies or sequencing concerns that must be clarified before finalizing Version 2. For each dependency, quantify its potential impact if incorrectly sequenced (e.g., timeline delays, increased costs), clearly explain its interaction with previously identified risks or actions, and recommend a concrete action to address the sequencing or dependency concern."),
            ("Financial Strategy", "Identify exactly three long-term financial strategy questions that must be clarified in Version 2. For each, quantify the financial or strategic impact of leaving it unanswered, explain clearly how each interacts with previously identified assumptions or risks, and recommend actionable steps to clarify or address the question."),
            ("Motivation Factors", "Identify exactly three factors essential to maintaining motivation and ensuring consistent progress toward the project's goals. For each factor, quantify potential setbacks if motivation falters (e.g., delays, reduced success rates, increased costs), clearly explain how it interacts with previously identified risks or assumptions, and provide a brief, actionable recommendation on maintaining motivation or addressing motivational barriers."),
            ("Automation Opportunities", "Identify exactly three opportunities within the project where tasks or processes can be automated or streamlined for improved efficiency. For each opportunity, quantify the potential savings (time, resources, or cost), explain how this interacts with previously identified timelines or resource constraints, and recommend a clear, actionable approach for implementing each automation or efficiency improvement."),
        ]
        if speed_vs_detail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            title_question_list = title_question_list[:3]
            logger.info("Running in FAST_BUT_SKIP_DETAILS mode. Omitting some questions.")
        else:
            logger.info("Running in ALL_DETAILS_BUT_SLOW mode. Processing all questions.")

        chat_message_list = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            )
        ]

        question_answers_list = []
        metadata_list = []

        durations = []
        response_byte_counts = []

        for index, title_question in enumerate(title_question_list, start=1):
            title, question = title_question
            logger.debug(f"Question {index} of {len(title_question_list)}: {question}")
            chat_message_list.append(ChatMessage(
                role=MessageRole.USER,
                content=question,
            ))

            def execute_function(llm: LLM) -> ReviewPlanRunResult:
                sllm = llm.as_structured_llm(DocumentDetails)
                chat_response = sllm.chat(chat_message_list)
                metadata = dict(llm.metadata)
                metadata["llm_classname"] = llm.class_name()
                return ReviewPlanRunResult(
                    chat_response=chat_response,
                    metadata=metadata
                )

            start_time = time.perf_counter()
            try:
                review_plan_run_result = llm_executor.run(execute_function)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.debug(f"Question {index} of {len(title_question_list)}. LLM chat interaction failed: {e}")
                logger.error(f"Question {index} of {len(title_question_list)}. LLM chat interaction failed.", exc_info=True)
                raise ValueError("LLM chat interaction failed.") from e
            if not isinstance(review_plan_run_result, ReviewPlanRunResult):
                raise ValueError(f"Expected a ReviewPlanRunResult instance. {review_plan_run_result!r}")

            end_time = time.perf_counter()
            duration = int(ceil(end_time - start_time))
            durations.append(duration)
            response_byte_count = len(review_plan_run_result.chat_response.message.content.encode('utf-8'))
            response_byte_counts.append(response_byte_count)
            logger.info(f"Question {index} of {len(title_question_list)}. LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")

            json_response = review_plan_run_result.chat_response.raw.model_dump()
            logger.debug(json.dumps(json_response, indent=2))

            question_answers_list.append({
                "title": title,
                "question": question,
                "answers": review_plan_run_result.chat_response.raw.bullet_points,
            })

            metadata_list.append(review_plan_run_result.metadata)

            chat_message_list.append(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=review_plan_run_result.chat_response.message.content,
            ))

        response_byte_count_total = sum(response_byte_counts)
        response_byte_count_average = response_byte_count_total / len(title_question_list)
        response_byte_count_max = max(response_byte_counts)
        response_byte_count_min = min(response_byte_counts)
        duration_total = sum(durations)
        duration_average = duration_total / len(title_question_list)
        duration_max = max(durations)
        duration_min = min(durations)

        metadata = {}
        metadata["duration_total"] = duration_total
        metadata["duration_average"] = duration_average
        metadata["duration_max"] = duration_max
        metadata["duration_min"] = duration_min
        metadata["response_byte_count_total"] = response_byte_count_total
        metadata["response_byte_count_average"] = response_byte_count_average
        metadata["response_byte_count_max"] = response_byte_count_max
        metadata["response_byte_count_min"] = response_byte_count_min
        metadata["metadata_list"] = metadata_list
        markdown = cls.convert_to_markdown(question_answers_list)

        result = ReviewPlan(
            system_prompt=system_prompt,
            question_answers_list=question_answers_list,
            metadata=metadata,
            markdown=markdown
        )
        return result
    
    def to_dict(self, include_metadata=True, include_system_prompt=True) -> dict:
        d = {}
        d['question_answers_list'] = self.question_answers_list
        if include_metadata:
            d['metadata'] = self.metadata
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def convert_to_markdown(question_answers_list: list[dict]) -> str:
        """
        Convert the question answers list to markdown.
        """
        rows = []

        for question_index, question_answers in enumerate(question_answers_list, start=1):
            if question_index > 1:
                rows.append("\n")
            title = question_answers.get('title', None)
            if title is None:
                logger.warning("Title is None.")
                continue
            answers = question_answers.get('answers', None)
            if answers is None:
                logger.warning("Answers are None.")
                continue
            rows.append(f"## Review {question_index}: {title}\n")
            for answer_index, answer in enumerate(answers, start=1):
                if answer_index > 1:
                    rows.append("\n")
                rows.append(f"{answer_index}. {answer}")

        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    llm_models_names = [
        # "ollama-llama3.1",
        "openrouter-paid-gemini-2.0-flash-001",
        "openrouter-paid-openai-gpt-4o-mini",
        "openrouter-paid-qwen3-30b-a3b"
    ]
    llm_models = LLMModelFromName.from_names(llm_models_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    path = os.path.join(os.path.dirname(__file__), 'test_data', "deadfish_assumptions.md")
    with open(path, 'r', encoding='utf-8') as f:
        assumptions_markdown = f.read()

    query = (
        f"File 'assumptions.md':\n{assumptions_markdown}"
    )
    print(f"Query:\n{query}\n\n")

    result = ReviewPlan.execute(llm_executor, SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS, query)
    json_response = result.to_dict(include_system_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

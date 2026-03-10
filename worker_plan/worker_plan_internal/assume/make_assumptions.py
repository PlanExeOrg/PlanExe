"""
Analyze a vague description, generate relevant questions for clarification, make reasonable assumptions where necessary.

PROMPT> python -m worker_plan_internal.assume.make_assumptions
"""
import json
import time
from datetime import datetime
import logging
from math import ceil
from typing import Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage, MessageRole
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class QuestionAssumptionItem(BaseModel):
    item_index: int = Field(description="Index in the list")
    question: str = Field(description="Question to clarify and refine the user's description")
    assumptions: str = Field(description="Reasonable assumptions made to fill in the gaps or missing details in the user's description.")
    assessments: str = Field(description="Detailed information about the assessments, including key findings and recommendations. *max 3 assessments*.")

class ExpertDetails(BaseModel):
    question_assumption_list: list[QuestionAssumptionItem] = Field(description="Questions and assumptions")

SYSTEM_PROMPT_1 = """
You are an intelligent **Planning Assistant** designed to help users develop detailed plans from vague or high-level descriptions.

**Your primary tasks are to:**

1. **Identify Potential Questions:**
   - **Analyze the provided description.**
   - **List exactly eight relevant questions** that need to be answered to clarify the requirements, scope, and objectives of the project or task.
   - **Each question must correspond to one of the eight critical areas** to ensure comprehensive and focused planning.

2. **Make Reasonable Assumptions:**
   - **For any information that is unclear, incomplete, or missing from the description, make logical and reasonable assumptions.**
   - **Clearly label these as assumptions** to distinguish them from the user's original input.
   - **Each assumption must directly correspond to its respective question,** providing clear and detailed guidance for planning without unnecessary complexity.
   - **Ensure all assumptions are realistic and feasible** based on industry standards and practical considerations.

3. **Conduct Assessments:**
   - **Perform evaluations** based on the identified questions and assumptions.
   - **Provide insights** into potential risks, feasibility, environmental impact, financial viability, and other relevant factors.
   - **Ensure exactly eight assessments** are provided, each corresponding to one of the eight critical areas of the project.
   - **Each assessment must include:**
     - **Title:** A concise title for the assessment (e.g., Risk Assessment).
     - **Description:** A brief overview of the assessment focus.
     - **Details:** Specific insights, including likelihood, impact, and mitigation strategies.

**Guidelines:**

- **Clarity & Precision:** Ensure that all questions, assumptions, and assessments are clear, relevant, and aimed at uncovering essential details that will aid in planning.

- **Comprehensive Coverage:** Address the following eight critical areas:
  1. **Funding & Budget:** Sources, allocation, and financial planning.
  2. **Timeline & Milestones:** Project phases, deadlines, and key milestones.
  3. **Resources & Personnel:** Required materials, technologies, and team members.
  4. **Governance & Regulations:** Rules, policies, and compliance requirements.
  5. **Safety & Risk Management:** Potential risks, safety measures, and contingency plans.
  6. **Environmental Impact:** Sustainability practices and environmental considerations.
  7. **Stakeholder Involvement:** Key stakeholders, their roles, and communication strategies.
  8. **Operational Systems:** Essential systems for functionality (e.g., power, water, air).

- **Logical Assumptions:** Base all assumptions on common-sense reasoning, industry benchmarks, and any implicit information present in the description. Avoid introducing unrelated or speculative elements.

- **Realism and Feasibility:** Ensure that all assumptions are grounded in realistic scenarios by referencing industry benchmarks, historical data, and practical constraints. Avoid speculative figures unless explicitly justified by the project context.

- **Alignment:** Ensure each assumption is directly tied to its corresponding question, providing a coherent and logical foundation for planning.

- **Neutral Tone:** Maintain an objective and neutral tone, avoiding any bias or subjective opinions.

- **Conciseness:** Keep questions, assumptions, and assessments concise and to the point, ensuring they are easily understandable while still being sufficiently detailed.

- **Strict Item Limit:** Do not exceed eight items in each section. If the content naturally exceeds this limit, prioritize the most critical aspects and omit less essential details.
"""

SYSTEM_PROMPT_2 = """
You are an expert **Planning Assistant** designed to transform vague descriptions into detailed, actionable plans. Your process is rigorous, structured, and ensures comprehensive coverage across all critical project areas.

**Your primary tasks are to perform the following in a strictly ordered sequence:**

1.  **Clarify Requirements with Focused Questions:**
    -   **Analyze the provided description** to identify its core objectives and constraints.
    -   **Generate exactly eight (8) targeted questions** designed to elicit essential details necessary for planning.
    -   **Each question MUST directly address one of the eight (8) critical planning areas** listed below, ensuring no area is overlooked.
    -   **Questions should be concise, specific, and directly related to the provided description.** Avoid overly generic or broad questions.
    -   **Output:** Present each question with an `item_index` (e.g., `item_index: 1`). The `item_index` is solely for output formatting and should *not* be used to reference other parts of your response.

2.  **Formulate Specific and Justifiable Assumptions:**
    -   **For every question posed, formulate a corresponding assumption.** These assumptions should bridge any gaps in the provided description and be directly related to the respective question.
    -   **Each assumption MUST be realistic, feasible, and based on industry benchmarks or common sense.** Justify each assumption briefly, referencing industry standards or practical considerations where applicable.
    -   **Label each assumption as "Assumption:"** to clearly distinguish it from user-provided information.
    -   **Output:** Present each assumption with a matching `item_index` (e.g., `item_index: 1`). The `item_index` is solely for output formatting and should *not* be used to reference other parts of your response.

3.  **Provide Balanced and Actionable Assessments:**
    -   **For every question and assumption**, conduct a comprehensive evaluation, analyzing its implications, including potential benefits, risks, and opportunities.
    -   **Provide exactly eight (8) assessments**, each directly linked to one question and assumption, and covering one of the Critical Planning Areas.
     -   **Each assessment MUST be a single string** containing:
         - A concise `Title:` (e.g., "Financial Feasibility Assessment").
         - A brief `Description:` of the assessment's focus.
         - `Details:` Specific insights into potential risks, impacts, mitigation strategies, potential benefits, and opportunities. Focus on actionable intelligence that can drive planning decisions. Include quantifiable metrics where applicable.
     -   **Output:** Present each assessment with a matching `item_index` (e.g., `item_index: 1`). The `item_index` is solely for output formatting and should *not* be used to reference other parts of your response.

**Critical Planning Areas (MUST be covered by one question, assumption, and assessment each):**

*   Funding & Budget
*   Timeline & Milestones
*   Resources & Personnel
*   Governance & Regulations
*   Safety & Risk Management
*   Environmental Impact
*   Stakeholder Involvement
*   Operational Systems

**Guidelines (Strictly Follow):**

*   **Strict Ordering:** Follow the sequence of tasks (questions, assumptions, assessments) and output the results in the same order.
*   **Strict Item Limit:** Do not exceed eight items in each section. If the content naturally exceeds this limit, prioritize the most critical aspects and omit less essential details.
*   **Direct Correspondence:** Maintain a one-to-one relationship between each question, assumption, and assessment.
*   **Realism and Feasibility:** Ensure assumptions are realistic, justifiable, and based on real-world considerations.
*   **Do not reference any item by index (e.g., "Assumption: 3.2").** The `item_index` is solely for output formatting.
*   **Balanced Insights:** Assessments should provide a balanced perspective, including potential benefits, opportunities, risks, and actionable mitigation strategies.
*   **Neutral Tone:** Maintain an objective, unbiased, and professional tone.
*   **Conciseness:** Be concise and direct. Prioritize the most critical information.
*   **No Exceeding Item Limit:** Strictly adhere to the 8-item limit for each task.
*   **Explicit Labeling:** All assumptions must be explicitly labeled with the prefix "Assumption:".
*   **Quantifiable Metrics:** Include specific numbers, measurements, or metrics in assumptions and assessments whenever possible to enhance precision.
*   **Justifications:** Briefly justify assumptions using common sense, industry standards, or practical considerations.
*   **Example of Assessment Output:**
    ```
    Title: Financial Feasibility Assessment
    Description: Evaluation of the project's financial viability.
    Details: Funding will come from government grants and private investors. The project has a high chance of success.
    ```
"""

SYSTEM_PROMPT_3 = """
You are an expert **Planning Assistant** designed to transform vague descriptions into detailed, actionable plans. Your process is rigorous, structured, and ensures comprehensive coverage across all critical project areas.

**Your primary tasks are to perform the following in a strictly ordered sequence:**

1. **Clarify Requirements with Focused Questions:**
    - **Analyze the provided description** to identify its core objectives and constraints.
    - **Generate exactly eight (8) targeted questions** designed to elicit essential details necessary for planning.
    - **Each question MUST directly address one of the eight (8) Critical Planning Areas** listed below, ensuring no area is overlooked.
    - **Questions should be concise, specific, and directly related to the provided description.** Avoid overly generic or broad questions.
    - **Output:** Present each question with an `item_index` (e.g., `item_index: 1`). The `item_index` is solely for output formatting and should *not* be used to reference other parts of your response.

2. **Formulate Specific and Justifiable Assumptions:**
    - **For every question posed, formulate a corresponding assumption.** These assumptions should bridge any gaps in the provided description and be directly related to the respective question.
    - **Each assumption MUST be realistic, feasible, and based on industry benchmarks or common sense.** Justify each assumption briefly, referencing industry standards or practical considerations where applicable.
    - **Label each assumption as "Assumption:"** to clearly distinguish it from user-provided information.
    - **Output:** Present each assumption with a matching `item_index` (e.g., `item_index: 1`). The `item_index` is solely for output formatting and should *not* be used to reference other parts of your response.

3. **Provide Balanced and Actionable Assessments:**
    - **For every question and assumption**, conduct a comprehensive evaluation, analyzing its implications, including potential benefits, risks, and opportunities.
    - **Provide exactly eight (8) assessments**, each directly linked to one question and assumption, and covering one of the Critical Planning Areas.
    - **Each assessment MUST be a single string** containing:
        - A concise `Title:` (e.g., "Financial Feasibility Assessment").
        - A brief `Description:` of the assessment's focus.
        - `Details:` Specific insights into potential risks, impacts, mitigation strategies, potential benefits, and opportunities. Focus on actionable intelligence that can drive planning decisions. Include quantifiable metrics where applicable.
    - **Output:** Present each assessment with a matching `item_index` (e.g., `item_index: 1`). The `item_index` is solely for output formatting and should *not* be used to reference other parts of your response.

**Critical Planning Areas (MUST be covered by one question, assumption, and assessment each):**

* Funding & Budget
* Timeline & Milestones
* Resources & Personnel
* Governance & Regulations
* Safety & Risk Management
* Environmental Impact
* Stakeholder Involvement
* Operational Systems

**Output Format:**

The output must be a JSON object with two keys:

1. `"question_assumption_list"`: An array of exactly eight objects, each containing:
    - `item_index`: Integer from 1 to 8.
    - `question`: String.
    - `assumptions`: String, starting with "Assumption:".
    - `assessments`: String containing Title, Description, and Details.

2. `"metadata"`: An object containing relevant metadata about the response.

**Example JSON Output:**

{
  "question_assumption_list": [
    {
      "item_index": 1,
      "question": "What is the size of the square and the yellow ball?",
      "assumptions": "Assumption: The square has a side length of 500 pixels. The yellow ball has a diameter of 50 pixels.",
      "assessments": "Title: Collision Detection Assessment\nDescription: Evaluation of collision between the ball and the square.\nDetails: If the ball's center x-coordinate is less than or equal to the square's left edge, or greater than or equal to the square's right edge, the ball will bounce back. Similarly, if the ball's center y-coordinate is less than or equal to the square's top edge, or greater than or equal to the square's bottom edge, the ball will bounce up or down."
    },
    // ... seven more items
  ]
}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_3

@dataclass
class MakeAssumptions:
    system_prompt: Optional[str]
    user_prompt: str
    response: dict
    metadata: dict
    assumptions: list
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'MakeAssumptions':
        """
        Invoke LLM and make assumptions based on the user prompt.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid query.")

        # Obtain the current year as a string, eg. "1984"
        current_year_int = datetime.now().year
        current_year = str(current_year_int)

        # Replace the placeholder in the system prompt with the current year
        system_prompt = SYSTEM_PROMPT.strip()
        system_prompt = system_prompt.replace("CURRENT_YEAR_PLACEHOLDER", current_year)

        chat_message_list = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=user_prompt,
            )
        ]

        sllm = llm.as_structured_llm(ExpertDetails)

        logger.debug("Starting LLM chat interaction.")
        start_time = time.perf_counter()
        try:
            chat_response = sllm.chat(chat_message_list)
        except Exception as e:
            llm_error = LLMChatError(cause=e)
            logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
            logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e
        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))
        response_byte_count = len(chat_response.message.content.encode('utf-8'))
        logger.info(f"LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        expert_details: ExpertDetails = chat_response.raw
        if expert_details is None:
            raise ValueError(
                "Structured LLM returned None for ExpertDetails. "
                "The model likely echoed the schema instead of producing values. "
                "Check model compatibility with structured output."
            )

        # Build assumption list from structured output (no JSON parsing needed)
        assumption_list = []
        for item in expert_details.question_assumption_list:
            assumption_item = {
                "question": item.question,
                "assumptions": item.assumptions,
                "assessments": item.assessments,
            }
            assumption_list.append(assumption_item)

        json_response = expert_details.model_dump()

        markdown = cls.convert_to_markdown(chat_response.raw)

        result = MakeAssumptions(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            assumptions=assumption_list,
            markdown=markdown
        )
        logger.debug("MakeAssumptions instance created successfully.")
        return result    

    def to_dict(self, include_metadata=True, include_system_prompt=True, include_user_prompt=True) -> dict:
        d = self.response.copy()
        if include_metadata:
            d['metadata'] = self.metadata
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        if include_user_prompt:
            d['user_prompt'] = self.user_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=2))

    def save_assumptions(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.assumptions, indent=2))

    @staticmethod
    def convert_to_markdown(expert_details: ExpertDetails) -> str:
        """
        Convert the raw document details to markdown.
        """
        rows = []

        if len(expert_details.question_assumption_list) > 0:
            for index, item in enumerate(expert_details.question_assumption_list, start=1):
                rows.append(f"\n## Question {index} - {item.question}")
                rows.append(f"\n**Assumptions:** {item.assumptions}")
                rows.append(f"\n**Assessments:** {item.assessments}")
        else:
            rows.append("The 'question-assumption-list' is empty. Finding zero questions for a plan is unusual, this is likely a bug. Please report this issue to the developer of PlanExe.")

        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

if __name__ == "__main__":
    import logging
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    plan_prompt = find_plan_prompt("4dc34d55-0d0d-4e9d-92f4-23765f49dd29")
    query = (
        f"{plan_prompt}\n\n"
        "Today's date:\n2025-Jan-26\n\n"
        "Project start ASAP"
    )

    llm = get_llm("ollama-llama3.1")
    # llm = get_llm("deepseek-chat", max_tokens=8192)

    print(f"Query: {query}")
    result = MakeAssumptions.execute(llm, query)

    print("\n\nResponse:")
    print(json.dumps(result.to_dict(include_system_prompt=False, include_user_prompt=False), indent=2))

    print("\n\nAssumptions:")
    print(json.dumps(result.assumptions, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

"""
Review the team.

PROMPT> python -m worker_plan_internal.team.review_team
"""
import os
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class ReviewItem(BaseModel):
    issue: str = Field(
        description="A brief title or name for the omission/improvement."
    )
    explanation: str = Field(
        description="A concise description of why this issue is important."
    )
    recommendation: str = Field(
        description="Specific suggestions on how to address the issue."
    )

class DocumentDetails(BaseModel):
    omissions: list[ReviewItem] = Field(
        description="The most significant omissions."
    )
    potential_improvements: list[ReviewItem] = Field(
        description="Suggestions and recommendations."
    )

REVIEW_TEAM_SYSTEM_PROMPT = """
You are an expert in designing and evaluating team structures for projects of all scales—from personal or trivial endeavors to large, complex initiatives. Your task is to review a team document that includes a project plan, detailed team roles, and sections on omissions and potential improvements.

In your analysis, please:

1. **Review the Team Composition:**
   - Examine the team roles described, including details such as contract types, typical activities, background stories, and resource needs.
   - Consider whether the roles sufficiently cover all aspects of the project given its scope.

2. **Identify Omissions:**
   - Highlight any significant missing roles, support functions, or expertise areas that are critical for the project's success.
   - **Important:** When the project is personal or trivial, avoid suggesting overly formal or business-oriented roles (e.g., Marketing Specialist, Legal Advisor, Technical Support Specialist). Instead, suggest simpler or integrative adjustments suitable for a personal context.

3. **Suggest Potential Improvements:**
   - Recommend actionable changes that enhance the team's overall effectiveness, communication, and clarity.
   - Focus on clarifying responsibilities and reducing overlap.
   - For personal or non-commercial projects, tailor your recommendations to be straightforward and avoid introducing new formal roles that are unnecessary.

4. **Provide Actionable Recommendations:**
   - For each identified omission or improvement, offer specific, practical advice on how to address the issue.
   - Ensure your recommendations are scaled appropriately to the project's nature.

Your output must be a JSON object with two top-level keys: "omissions" and "potential_improvements". Each key should map to an array of objects, where each object contains:
- `"issue"`: A brief title summarizing the omission or improvement.
- `"explanation"`: A concise explanation of why this issue is significant in relation to the project's goals.
- `"recommendation"`: Specific, actionable advice on how to address the issue.

Ensure your JSON output strictly follows this structure without any additional commentary or text.
"""

@dataclass
class ReviewTeam:
    """
    Take a look at the proposed team and provide feedback on potential omissions and improvements.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict

    @classmethod
    def format_query(cls, job_description: str, team_document_markdown: str) -> str:
        if not isinstance(job_description, str):
            raise ValueError("Invalid job_description.")
        if not isinstance(team_document_markdown, str):
            raise ValueError("Invalid team_document_markdown.")

        query = (
            f"Project description:\n{job_description}\n\n"
            f"Document with team members:\n{team_document_markdown}"
        )
        return query

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'ReviewTeam':
        """
        Invoke LLM with the project description and team document to be reviewed.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = REVIEW_TEAM_SYSTEM_PROMPT.strip()

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

        sllm = llm.as_structured_llm(DocumentDetails)
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

        json_response = chat_response.raw.model_dump()

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        result = ReviewTeam(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
        )
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

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm

    llm = get_llm("ollama-llama3.1")

    path = os.path.join(os.path.dirname(__file__), 'test_data', "solarfarm_team_without_review.md")
    with open(path, 'r', encoding='utf-8') as f:
        team_document_markdown = f.read()
    job_description = "Establish a solar farm in Denmark."

    query = ReviewTeam.format_query(job_description, team_document_markdown)
    print(f"Query:\n{query}\n\n")

    review_team = ReviewTeam.execute(llm, query)
    json_response = review_team.to_dict(include_system_prompt=False, include_user_prompt=False)
    print(json.dumps(json_response, indent=2))

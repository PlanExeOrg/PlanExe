"""
Suggest similar past or existing projects that can be used as a reference for the current project.

PROMPT> python -m worker_plan_internal.plan.related_resources
"""
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

class SuggestionItem(BaseModel):
    item_index: int = Field(
        description="Enumeration, starting from 1."
    )
    project_name: str = Field(
        description="The name of the project."
    )
    project_description: str = Field(
        description="A description of the project."
    )
    success_metrics: list[str] = Field(
        description="Indicators of success, challenges encountered, and project outcomes."
    )
    risks_and_challenges_faced: list[str] = Field(
        description="Explain how each project overcame or mitigated these challenges to provide practical guidance."
    )
    where_to_find_more_information: list[str] = Field(
        description="Links to online resources, articles, official documents, or industry reports where more information can be found."
    )
    actionable_steps: list[str] = Field(
        description="Clear instructions on how the user might directly contact key individuals or organizations from those projects, if they desire deeper insights."
    )
    rationale_for_suggestion: str = Field(
        description="Explain why this particular project is suggested."
    )

class DocumentDetails(BaseModel):
    suggestion_list: list[SuggestionItem] = Field(
        description="List of suggestions."
    )
    summary: str = Field(
        description="Providing a high level context."
    )

RELATED_RESOURCES_SYSTEM_PROMPT = """
You are an expert project analyst tasked with recommending highly relevant past or existing projects as references for a user's described project.

Your goal is to always provide at least **three detailed and insightful recommendations**, strictly adhering to the following guidelines:

- **Primary Suggestions (at least 2):**
  - Must be **real and verifiable past or existing projects**—no hypothetical, fictional, or speculative examples.
  - Include exhaustive detail:
    - **Project Name:** Clearly state the official name.
    - **Project Description:** Concise yet comprehensive description of objectives, scale, timeline, industry, location, and outcomes.
    - **Rationale for Suggestion:** Explicitly highlight similarities in technology, objectives, operational processes, geographical, economic, or cultural aspects.
    - **Risks and Challenges Faced:** Explicitly list major challenges and clearly explain how each was overcome or mitigated.
    - **Success Metrics:** Measurable outcomes such as economic impact, production volume, customer satisfaction, timeline adherence, or technology breakthroughs.
    - **Where to Find More Information:** Direct and authoritative links (official websites, reputable publications, scholarly articles).
    - **Actionable Steps:** Clearly specify roles, names, and robust communication channels (emails, LinkedIn, organizational contacts).

- **Secondary Suggestions (optional but encouraged, at least 1):**
  - Must also be real projects but may contain fewer details.
  - Mark explicitly as secondary suggestions.

**Priority for Relevance:**
- Emphasize geographical or cultural proximity first, but clearly justify including geographically distant examples if necessary.
- If geographically or culturally similar projects are limited, explicitly state this in the rationale.

**Important:** Avoid any hypothetical, speculative, or fictional suggestions. Only include real, documented projects.

Your recommendations should collectively provide the user with robust insights, actionable guidance, and practical contacts for successful execution.
"""

@dataclass
class RelatedResources:
    """
    Identify similar past or existing projects that can be used as a reference for the current project.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'RelatedResources':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = RELATED_RESOURCES_SYSTEM_PROMPT.strip()

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

        markdown = cls.convert_to_markdown(chat_response.raw)

        result = RelatedResources(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            markdown=markdown
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

    def save_raw(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def convert_to_markdown(document_details: DocumentDetails) -> str:
        """
        Convert the raw document details to markdown.
        """
        rows = []

        for item_index, suggestion in enumerate(document_details.suggestion_list, start=1):
            rows.append(f"## Suggestion {item_index} - {suggestion.project_name}\n")
            rows.append(suggestion.project_description)

            success_metrics = "\n".join(suggestion.success_metrics)
            rows.append(f"\n### Success Metrics\n\n{success_metrics}")

            risks_and_challenges_faced = "\n".join(suggestion.risks_and_challenges_faced)
            rows.append(f"\n### Risks and Challenges Faced\n\n{risks_and_challenges_faced}")

            where_to_find_more_information = "\n".join(suggestion.where_to_find_more_information)
            rows.append(f"\n### Where to Find More Information\n\n{where_to_find_more_information}")

            actionable_steps = "\n".join(suggestion.actionable_steps)
            rows.append(f"\n### Actionable Steps\n\n{actionable_steps}")

            rows.append(f"\n### Rationale for Suggestion\n\n{suggestion.rationale_for_suggestion}")

        rows.append(f"\n## Summary\n\n{document_details.summary}")
        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    llm = get_llm("ollama-llama3.1")

    plan_prompt = find_plan_prompt("de626417-4871-4acc-899d-2c41fd148807")
    query = (
        f"{plan_prompt}\n\n"
        "Today's date:\n2025-Feb-27\n\n"
        "Project start ASAP"
    )
    print(f"Query: {query}")

    result = RelatedResources.execute(llm, query)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

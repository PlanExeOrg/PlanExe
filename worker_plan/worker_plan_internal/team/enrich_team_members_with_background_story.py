"""
Enrich each team member with a fictional background story and typical job activities.

PROMPT> python -m worker_plan_internal.team.enrich_team_members_with_background_story
"""
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.format_json_for_use_in_query import format_json_for_use_in_query
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class TeamMember(BaseModel):
    """A human with domain knowledge."""
    id: int = Field(
        description="A unique id for the job_category."
    )
    job_background_story_of_employee: str = Field(
        description="Provide a fictional story for this person."
    )
    typical_job_activities: str = Field(
        description="Describe some typical activities in the job."
    )

class TeamDetails(BaseModel):
    team_members: list[TeamMember] = Field(
        description="The experts with domain knowledge about the problem."
    )

ENRICH_TEAM_MEMBERS_SYSTEM_PROMPT = """
For each team member provided, enrich them with a fictional background story and typical job activities.

Write a fictional background story about the person. It must be one paragraph that covers: 
- First name and last name.
- Location.
- What education, experience, and skills does this person have.
- Familiarity with the task.
- Why is this particular person is relevant.

The typical_job_activities describes relevant skills needed for this project.
"""

@dataclass
class EnrichTeamMembersWithBackgroundStory:
    """
    Enrich each team member with a fictional background story and typical job activities.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    team_member_list: list[dict]

    @classmethod
    def format_query(cls, job_description: str, team_member_list: list[dict]) -> str:
        if not isinstance(job_description, str):
            raise ValueError("Invalid job_description.")
        if not isinstance(team_member_list, list):
            raise ValueError("Invalid team_member_list.")

        query = (
            f"Project description:\n{job_description}\n\n"
            f"Here is the list of team members that needs to be enriched:\n{format_json_for_use_in_query(team_member_list)}"
        )
        return query

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str, team_member_list: list[dict]) -> 'EnrichTeamMembersWithBackgroundStory':
        """
        Invoke LLM with each team member.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")
        if not isinstance(team_member_list, list):
            raise ValueError("Invalid team_member_list.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = ENRICH_TEAM_MEMBERS_SYSTEM_PROMPT.strip()

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

        sllm = llm.as_structured_llm(TeamDetails)
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

        team_member_list_enriched = cls.cleanup_enriched_team_members_and_merge_with_team_members(chat_response.raw, team_member_list)

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        result = EnrichTeamMembersWithBackgroundStory(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            team_member_list=team_member_list_enriched,
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

    def cleanup_enriched_team_members_and_merge_with_team_members(team_details: TeamDetails, team_member_list: list[dict]) -> list[dict]:
        result_team_member_list = team_member_list.copy()
        enriched_team_member_list = team_details.team_members
        id_to_enriched_team_member = {item.id: item for item in enriched_team_member_list}
        for team_member_index, team_member in enumerate(result_team_member_list):
            if 'id' not in team_member:
                logger.warning(f"Team member #{team_member_index} does not have an id")
                continue
            id = team_member['id']
            enriched_team_member = id_to_enriched_team_member.get(id)
            if enriched_team_member:
                team_member['typical_job_activities'] = enriched_team_member.typical_job_activities
                team_member['background_story'] = enriched_team_member.job_background_story_of_employee
        return result_team_member_list

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm

    llm = get_llm("ollama-llama3.1")
    # llm = get_llm("deepseek-chat")

    job_description = "Investigate outbreak of a deadly new disease in the jungle."

    team_member_list = [
        {
            "id": 1,
            "category": "Medical Expertise",
            "explanation": "To identify and understand the pathogen, its symptoms, transmission methods, and potential treatments or vaccines."
        },
        {
            "id": 2,
            "category": "Epidemiologist",
            "explanation": "To track the spread of the disease, analyze patterns, and develop strategies to contain it."
        },
        {
            "id": 3,
            "category": "Field Research Specialist",
            "explanation": "To conduct on-the-ground investigations in challenging jungle environments, collect samples, and interact with affected communities."
        },
        {
            "id": 4,
            "category": "Logistics Coordinator",
            "explanation": "To manage the supply chain for equipment, medical supplies, and personnel transport to remote locations."
        }
    ]

    query = EnrichTeamMembersWithBackgroundStory.format_query(job_description, team_member_list)
    print(f"Query:\n{query}\n\n")

    enrich_team_members_with_background_story = EnrichTeamMembersWithBackgroundStory.execute(llm, query, team_member_list)
    json_response = enrich_team_members_with_background_story.to_dict(include_system_prompt=False, include_user_prompt=False)
    print(json.dumps(json_response, indent=2))

    print("\n\nTeam members with extra details:")
    enriched_json = enrich_team_members_with_background_story.team_member_list
    print(json.dumps(enriched_json, indent=2))

"""
Enrich the team members with what kind of contract type they have.

PROMPT> python -m worker_plan_internal.team.enrich_team_members_with_contract_type
"""
import json
import time
import logging
from enum import Enum
from math import ceil
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.format_json_for_use_in_query import format_json_for_use_in_query
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class ContractType(str, Enum):
    # Full-Time Employee. The individual is a permanent member of the organization.
    full_time_employee = 'full_time_employee'
    # Part-Time Employee. Similar to a full-time employee, but they work fewer hours per week.
    part_time_employee = 'part_time_employee'
    # Independent Contractor / Consultant. A self-employed individual or business that provides services to your company on a contractual basis.
    independent_contractor = 'independent_contractor'
    # Temporary Employee through an Agency. The individual is employed by a staffing agency, who then assigns them to work at your company for a set period.
    agency_temp = 'agency_temp'
    # Other. When the contract type does not fit into the above categories.
    other = 'other'

class TeamMember(BaseModel):
    """A human with domain knowledge."""
    id: int = Field(
        description="A unique id for the job_category."
    )
    contract_type: Literal["full_time_employee", "part_time_employee", "independent_contractor", "agency_temp", "other"] = Field(
        description="The legal and financial agreement."
    )
    justification: str = Field(
        description="Brief explanation for why that contract type was chosen. Helps justify decisions and allows for easy review."
    )

class DocumentDetails(BaseModel):
    team_members: list[TeamMember] = Field(
        description="The experts with domain knowledge about the problem."
    )

ENRICH_TEAM_MEMBERS_CONTRACT_TYPE_SYSTEM_PROMPT = """
You are an expert at determining what contract type are needed for different job roles given a project description.

"Contract Type" refers to the legal and financial agreement you have with each individual working on the project. 
It dictates their employment status, compensation, benefits, and overall relationship with your organization (or the project). 

The "contract_type" for each team member is crucial for the following reasons:
- Drives Cost Calculations: The type of employment agreement dictates a huge portion of project labor costs. Whether you're paying salary + benefits, or a fixed project fee, this is foundational information.
- Impacts Availability and Control: The contract type determines how much control you have over the person and how readily available they will be.
- Informs Resource Planning: It influences long-term versus short-term resource commitments.

Allowed values: "full_time_employee", "part_time_employee", "independent_contractor", "agency_temp"

For each team member provided, identify the contract_type considering the given project description. Provide concise but descriptive answers.
"""

@dataclass
class EnrichTeamMembersWithContractType:
    """
    Enrich each team member with more info.
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
    def execute(cls, llm: LLM, user_prompt: str, team_member_list: list[dict]) -> 'EnrichTeamMembersWithContractType':
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

        system_prompt = ENRICH_TEAM_MEMBERS_CONTRACT_TYPE_SYSTEM_PROMPT.strip()

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

        team_member_list_enriched = cls.cleanup_and_merge_with_team_members(chat_response.raw, team_member_list)

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        result = EnrichTeamMembersWithContractType(
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

    def cleanup_and_merge_with_team_members(document_details: DocumentDetails, team_member_list: list[dict]) -> list:
        result_team_member_list = team_member_list.copy()
        enriched_team_member_list = document_details.team_members
        id_to_enriched_team_member = {item.id: item for item in enriched_team_member_list}
        for team_member_index, team_member in enumerate(result_team_member_list):
            if 'id' not in team_member:
                logger.warning(f"Team member #{team_member_index} does not have an id")
                continue
            id = team_member['id']
            enriched_team_member = id_to_enriched_team_member.get(id)
            if enriched_team_member:
                team_member['contract_type'] = enriched_team_member.contract_type
                team_member['contract_type_justification'] = enriched_team_member.justification
        return result_team_member_list

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm

    llm = get_llm("ollama-llama3.1")
    # llm = get_llm("deepseek-chat")

    job_description = "Establish a new police station in a high crime area."

    team_member_list = [
        {
            "id": 1,
            "category": "Law Enforcement",
            "explanation": "Police officers and detectives are essential for patrolling, investigation, and maintaining public safety."
        },
        {
            "id": 2,
            "category": "Administration",
            "explanation": "Administrative staff manage paperwork, scheduling, and coordination of police activities."
        },
        {
            "id": 3,
            "category": "Forensics",
            "explanation": "Forensic experts analyze crime scene evidence to support investigations."
        },
        {
            "id": 4,
            "category": "Community Relations",
            "explanation": "Officers or liaisons engage with the community to build trust and cooperation."
        }
    ]

    query = EnrichTeamMembersWithContractType.format_query(job_description, team_member_list)
    print(f"Query:\n{query}\n\n")

    enrich_team_members_with_environment_info = EnrichTeamMembersWithContractType.execute(llm, query, team_member_list)
    json_response = enrich_team_members_with_environment_info.to_dict(include_system_prompt=False, include_user_prompt=False)
    print(json.dumps(json_response, indent=2))

    print("\n\nTeam members:")
    json_team = enrich_team_members_with_environment_info.team_member_list
    print(json.dumps(json_team, indent=2))

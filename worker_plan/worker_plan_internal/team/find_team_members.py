"""
From a project description, find a team for solving the job.

PROMPT> python -m worker_plan_internal.team.find_team_members
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

class TeamMember(BaseModel):
    job_category_title: str = Field(
        description="Human readable title"
    )
    short_explanation: str = Field(
        description="Why that category of expert is relevant to solving the task."
    )
    people_needed: str = Field(
        description="Number of people needed."
    )
    consequences_of_not_having_this_role: str = Field(
        description="Consequences of not having this role."
    )

class DocumentDetails(BaseModel):
    brainstorm_of_needed_team_members: list[TeamMember] = Field(
        description="What experts may be needed with domain knowledge about the problem."
    )

FIND_TEAM_MEMBERS_SYSTEM_PROMPT = """
You are a versatile project planning assistant and team architect. Your goal is to analyze the user's project description and decompose it into a comprehensive plan with a focus on human roles and resource allocation—**do not generate any code or technical implementation details.**

If the project description involves programming tasks or includes requests for code, treat it as a planning challenge. Instead of writing a script or providing code, break down the project into essential phases and identify the key human roles needed to successfully complete the project.

Based on the user's project description, brainstorm a team of potential human support roles that cover all crucial aspects of the project, including planning & preparation, execution, monitoring & adjustment, and maintenance & sustainability.

**Output Requirements:**

1. **Team Size:**  
   Your output **must include exactly 8 candidate roles**.  
   - If your initial analysis identifies fewer than 8 distinct roles, create additional meaningful roles to reach exactly 8.  
   - If your analysis results in more than 8 roles, consolidate or combine roles so that the final output contains exactly 8 candidates.

2. **Role Titles:**  
   Provide a clear and concise `job_category_title` that accurately describes the role's primary contribution.

3. **Role Explanations:**  
   Briefly explain each role’s purpose, key responsibilities, and how it contributes actively throughout the project.

4. **Consequences:**  
   For each role, note potential risks or consequences of omitting that role.

5. **People Count / Resource Level:** 
   Use the `people_needed` field to indicate the number of people required for each role. **Do not simply default to "1" for every role.** Instead, evaluate the complexity and workload of the role relative to the project's scale:
   - **Single Resource:** If one person is clearly sufficient, use "1".
   - **Fixed Level:** If the role consistently requires a specific number of people (e.g., "2" or "3"), use that fixed number.
   - **Variable Level:** If the required support may vary based on factors like project scale, workload, or budget, specify a range. For example, instead of "1", you might write "min 1, max 3, depending on project scale and workload." Be sure to justify why the role may require more than one person.

6. **Project Phases / Support Stages:**  
   Ensure the roles collectively address the following phases:
    - **Planning & Preparation**
    - **Execution**
    - **Monitoring & Adjustment**
    - **Maintenance & Sustainability**

**Essential Considerations for EVERY Role:**

- **Specific Expertise**
- **Key Responsibilities**
- **Direct Impact (if applicable)**
- **Project Dependencies**
- **Relevant Skills**
- **Role Priority**

**Important:** 
- Do not provide any code or implementation details—even if the prompt is programming-related. Focus solely on planning, decomposing the work, and identifying the essential human roles.
- **For personal, trivial, or non-commercial projects, avoid suggesting overly formal or business-oriented roles (e.g., Marketing Specialist, Legal Advisor, Technical Support Specialist) unless they are absolutely necessary.** In such cases, prefer roles that can be integrated or scaled down to suit the project's nature.
"""

@dataclass
class FindTeamMembers:
    """
    From a project description, find a team for solving the job.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    team_member_list: list[dict]

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'FindTeamMembers':
        """
        Invoke LLM to find a team.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = FIND_TEAM_MEMBERS_SYSTEM_PROMPT.strip()

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

        team_member_list = cls.cleanup_team_members_and_assign_id(chat_response.raw)

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        result = FindTeamMembers(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            team_member_list=team_member_list,
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

    def cleanup_team_members_and_assign_id(document_details: DocumentDetails) -> list:
        result_list = []
        team_members = document_details.brainstorm_of_needed_team_members
        for i, team_member in enumerate(team_members, start=1):
            item = {
                "id": i,
                "category": team_member.job_category_title,
                "explanation": team_member.short_explanation,
                "consequences": team_member.consequences_of_not_having_this_role,
                "count": team_member.people_needed,
            }
            result_list.append(item)
        return result_list
    
if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    llm = get_llm("ollama-llama3.1")
    plan_prompt = find_plan_prompt("4dc34d55-0d0d-4e9d-92f4-23765f49dd29")

    print(f"Query:\n{plan_prompt}\n\n")
    result = FindTeamMembers.execute(llm, plan_prompt)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print(json.dumps(json_response, indent=2))

    print("\n\nTeam members:")
    json_team = result.team_member_list
    print(json.dumps(json_team, indent=2))

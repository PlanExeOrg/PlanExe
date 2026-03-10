"""
Internal Governance Bodies: 

Gemini 2.5 has this opinion about llama3.1's response.
"Llama 3.1 is not suitable for this task."

PROMPT> python -m worker_plan_internal.governance.governance_phase2_bodies
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

class InternalGovernanceBody(BaseModel):
    name: str = Field(description="Name of the internal governance body.")
    rationale_for_inclusion: str = Field(
        description="Brief justification explaining *why* this specific type of internal governance body (e.g., Steering Committee, PMO, Ethics Committee) is necessary or appropriate for *this particular project*, based on its description, scale, or key challenges."
    )
    responsibilities: list[str] = Field(description="Key tasks or responsibilities of this internal body.")
    initial_setup_actions: list[str] = Field(description="Key initial actions this body needs to take upon formation (e.g., 'Finalize Terms of Reference', 'Elect Chair', 'Set meeting schedule').")
    membership: list[str] = Field(description="Roles or titles of individuals *within the project/organization* forming this internal body.")
    decision_rights: str = Field(description="Type and scope of decisions this internal body is empowered to make.")
    decision_mechanism: str = Field(description="How decisions are typically made within this internal body. Specify tie-breaker if applicable.")
    meeting_cadence: str = Field(description="How often this internal body meets.")
    typical_agenda_items: list[str] = Field(description="Example recurring items for this internal body's meetings.")
    escalation_path: str = Field(description="Where or to which *other internal body or senior project/organization role* issues exceeding authority are escalated.")

class DocumentDetails(BaseModel):
    internal_governance_bodies: list[InternalGovernanceBody] = Field(
        description="List of all internal governance bodies with roles, responsibilities, and membership."
    )

GOVERNANCE_PHASE2_BODIES_SYSTEM_PROMPT = """
You are an expert in project governance and organizational design. Your task is to analyze the provided project description and propose a suitable and robust structure of **distinct INTERNAL project governance bodies** required to effectively oversee and manage the project internally.

**Consider these distinct governance body types and select/adapt those most appropriate:**

* **Strategic Oversight:** (e.g., Project Steering Committee, Project Board) – Provides high-level strategic direction, approves significant project milestones, budgets above a clearly defined threshold, and strategic risk oversight.
* **Operational Management:** (e.g., Project Management Office, Core Project Team) – Manages day-to-day execution, operational risk management, and decisions below strategic thresholds.
* **Specialized Advisory/Assurance:** (e.g., Technical Advisory Group, Ethics & Compliance Committee, Stakeholder Engagement Group) – Provides specialized input or assurance on key project aspects (technical, ethical, compliance, or stakeholder perspectives).

**Ensure clear logical separation of roles:**
- Clearly differentiate strategic oversight from operational management.
- Avoid overlapping memberships that could lead to conflicts of interest.
- Include independent or external members in oversight or specialized assurance bodies to maintain impartial governance.

**Explicitly define governance details:**
- Set clear financial thresholds or decision criteria distinguishing strategic from operational decisions.
- Clearly outline escalation paths and explicit conflict resolution mechanisms for when consensus or majority votes cannot be reached.
- Explicitly integrate risk management across all governance bodies, detailing how risks inform strategic and operational decisions.
- Assign explicit responsibility for comprehensive compliance oversight, including GDPR, ethical standards, and relevant regulations, to a dedicated governance body.

Define a list named `internal_governance_bodies`, each element strictly adhering to the `InternalGovernanceBody` schema:

1. **`name`:** Name of the internal governance body.
2. **`rationale_for_inclusion`:** Explicit justification based on project complexity, scale, and risks.
3. **`responsibilities`:** Clearly defined tasks and oversight roles.
4. **`initial_setup_actions`:** Essential initial steps upon formation.
5. **`membership`:** Clearly specified internal roles, explicitly identifying independent or external roles.
6. **`decision_rights`:** Defined scope and threshold for decision-making authority.
7. **`decision_mechanism`:** Explicit decision-making process with defined tie-breakers.
8. **`meeting_cadence`:** Clearly defined meeting frequency appropriate to responsibilities.
9. **`typical_agenda_items`:** Clearly articulated recurring agenda items relevant to the governance body's role.
10. **`escalation_path`:** Clearly defined next internal body or senior role for unresolved issues, specifying criteria for escalation.

Your output must strictly adhere to the provided Pydantic schema `DocumentDetails`, containing *only* the `internal_governance_bodies` list following the `InternalGovernanceBody` schema.
"""

@dataclass
class GovernancePhase2Bodies:
    """
    Take a look at the almost finished plan and propose a governance structure, focus only on the governance bodies part.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'GovernancePhase2Bodies':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = GOVERNANCE_PHASE2_BODIES_SYSTEM_PROMPT.strip()

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

        result = GovernancePhase2Bodies(
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
        
        for i, body in enumerate(document_details.internal_governance_bodies, 1):
            rows.append(f"### {i}. {body.name}")
            rows.append(f"\n**Rationale for Inclusion:** {body.rationale_for_inclusion}")
            rows.append("\n**Responsibilities:**\n")
            for resp in body.responsibilities:
                rows.append(f"- {resp}")
                
            rows.append("\n**Initial Setup Actions:**\n")
            for action in body.initial_setup_actions:
                rows.append(f"- {action}")
                
            rows.append("\n**Membership:**\n")
            for member in body.membership:
                rows.append(f"- {member}")
                
            rows.append(f"\n**Decision Rights:** {body.decision_rights}")
            rows.append(f"\n**Decision Mechanism:** {body.decision_mechanism}")
            rows.append(f"\n**Meeting Cadence:** {body.meeting_cadence}")
            
            rows.append("\n**Typical Agenda Items:**\n")
            for item in body.typical_agenda_items:
                rows.append(f"- {item}")
                
            rows.append(f"\n**Escalation Path:** {body.escalation_path}")

        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    llm = get_llm("ollama-llama3.1")

    plan_prompt = find_plan_prompt("4060d2de-8fcc-4f8f-be0c-fdae95c7ab4f")
    query = (
        f"{plan_prompt}\n\n"
        "Today's date:\n2025-Mar-23\n\n"
        "Project start ASAP"
    )
    print(f"Query: {query}")

    result = GovernancePhase2Bodies.execute(llm, query)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

"""
Decision Escalation Matrix

To define how specific types of important issues or decisions are escalated beyond the body or role 
where they initially arise or cannot be resolved. This provides clarity and prevents bottlenecks.

Gemini 2.5 doesn't like the response from Llama 3.1.
"This confirms definitively that Llama 3.1 is unsuitable for generating a Decision Escalation Matrix based on the provided inputs and prompting strategies attempted. 
It demonstrates a fundamental inability to distinguish between project setup activities and issue escalation scenarios, regardless of how the prompt is structured or simplified."

PROMPT> python -m worker_plan_internal.governance.governance_phase4_decision_escalation_matrix
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

class DecisionEscalationItem(BaseModel):
    issue_type: str = Field(
        description="Type of issue (e.g., budget overruns, ethical concerns, strategic pivot)."
    )
    escalation_level: str = Field(
        description="Indicates the governance body or role to which the issue is escalated (e.g., Steering Committee)."
    )
    approval_process: str = Field(
        description="Outlines how an issue is approved or resolved once escalated (e.g., majority vote)."
    )
    rationale: str = Field(
        description="Explains *why* this issue triggers escalation—i.e. potential impacts or risks that justify escalating."
    )
    negative_consequences: str = Field(
        description="Describes the potential adverse outcomes or risks if this issue remains unresolved or is not escalated in a timely manner."
    )

class DocumentDetails(BaseModel):
    decision_escalation_matrix: list[DecisionEscalationItem] = Field(
        description="Clear escalation paths for various issues."
    )

GOVERNANCE_PHASE4_DECISION_ESCALATION_MATRIX_SYSTEM_PROMPT = """
You are an expert in project governance. Your task is to create a Decision Escalation Matrix. **This matrix describes what happens when specific PROBLEMS occur or when DECISIONS exceed the authority of a lower level.**

**You will be provided with:**
1.  The overall project description.
2.  A list of defined `internal_governance_bodies` (e.g., PMO, Project Steering Committee, Executive Sponsor) showing their typical hierarchy.

**Your goal is to generate the `decision_escalation_matrix` list.** Identify **at least 5 different SCENARIOS** where a problem or decision needs to move to a higher level.

**Think about triggers:** What specific event causes the escalation?
    *   **Trigger Example 1:** A budget request is *too large* for the PMO to approve alone.
    *   **Trigger Example 2:** A *critical risk* happens that the PMO cannot handle with existing resources.
    *   **Trigger Example 3:** The PMO *cannot agree* on a key operational decision.
    *   **Trigger Example 4:** A *major change* to the project scope is proposed.
    *   **Trigger Example 5:** An *ethical violation* is reported.

**For each scenario (`DecisionEscalationItem`), fill in these details:**
1.  **`issue_type`:** Describe the **specific problem or decision trigger** requiring escalation. Use the examples above as a guide. (e.g., 'Budget Request Exceeding PMO Authority', 'Critical Risk Materialization', 'PMO Deadlock on Vendor Selection', 'Proposed Major Scope Change', 'Reported Ethical Concern'). **DO NOT list routine tasks like 'Vendor Selection' or setup steps.**
2.  **`escalation_level`:** State the **specific name** of the *next higher* `InternalGovernanceBody` or senior role (from the provided structure) that handles this escalated issue.
3.  **`approval_process`:** Briefly describe how the decision is likely made *at that higher level* (e.g., 'Steering Committee Vote', 'Sponsor Approval', 'Ethics Committee Investigation & Recommendation').
4.  **`rationale`:** Briefly explain *why* this **trigger** requires escalation (e.g., 'Exceeds financial limit', 'Strategic impact', 'Needs independent review', 'Requires higher authority').
5.  **`negative_consequences`:** Briefly state the risk if the **escalated issue** is not resolved properly (e.g., 'Budget overrun', 'Project failure', 'Legal penalty', 'Reputational damage').

Focus *only* on generating the `decision_escalation_matrix` list based on the provided project description and governance bodies. Ensure the scenarios represent **escalations due to exceeding limits, disagreements, or critical events.**

Ensure your output strictly adheres to the provided Pydantic schema `DocumentDetails` containing *only* the `decision_escalation_matrix` list, where each element follows the `DecisionEscalationItem` schema.
"""

@dataclass
class GovernancePhase4DecisionEscalationMatrix:
    """
    Take a look at the almost finished plan and propose a governance structure, focus only on the decision escalation matrix.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'GovernancePhase4DecisionEscalationMatrix':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = GOVERNANCE_PHASE4_DECISION_ESCALATION_MATRIX_SYSTEM_PROMPT.strip()

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

        result = GovernancePhase4DecisionEscalationMatrix(
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
        
        for i, item in enumerate(document_details.decision_escalation_matrix, 1):
            if i > 1:
                rows.append("")
            rows.append(f"**{item.issue_type}**")
            rows.append(f"Escalation Level: {item.escalation_level}")
            rows.append(f"Approval Process: {item.approval_process}")
            rows.append(f"Rationale: {item.rationale}")
            rows.append(f"Negative Consequences: {item.negative_consequences}")

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

    result = GovernancePhase4DecisionEscalationMatrix.execute(llm, query)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

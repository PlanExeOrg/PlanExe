"""
Governance extra fields

PROMPT> python -m worker_plan_internal.governance.governance_phase6_extra
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

class DocumentDetails(BaseModel):
    governance_validation_checks: list[str] = Field(
        description="A rigorous check of the generated governance components for completeness, consistency, and potential gaps based on the inputs and standard practices."
    )
    tough_questions: list[str] = Field(
        description="Representative questions leadership should regularly ask (e.g., 'Are we on budget?')."
    )
    summary: str = Field(
        description="High-level context or summary of governance approach."
    )
 
GOVERNANCE_PHASE6_EXTRA_SYSTEM_PROMPT = """
You are an expert in project governance quality assurance, risk management, and strategic oversight. Your task is to **critically validate** the previously generated components of the project governance framework, identify **specific areas needing further detail or clarification**, generate insightful accountability questions, and provide an overall summary.

**You will be provided with (as context):**
1.  The overall project description (including objectives, critical factors, risks).
2.  The defined `internal_governance_bodies` (Stage 2).
3.  The `governance_implementation_plan` (Stage 3).
4.  The `decision_escalation_matrix` (Stage 4).
5.  The `monitoring_progress` plan (Stage 5).
6.  (Potentially) `AuditDetails` (Stage 1).

**Based on reviewing and VALIDATING ALL the provided governance context, your goal is to generate:**

1.  **`governance_validation_checks`:**
    *   Perform a **rigorous consistency and completeness check**.
    *   **Point 1: Completeness Confirmation:** State clearly if all core requested components appear generated.
    *   **Point 2: Internal Consistency Check:** Verify logical alignment between stages (e.g., Implementation Plan uses correct bodies, Escalation Matrix follows hierarchy, Monitoring roles exist). Briefly confirm consistency or note specific discrepancies found.
    *   **Point 3: Potential Gaps / Areas for Enhancement:** Critically review the *details* within the generated components. **Identify specific, nuanced gaps or areas where more detail, process definition, or clarification would significantly strengthen the framework.** Examples of areas to scrutinize:
        *   *Clarity of roles:* Are responsibilities and expected contributions of **all members, especially advisors or independent roles,** clearly defined? Is the role/authority of the ultimate **Project Sponsor** clear within the structure?
        *   *Process Depth:* Are key operational or ethical processes (like **conflict of interest management, whistleblower investigation, change control, stakeholder communication protocols**) sufficiently detailed or just mentioned at a high level?
        *   *Thresholds/Delegation:* Is delegated authority clear and practical? Are there opportunities for **more granular delegation** below the main committee levels (e.g., for specific coordinators) with defined parameters?
        *   *Integration:* Are related components well-integrated (e.g., audit procedures linked to monitoring or E&C responsibilities)? Is the flow of information between committees clear?
        *   *Specificity:* Are any parts too vague (e.g., **escalation path endpoints like 'Senior Management'**, adaptation triggers, membership criteria)?
        **Aim for at least 3-5 specific points identifying areas needing more detail or clarification.**

2.  **`tough_questions`:**
    *   Generate **at least 7 critical, probing questions** demanding specific data, evidence, forecasts, contingency plans, or verification of processes. Frame them to challenge assumptions and ensure proactive management. Link questions directly to the project's critical factors, risks, and compliance needs.
    *   *(Provide specific examples here if desired, e.g., 'What is the current probability-weighted forecast for [Critical Target]?', 'Show evidence of [Compliance Action] verification.', etc.)*

3.  **`summary`:**
    *   Write a brief, high-level concluding paragraph summarizing the overall governance approach and its key strengths or focus areas.

Focus *only* on generating `governance_validation_checks`, `tough_questions`, and `summary`. Base your validation and questions on the governance details provided.

Ensure your output strictly adheres to the provided Pydantic schema `DocumentDetails` containing *only* `governance_validation_checks`, `tough_questions`, and `summary`.
"""

@dataclass
class GovernancePhase6Extra:
    """
    Take a look at the almost finished plan and propose a governance structure, focus only on extra fields.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'GovernancePhase6Extra':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = GOVERNANCE_PHASE6_EXTRA_SYSTEM_PROMPT.strip()

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

        result = GovernancePhase6Extra(
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
                
        rows.append("## Governance Validation Checks")
        for i, item in enumerate(document_details.governance_validation_checks, 1):
            if i == 1:
                rows.append("")
            rows.append(f"{i}. {item}")

        rows.append("\n## Tough Questions")
        for i, question in enumerate(document_details.tough_questions, 1):
            if i == 1:
                rows.append("")
            rows.append(f"{i}. {question}")
        
        rows.append(f"\n## Summary\n\n{document_details.summary}")
        
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

    result = GovernancePhase6Extra.execute(llm, query)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

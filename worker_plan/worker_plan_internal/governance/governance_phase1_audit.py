"""
Governance - Audit Framework: 
- Corruption.
- Misallocation Risks. 
- Audit procedures.
- Corruption countermeasures.

PROMPT> python -m worker_plan_internal.governance.governance_phase1_audit
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
    corruption_list: list[str] = Field(
        description="Corruption risks in this project: bribery, nepotism, etc."
    )
    misallocation_list: list[str] = Field(
        description="Ways resources can be misallocated: budget misuse, double spending, etc."
    )
    audit_procedures: list[str] = Field(
        description="Procedures for conducting regular and ad-hoc audits (e.g., quarterly external audits)."
    )
    transparency_measures: list[str] = Field(
        description="Mechanisms to ensure transparency (e.g., public dashboards, published meeting minutes)."
    )

GOVERNANCE_PHASE1_AUDIT_SYSTEM_PROMPT = """
You are an expert in project governance, risk management, and auditing. Your task is to analyze the provided project description and identify potential audit-related risks and associated control measures relevant to that specific project.

Based *only* on the **project description provided by the user**, generate the following details:

1.  **Corruption Risks:** Identify specific ways corruption (like bribery, nepotism, conflicts of interest, kickbacks, information misuse, trading favors) could manifest **within the context of the described project**. Consider potential interactions with suppliers, contractors, regulators, stakeholders, or internal personnel. Aim to list **at least 5 distinct and plausible risks relevant to the project type and scale**. List these as `corruption_list`.
2.  **Misallocation Risks:** Identify specific ways resources (budget, time, materials, personnel effort) could be misallocated or misused **in this specific project** (like budget misuse for personal gain, double spending, inefficient allocation, unauthorized use of assets, poor record keeping, misreporting progress or results). Aim to list **at least 5 distinct and plausible risks relevant to the project type and scale**. List these as `misallocation_list`.
3.  **Audit Procedures:** Recommend specific, practical procedures for auditing project activities, finances, and compliance **relevant to the described project**. Include frequency and potential responsibility where appropriate (e.g., periodic internal reviews, post-project external audit, contract review thresholds, expense workflows, compliance checks relevant to the project domain). Aim to list **at least 5 distinct and practical procedures**. List these as `audit_procedures`.
4.  **Transparency Measures:** Recommend specific mechanisms to ensure transparency in project operations, finances, and decision-making, fostering accountability **appropriate for the project's context**. (e.g., progress/budget dashboards [specify type if possible], published key meeting minutes [specify which governing body if known/applicable], whistleblower mechanisms, public access to relevant policies/reports, documented selection criteria for major decisions/vendors). Aim to list **at least 5 distinct and practical measures**. List these as `transparency_measures`.

Focus *only* on these four aspects. Provide detailed and context-specific examples inferred directly from the **user's project description**. Do not generate information about governance bodies, implementation plans, decision-making, or other topics beyond these four audit/control points. Do not invent details not supported by the input project description.

Ensure your output strictly adheres to the provided Pydantic schema `DocumentDetails` containing only `corruption_list`, `misallocation_list`, `audit_procedures`, and `transparency_measures`.
"""

@dataclass
class GovernancePhase1Audit:
    """
    Take a look at the almost finished plan and propose a governance structure, focus only on the audit part.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'GovernancePhase1Audit':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = GOVERNANCE_PHASE1_AUDIT_SYSTEM_PROMPT.strip()

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

        result = GovernancePhase1Audit(
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
        
        # Add audit details section
        rows.append("\n## Audit - Corruption Risks\n")
        for item in document_details.corruption_list:
            rows.append(f"- {item}")
            
        rows.append("\n## Audit - Misallocation Risks\n")
        for item in document_details.misallocation_list:
            rows.append(f"- {item}")
            
        rows.append("\n## Audit - Procedures\n")
        for item in document_details.audit_procedures:
            rows.append(f"- {item}")
            
        rows.append("\n## Audit - Transparency Measures\n")
        for item in document_details.transparency_measures:
            rows.append(f"- {item}")
                
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

    result = GovernancePhase1Audit.execute(llm, query)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

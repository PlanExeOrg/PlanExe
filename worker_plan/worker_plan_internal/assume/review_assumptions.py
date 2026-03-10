"""
Review the assumptions. Are they too low/high? Are they reasonable? Are there any missing assumptions?

PROMPT> python -m worker_plan_internal.assume.review_assumptions
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
        description="A brief title or name."
    )
    explanation: str = Field(
        description="A concise description of why this issue is important."
    )
    recommendation: str = Field(
        description="Specific suggestions on how to address the issue."
    )
    sensitivity: str = Field(
        default="",
        description="Optional: Provide any sensitivity analysis insights related to this issue."
    )

class DocumentDetails(BaseModel):
    expert_domain: str = Field(
        description="The domain of the expert reviewer."
    )
    domain_specific_considerations: list[str] = Field(
        description="Key factors and areas of focus relevant to the specific project domain, which this review should prioritize."
    )
    issues: list[ReviewItem] = Field(
        description="The most significant issues."
    )
    conclusion: str = Field(
        description="Summary of the most important issues."
    )

REVIEW_ASSUMPTIONS_SYSTEM_PROMPT = """
You are a world-class planning expert specializing in the success of projects. Your task is to critically review the provided assumptions and identify potential weaknesses, omissions, or unrealistic elements that could significantly impact project success. Your analysis should be tailored to the project’s scale and context, while considering standard project management best practices. Be creative and innovative in your analysis, considering risks and opportunities that might be overlooked by others.

**Crucial Focus: Missing Assumptions and Impact Assessment**

Your primary goal is to identify *critical missing assumptions* that have not been explicitly stated, but are vital for successful project planning and execution. For each missing assumption, estimate its potential impact on the project's key performance indicators (KPIs) such as ROI, timeline, budget, or quality. This impact assessment should be quantitative wherever possible. For instance, if a missing assumption relates to regulatory approval, estimate the potential delay in project completion and the associated cost implications.

**Consider the Following Project Aspects:**

When reviewing the assumptions, actively consider these areas. Look for explicit *or* implicit assumptions that impact these areas.

-   **Financial:** Funding sources, cost estimates (initial and operational), revenue projections, pricing strategy, profitability, economic viability, return on investment (ROI), cost of capital, financial risks (e.g., currency fluctuations, interest rate changes), insurance costs.
-   **Timeline:** Project duration, key milestones, task dependencies, resource allocation over time, critical path analysis, potential delays (e.g., permitting, supply chain), seasonality effects, weather-related risks.
-   **Resources:** Human resources (skill availability, labor costs), material resources (supply availability, raw material costs), equipment (availability, maintenance costs), technology (availability, licensing costs), land (acquisition costs, suitability).
-   **Regulations:** Compliance with local, regional, and national laws, environmental regulations, permitting requirements, zoning ordinances, safety standards, data privacy regulations, industry-specific standards, political risks.
-   **Infrastructure:** Availability and capacity of transportation, utilities (electricity, water, gas), communication networks, cybersecurity risks.
-   **Environment:** Potential environmental impacts (e.g., emissions, waste generation, habitat disruption), mitigation strategies, climate change risks, sustainability practices, resource consumption.
-   **Stakeholders:** Community acceptance, government support, customer needs, supplier relationships, investor expectations, media relations, political influence, key partner dependencies.
-   **Technology:** Technology selection, innovation, integration, obsolescence, intellectual property rights, data security, scalability, maintenance, licensing.
-   **Market:** Market demand, competitive landscape, pricing pressure, customer preferences, economic trends, technological disruption, new market entrants, black swan events.
-   **Risk:** Credit risk, operational risk, strategic risk, compliance risk, political risk, insurance needs, cost of capital, inflation. Examples of risks are: the NLP algorithm has a bug and must be rewritten, funding dries up due to a market crash, etc.

**Your Analysis MUST:**

1.  **Identify Critical Missing Assumptions:** Explicitly state any crucial assumptions that are missing from the provided input. Clearly explain why each missing assumption is critical to the project's success.
2.  **Highlight Under-Explored Assumptions:** Point out areas where the existing assumptions lack sufficient detail or supporting evidence.
3.  **Challenge Questionable or Unrealistic Assumptions:** Identify any assumptions that seem unrealistic or based on flawed logic.
4.  **Discuss Sensitivity Analysis for key variables:** Quantify the potential impact of changes in key variables (e.g., a delay in permitting, a change in energy prices) on the project's overall success. For each issue, consider a plausible range for the key driving variables, and quantify the impact on the project's Return on Investment (ROI) or total project cost. Use percentages or hard numbers! Example of an analysis range of key variables is: The project may experience challenges related to a lack of data privacy considerations. A failure to uphold GDPR principles may result in fines ranging from 5-10% of annual turnover. The cost of a human for the project can be based on a 40/hr for 160 hours and would require a computer, this could be from 6000 to 7000 per month. The variance should not be double the base value.
5.  **Prioritize Issues:** Focus on the *three most critical* issues, providing detailed and actionable recommendations for addressing them.

**Guidance for identifying missing assumptions:**
Think about all the things that must be true for this project to succeed. Are all of these things in the existing list of assumptions?
* Resources: Financial, Human, Data, Time, etc.
* Pre-Existing Work: Benchmarks, Data Sets, Algorithms, Existing papers, etc.
* Outside Forces: Community Buy-In, Funding, New laws, weather, etc.
* Metrics: Clear, measurable success conditions.
* Technical Considerations: Hardware, Software, Algorithms, Scalability, Data security, etc.

Please limit your output to no more than 800 words.

Return your response as a JSON object with the following structure:
{
  "expert_domain": "The area of expertise most relevant for this review",
  "domain_specific_considerations": ["List", "of", "relevant", "considerations"],
  "issues": [
    {
      "issue": "Title of the issue",
      "explanation": "Explanation of why this issue is important",
      "recommendation": "Actionable recommendations to address the issue.  Be specific. Include specific steps, quantifiable targets, or examples of best practices whenever possible.",
      "sensitivity": "Quantitative sensitivity analysis details. Express the impact as a *range* of values on the project's ROI, total project cost, or project completion date, and include the *baseline* for comparison. Here are examples: *  'A delay in obtaining necessary permits (baseline: 6 months) could increase project costs by \u20ac100,000-200,000, or delay the ROI by 3-6 months.' *  'A 15% increase in the cost of solar panels (baseline: \u20ac1 million) could reduce the project's ROI by 5-7%.' *  'If we underestimate cloud computing costs, the project could be delayed by 3-6 months, or the ROI could be reduced by 10-15%'"
    },
    ...
  ],
  "conclusion": "Summary of main findings and recommendations"
}
"""

@dataclass
class ReviewAssumptions:
    """
    Take a look at the assumptions and provide feedback on potential omissions and improvements.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'ReviewAssumptions':
        """
        Invoke LLM with the project description and assumptions to be reviewed.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = REVIEW_ASSUMPTIONS_SYSTEM_PROMPT.strip()

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

        result = ReviewAssumptions(
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

        rows.append(f"## Domain of the expert reviewer\n{document_details.expert_domain}")

        if len(document_details.domain_specific_considerations) > 0:
            rows.append("\n## Domain-specific considerations\n")
            for item in document_details.domain_specific_considerations:
                rows.append(f"- {item}")
        else:
            rows.append("\n## Domain-specific considerations - None\n")

        if len(document_details.issues) > 0:
            for index, item in enumerate(document_details.issues, start=1):
                rows.append(f"\n## Issue {index} - {item.issue}")
                rows.append(item.explanation)
                rows.append(f"\n**Recommendation:** {item.recommendation}")
                rows.append(f"\n**Sensitivity:** {item.sensitivity}")
        else:
            rows.append("## Issues - None. This is unusual. Please report this to the developer of PlanExe.")

        rows.append(f"\n## Review conclusion\n{document_details.conclusion}")
        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.utils.concat_files_into_string import concat_files_into_string

    llm = get_llm("ollama-llama3.1")

    base_path = os.path.join(os.path.dirname(__file__), 'test_data', 'review_assumptions1')

    all_documents_string = concat_files_into_string(base_path)
    print(all_documents_string)

    result = ReviewAssumptions.execute(llm, all_documents_string)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

"""
Identify what data is needed for the plan. Such as for validating demand, the plan needs data from audience research or simulations.

Data Areas to Cover:
- Market Research and Participant Feedback
- Financial Estimates and Revenue Streams
- Resource and Staffing Needs
- Operational Logistics Simulations
- Regulatory Requirements & Ethical Considerations

Alternative names for this topic:
- Prepare Data
- Validate Before Execute
- Data-Driven Validation
- Gather Data
- Validation

PROMPT> python -m worker_plan_internal.plan.data_collection
"""
import json
import time
import logging
from math import ceil
from enum import Enum
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

logger = logging.getLogger(__name__)

class SensitivityScore(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'

    def human_readable(self) -> str:
        return self.value.capitalize()

class AssumptionItem(BaseModel):
    item_index: int = Field(
        description="Enumeration, starting from 1."
    )
    assumption: str = Field(
        description="The assumption to be validated."
    )
    sensitivity_score: Literal["low", "medium", "high"] = Field(
        description="The sensitivity score of the assumption."
    )

class PlannedDataCollectionItem(BaseModel):
    item_index: int = Field(
        description="Enumeration, starting from 1."
    )
    title: str = Field(
        description="Brief title, such as 'Venue Cost Estimates', 'Sponsorship & Revenue Streams'."
    )
    data_to_collect: list[str] = Field(
        description="What data to collect"
    )
    simulation_steps: list[str] = Field(
        description="How to simulate the data and what tools to use."
    )
    expert_validation_steps: list[str] = Field(
        description="Human expert validation steps."
    )
    rationale: str = Field(
        description="Explain why this particular data is to be collected."
    )
    responsible_parties: list[str] = Field(
        description="Who specifically should be involved or responsible."
    )
    assumptions: list[AssumptionItem] = Field(
        description="What assumptions are made about data, validation, collection, etc."
    )
    smart_validation_objective: str = Field(
        description="Explicit SMART objectives for validation."
    )
    notes: list[str] = Field(
        description="Insights and notes."
    )

class DocumentDetails(BaseModel):
    data_collection_list: list[PlannedDataCollectionItem] = Field(
        description="List of data to be collected."
    )
    summary: str = Field(
        description="Providing a high level context."
    )

DATA_COLLECTION_SYSTEM_PROMPT = """
You are an automated project planning assistant generating structured project plans.

When given a project query:
  - Identify crucial data collection areas necessary to achieve the project's objectives.
  - Clearly define what data needs to be collected for each area.
  - Specify detailed simulation steps (e.g., software tools or online resources) to preliminarily validate data before expert consultation.
  - Specify expert validation steps explicitly, detailing experts or authoritative bodies to consult.
  - Clearly state a concise rationale explaining the criticality of each data collection area.
  - List the responsible parties who will carry out or oversee the data collection.
  - Explicitly state underlying assumptions, labeling each assumption with a sensitivity score ("low", "medium", "high" — exact lowercase; use only these values, no synonyms) based on potential project impact if incorrect.
  - Write SMART (Specific, Measurable, Achievable, Relevant, Time-bound) validation objectives for each area.
  - Include a rough cost estimate for validation activities when possible.
  - Generate a clear validation results template for each data collection area, containing fields for original assumption, SMART objective, actual data collected, data source, comparison against assumption, conclusion (Validated, Partially Validated, Invalidated), recommended escape hatch or contingency if invalidated, and triage actions if partially validated.
  - Explicitly mention uncertainties, risks, or missing data.
  - Provide a concise summary of immediate actionable tasks focusing on validating the most sensitive assumptions first.

Ensure every "data collection item" explicitly includes BOTH simulation_steps and expert_validation_steps. Simulation_steps must always specify tools or software. Expert_validation_steps must clearly define human experts or authorities for verification. Never leave these steps empty.

Provide a concise and meaningful summary outlining critical next steps and immediately actionable tasks, guiding stakeholders clearly on what must be done next.

OUTPUT FORMAT — STRICT JSON ONLY
Return exactly one JSON object that conforms to the provided JSON schema. Output nothing before or after it.

Rules:
1) No markdown or code fences.
2) Valid RFC 8259 JSON: double-quoted keys/strings, proper escaping, no trailing commas.
3) Arrays/objects: separate items with a single comma; never place a comma immediately before '}' or ']'.
4) Use only ASCII JSON punctuation: " , : [ ] { } (no full-width or locale punctuation).
5) End immediately after the final '}' of that single JSON object.
6) SELF-CHECK (silent): ensure the JSON parses (e.g., JSON.parse) before sending; if it would fail (missing/extra commas, bad quotes), fix it first.
"""

@dataclass
class DataCollection:
    """
    Identify what data is needed for the plan.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'DataCollection':
        """
        Invoke LLM with the project details.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        system_prompt = DATA_COLLECTION_SYSTEM_PROMPT.strip()

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
            logger.debug(f"LLM chat interaction failed: {e}")
            logger.error("LLM chat interaction failed.", exc_info=True)
            raise ValueError("LLM chat interaction failed.") from e

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

        result = DataCollection(
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
    def _format_bullet_list(items: list[str]) -> str:
        """
        Format a list of strings into a markdown bullet list.
        
        Args:
            items: List of strings to format as bullet points
            
        Returns:
            Formatted markdown bullet list
        """
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def convert_to_markdown(document_details: DocumentDetails) -> str:
        """
        Convert the raw document details to markdown.
        """
        rows = []

        for item_index, data_collection_item in enumerate(document_details.data_collection_list, start=1):
            if item_index > 1:
                rows.append("\n")
            rows.append(f"## {item_index}. {data_collection_item.title}\n")
            rows.append(data_collection_item.rationale)

            data_to_collect = DataCollection._format_bullet_list(data_collection_item.data_to_collect)
            rows.append(f"\n### Data to Collect\n\n{data_to_collect}")

            simulation_steps = DataCollection._format_bullet_list(data_collection_item.simulation_steps)
            rows.append(f"\n### Simulation Steps\n\n{simulation_steps}")

            expert_validation_steps = DataCollection._format_bullet_list(data_collection_item.expert_validation_steps)
            rows.append(f"\n### Expert Validation Steps\n\n{expert_validation_steps}")

            responsible_parties = DataCollection._format_bullet_list(data_collection_item.responsible_parties)
            rows.append(f"\n### Responsible Parties\n\n{responsible_parties}")

            assumption_list = [f"**{item.sensitivity_score.human_readable()}:** {item.assumption}" for item in data_collection_item.assumptions]
            assumptions = DataCollection._format_bullet_list(assumption_list)
            rows.append(f"\n### Assumptions\n\n{assumptions}")

            rows.append(f"\n### SMART Validation Objective\n\n{data_collection_item.smart_validation_objective}")

            notes = DataCollection._format_bullet_list(data_collection_item.notes)
            rows.append(f"\n### Notes\n\n{notes}")

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

    result = DataCollection.execute(llm, query)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

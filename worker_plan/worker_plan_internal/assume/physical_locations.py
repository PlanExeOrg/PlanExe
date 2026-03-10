"""
Pick suitable physical locations for the project plan. 
- If the plan is purely digital and can be executed without any physical location, then there is no need to run this step.
- If the user prompt already includes the physical location, then include that location in the response.
- If the user prompt does not mention any location, then the expert should suggest suitable locations based on the project requirements.
- There may be multiple locations, in case a bridge is to be built between two countries.

PROMPT> python -m worker_plan_internal.assume.physical_locations
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

class PhysicalLocationItem(BaseModel):
    item_index: int = Field(
        description="Enumeration of the locations, starting from 1."
    )
    physical_location_broad: str = Field(
        description="A broad location for the project, such as a country or region. Use 'Global' if applicable."
    )
    physical_location_detailed: str = Field(
        description="Narrow down the physical location even more, such as a city name."
    )
    physical_location_specific: str = Field(
        description="Narrow down the physical location even more, such as a city name, region, or type of location (e.g., 'Oceanographic Research Centers')."
    )
    rationale_for_suggestion: str = Field(
        description="Explain why this particular physical location is suggested."
    )

class DocumentDetails(BaseModel):
    has_location_in_plan: bool = Field(
        description="Is the location specified in the plan."
    )
    requirements_for_the_physical_locations: list[str] = Field(
        description="List of requirements/constraints for well suited locations."
    )
    physical_locations: list[PhysicalLocationItem] = Field(
        description="List of physical locations."
    )
    location_summary: str = Field(
        description="Providing a high level context."
    )

PHYSICAL_LOCATIONS_SYSTEM_PROMPT = """
You are a world-class planning expert specializing in real-world physical locations. Your goal is to generate a JSON response that follows the `DocumentDetails` and `PhysicalLocationItem` models precisely. 

Use the following guidelines:

## JSON Models

### DocumentDetails
- **has_location_in_plan** (bool):
  - `true` if the user’s prompt *explicitly mentions or strongly implies* a physical location. This includes named locations (e.g., "Paris", "my office"), specific landmarks (e.g., "Eiffel Tower," "Grand Canyon"), or clear activities that inherently tie the plan to a location (e.g., "build a house", "open a restaurant"). **If the user's plan can *only* occur in a specific geographic area, consider it to have a location in the plan.**
  - `false` if the user’s prompt does not specify any location.

- **requirements_for_the_physical_locations** (list of strings):
  - Key criteria or constraints relevant to location selection (e.g., "cheap labor", "near highways", "near harbor", "space for 10-20 people").

- **physical_locations** (list of PhysicalLocationItem):
  - A list of recommended or confirmed physical sites. 
  - If the user’s prompt does not require any location, then you **MUST** suggest **three** well-reasoned suggestions.
  - If the user does require a new site (and has no location in mind), you **MUST** provide **three** well-reasoned suggestions. 
  - If the user’s prompt already includes a specific location but does not need other suggestions, you may list just that location, or clarify it in one `PhysicalLocationItem` in addition to providing the other **three** well-reasoned suggestions.
  - When suggesting locations, consider a variety of factors, such as accessibility, cost, zoning regulations, and proximity to relevant resources or amenities.

- **location_summary** (string):
  - A concise explanation of why the listed sites (if any) are relevant, or—if no location is provided—why no location is necessary (e.g., “All tasks can be done with the user’s current setup; no new site required.”).

### PhysicalLocationItem
- **item_index** (string):
  - A unique integer (e.g., 1, 2, 3) for each location.
- **physical_location_broad** (string):
  - A country or wide region (e.g., "USA", "Region of North Denmark").
- **physical_location_detailed** (string):
  - A more specific subdivision (city, district).
- **physical_location_specific** (string):
  - A precise address, if relevant.
- **rationale_for_suggestion** (string):
  - Why this location suits the plan (e.g., "near raw materials", "close to highways", "existing infrastructure").

## Additional Instructions

1. **When the User Already Has a Location**  
   - If `has_location_in_plan = true` and the user explicitly provided a place (e.g., "my home", "my shop"), you can either:
     - Use a single `PhysicalLocationItem` to confirm or refine that address in addition to the other **three** well-reasoned suggestions, **or**  
     - Provide **three** location items of suggestions if the user is open to alternatives or further detail within the same area.  

2. **When the User Needs Suggestions**  
   - If `has_location_in_plan = false`, you **MUST** propose **three** distinct sites that satisfy the user’s requirements.

3. **location_summary** Consistency  
   - Always provide a summary that matches the `physical_locations` array. 
   - If multiple locations are provided, summarize how each meets the user’s needs.

---

Example scenarios:

- **Implied Physical Location - Eiffel Tower:**
  Given "Visit the Eiffel Tower."
  The correct output is:
  {
    "has_location_in_plan": true,
    "requirements_for_the_physical_locations": [],
    "physical_locations": [
      {
        "item_index": 1,
        "physical_location_broad": "France",
        "physical_location_detailed": "Eiffel Tower, Paris",
        "physical_location_specific": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        "rationale_for_suggestion": "The plan is to visit the Eiffel Tower, which is located in Paris, France."
      },
      {
        "item_index": 2,
        "physical_location_broad": "France",
        "physical_location_detailed": "Near Eiffel Tower, Paris",
        "physical_location_specific": "5 Avenue Anatole France, 75007 Paris, France",
        "rationale_for_suggestion": "A location near the Eiffel Tower would provide convenient access for individuals who also plan to visit the landmark."
      },
      {
        "item_index": 3,
        "physical_location_broad": "France",
        "physical_location_detailed": "Central Paris",
        "physical_location_specific": "Various locations in Central Paris",
        "rationale_for_suggestion": "Central Paris offers a vibrant and accessible environment with numerous transportation options."
      }
    ],
    "location_summary": "The plan is to visit the Eiffel Tower, which is located in Paris, France, in addition to a location near the Eiffel Tower and Central Paris."
  }
"""

@dataclass
class PhysicalLocations:
    """
    Take a look at the vague plan description and suggest physical locations.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'PhysicalLocations':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = PHYSICAL_LOCATIONS_SYSTEM_PROMPT.strip()

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

        result = PhysicalLocations(
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

        if document_details.has_location_in_plan:
            rows.append("This plan implies one or more physical locations.")
        else:
            rows.append("This plan **does not** imply any physical location.")

        if len(document_details.requirements_for_the_physical_locations) > 0:
            rows.append("\n## Requirements for physical locations\n")
            for requirement in document_details.requirements_for_the_physical_locations:
                rows.append(f"- {requirement}")
        else:
            rows.append("No requirements for the physical location.")
        
        for location_index, location in enumerate(document_details.physical_locations, start=1):
            rows.append(f"\n## Location {location_index}")
            physical_location_broad = location.physical_location_broad.strip()
            physical_location_detailed = location.physical_location_detailed.strip()
            physical_location_specific = location.physical_location_specific.strip()
            missing_location = (len(physical_location_broad) + len(physical_location_detailed) + len(physical_location_specific)) == 0
            if len(physical_location_broad) > 0:
                rows.append(f"{physical_location_broad}\n")
            if len(physical_location_detailed) > 0:
                rows.append(f"{physical_location_detailed}\n")
            if len(physical_location_specific) > 0:
                rows.append(f"{physical_location_specific}\n")
            if missing_location:
                rows.append("Missing location info.\n")
            rows.append(f"**Rationale**: {location.rationale_for_suggestion}")
        
        rows.append(f"\n## Location Summary\n{document_details.location_summary}")
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

    physical_locations = PhysicalLocations.execute(llm, query)
    json_response = physical_locations.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{physical_locations.markdown}")

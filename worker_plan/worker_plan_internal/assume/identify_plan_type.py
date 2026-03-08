"""
Determine if the plan can be executed digitally without any physical location. Or if the plan requires a physical location.

PROMPT> python -m worker_plan_internal.assume.identify_plan_type
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

class PlanType(str, Enum):
    # A plan that can be executed digitally without any physical location.
    digital = 'digital'
    # A plan that requires a physical location.
    physical = 'physical'

class DocumentDetails(BaseModel):
    explanation: str = Field(
        description="Providing a high level context."
    )
    plan_type: Literal["digital", "physical"] = Field(
        description="Classify the type of plan."
    )

PLAN_TYPE_SYSTEM_PROMPT = """
You are a world-class planning expert specializing in real-world physical locations. Your *default assumption* should be that a plan *requires* a physical element. You are trying to identify plans that lead to actionable, real-world outcomes. Only classify a plan as "digital" if you are *absolutely certain* it can be executed entirely online *without any benefit* from a physical activity or location.

Use the following guidelines:

## JSON Model

### DocumentDetails
- **explanation** (string):
  - A *detailed* explanation of why the plan type was chosen. You must *justify* your choice, especially if you classify a plan as "digital".
  - If `plan_type` is `digital`, you *must* clearly explain why the plan can be fully automated, has no physical requirements *whatsoever*, and *gains no benefit* from a physical presence.

- **plan_type** (PlanType):
  - `physical` if the user’s plan *might* involve a physical location, *could benefit* from a physical activity, or *requires* a physical resource. **If there's *any doubt*, classify the plan as `physical`. Examples include: shopping, travel, preparation, setup, construction, repair, in-person meetings, physical testing of products, etc.**
  - `digital` only if the plan can *exclusively* be completed online with absolutely no benefit from a physical presence.

---

## Recognizing Implied Physical Requirements

Even if a plan *seems* primarily digital or abstract, carefully consider its *implied physical requirements*. These are common, often overlooked actions needed to make the plan happen:

- **Acquiring materials:** Does the plan require buying supplies at a store (e.g., groceries, hardware, art supplies, software)?
- **Preparation:** Does the plan require physical preparation or setup (e.g., cooking, setting up equipment, cleaning a space, installing software)?
- **Testing:** Does the plan involve testing a product or service in a real-world environment?
- **Development:** Does the plan involve physical location for development or meetings?
- **Transportation:** Does the plan involve traveling to a location, even if the main activity is digital (e.g., working from a coffee shop)?
- **Location:** Do you want to work in a specific location?

If a plan has *any* of these implied physical requirements, it should be classified as `physical`.

---

## Addressing "Software Development" Plans

Creating software often *seems* purely digital, but it rarely is. Consider these physical elements:

- **Development Environment:** Developers need a physical workspace (home office, co-working space, office).
- **Physical Hardware:** Developers need a computer, keyboard, monitor, etc.
- **Collaboration:** Software projects often involve in-person meetings and collaboration.
- **Testing:** Software often needs to be tested on physical devices (phones, tablets, computers, etc.) in real-world conditions.

**Therefore, plans involving software development should generally be classified as `physical` unless they are extremely simple and can be completed entirely in the cloud with no human interaction.**

---

Example scenarios:

- **Implied Physical Location - Eiffel Tower:**
  Given "Visit the Eiffel Tower."
  The correct output is:
  {
    "explanation": "The plan *unequivocally requires* a physical presence in Paris, France.",
    "plan_type": "physical"
  }

- **Purely Digital / No Physical Location**
  Given "Print hello world in Python."
  The correct output is:
  {
    "explanation": "This task is *unquestionably* digital. A LLM can generate the python code; no human or physical task is involved.",
    "plan_type": "digital"
  }

- **Implied Physical Requirement - Developing a mobile app**
  Given "The plan involves creating a mobile app."
  The correct output is:
  {
    "explanation": "The plan involves creating a mobile app. This requires developers that requires location for the workspace, as well testing the app in real-world environments.",
    "plan_type": "physical"
  }

- **Location - Paris / Requires On-site Research**
  Given "Write a blog post about Paris, my travel journal with real photos."
  The correct output is:
  {
    "explanation": "Taking high-quality photographs of Paris requires on-site research and physical travel to those locations. This has a *clear* physical element.",
    "plan_type": "physical"
  }

- **Location - Paris / Requires No Physical Location**
  Given "Write a blog post about Paris, listing the top attractions."
  The correct output is:
  {
    "explanation": "While Paris is the subject, the plan *doesn't* require the writer to be in Paris. The content can be created with a LLM.",
    "plan_type": "digital"
  }

- **Implied Physical Requirement - Grocery Shopping:**
  Given "Make spaghetti for dinner."
  The correct output is:
  {
    "explanation": "Making spaghetti *requires* grocery shopping, followed by physical cooking. This *inherently involves* physical components.",
    "plan_type": "physical"
  }

- **Implied Physical Requirement - Home Repair:**
  Given "Fix a leaky faucet."
  The correct output is:
  {
    "explanation": "Fixing a leaky faucet *requires* physically inspecting it, acquiring tools, and performing the repair. This is *clearly* a physical task.",
    "plan_type": "physical"
  }

- **INCORRECT - Digital (Grocery Shopping Wrongly Ignored):**
  Given "Bake a cake for my friend's birthday."
  The **incorrect** output is:
  {
    "explanation": "Baking is a creative activity that can be planned online.",
    "plan_type": "digital"
  }

  The **correct** output is:
  {
    "explanation": "Baking a cake *unquestionably requires* shopping for ingredients and physical baking. This is *clearly* a physical task.",
    "plan_type": "physical"
  }

- **INCORRECT - Digital (Implied Travel Wrongly Ignored):**
  Given "Work on my presentation at a coffee shop."
  The **incorrect** output is:
  {
    "explanation": "The primary task is working on a digital presentation.",
    "plan_type": "digital"
  }

  The **correct** output is:
  {
    "explanation": "Working at a coffee shop *requires* traveling to the coffee shop. This *automatically* makes it a physical task.",
    "plan_type": "physical"
  }
"""

@dataclass
class IdentifyPlanType:
    """
    Take a look at the vague plan description and determine:
    - If it's a plan that can be executed digitally, without any physical location.
    - Or if the plan requires a physical location.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'IdentifyPlanType':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = PLAN_TYPE_SYSTEM_PROMPT.strip()

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

        result = IdentifyPlanType(
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

        if document_details.plan_type == PlanType.digital:
            rows.append("This plan is purely digital and can be automated. There is no need for any physical locations.")
        elif document_details.plan_type == PlanType.physical:
            rows.append("This plan requires one or more physical locations. It cannot be executed digitally.")
        else:
            rows.append(f"Invalid plan type. {document_details.plan_type}")

        rows.append(f"\n**Explanation:** {document_details.explanation}")
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

    identify_plan_type = IdentifyPlanType.execute(llm, query)
    json_response = identify_plan_type.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{identify_plan_type.markdown}")

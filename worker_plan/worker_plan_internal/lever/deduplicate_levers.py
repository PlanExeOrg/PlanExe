"""
The identify_potential_levers.py script creates a list of levers, some of which are duplicates.
This script deduplicates the list.

PROMPT> python -m worker_plan_internal.lever.deduplicate_levers

PROBLEM: A frequent problem is that the deduplicated levers is an empty list, despite having multiple input levers.
002-11-deduplicated_levers_raw.json
I often see output like this:
"deduplicated_levers": []
It's never supposed to be an empty list. It's supposed to be a list of multiple levers. I need to fix this.
"""
from enum import Enum
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field, ValidationError
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested

logger = logging.getLogger(__name__)

class LeverClassification(str, Enum):
    keep   = "keep"
    absorb = "absorb"
    remove = "remove"

class LeverDecision(BaseModel):
    lever_id: str = Field(
        description="The uuid of the lever."
    )
    classification: LeverClassification = Field(
        description="What should happen to this lever."
    )
    justification: str = Field(
        description="A concise justification for the classification. Use the lever_id to reference the lever that is being kept in its place. Use ~80 words."
    )

class DeduplicationAnalysis(BaseModel):
    decisions: List[LeverDecision] = Field(
        description="A list of all levers with their classification and justification."
    )

class InputLever(BaseModel):
    """Represents a single lever loaded from the initial brainstormed file."""
    lever_id: str
    name: str
    consequences: str
    options: List[str]
    review: str

class OutputLever(InputLever):
    """The InputLever and the deduplication justification."""
    deduplication_justification: str


DEDUPLICATE_SYSTEM_PROMPT = """
Evaluate each of the provided strategic levers individually. Classify every lever explicitly into one of:

- keep: Lever is distinct, unique, and essential.
- absorb: Lever overlaps significantly with another lever. Explicitly state the lever ID it should be merged into.
- remove: Lever is fully redundant. Removing it loses no meaningful detail. Use this sparingly.

Provide concise, explicit justifications mentioning lever IDs clearly. Always prefer "absorb" over "remove" to retain important details.

Always provide a justification for the classification. Explain why the lever is distinct from others. Don't use the same uninformative boilerplate.

Respect Hierarchy: When absorbing, merge the more specific lever into the more general one.
Don't take the more general lever and absorb it into a narrower one.
Also compare a lever against the group of already-merged levers.

Use "keep" if you lack understanding of what the lever is doing. This way a potential important lever is not getting removed.
Describe what the issue is in the justification.

Don't play it too safe, so you fail to perform the core task: consolidate the levers and get rid of the duplicates.

You must classify and justify **every lever** provided in the input.
"""

@dataclass
class DeduplicateLevers:
    """Holds the results of the deduplication."""
    user_prompt: str
    system_prompt: str
    response: DeduplicationAnalysis
    deduplicated_levers: List[OutputLever]
    metadata: Dict[str, Any]

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, project_context: str, raw_levers_list: List[dict]) -> 'DeduplicateLevers':
        """
        Executes the deduplication process.

        Args:
            llm_executor: The configured LLMExecutor instance.
            raw_levers_list: A list of dictionaries, each representing a lever.

        Returns:
            An instance of DeduplicateLevers containing the results.
        """
        try:
            input_levers = [InputLever(**lever) for lever in raw_levers_list]
        except ValidationError as e:
            raise ValueError(f"Invalid input lever data: {e}")

        if not input_levers:
            raise ValueError("No input levers to deduplicate.")

        logger.info(f"Starting deduplication for {len(input_levers)} levers.")

        levers_json = json.dumps([lever.model_dump() for lever in input_levers], indent=2)        
        user_prompt = (
            f"**Project Context:**\n{project_context}\n\n"
            "Here is the full list of strategic levers. Please analyze them for duplicates.\n\n"
            f"{levers_json}"
        )

        system_prompt = DEDUPLICATE_SYSTEM_PROMPT.strip()
        chat_message_list = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt)
        ]

        def execute_function(llm: LLM) -> dict:
            sllm = llm.as_structured_llm(DeduplicationAnalysis)
            chat_response = sllm.chat(chat_message_list)
            metadata = dict(llm.metadata)
            return {"chat_response": chat_response, "metadata": metadata}

        try:
            result = llm_executor.run(execute_function)
            analysis_result: DeduplicationAnalysis = result["chat_response"].raw
            metadata = result["metadata"]
        except PipelineStopRequested:
            raise
        except Exception as e:
            logger.error("Deduplication failed.", exc_info=True)
            raise ValueError("Deduplication failed.") from e

        # The LLM is supposed to return the same number of levers as the input.
        # However sometimes LLMs skips some levers. So I cannot assume that all the levers in the input are returned.
        # In case a lever is not returned, then I want to `keep` it. Otherwise, I might lose an important lever.

        # Perform the deduplication.
        output_levers = []
        for lever in input_levers:
            # Find the decision for this lever
            decision = None
            for decision_item in analysis_result.decisions:
                if decision_item.lever_id == lever.lever_id:
                    decision = decision_item
                    break
            if not decision:
                # Missing decision for this lever. Keep it.
                deduplication_justification = "Missing deduplication justification. Keeping this lever."
                output_lever = OutputLever(
                    **lever.model_dump(),
                    deduplication_justification=deduplication_justification
                )
                output_levers.append(output_lever)
                continue

            # Check if this is a keeper
            if decision.classification != LeverClassification.keep:
                # This is not a keeper
                continue

            # This is a keeper
            deduplication_justification = decision.justification.strip()
            if len(deduplication_justification) == 0:
                deduplication_justification = "Empty explanation. Keeping this lever."

            output_lever = OutputLever(
                **lever.model_dump(),
                deduplication_justification=deduplication_justification
            )
            output_levers.append(output_lever)

        return cls(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response=analysis_result,
            deduplicated_levers=output_levers,
            metadata=metadata
        )

    def to_dict(self, include_response=True, include_deduplicated_levers=True, include_metadata=True, include_system_prompt=True, include_user_prompt=True) -> dict:
        d = {}
        if include_response:
            d["response"] = self.response.model_dump()
        if include_deduplicated_levers:
            d['deduplicated_levers'] = [lever.model_dump() for lever in self.deduplicated_levers]
        if include_metadata:
            d['metadata'] = self.metadata
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        if include_user_prompt:
            d['user_prompt'] = self.user_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2))

    def save_clean(self, file_path: Path) -> None:
        """Saves the final, deduplicated list of levers to a JSON file."""
        output_data = [lever.model_dump() for lever in self.deduplicated_levers]
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Successfully saved {len(output_data)} deduplicated levers to {file_path!r}.")
        except IOError as e:
            logger.error(f"Failed to write output to {file_path!r}: {e}")

if __name__ == "__main__":
    from worker_plan_internal.prompt.prompt_catalog import PromptCatalog
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    prompt_catalog = PromptCatalog()
    prompt_catalog.load_simple_plan_prompts()

    prompt_id = "19dc0718-3df7-48e3-b06d-e2c664ecc07d"
    # prompt_id = "b9afce6c-f98d-4e9d-8525-267a9d153b51"
    prompt_item = prompt_catalog.find(prompt_id)
    if not prompt_item:
        raise ValueError("Prompt item not found.")
    project_context = prompt_item.prompt

    # This file is created by identify_potential_levers.py
    input_file = os.path.join(os.path.dirname(__file__), 'test_data', f'identify_potential_levers_{prompt_id}.json')
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_levers_data = json.load(f)

    output_file = f"deduplicate_levers_{prompt_id}.json"

    model_names = ["ollama-llama3.1"]
    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    # --- Run Deduplication ---
    result = DeduplicateLevers.execute(
        llm_executor=llm_executor,
        project_context=project_context,
        raw_levers_list=raw_levers_data
    )

    d = result.to_dict(include_response=True, include_deduplicated_levers=True, include_metadata=True, include_system_prompt=False, include_user_prompt=False)
    d_json = json.dumps(d, indent=2)
    logger.info(f"Deduplication result: {d_json}")
    logger.info(f"Lever count after deduplication: {len(result.deduplicated_levers)}.")

    # --- Save Output ---
    result.save_clean(output_file)

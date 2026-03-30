"""
Enrich the potential levers with fields such as: "description", "synergy_text", "conflict_text".

- This module takes the raw, brainstormed list of potential levers.
- For each lever, it generates three critical fields:
  1. `description`: A clear explanation of the lever's purpose and scope.
  2. `synergy_text`: A summary of its most significant positive interactions with other levers in the system.
  3. `conflict_text`: A summary of its most significant negative interactions and trade-offs.
- This creates a highly context-rich dataset of levers, ready for the filtering step.

PROMPT> python -m worker_plan_internal.lever.enrich_potential_levers
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field, ValidationError

from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested

logger = logging.getLogger(__name__)

OPTIMIZE_INSTRUCTIONS = """\
Goal: enrich each deduplicated lever with a description, synergy_text, and
conflict_text so that downstream steps (FocusOnVitalFewLevers, scenario
generation) can make informed prioritization and trade-off decisions.

Pipeline context
----------------
This step (EnrichLevers) is part of a 6-step solution-space exploration
pipeline inside run_plan_pipeline.py:

  1. IdentifyPotentialLevers  — brainstorms 15-20 raw levers
  2. DeduplicateLevers        — removes near-duplicate levers
  3. EnrichLevers             ← you are here
  4. FocusOnVitalFewLevers    — filters down to 4-6 high-impact levers
  5. ScenarioGeneration       — builds 3 scenarios (aggressive, medium, safe)
  6. ScenarioSelection        — picks the best-fitting scenario

Input: deduplicated levers (primary + secondary) from step 2.
Output: the same levers plus three new fields — description, synergy_text,
conflict_text. Processing happens in batches of BATCH_SIZE levers per LLM
call, with the full lever list provided as context in every batch.

Step 4 uses description for filtering and synergy/conflict for scenario
construction. Vague or generic enrichment text propagates downstream and
degrades scenario quality.

Known problems to guard against
--------------------------------
- Boilerplate descriptions. Weak models produce nearly identical
  description paragraphs for every lever, differing only in the lever
  name. Each description should reflect the lever's specific purpose,
  scope, and success metrics — not a generic template.
- Self-referential synergy/conflict. Models sometimes name the lever
  itself in its own synergy_text or conflict_text ("This lever synergizes
  with itself"). The fields must reference OTHER levers from the full
  list by name.
- Phantom lever references. Models invent lever names that do not exist
  in the full list. Every lever name mentioned in synergy_text or
  conflict_text must match an actual lever_id/name from the input.
- Symmetric parroting. When lever A says it synergizes with lever B,
  lever B's synergy_text often copies the same sentence verbatim with
  names swapped. Each lever's text should describe the relationship
  from its own perspective with distinct reasoning.
- Word-count padding. Models inflate to hit the 80-100 word target with
  filler phrases ("It is important to note that...", "This lever plays
  a crucial role in..."). Prefer concrete, information-dense sentences.
- Missing conflict_text. Models default to positive framing and produce
  empty or trivially short conflict_text ("No significant conflicts").
  Every lever has trade-offs — the conflict_text must identify at least
  one genuine tension.
- Batch boundary blindness. When processing batches, the LLM may fail
  to reference levers from other batches in synergy/conflict text. The
  full lever list is provided as context precisely to prevent this —
  ensure the prompt makes the full list visually prominent.
- Consequence echoing without elaboration. When consequences and review
  are provided in the batch prompt, weak models (e.g. llama3.1)
  summarize consequences verbatim as the description instead of using
  them as grounding for a richer explanation of purpose, scope, and
  success metrics. The description should go beyond what consequences
  already states.
- UUID cross-reference format inconsistency. The full_lever_context_str
  includes lever_id UUIDs, causing models to copy UUIDs into
  synergy_text and conflict_text in varying formats (full UUID, 8-char
  truncated, backtick-quoted name, plain name). Models should reference
  levers by name only in free-text fields.
"""

# The number of levers to process in a single call to the LLM.
BATCH_SIZE = 5

# Retry guards: limit how aggressively failed batches are re-split.
MAX_RETRY_DEPTH = 1          # max number of splits before skipping
MAX_RETRY_BUDGET_SECONDS = 300  # stop retrying after this many seconds

# Models with a small context window get a smaller batch size to avoid
# output overflow on structured JSON responses.  context_window is the
# correct metric — num_output reflects the configured max_tokens cap,
# which can be much larger than what the model actually supports.
SMALL_CONTEXT_THRESHOLD = 6000
SMALL_CONTEXT_BATCH_SIZE = 2

# --- Pydantic Models for Data Structuring ---

class InputLever(BaseModel):
    """Represents a single lever loaded from the deduplicated file."""
    lever_id: str
    name: str
    consequences: str
    options: List[str]
    review: str
    classification: Optional[str] = None
    deduplication_justification: str

class LeverCharacterization(BaseModel):
    """Structured response for a single lever's enrichment from the LLM."""
    lever_id: str = Field(description="The uuid of the lever")
    description: str = Field(
        description="A comprehensive description (80-100 words) of the lever's purpose, scope, and key success metrics."
    )
    synergy_text: str = Field(
        description="A free-form text (40-60 words) describing this lever's most prominent synergistic effects with other levers in the full list. Name the specific levers it enhances."
    )
    conflict_text: str = Field(
        description="A free-form text (40-60 words) describing this lever's most prominent conflicts or trade-offs with other levers in the full list. Name the specific levers it constrains."
    )

class BatchCharacterizationResult(BaseModel):
    """The expected JSON structure for a batch of characterizations from the LLM."""
    characterizations: List[LeverCharacterization] = Field(
        description="A list containing the full characterization for each requested lever in the batch."
    )

class CharacterizedLever(InputLever):
    """The final, enriched and characterized lever model."""
    description: str
    synergy_text: str
    conflict_text: str

# --- LLM Prompts ---

ENRICH_LEVERS_SYSTEM_PROMPT = """
You are an expert systems analyst and strategist. Your task is to enrich a list of strategic levers by characterizing their role within the broader system of all levers for a project.

**Goal:** For each lever provided in the current batch, you will generate a `description`, a `synergy_text`, and a `conflict_text`.

**Full Context:** You will be given the overall project plan and the FULL list of ALL levers for context. You must analyze each lever in the batch against this full list.

**Output Requirements (for each lever in the batch):**
1.  **`description`:** (80-100 words) Clearly explain the lever's purpose, what it controls, its objectives, and key success metrics.
2.  **`synergy_text`:** (40-60 words) Describe its most important POSITIVE interactions. How does this lever amplify or enable others? You MUST explicitly name one or two other levers from the full list that it has strong synergy with.
3.  **`conflict_text`:** (40-60 words) Describe its most important NEGATIVE interactions or trade-offs. What difficult choices does this lever create? Which other levers does it constrain? You MUST explicitly name one or two other levers from the full list that it has a strong conflict with.

You MUST respond with a single JSON object that strictly adheres to the `BatchCharacterizationResult` schema. Provide a full characterization for every single lever requested in the user prompt.
"""

@dataclass
class EnrichPotentialLevers:
    """Holds the results of the characterization process."""
    characterized_levers: List[CharacterizedLever]
    metadata: List[Dict[str, Any]]
    batches_succeeded: int = 0
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, project_context: str, raw_levers_list: list[dict]) -> 'EnrichPotentialLevers':
        levers_to_characterize = [InputLever(**lever) for lever in raw_levers_list]

        if not levers_to_characterize:
            raise ValueError("The list of levers to characterize cannot be empty.")

        # Adaptive batch size — use context_window (the model's actual token
        # budget) rather than num_output (which just reflects max_tokens config).
        batch_size = BATCH_SIZE
        try:
            if llm_executor.llm_models:
                probe_llm = llm_executor.llm_models[0].create_llm()
                context_window = probe_llm.metadata.context_window
                if context_window < SMALL_CONTEXT_THRESHOLD:
                    batch_size = SMALL_CONTEXT_BATCH_SIZE
                    logger.info(f"Adaptive batch_size={batch_size} for model with context_window={context_window}")
        except Exception:
            logger.debug("Could not probe model metadata for adaptive batch size, using default", exc_info=True)

        logger.info(f"Characterizing {len(levers_to_characterize)} levers in batches of {batch_size}.")

        # Prepare the full list of lever names and IDs for context in the prompt
        full_lever_context_str = "\n".join([f"- {lever.lever_id}: {lever.name}" for lever in levers_to_characterize])

        enriched_levers_map = {lever.lever_id: lever.model_dump() for lever in levers_to_characterize}
        all_metadata = []

        system_message = ChatMessage(role=MessageRole.SYSTEM, content=ENRICH_LEVERS_SYSTEM_PROMPT.strip())

        # Process levers in batches with retry on failure.
        # On failure, split the batch in half and retry each sub-batch.
        # Guards: depth limit (MAX_RETRY_DEPTH) and time budget (MAX_RETRY_BUDGET_SECONDS).
        batches_succeeded = 0
        errors: list[dict[str, Any]] = []
        retry_start_time = time.monotonic()
        pending_batches: list[tuple[list[InputLever], int]] = []  # (batch, depth)
        for i in range(0, len(levers_to_characterize), batch_size):
            batch = levers_to_characterize[i:i + batch_size]
            if batch:
                pending_batches.append((batch, 0))

        while pending_batches:
            batch, depth = pending_batches.pop(0)
            batch_label = f"batch of {len(batch)} levers (depth={depth})"
            logger.info(f"Processing {batch_label}...")

            lever_details_for_prompt = "\n\n".join([
                f"Lever ID: {lever.lever_id}\n"
                f"Name: {lever.name}\n"
                f"Consequences: {lever.consequences}\n"
                f"Options: {json.dumps(lever.options)}\n"
                f"Review: {lever.review}"
                for lever in batch
            ])

            user_prompt = (
                f"**Project Context:**\n{project_context}\n\n"
                f"**Full List of All Levers (for context):**\n{full_lever_context_str}\n\n"
                "---\n\n"
                f"**Levers to Characterize in this Batch:**\n"
                f"Please provide the `description`, `synergy_text`, and `conflict_text` for the following {len(batch)} levers. "
                f"Analyze them against the full list provided above.\n\n"
                f"{lever_details_for_prompt}"
            )

            chat_message_list = [system_message, ChatMessage(role=MessageRole.USER, content=user_prompt)]

            def execute_function(llm: LLM) -> dict:
                sllm = llm.as_structured_llm(BatchCharacterizationResult)
                chat_response = sllm.chat(chat_message_list)
                metadata = dict(llm.metadata)
                metadata["llm_classname"] = llm.class_name()
                return {"chat_response": chat_response, "metadata": metadata}

            try:
                result = llm_executor.run(execute_function)
                batch_result: BatchCharacterizationResult = result["chat_response"].raw
                all_metadata.append(result["metadata"])
                batches_succeeded += 1

                for char in batch_result.characterizations:
                    if char.lever_id in enriched_levers_map:
                        enriched_levers_map[char.lever_id].update({
                            'description': char.description,
                            'synergy_text': char.synergy_text,
                            'conflict_text': char.conflict_text
                        })
                    else:
                        logger.warning(f"LLM returned characterization for an unknown lever_id: '{char.lever_id}'")
                        errors.append({"type": "unknown_lever_id", "lever_id": char.lever_id})

            except PipelineStopRequested:
                raise
            except Exception as e:
                lever_ids = [lever.lever_id for lever in batch]
                elapsed = time.monotonic() - retry_start_time
                can_retry = (
                    len(batch) > 1
                    and depth < MAX_RETRY_DEPTH
                    and elapsed < MAX_RETRY_BUDGET_SECONDS
                )
                error_str = f"{type(e).__name__}: {e}"
                if can_retry:
                    mid = len(batch) // 2
                    logger.warning(
                        f"Batch failed for {lever_ids}, splitting into sub-batches of "
                        f"{mid} and {len(batch) - mid} and retrying (depth={depth + 1}, elapsed={elapsed:.0f}s)."
                    )
                    errors.append({"type": "batch_retry", "lever_ids": lever_ids, "depth": depth, "error": error_str})
                    pending_batches.insert(0, (batch[mid:], depth + 1))
                    pending_batches.insert(0, (batch[:mid], depth + 1))
                else:
                    if len(batch) == 1:
                        logger.error(
                            f"Single-lever batch failed for {lever_ids[0]}, skipping.", exc_info=True
                        )
                    else:
                        logger.error(
                            f"Batch failed for {lever_ids}, skipping (depth={depth}, elapsed={elapsed:.0f}s).",
                            exc_info=True,
                        )
                    errors.append({"type": "batch_skipped", "lever_ids": lever_ids, "depth": depth, "error": error_str})

        final_characterized_levers = []
        for lever_id, data in enriched_levers_map.items():
            if all(k in data for k in ['description', 'synergy_text', 'conflict_text']):
                try:
                    final_characterized_levers.append(CharacterizedLever(**data))
                except ValidationError as e:
                    logger.error(f"Pydantic validation failed for characterized lever '{lever_id}'. Error: {e}")
                    errors.append({"type": "validation_error", "lever_id": lever_id, "error": str(e)})
            else:
                logger.error(f"Characterization incomplete for lever '{lever_id}'. Skipping this lever.")
                errors.append({"type": "incomplete", "lever_id": lever_id})

        return cls(
            characterized_levers=final_characterized_levers,
            metadata=all_metadata,
            batches_succeeded=batches_succeeded,
            errors=errors,
        )
    
    def save_raw(self, file_path: str) -> None:
        """Saves the characterized levers to a JSON file."""
        output_data = {
            "metadata": self.metadata,
            "errors": self.errors,
            "characterized_levers": [lever.model_dump() for lever in self.characterized_levers]
        }
        with open(file_path, 'w') as f:
            json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName
    from worker_plan_internal.prompt.prompt_catalog import PromptCatalog
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    prompt_catalog = PromptCatalog()
    prompt_catalog.load_simple_plan_prompts()

    prompt_id = "19dc0718-3df7-48e3-b06d-e2c664ecc07d"
    prompt_item = prompt_catalog.find(prompt_id)
    if not prompt_item:
        raise ValueError("Prompt item not found.")
    project_plan = prompt_item.prompt

    # This file is created by deduplicate_levers.py
    input_file = os.path.join(os.path.dirname(__file__), 'test_data', f'deduplicate_levers_{prompt_id}.json')
    output_file = f"enrich_potential_levers_{prompt_id}.json"
    
    if not os.path.exists(input_file):
        logger.error(f"Input data file not found at: {input_file}")
        exit(1)

    # Parse the multi-part input file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_levers = json.load(f)
        
    logger.info(f"Successfully loaded {len(input_levers)} levers from '{input_file}'.")

    model_names = ["ollama-llama3.1"]
    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    result = EnrichPotentialLevers.execute(
        llm_executor=llm_executor,
        project_context=project_plan,
        raw_levers_list=input_levers
    )

    print(f"\nSuccessfully processed. Characterized {len(result.characterized_levers)} out of {len(input_levers)} levers.")
    
    if not result.characterized_levers:
        raise ValueError("No levers were successfully characterized.")
    
    print("\n--- Example Characterized Lever ---")
    example_lever = result.characterized_levers[0]
    print(json.dumps(example_lever.model_dump(), indent=2))

    result.save_raw(output_file)
    logger.info(f"Full list of {len(result.characterized_levers)} characterized levers saved to '{output_file}'.")

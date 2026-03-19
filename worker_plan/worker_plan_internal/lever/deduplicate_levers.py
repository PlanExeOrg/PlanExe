"""
The identify_potential_levers.py script creates a list of levers, some of which are duplicates.
This script deduplicates the list.

PROMPT> python -m worker_plan_internal.lever.deduplicate_levers

"""
from enum import Enum
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field, ValidationError
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OPTIMIZE_INSTRUCTIONS — read by self_improve analysis to understand known
# failure modes. Keep this up to date as iterations reveal new patterns.
# ---------------------------------------------------------------------------
OPTIMIZE_INSTRUCTIONS = """
Known failure modes for deduplicate_levers (discovered in iteration 42):

- Blanket-primary. Weak models (llama3.1) classify nearly every lever as
  "primary" and perform zero absorb/remove — effectively skipping
  deduplication. Root cause: the safety-valve instruction ("Use primary
  if you lack understanding") gives an easy escape. Mitigation: narrow
  the safety valve and add a calibration hint for expected absorb/remove
  counts.

- Over-inclusion. Mid-tier models (gpt-4o-mini) keep 10-12 of 15 levers
  where stronger models keep 5-8. The prompt lacks concrete guidance on
  what qualifies as secondary vs primary. Mitigation: add worked examples
  of secondary levers.

- Hierarchy-direction errors. Some models absorb a general lever into a
  narrow one instead of the reverse. The prompt says "merge specific into
  general" but does not demonstrate it with an example.

- Chain absorption. A model absorbs lever A into B, then absorbs B into C,
  but only C survives — the detail from A is lost. Currently no detection
  or warning for multi-hop absorption chains.
"""


class LeverClassification(str, Enum):
    primary   = "primary"
    secondary = "secondary"
    absorb    = "absorb"
    remove    = "remove"

class LeverClassificationDecision(BaseModel):
    """Minimal per-lever schema. lever_id is assigned by code, not the LLM."""
    classification: Literal["primary", "secondary", "absorb", "remove"] = Field(
        description=(
            "What should happen to this lever: "
            "primary (distinct, essential strategic lever), "
            "secondary (distinct but supporting/operational), "
            "absorb (overlaps another lever — state which lever id it merges into), "
            "or remove (fully redundant)."
        )
    )
    justification: str = Field(
        description="A concise justification for the classification (~80 words). If absorbing, state which lever id it merges into."
    )

class LeverDecision(BaseModel):
    lever_id: str
    classification: Literal["primary", "secondary", "absorb", "remove"]
    justification: str

class InputLever(BaseModel):
    """Represents a single lever loaded from the initial brainstormed file."""
    lever_id: str
    name: str
    consequences: str
    options: List[str]
    review: str

class OutputLever(InputLever):
    """The InputLever and the deduplication justification."""
    classification: Literal["primary", "secondary"] = Field(
        description="Whether this lever is a primary strategic lever or a secondary/supporting one."
    )
    deduplication_justification: str


def _build_compact_history(
    system_message_with_context: str,
    prior_decisions: List[LeverDecision],
) -> List[ChatMessage]:
    """Option C: replace full conversation history with a compact summary in the system message."""
    summary = "\n".join(
        f"- [{d.lever_id}] {d.classification}: {d.justification[:80]}{'...' if len(d.justification) > 80 else ''}"
        for d in prior_decisions
    )
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=(
            f"{system_message_with_context}\n\n"
            f"**Prior decisions (compacted):**\n{summary}"
        )),
    ]


def _call_llm(chat_message_list: List[ChatMessage], llm: LLM) -> dict:
    """Execute a structured LLM call for a single lever classification."""
    sllm = llm.as_structured_llm(LeverClassificationDecision)
    chat_response = sllm.chat(chat_message_list)
    return {"chat_response": chat_response, "metadata": dict(llm.metadata)}


DEDUPLICATE_SYSTEM_PROMPT = """
Evaluate each of the provided strategic levers individually. Classify every lever explicitly into one of:

- primary: Lever is a distinct, essential strategic decision — it directly shapes the project's success or failure. Methodology, governance, and high-stakes execution levers belong here.
- secondary: Lever is distinct and useful but supporting or operational — it matters for delivery but is not a top-level strategic choice. Examples of secondary levers: marketing campaign timing, internal reporting cadence, team communication tooling, documentation formatting standards.
- absorb: Lever overlaps significantly with another lever. Explicitly state the lever ID it should be merged into.
- remove: Lever is fully redundant. Removing it loses no meaningful detail. Use this sparingly.

Provide concise, explicit justifications mentioning lever IDs clearly. Always prefer "absorb" over "remove" to retain important details.

Always provide a justification for the classification. Explain why the lever is distinct from others. Don't use the same uninformative boilerplate.

Respect Hierarchy: When absorbing, merge the more specific lever into the more general one. Don't take the more general lever and absorb it into a narrower one. Also compare a lever against the group of already-merged levers.

Use "primary" only as a last resort — if you genuinely cannot determine a lever's strategic role after reading the full context. Describe what is unclear in the justification.

In a well-formed set of 15 levers, expect 4–8 to be absorbed or removed. If you find zero absorb/remove decisions, reconsider: the input almost always contains near-duplicates. Do not keep every lever.

You must classify and justify **every lever** provided in the input.
"""

@dataclass
class DeduplicateLevers:
    """Holds the results of the deduplication."""
    user_prompt: str
    system_prompt: str
    response: List[LeverDecision]
    deduplicated_levers: List[OutputLever]
    metadata: List[Dict[str, Any]]

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

        system_prompt = DEDUPLICATE_SYSTEM_PROMPT.strip()

        # Build a summary of all levers for comparison context (shared across all per-lever calls).
        all_levers_summary = "\n".join(
            f"- [{lever.lever_id}] {lever.name}: {lever.consequences[:120]}..."
            for lever in input_levers
        )

        decisions: List[LeverDecision] = []
        metadata_list: List[dict] = []

        # Initialise conversation with full context in the system message (option A).
        # System message carries project context + lever summary so the first USER
        # message is the first lever — no dangling USER→USER before the first ASSISTANT.
        system_message_with_context = (
            f"{system_prompt}\n\n"
            f"**Project Context:**\n{project_context}\n\n"
            f"**All levers under review:**\n{all_levers_summary}"
        )
        chat_message_list: List[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_message_with_context),
        ]

        # Closure captures chat_message_list by variable reference, so rebinding
        # after compaction is visible on the next call without redefining the function.
        def execute_function(llm: LLM) -> dict:
            return _call_llm(chat_message_list, llm)

        for lever in input_levers:
            lever_json = json.dumps(lever.model_dump(), indent=2)
            lever_prompt = (
                f"Classify this lever (primary / secondary / absorb / remove) with a justification:\n{lever_json}"
            )
            chat_message_list.append(ChatMessage(role=MessageRole.USER, content=lever_prompt))

            decision: LeverClassificationDecision | None = None
            result = None

            # First attempt with full conversation history.
            try:
                result = llm_executor.run(execute_function)
                metadata_list.append(result.get("metadata", {}))
            except PipelineStopRequested:
                raise
            except Exception as e:
                # Option C: compact history and retry once.
                logger.warning(f"Lever {lever.lever_id}: call failed ({e}). Compacting history and retrying.")
                chat_message_list = _build_compact_history(system_message_with_context, decisions)
                chat_message_list.append(ChatMessage(role=MessageRole.USER, content=lever_prompt))

            # Second attempt with compacted history (only reached if first attempt failed).
            if result is None:
                try:
                    result = llm_executor.run(execute_function)
                    metadata_list.append(result.get("metadata", {}))
                except PipelineStopRequested:
                    raise
                except Exception as e2:
                    logger.warning(f"Lever {lever.lever_id}: failed after compaction ({e2}). Skipping lever.")

            # Process whichever attempt succeeded.
            if result is not None:
                raw = result["chat_response"].raw
                if raw is not None:
                    decision = raw
                    chat_message_list.append(ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=json.dumps({"classification": decision.classification, "justification": decision.justification}),
                    ))
                else:
                    logger.warning(f"Lever {lever.lever_id}: returned None raw.")

            if decision is None:
                logger.warning(f"Lever {lever.lever_id}: classification failed. Defaulting to primary.")
                decision = LeverClassificationDecision(
                    classification=LeverClassification.primary,
                    justification="Classification failed after retries. Keeping this lever to avoid data loss."
                )
                chat_message_list.append(ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps({"classification": decision.classification, "justification": decision.justification}),
                ))

            decisions.append(LeverDecision(
                lever_id=lever.lever_id,
                classification=decision.classification,
                justification=decision.justification,
            ))

        # Perform the deduplication.
        keep_classifications = {LeverClassification.primary, LeverClassification.secondary}
        decisions_by_id = {d.lever_id: d for d in decisions}
        output_levers = []
        for lever in input_levers:
            lever_decision = decisions_by_id.get(lever.lever_id)
            if not lever_decision:
                # Missing decision for this lever. Keep it as primary.
                output_lever = OutputLever(
                    **lever.model_dump(),
                    classification=LeverClassification.primary,
                    deduplication_justification="Missing deduplication justification. Keeping this lever."
                )
                output_levers.append(output_lever)
                continue

            # Only primary and secondary survive
            if lever_decision.classification not in keep_classifications:
                continue

            deduplication_justification = lever_decision.justification.strip()
            if len(deduplication_justification) == 0:
                deduplication_justification = "Empty explanation. Keeping this lever."

            output_lever = OutputLever(
                **lever.model_dump(),
                classification=lever_decision.classification,
                deduplication_justification=deduplication_justification
            )
            output_levers.append(output_lever)

        return cls(
            user_prompt=levers_json,
            system_prompt=system_prompt,
            response=decisions,
            deduplicated_levers=output_levers,
            metadata=metadata_list
        )

    def to_dict(self, include_response=True, include_deduplicated_levers=True, include_metadata=True, include_system_prompt=True, include_user_prompt=True) -> dict:
        d = {}
        if include_response:
            d["response"] = [item.model_dump() for item in self.response]
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

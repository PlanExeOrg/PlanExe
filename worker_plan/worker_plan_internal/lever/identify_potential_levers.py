"""
Brainstorm what key "levers" can be pulled to change the outcome of the plan.

Don’t focus on hitting exactly 15 levers. It’s more important that there is 15..20 levers.
If there are too many levers, I don’t want to blindly discard the extra ones, this is what the deduplicate levers are made for.Downstream there is a deduplicate levers, that gets rid of near duplicates.
It’s more important that the quality of the text content of the levers are getting improved on.

The output contains near duplicates, these have to be deduplicated. A few lever names appear twice.
The deduplication is done in the deduplicate_levers.py script.

PROMPT> python -m worker_plan_internal.lever.identify_potential_levers
"""
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import uuid
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field, field_validator
from llama_index.core.llms import ChatMessage, MessageRole
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

OPTIMIZE_INSTRUCTIONS = """\
Goal: produce levers that lead to realistic, feasible, actionable plans that
humans or AI agents can actually execute.

Pipeline context
----------------
This step (IdentifyPotentialLevers) is part of a 6-step solution-space
exploration pipeline inside run_plan_pipeline.py:

  1. IdentifyPotentialLevers  ← you are here
  2. DeduplicateLevers        — removes near-duplicate levers
  3. EnrichLevers             — adds description, synergy, and conflict text
  4. FocusOnVitalFewLevers    — filters down to 4-6 high-impact levers
  5. ScenarioGeneration       — builds 3 scenarios (aggressive, medium, safe)
  6. ScenarioSelection        — picks the best-fitting scenario

Over-generation is fine; step 2 handles extras. Quality of content matters more
than hitting an exact count.

Known problems to guard against
--------------------------------
- Overly optimistic scenarios. The downstream scenario picker tends to choose
  the most ambitious option unless the levers themselves offer grounded,
  pragmatic choices. Each lever's options should include at least one
  conservative, low-risk path — not just aspirational moonshots.
- Fabricated numbers. Do not invent percentages, cost savings, market-share
  figures, or performance deltas. If the project context supplies a number,
  cite it; otherwise use qualitative language.
- Hype and marketing copy. Words like "game-changing", "revolutionary",
  "cutting-edge", "disruptive", and "breakthrough" erode credibility.
  Use plain, concrete language instead.
- Vague aspirations posing as options. Each option must be a specific,
  actionable approach — something a project manager could actually schedule
  and resource — not a slogan.
- Fragile English-only validation. PlanExe receives initial prompts in many
  non-English languages (Chinese, Japanese, Arabic, German, etc.). Validators
  and auto-correct logic must not rely on English keywords like "Controls",
  "Weakness:", "versus"/"vs." being present in the LLM output. Hard-coded
  English substring checks (e.g. `'Controls ' not in response_str`) will reject
  perfectly valid levers whenever the model responds in the prompt's
  language. Prefer structural checks (field count, JSON shape) over
  language-dependent string matching.
"""

class Lever(BaseModel):
    lever_index: int = Field(
        description="Index of this lever."
    )
    name: str = Field(
        description="Name of this lever."
    )
    consequences: str = Field(
        description=(
            "What happens when this lever is pulled? Describe the direct effect and "
            "at least one downstream implication or trade-off. Be concise and grounded — "
            "only cite numbers if the project context provides evidence for them. "
            "Do NOT include 'Controls ... vs.', 'Weakness:', or other review/critique text in this field — "
            "those belong exclusively in review_lever. "
            "Target length: 2–4 sentences."
        )
    )
    options: list[str] = Field(
        description="Exactly 3 options for this lever. No more, no fewer. Each option must be a complete "
                    "strategic approach (a full sentence with an action verb), not a label."
    )
    review_lever: str = Field(
        description=(
            "Two sentences. First sentence names the core tension this lever "
            "controls. Second sentence identifies a weakness the options miss. "
            "Example: 'Controls centralization vs. local autonomy. "
            "Weakness: The options fail to account for transition costs.' "
            "Do not use square brackets or placeholder text."
        )
    )

    @field_validator('options', mode='before')
    @classmethod
    def parse_options(cls, v):
        """Handle cases where LLMs return options as a stringified JSON array."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return v

    @field_validator('options', mode='after')
    @classmethod
    def check_option_count(cls, v):
        """Reject levers that don't have exactly 3 options.

        Run 89 (llama, hong_kong_game) produced levers with 2 options that
        silently passed validation and shipped to downstream tasks which
        assume exactly 3 options per lever.
        """
        if len(v) != 3:
            raise ValueError(f"options must have exactly 3 items, got {len(v)}")
        return v

    @field_validator('review_lever', mode='after')
    @classmethod
    def check_review_format(cls, v):
        """Structural validation only — no English keyword checks.

        PlanExe receives prompts in many non-English languages, so the
        validator must not rely on English markers like "Controls" or
        "Weakness:". Instead we enforce structural properties:
        - minimum length (at least 20 characters)
        - no square-bracket placeholders (e.g. [Tension A])
        """
        if len(v) < 20:
            raise ValueError(f"review_lever is too short ({len(v)} chars); expected at least 20")
        if '[' in v or ']' in v:
            raise ValueError("review_lever must not contain square-bracket placeholders")
        return v

class DocumentDetails(BaseModel):
    strategic_rationale: Optional[str] = Field(
        default=None,
        description="A concise strategic analysis (around 100 words) of the project's core tensions and trade-offs. This rationale must JUSTIFY why the selected levers are the most critical levers for decision-making. For example, explain how the chosen levers navigate the fundamental conflicts between speed, cost, scope, and quality."
    )
    # No max_length constraint: if a model returns more than 7 levers, the downstream
    # DeduplicateLeversTask handles extras. A hard cap would discard the entire response
    # and waste tokens retrying.
    levers: list[Lever] = Field(
        min_length=5,
        description="Propose 5 to 7 levers."
    )
    summary: str = Field(
        description="One sentence prescribing a concrete addition to a specific lever. Example: \"Add 'partner with a regional distributor' to Supply Chain Strategy.\""
    )

class LeverCleaned(BaseModel):
    """
    The Lever class has some ugly field names, that guide the LLM for what to generate. Changing them and the LLM can't generate as good results.
    This class has nicer field names for the final output.
    """
    lever_id: str = Field(
        description="A uuid that identifies this lever. The levers can be deduplicated and preserve their lever_id without leaving gaps in the numbering."
    )
    name: str = Field(
        description="Name of this lever."
    )
    consequences: str = Field(
        description=(
            "What happens when this lever is pulled? Describe the direct effect and "
            "at least one downstream implication or trade-off. Be concise and grounded — "
            "only cite numbers if the project context provides evidence for them. "
            "Do NOT include 'Controls ... vs.', 'Weakness:', or other review/critique text in this field — "
            "those belong exclusively in review_lever. "
            "Target length: 2–4 sentences."
        )
    )
    options: list[str] = Field(
        description="Exactly 3 options for this lever. No more, no fewer. Each option must be a complete "
                    "strategic approach (a full sentence with an action verb), not a label."
    )
    review: str = Field(
        description=(
            "Two sentences. First sentence names the core tension this lever "
            "controls. Second sentence identifies a weakness the options miss. "
            "Example: 'Controls centralization vs. local autonomy. "
            "Weakness: The options fail to account for transition costs.' "
            "Do not use square brackets or placeholder text."
        )
    )

IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT = """
You are an expert strategic analyst. Generate solution space parameters following these directives:

1. **Output Requirements**
   - You must generate 5 to 7 levers per response.
   - Each lever's `options` field must contain exactly 3 qualitative strategic choices as plain strings.

2. **Lever Quality Standards**
   - Consequences: describe the direct effect of pulling this lever, then at least one downstream implication or trade-off. Be concise and grounded — only cite specific numbers if the project context provides evidence for them. Do not fabricate percentages or cost estimates. Target length: 2–4 sentences.
   - Options MUST:
     • Represent genuinely distinct strategic pathways (not just labels)
     • Include at least one unconventional or non-obvious approach
     • NO prefixes (e.g., "Option A:", "Choice 1:")

3. **Strategic Framing**
   - Name each lever using language drawn directly from the project's own domain — avoid formulaic patterns or repeated prefixes
   - Frame options as complete strategic approaches
   - Ensure levers challenge core project assumptions

4. **Validation Protocols**
   - For `review_lever` (one field, two sentences):
     First sentence names the core tension. Second sentence identifies a weakness.
     Example: "Controls centralization vs. local autonomy. Weakness: The options fail to account for transition costs."
     Do not use square brackets or placeholder text.
   - For `summary`:
     One sentence prescribing a concrete addition to a specific lever.
     Example: "Add 'partner with a regional distributor for last-mile logistics' to Supply Chain Strategy."

5. **Prohibitions**
   - NO prefixes/labels in options (e.g., "Option A:", "Choice 1:")
   - NO generic option labels (e.g., "Optimize X", "Tolerate Y")
   - NO placeholder consequences or bracket-wrapped templates
   - NO fabricated statistics or percentages without evidence from the project context
   - NO marketing language (e.g., "game-changing", "cutting-edge", "revolutionary")

6. **Option Structure**
   - Maintain parallel grammatical structure across options
   - Ensure options are self-contained descriptions
   - Each option should be a concrete, actionable approach — not a vague aspiration
"""

@dataclass
class IdentifyPotentialLevers:
    system_prompt: Optional[str]
    user_prompt: str
    responses: list[DocumentDetails]
    levers: list[LeverCleaned]
    metadata: dict

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, user_prompt: str, system_prompt: Optional[str] = None) -> 'IdentifyPotentialLevers':
        if not isinstance(llm_executor, LLMExecutor):
            raise ValueError("Invalid LLMExecutor instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        if system_prompt is None:
            system_prompt = IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT.strip()
        else:
            system_prompt = system_prompt.strip()
        system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

        total_calls = 3
        responses: list[DocumentDetails] = []
        metadata_list: list[dict] = []
        generated_lever_names: list[str] = []

        for call_index in range(1, total_calls + 1):
            if call_index == 1:
                prompt_content = user_prompt
            else:
                names_list = ", ".join(f'"{n}"' for n in generated_lever_names)
                prompt_content = (
                    f"Generate 5 to 7 MORE levers with completely different names. "
                    f"Do NOT reuse any of these already-generated names: [{names_list}]\n"
                    f"Each option must be a complete strategic sentence (at least 15 words with an action verb), not a short label.\n"
                    f"Do not invent percentages, cost savings, or performance deltas — use qualitative language unless the project document supplies the number.\n\n"
                    f"{user_prompt}"
                )

            logger.info(f"Processing call {call_index} of {total_calls}")
            call_messages = [
                system_message,
                ChatMessage(
                    role=MessageRole.USER,
                    content=prompt_content,
                ),
            ]

            messages_snapshot = list(call_messages)

            def execute_function(llm: LLM) -> dict:
                sllm = llm.as_structured_llm(DocumentDetails)
                chat_response = sllm.chat(messages_snapshot)
                metadata = dict(llm.metadata)
                metadata["llm_classname"] = llm.class_name()
                return {
                    "chat_response": chat_response,
                    "metadata": metadata
                }

            try:
                result = llm_executor.run(execute_function)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                llm_error = LLMChatError(cause=e)
                logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
                logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
                # If earlier calls succeeded, keep their levers instead of
                # discarding everything. A single validator rejection (e.g.,
                # one lever with 2 options) should not wipe out 10+ valid
                # levers from prior calls.
                if len(responses) == 0:
                    raise llm_error from e
                logger.warning(
                    f"Call {call_index} of {total_calls} failed [{llm_error.error_id}], "
                    f"continuing with {len(responses)} prior call(s)."
                )
                continue

            generated_lever_names.extend(lever.name for lever in result["chat_response"].raw.levers)
            responses.append(result["chat_response"].raw)
            metadata_list.append(result["metadata"])

        # from the raw_responses, extract the levers into a flatten list
        levers_raw: list[Lever] = []
        for response in responses:
            levers_raw.extend(response.levers)

        # Clean the raw levers, skipping duplicates
        seen_names: set[str] = set()
        levers_cleaned: list[LeverCleaned] = []
        for i, lever in enumerate(levers_raw, start=1):
            if lever.name in seen_names:
                logger.warning(f"Duplicate lever name '{lever.name}', skipping.")
                continue
            seen_names.add(lever.name)

            lever_id = str(uuid.uuid4())
            lever_cleaned = LeverCleaned(
                lever_id=lever_id,
                name=lever.name,
                consequences=lever.consequences,
                options=lever.options,
                review=lever.review_lever,
            )
            levers_cleaned.append(lever_cleaned)

        metadata = {}
        for metadata_index, metadata_item in enumerate(metadata_list, start=1):
            metadata[f"metadata_{metadata_index}"] = metadata_item

        result = IdentifyPotentialLevers(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            responses=responses,
            levers=levers_cleaned,
            metadata=metadata,
        )
        return result    

    def to_dict(self, include_responses=True, include_cleaned_levers=True, include_metadata=True, include_system_prompt=True, include_user_prompt=True) -> dict:
        d = {}
        if include_responses:
            d["responses"] = [response.model_dump() for response in self.responses]
        if include_cleaned_levers:
            d['levers'] = [lever.model_dump() for lever in self.levers]
        if include_metadata:
            d['metadata'] = self.metadata
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        if include_user_prompt:
            d['user_prompt'] = self.user_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2))

    def lever_item_list(self) -> list[dict]:
        """
        Return a list of dictionaries, each representing a lever.
        """
        return [lever.model_dump() for lever in self.levers]
    
    def save_clean(self, file_path: str) -> None:
        levers_dict = self.lever_item_list()
        Path(file_path).write_text(json.dumps(levers_dict, indent=2))
    
if __name__ == "__main__":
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName
    from worker_plan_internal.prompt.prompt_catalog import PromptCatalog

    logging.basicConfig(level=logging.DEBUG)

    prompt_catalog = PromptCatalog()
    prompt_catalog.load_simple_plan_prompts()

    # prompt_id = "b9afce6c-f98d-4e9d-8525-267a9d153b51"
    # prompt_id = "a6bef08b-c768-4616-bc28-7503244eff02"
    # prompt_id = "19dc0718-3df7-48e3-b06d-e2c664ecc07d"
    prompt_id = "e42eafce-5c8c-4801-b9f1-b8b2a402cd78"
    prompt_item = prompt_catalog.find(prompt_id)
    if not prompt_item:
        raise ValueError("Prompt item not found.")
    query = prompt_item.prompt

    model_names = [
        "ollama-llama3.1",
        # "openrouter-paid-gemini-2.0-flash-001",
        # "openrouter-paid-qwen3-30b-a3b"
    ]
    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    print(f"Query: {query}")
    result = IdentifyPotentialLevers.execute(llm_executor, query)

    print("\nResult:")
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print(json.dumps(json_response, indent=2))

    test_data_filename = f"identify_potential_levers_{prompt_id}.json"
    result.save_clean(Path(test_data_filename))
    print(f"Test data saved to: {test_data_filename!r}")

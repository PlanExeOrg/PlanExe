"""
Brainstorm what key "levers" can be pulled to change the outcome of the plan.

Don’t focus on hitting exactly 15 levers. It’s more important that there is 15..20 levers.
If there are too many levers, I don’t want to blindly discard the extra ones, this is what the deduplicate levers are made for.Downstream there is a deduplicate levers, that gets rid of near duplicates.
It’s more important that the quality of the text content of the levers are getting improved on.

The output contains near duplicates, these have to be deduplicated. A few lever names appear twice.
The triage is done in the triage_levers.py script.

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
  2. TriageLevers             — removes near-duplicate levers
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
- Single-example template lock. When the prompt provides exactly one
  review_lever example, weaker models reproduce that exact syntax 90–100%
  of the time. Always provide at least two structurally distinct examples
  to give models variety to draw from.
- Template-lock migration. Replacing a copyable opener does not eliminate
  template lock — weaker models shift to copying subphrases within the
  new examples (e.g. "the options neglect", "the options assume").
  Examples must avoid reusable transitional phrases that fit any domain.
  Each example must name a domain-specific mechanism or constraint
  directly rather than referencing "the options" as grammatical subject.
  No two examples should share a sentence pattern or rhetorical
  structure. Span at least three distinct domains. Do NOT add explicit
  prohibitions naming banned phrases — small models treat the
  prohibition text as a template and copy the banned phrases.
- Verbosity amplification. Models mirror example verbosity, not just
  structure. Keep review_lever examples concise and enforce a length
  cap in the system prompt to prevent output overflow.
- Field-description template lock. A Pydantic field description
  containing a structural phrase (e.g. "name the core tension", "the
  gap the three options leave unaddressed") is read as a literal
  instruction — models start every output with that exact phrase.
  Replacing one structural phrase with another just migrates the lock
  (confirmed: "the proposed options collectively do not resolve" caused
  qwen3-30b to lock at 7/20 and worsened gpt-oss-20b). The fix is to
  strip the field description to content-type and word count only
  (e.g. "Critical review of this lever (one sentence, 20–40 words)")
  and let the system-prompt examples teach the style. PR #484 proved
  this: haiku template lock 24.7% → 0% with zero regressions.
- Negative prohibitions activate the banned pattern. "Do NOT include
  'Controls ... vs.'" causes small models (llama3.1) to copy the
  banned phrase. "Never invent percentages" and "NO fabricated
  statistics" both increase fabricated numbers by drawing model
  attention to numbers. Use positive framing instead: "Save critical
  assessments for the review_lever field" or "only cite numbers the
  project context provides directly." Confirmed across PRs #458,
  #460, #475.
- Consequence parroting in later calls. When the adaptive loop makes
  calls 2+, the subsequent-call prompt focuses on name novelty with no
  review-quality guidance. llama3.1 copies the consequences field
  verbatim into review_lever with only modal verb substitution
  ("could" → "can"). An anti-parrot sentence in the subsequent-call
  prompt eliminates this for call-2+ but does not reach call-1.
- Field descriptions vs system prompt: keep both consistent. If the
  field description says "2–3 sentences" but the system prompt says
  "2–4 sentences", haiku follows the system prompt (the looser
  target). Models that use structured-output JSON schemas (haiku,
  gpt-4o-mini) weight the field description; text-completion models
  (llama3.1, gpt-oss-20b) weight the system prompt. Both must agree.
- Stripping field descriptions too far breaks weak models. Removing
  structural anchors like "one sentence" and "See system prompt
  section 4 for examples" from review_lever causes llama3.1 to
  produce exact-copy parrots (review ≡ consequence). Weak models
  need minimal structural cues in the field description; strong
  models can work from system-prompt examples alone. The sweet spot:
  content-type + word count + section pointer, no sentence templates.
"""

class Lever(BaseModel):
    """Schema sent to the LLM for structured output generation.

    Field names and descriptions here directly influence LLM output quality.
    Renaming fields or changing descriptions is a prompt change — test with
    a self_improve iteration before merging. See OPTIMIZE_INSTRUCTIONS above.
    """
    # Counting aid: models use this to track how many levers they've generated.
    # Without it, models over-generate 8-12 per call instead of 5-7.
    # Not passed to LeverCleaned — its value is in regulating output count.
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
            "Critical review of this lever (one sentence, 20–40 words). "
            "See system prompt section 4 for examples. "
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
        """Reject levers with fewer than 3 options.

        Some models produce levers with 2 options that
        silently passed validation and shipped to downstream tasks which
        assume at least 3 options per lever. Over-generation (>3) is
        tolerable; under-generation is not.
        """
        if len(v) < 3:
            raise ValueError(f"options must have at least 3 items, got {len(v)}")
        return v

    @field_validator('review_lever', mode='after')
    @classmethod
    def check_review_format(cls, v):
        """Structural validation only — no English keyword checks.

        PlanExe receives prompts in many non-English languages, so the
        validator must not rely on English markers like "Controls" or
        "Weakness:". Instead we enforce structural properties:
        - minimum length: 10 chars (hard limit). The enrich step adds
          detail later, so 50 chars is the soft target but not enforced here.
        - no square-bracket placeholders (e.g. [Tension A])
        """
        if len(v) < 10:
            raise ValueError(f"review_lever is too short ({len(v)} chars); expected at least 10")
        if '[' in v or ']' in v:
            raise ValueError("review_lever must not contain square-bracket placeholders")
        return v

class DocumentDetails(BaseModel):
    # Chain-of-thought field: forces the LLM to reason about the project's
    # trade-offs before generating levers. Not passed downstream — its value
    # is in improving lever quality, not in being read. Do NOT remove.
    strategic_rationale: Optional[str] = Field(
        default=None,
        description="A concise strategic analysis (around 100 words) of the project's core tensions and trade-offs. This rationale must JUSTIFY why the selected levers are the most critical levers for decision-making. For example, explain how the chosen levers navigate the fundamental conflicts between speed, cost, scope, and quality."
    )
    # No max_length constraint: if a model returns more than 7 levers, the downstream
    # TriageLeversTask handles extras. A hard cap would discard the entire response
    # and waste tokens retrying.
    levers: list[Lever] = Field(
        min_length=5,
        description="Propose 5 to 7 levers."
    )

class LeverCleaned(BaseModel):
    """Cleaned output schema — never sent to an LLM.

    Maps Lever's LLM-facing field names (e.g. review_lever) to cleaner
    names (e.g. review) for downstream consumption. Field descriptions
    here are for documentation only and have no effect on LLM output.
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
        description="Critical review of this lever."
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
   - For `review_lever`:
     A one-sentence critical review (20–40 words).
     Examples:
     - "Switching from seasonal contract labor to year-round employees stabilizes harvest quality, but the idle-wage burden during the 5-month off-season adds a fixed cost that erases the per-unit savings unless utilization reaches year-round levels."
     - "Each additional clinical site requires its own IRB approval, site-initiation visit, and staff credentialing — a sequential overhead that compounds rather than parallelizes, so doubling site count does not halve enrollment time."
     - "Pooling catastrophe risk across three coastal regions reduces expected annual loss on paper, but a single regional hurricane season can correlate all three simultaneously, turning the diversification assumption into a concentration risk at the worst possible moment."
     Do not use square brackets or placeholder text.

5. **Prohibitions**
   - NO prefixes/labels in options (e.g., "Option A:", "Choice 1:")
   - NO generic option labels (e.g., "Optimize X", "Tolerate Y")
   - NO placeholder consequences or bracket-wrapped templates
   - NO fabricated statistics or percentages without evidence from the project context
   - NO marketing language (e.g., "game-changing", "cutting-edge", "revolutionary")

6. **Length Limits**
   - Keep each `review_lever` to one sentence (20–40 words). State the trade-off and the gap concisely.
   - Each option should be a concrete, actionable approach (at least 15 words with an action verb) — not a short label or vague aspiration
   - Maintain parallel grammatical structure across options
   - Ensure options are self-contained descriptions
"""

def raise_if_no_levers_survived(levers_cleaned: list, all_constraint_checks: list[dict]) -> None:
    """Fail loud when constraint checking rejected every generated lever.

    This happens when the plan prompt is self-contradictory: a negative
    constraint bans a concept that is also the plan's core subject. The
    clearest example is a plan explicitly about "AI agents" that lists "AI"
    among its banned words — the ConstraintChecker then flags every lever as
    violating "Do not use AI", so all of them are filtered out and nothing
    remains.

    Returning an empty list here is dangerous: PotentialLeversTask would
    "succeed" with an empty potential_levers.json, and the failure would only
    surface two stages later in TriageLeversTask as the misleading "No input
    levers to deduplicate." Raising here blames the real cause and names the
    constraint(s) responsible.

    TODO: Detect a self-contradictory prompt earlier — ideally during
    constraint extraction or at the redline gate — so the run stops before
    spending tokens generating levers that are guaranteed to be rejected by
    their own banned words.
    """
    if levers_cleaned:
        return

    # Count how many distinct levers each constraint rejected. A single lever's
    # check may list the same constraint more than once, so collapse to a set
    # per lever before tallying — otherwise the "rejected N lever(s)" count can
    # exceed the number of levers generated.
    violation_counts: dict[str, int] = {}
    for check in all_constraint_checks:
        rejected_by = {
            violation.get("constraint_text", "(unknown constraint)")
            for violation in check.get("constraint_violations", [])
            if violation.get("status") == "violated"
        }
        for text in rejected_by:
            violation_counts[text] = violation_counts.get(text, 0) + 1

    generated_count = len(all_constraint_checks)
    if violation_counts:
        ranked = sorted(violation_counts.items(), key=lambda item: item[1], reverse=True)
        dominant = "; ".join(f'"{text}" (rejected {count} lever(s))' for text, count in ranked[:3])
        raise ValueError(
            f"All {generated_count} generated levers were rejected by the constraint check, "
            f"leaving none for the downstream tasks. Most rejections came from: {dominant}. "
            f"This usually means a negative constraint contradicts the plan's core subject "
            f"(e.g. banning \"AI\" in a plan that is explicitly about AI agents). "
            f"Revise the plan's banned words / negative constraints so they do not exclude the plan's main topic."
        )
    raise ValueError(
        f"IdentifyPotentialLevers produced 0 levers "
        f"(generated {generated_count}, none survived filtering)."
    )


@dataclass
class IdentifyPotentialLevers:
    system_prompt: Optional[str]
    user_prompt: str
    responses: list[DocumentDetails]
    levers: list[LeverCleaned]
    metadata: dict
    constraint_checks: list[dict] = None  # Constraint checker results per lever

    @classmethod
    def _check_constraints_on_response(
        cls,
        llm_executor: LLMExecutor,
        constraints_markdown: str,
        response_doc: 'DocumentDetails',
    ) -> tuple[list['Lever'], list[tuple['Lever', list[dict]]], list[dict]]:
        """Run ConstraintChecker on each lever individually.

        Returns (accepted_levers, rejected_levers, all_check_results) where:
        - accepted_levers: levers that passed the constraint check
        - rejected_levers: list of (lever, violations) tuples
        - all_check_results: full constraint checker response for each lever
        """
        from worker_plan_internal.diagnostics.constraint_checker import ConstraintChecker

        constraints_json = json.dumps({"constraints_markdown": constraints_markdown})

        accepted: list[Lever] = []
        rejected: list[tuple[Lever, list[dict]]] = []
        all_check_results: list[dict] = []

        for lever in response_doc.levers:
            lever_json = json.dumps({
                "lever_name": lever.name,
                "consequences": lever.consequences,
                "options": lever.options,
                "review": lever.review_lever,
            }, indent=2)

            # Capture loop variables in a closure that returns a single-arg function,
            # because LLMExecutor validates that execute_function has exactly 1 parameter.
            cj, lj, ln = constraints_json, lever_json, lever.name
            def make_check_function(cj, lj, ln):
                def check_function(llm: LLM) -> dict:
                    result = ConstraintChecker.execute(llm, cj, lj, f"lever: {ln}")
                    return result.response
                return check_function

            try:
                check_result = llm_executor.run(make_check_function(cj, lj, ln))
            except Exception as e:
                logger.warning(f"Constraint check failed for lever '{lever.name}': {e}. Accepting lever.")
                accepted.append(lever)
                all_check_results.append({
                    "lever_name": lever.name,
                    "status": "error",
                    "error": str(e),
                })
                continue

            all_check_results.append({
                "lever_name": lever.name,
                **check_result,
            })

            violations = [
                v for v in check_result.get("constraint_violations", [])
                if v.get("status") == "violated"
            ]
            if violations:
                rejected.append((lever, violations))
            else:
                accepted.append(lever)

        return accepted, rejected, all_check_results

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, user_prompt: str, constraints_markdown: str = "") -> 'IdentifyPotentialLevers':
        if not isinstance(llm_executor, LLMExecutor):
            raise ValueError("Invalid LLMExecutor instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        system_prompt = IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT.strip()
        system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

        # Adaptive loop: keep calling until we have enough levers.
        # Over-generation is fine — TriageLeversTask handles extras.
        min_levers = 15
        max_calls = 5
        responses: list[DocumentDetails] = []
        metadata_list: list[dict] = []
        generated_lever_names: list[str] = []
        # Accumulate constraint violation feedback across iterations so the
        # LLM learns from repeated rejections (e.g. "VR was rejected 3 times").
        constraint_rejection_history: list[str] = []
        all_constraint_checks: list[dict] = []

        for call_index in range(1, max_calls + 1):
            if call_index == 1:
                prompt_content = user_prompt
            else:
                names_list = ", ".join(f'"{n}"' for n in generated_lever_names)
                prompt_content = (
                    f"Generate 5 to 7 MORE levers with completely different names. "
                    f"Do NOT reuse any of these already-generated names: [{names_list}]\n\n"
                    f"{user_prompt}"
                )

            # Include accumulated constraint rejection feedback so the LLM
            # knows which levers were rejected and why.
            if constraint_rejection_history:
                rejection_text = "\n".join(constraint_rejection_history)
                prompt_content += (
                    f"\n\n## Constraint Violation History\n"
                    f"The following levers were REJECTED for violating constraints. "
                    f"Do NOT generate levers that repeat these violations:\n{rejection_text}"
                )

            logger.info(f"Processing call {call_index} of {max_calls} (have {len(generated_lever_names)} levers, need {min_levers})")
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
                # Let the adaptive loop retry — even first-call failures
                # may be stochastic (e.g. llama3.1 producing 2 options on
                # one attempt but 3 on the next). The loop has max_calls=5,
                # so persistent failures will exhaust retries naturally.
                if len(responses) == 0 and call_index == max_calls:
                    raise llm_error from e
                logger.warning(
                    f"Call {call_index} of {max_calls} failed [{llm_error.error_id}], "
                    f"continuing with {len(responses)} prior call(s)."
                )
                continue

            # Run constraint check on the response if constraints were provided.
            response_doc: DocumentDetails = result["chat_response"].raw
            if constraints_markdown.strip():
                accepted_levers, rejected_levers, check_results = cls._check_constraints_on_response(
                    llm_executor, constraints_markdown, response_doc
                )
                all_constraint_checks.extend(check_results)
                if rejected_levers:
                    for lever, violations in rejected_levers:
                        violation_details = "; ".join(
                            f"{v['constraint_text']}: {v['explanation']}"
                            for v in violations
                        )
                        constraint_rejection_history.append(
                            f"- Lever \"{lever.name}\" REJECTED: {violation_details}"
                        )
                    logger.info(
                        f"Call {call_index}: {len(rejected_levers)} lever(s) rejected by constraint check, "
                        f"{len(accepted_levers)} accepted."
                    )
                    # Use model_copy to skip revalidation: DocumentDetails.levers has
                    # min_length=5 to enforce LLM output quality, but our post-filter
                    # set may legitimately be smaller — the adaptive loop will top up
                    # via additional calls until min_levers is met.
                    response_doc = response_doc.model_copy(update={"levers": accepted_levers})

            generated_lever_names.extend(lever.name for lever in response_doc.levers)
            responses.append(response_doc)
            metadata_list.append(result["metadata"])

            if len(generated_lever_names) >= min_levers:
                logger.info(f"Reached {len(generated_lever_names)} levers after {call_index} calls, stopping.")
                break

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

        # Fail loud when constraint checking rejected every generated lever,
        # rather than silently returning an empty list (see
        # raise_if_no_levers_survived for why this matters).
        raise_if_no_levers_survived(levers_cleaned, all_constraint_checks)

        metadata = {}
        for metadata_index, metadata_item in enumerate(metadata_list, start=1):
            metadata[f"metadata_{metadata_index}"] = metadata_item

        result = IdentifyPotentialLevers(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            responses=responses,
            levers=levers_cleaned,
            metadata=metadata,
            constraint_checks=all_constraint_checks if all_constraint_checks else None,
        )
        return result

    def to_dict(self, include_responses=True, include_cleaned_levers=True, include_metadata=True, include_system_prompt=True, include_user_prompt=True) -> dict:
        d = {}
        if include_responses:
            d["responses"] = [response.model_dump() for response in self.responses]
        if include_cleaned_levers:
            d['levers'] = [lever.model_dump() for lever in self.levers]
        if self.constraint_checks is not None:
            d['constraint_checks'] = self.constraint_checks
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

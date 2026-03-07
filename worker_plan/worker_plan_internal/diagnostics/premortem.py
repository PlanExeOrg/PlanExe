"""
Premortem: "If we fail, here's how and why."

Imagine that the project has failed, and work backwards to identify plausible reasons why.

https://en.wikipedia.org/wiki/Pre-mortem
Premortem is a risk assessment method by Gary A. Klein
https://en.wikipedia.org/wiki/Gary_A._Klein

PROMPT> python -m worker_plan_internal.diagnostics.premortem

`assumptions_to_kill` are the INPUTS. They are the foundational beliefs held before the project begins. They represent the project's
most significant areas of uncertainty. The list of assumptions is, in itself, a high-value deliverable for a project kickoff.
It's the "here's what we believe to be true, but we need to prove it" list.

`failure_modes` are the potential OUTCOMES. They are the narrative stories of what could happen if an assumption proves false.
They explore the consequences and the causal chain of failure.

IDEA: Focus on top 3 failure modes. All failure modes are rated "High" or "Critical", which dilutes prioritization. This risks overwhelming the team with too many "critical" focus areas. Rank failure modes by priority (e.g., top 3: FM5, FM1, FM6) and allocate resources accordingly.

IDEA: The "Response Playbook" uses the "Contain, Assess, Respond" model. Enhance with a field for "Proactive Mitigation." The playbook is for when a tripwire is hit (reactive). Proactive mitigation would be the actions taken beforehand to prevent the tripwire from ever being hit. For example, for "The Empty Wallet Wasteland", the proactive mitigation is "Conduct a detailed bottom-up cost estimation." This task should be in the project plan from day one because of the risk identified in the Premortem.

IDEA: Add a recurring risk review cadence (e.g., quarterly) to update assumptions and tripwires based on new data.

IDEA: The premortem assumes a static risk landscape.

IDEA: add a low-probability, high-impact "external shock" scenario, "black swan" scenario.

IDEA: Use a reasoning model to validate the premortem section and fix issues.

"""
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested
from worker_plan_api.speedvsdetail import SpeedVsDetailEnum

logger = logging.getLogger(__name__)

class AssumptionItem(BaseModel):
    assumption_id: str = Field(description="A unique ID for this assumption, e.g. 'A1', 'A2', 'A3'.")
    statement: str = Field(description="The core assumption we are making that, if false, would kill the project.")
    test_now: str = Field(description="A concrete, immediate action to test if this assumption is true.")
    falsifier: str = Field(description="The specific result from the test that would prove the assumption false.")

class FailureModeItem(BaseModel):
    failure_mode_index: int = Field(description="Index of this failure mode, starting from 1.")
    root_cause_assumption_id: str = Field(description="The assumption_id of the assumption that is the root cause of this failure (e.g. 'A1').")
    failure_mode_archetype: str = Field(description="The archetype: 'Process/Financial', 'Technical/Logistical', or 'Market/Human'.")
    failure_mode_title: str = Field(description="A compelling, story-like title (e.g. 'The Gridlock Gamble').")
    risk_analysis: str = Field(description="Factual breakdown of causes, contributing factors, and impacts.")
    early_warning_signs: List[str] = Field(description="Measurable indicators that this failure may occur.")
    owner: Optional[str] = Field(None, description="The single role who owns this risk.")
    likelihood_5: Optional[int] = Field(None, description="Integer 1-5: likelihood of this failure occurring.")
    impact_5: Optional[int] = Field(None, description="Integer 1-5: impact if this failure occurs.")
    tripwires: Optional[List[str]] = Field(None, description="2-3 measurable thresholds signalling imminent failure (e.g. 'Permit delays exceed 90 days').")
    playbook: Optional[List[str]] = Field(None, description="Exactly 3 imperative actions: Contain, Assess, Respond.")
    stop_rule: Optional[str] = Field(None, description="Hard stop condition that would trigger project cancellation or major pivot.")

class ArchetypeAnalysis(BaseModel):
    """Single-archetype premortem: one assumption + one failure mode. Used per-call to reduce schema complexity."""
    assumption: AssumptionItem = Field(description="One critical assumption underlying this failure archetype.")
    failure_mode: FailureModeItem = Field(description="One failure mode for this archetype, linked to the assumption above.")

class PremortemAnalysis(BaseModel):
    assumptions_to_kill: List[AssumptionItem] = Field(description="Critical assumptions to test immediately.")
    failure_modes: List[FailureModeItem] = Field(description="Failure mode stories, one per archetype.")

ARCHETYPES = [
    ("Process/Financial", "A1", 1),
    ("Technical/Logistical", "A2", 2),
    ("Market/Human", "A3", 3),
]

PREMORTEM_SYSTEM_PROMPT_TEMPLATE = """
You are a senior project analyst. Imagine the project has failed completely.

Your task: analyse a single failure archetype - {archetype}.

Generate exactly ONE assumption and ONE failure mode for this archetype as a JSON object with two keys:
- "assumption": one AssumptionItem (assumption_id="{assumption_id}", statement, test_now, falsifier)
- "failure_mode": one FailureModeItem (failure_mode_index={index}, root_cause_assumption_id="{assumption_id}", failure_mode_archetype="{archetype}", plus all other fields)

Rules:
- failure_mode_archetype MUST be "{archetype}"
- root_cause_assumption_id MUST be "{assumption_id}"
- tripwires must use numeric thresholds (e.g. "Permit delays exceed 90 days")
- playbook must have exactly 3 steps: Contain, Assess, Respond
- Output ONLY the JSON object. No markdown, no explanation. Start with {{ and end with }}.
"""

@dataclass
class Premortem:
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, speed_vs_detail: SpeedVsDetailEnum, user_prompt: str) -> 'Premortem':
        if not isinstance(llm_executor, LLMExecutor):
            raise ValueError("Invalid LLMExecutor instance.")
        if not isinstance(speed_vs_detail, SpeedVsDetailEnum):
            raise ValueError("Invalid SpeedVsDetailEnum instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        # Decomposed approach: one independent call per archetype.
        # Each call produces exactly one assumption + one failure mode, eliminating
        # cross-linking constraints and reducing schema complexity per call.
        archetypes_to_run = ARCHETYPES
        if speed_vs_detail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            archetypes_to_run = ARCHETYPES[:1]
            logger.info("Running in FAST_BUT_SKIP_DETAILS mode. Only first archetype.")
        else:
            logger.info(f"Running in ALL_DETAILS_BUT_SLOW mode. Processing {len(ARCHETYPES)} archetypes.")

        assumptions_to_kill: list[AssumptionItem] = []
        failure_modes: list[FailureModeItem] = []
        metadata_list: list[dict] = []

        for archetype, assumption_id, index in archetypes_to_run:
            logger.info(f"Processing archetype {index}/{len(archetypes_to_run)}: {archetype}")
            system_prompt = PREMORTEM_SYSTEM_PROMPT_TEMPLATE.format(
                archetype=archetype,
                assumption_id=assumption_id,
                index=index,
            ).strip()

            chat_message_list = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]

            def execute_function(llm: LLM, _chat=chat_message_list) -> dict:
                sllm = llm.as_structured_llm(ArchetypeAnalysis)
                start_time = time.perf_counter()
                chat_response = sllm.chat(_chat)
                pydantic_response = chat_response.raw
                end_time = time.perf_counter()
                duration = int(ceil(end_time - start_time))
                metadata = dict(llm.metadata)
                metadata["llm_classname"] = llm.class_name()
                metadata["duration"] = duration
                return {
                    "pydantic_response": pydantic_response,
                    "metadata": metadata,
                    "duration": duration,
                }

            MAX_RETRIES = 5
            archetype_result = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = llm_executor.run(execute_function)
                    archetype_result = result["pydantic_response"]
                    metadata_list.append(result["metadata"])
                    logger.info(f"Archetype {archetype} succeeded on attempt {attempt}.")
                    break
                except PipelineStopRequested:
                    raise
                except Exception as e:
                    logger.warning(f"Archetype {archetype} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    if attempt == MAX_RETRIES:
                        logger.warning(f"Archetype {archetype} exhausted {MAX_RETRIES} attempts. Skipping — premortem will be partial.")

            if archetype_result is None:
                continue

            assumptions_to_kill.append(archetype_result.assumption)
            failure_modes.append(archetype_result.failure_mode)

        final_response = PremortemAnalysis(
            assumptions_to_kill=assumptions_to_kill,
            failure_modes=failure_modes
        )

        json_response = final_response.model_dump()
        response_byte_count = len(json.dumps(json_response).encode('utf-8'))

        logger.info(f"LLM chat interaction completed. Response byte count: {response_byte_count}")

        metadata = {}
        metadata["models"] = metadata_list
        metadata["response_byte_count"] = response_byte_count

        markdown = cls.convert_to_markdown(final_response)

        return Premortem(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            markdown=markdown
        )

    def to_dict(self, include_metadata=True, include_system_prompt=True, include_user_prompt=True, include_markdown=True) -> dict:
        d = self.response.copy()
        if include_metadata:
            d['metadata'] = self.metadata
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        if include_user_prompt:
            d['user_prompt'] = self.user_prompt
        if include_markdown:
            d['markdown'] = self.markdown
        return d

    def save_raw(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=2))

    def save_markdown(self, output_file_path: str):
        """Save the markdown output to a file."""
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

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
    def _calculate_risk_level_brief(likelihood: Optional[int], impact: Optional[int]) -> str:
        """Calculates a qualitative risk level from likelihood and impact scores."""
        if likelihood is None or impact is None:
            return "Not Scored"

        score = likelihood * impact
        if score >= 15:
            classification = "CRITICAL"
        elif score >= 9:
            classification = "HIGH"
        elif score >= 4:
            classification = "MEDIUM"
        else:
            classification = "LOW"

        return f"{classification} ({score}/25)"

    @staticmethod
    def _calculate_risk_level_verbose(likelihood: Optional[int], impact: Optional[int]) -> str:
        """Calculates a qualitative risk level from likelihood and impact scores."""
        if likelihood is None or impact is None:
            return f"Likelihood {likelihood}/5, Impact {impact}/5"

        score = likelihood * impact
        if score >= 15:
            classification = "CRITICAL"
        elif score >= 9:
            classification = "HIGH"
        elif score >= 4:
            classification = "MEDIUM"
        else:
            classification = "LOW"

        return f"{classification} {score}/25 (Likelihood {likelihood}/5 × Impact {impact}/5)"

    @staticmethod
    def convert_to_markdown(premortem_analysis: PremortemAnalysis) -> str:
        """
        Convert the premortem analysis to markdown format.
        """
        rows = []

        # Header
        rows.append("A premortem assumes the project has failed and works backward to identify the most likely causes.\n")

        # Assumptions to Kill
        rows.append("## Assumptions to Kill\n")
        rows.append("These foundational assumptions represent the project's key uncertainties. If proven false, they could lead to failure. Validate them immediately using the specified methods.\n")

        rows.append("| ID | Assumption | Validation Method | Failure Trigger |")
        rows.append("|----|------------|-------------------|-----------------|")
        for assumption in premortem_analysis.assumptions_to_kill:
            rows.append(f"| {assumption.assumption_id} | {assumption.statement} | {assumption.test_now} | {assumption.falsifier} |")
        rows.append("\n")

        # Failure Modes
        rows.append("## Failure Scenarios and Mitigation Plans\n")
        rows.append("Each scenario below links to a root-cause assumption and includes a detailed failure story, early warning signs, measurable tripwires, a response playbook, and a stop rule to guide decision-making.\n")

        # Summary Table for Failure Modes
        rows.append("### Summary of Failure Modes\n")
        rows.append("| ID | Title | Archetype | Root Cause | Owner | Risk Level |")
        rows.append("|----|-------|-----------|------------|-------|------------|")
        for index, failure_mode in enumerate(premortem_analysis.failure_modes, start=1):
            risk_level_str = Premortem._calculate_risk_level_brief(failure_mode.likelihood_5, failure_mode.impact_5)
            owner_str = failure_mode.owner or 'Unassigned'
            rows.append(f"| FM{index} | {failure_mode.failure_mode_title} | {failure_mode.failure_mode_archetype} | {failure_mode.root_cause_assumption_id} | {owner_str} | {risk_level_str} |")
        rows.append("\n")

        # Detailed Failure Modes
        rows.append("### Failure Modes\n")
        for index, failure_mode in enumerate(premortem_analysis.failure_modes, start=1):
            if index > 1:
                rows.append("---\n")
            rows.append(f"#### FM{index} - {failure_mode.failure_mode_title}\n")
            rows.append(f"- **Archetype**: {failure_mode.failure_mode_archetype}")
            rows.append(f"- **Root Cause**: Assumption {failure_mode.root_cause_assumption_id}")
            rows.append(f"- **Owner**: {failure_mode.owner or 'Unassigned'}")
            risk_level_str = Premortem._calculate_risk_level_verbose(failure_mode.likelihood_5, failure_mode.impact_5)
            rows.append(f"- **Risk Level:** {risk_level_str}\n")

            rows.append("##### Failure Story")
            rows.append(f"{failure_mode.risk_analysis}\n")

            rows.append("##### Early Warning Signs")
            rows.append(Premortem._format_bullet_list(failure_mode.early_warning_signs))

            rows.append("\n##### Tripwires")
            rows.append(Premortem._format_bullet_list(failure_mode.tripwires or ["No tripwires defined"]))

            rows.append("\n##### Response Playbook")
            rows.append(Premortem._format_bullet_list(failure_mode.playbook or ["No response actions defined"]))
            rows.append("\n")

            stop_rule_text = failure_mode.stop_rule or 'Not specified'
            rows.append(f"**STOP RULE:** {stop_rule_text}\n")

        return "\n".join(rows)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelFromName
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    model_names = [
        "ollama-llama3.1",
        # "openrouter-paid-gemini-2.0-flash-001",
        # "openrouter-paid-openai-gpt-oss-20b",
        # "openrouter-paid-openai-gpt-4o-mini",
        # "openrouter-paid-qwen3-30b-a3b",
    ]
    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    # prompt_id = "4dc34d55-0d0d-4e9d-92f4-23765f49dd29"
    prompt_id = "ab700769-c3ba-4f8a-913d-8589fea4624e"
    plan_prompt = find_plan_prompt(prompt_id)

    print(f"Query:\n{plan_prompt}\n\n")
    result = Premortem.execute(llm_executor=llm_executor, speed_vs_detail=SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW, user_prompt=plan_prompt)

    response_data = result.to_dict(include_metadata=True, include_system_prompt=False, include_user_prompt=False, include_markdown=False)

    print("\n\nResponse:")
    print(json.dumps(response_data, indent=2))

    print(f"\n\nMarkdown Output:")
    print(result.markdown)


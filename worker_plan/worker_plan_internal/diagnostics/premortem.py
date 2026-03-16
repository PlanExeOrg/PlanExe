"""
Premortem: “If we fail, here’s how and why.”

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

IDEA: Focus on top 3 failure modes. All failure modes are rated “High” or “Critical”, which dilutes prioritization. This risks overwhelming the team with too many “critical” focus areas. Rank failure modes by priority (e.g., top 3: FM5, FM1, FM6) and allocate resources accordingly.

IDEA: The "Response Playbook" uses the "Contain, Assess, Respond" model. Enhance with a field for "Proactive Mitigation." The playbook is for when a tripwire is hit (reactive). Proactive mitigation would be the actions taken beforehand to prevent the tripwire from ever being hit. For example, for "The Empty Wallet Wasteland", the proactive mitigation is "Conduct a detailed bottom-up cost estimation." This task should be in the project plan from day one because of the risk identified in the Premortem.

IDEA: Add a recurring risk review cadence (e.g., quarterly) to update assumptions and tripwires based on new data.

IDEA: The premortem assumes a static risk landscape.

IDEA: add a low-probability, high-impact “external shock” scenario, "black swan" scenario.

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
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

OPTIMIZE_INSTRUCTIONS = """\
Goal: produce a premortem that imagines specific, plausible failure scenarios
grounded in this plan's actual risks — not generic failure templates.

Pipeline context
----------------
Premortem runs near the end of the pipeline (after governance, team review,
and expert criticism). It receives the full accumulated plan context. This
makes it one of the highest context-pressure tasks in the pipeline, and one
of the most prone to truncation on smaller models.

The premortem has two distinct components:
- assumptions_to_kill: the foundational beliefs that, if false, kill the
  project. These are the INPUTS.
- failure_modes: the narrative stories of what happens if an assumption proves
  false. These are the OUTPUTS — causal chains, not just risk labels.

Known problems to guard against
---------------------------------
- Failure mode inflation. Rating all failure modes as "High" or "Critical"
  defeats the purpose of prioritisation. If everything is critical, nothing
  is. Calibrate honestly: a failure mode that is plausible but not project-
  ending should be rated lower. The top 2-3 failure modes should be
  genuinely distinct from the rest.
- Generic failure templates. "Budget overrun leads to project cancellation"
  and "stakeholder disengagement causes timeline slippage" are not premortem
  insights — they are defaults. Each failure mode must trace back to a
  specific assumption from this plan's assumptions_to_kill list.
- Reactive-only playbooks. The "Contain, Assess, Respond" structure addresses
  what to do when a tripwire is hit. Proactive mitigation — actions taken
  before the tripwire fires — is more valuable. At minimum, each failure mode
  should include one proactive step, not just a reactive response.
- Assumption-failure_mode mismatch. Every failure mode should be traceable to
  a specific assumption. If a failure mode appears that isn't connected to any
  listed assumption, the assumption list is incomplete. Fix the assumptions
  before adding failure modes.
- Context pressure and truncation. Premortem runs late in the pipeline with a
  large context window. This task has been observed to produce truncated JSON
  on models with smaller output windows (e.g., GLM 4.7 Flash during the
  Batman RICO v2 run). Prefer concise failure narratives (3-5 sentences per
  failure mode) over exhaustive multi-paragraph cascades. Completeness of
  structure matters more than verbosity.
- Static risk landscape assumption. The premortem models failure as if the
  plan's context is frozen. In practice, external conditions change. Where
  relevant, note if a failure mode is primarily driven by external factors
  (market, weather, regulatory change) rather than execution failures.
"""

class AssumptionItem(BaseModel):
    assumption_id: str = Field(description="Enumerate the assumption items starting from 'A1', 'A2', 'A3', 'A4', etc. Do not restart at A1.")
    statement: str = Field(description="The core assumption we are making that, if false, would kill the project.")
    test_now: str = Field(description="A concrete, immediate action to test if this assumption is true.")
    falsifier: str = Field(description="The specific result from the test that would prove the assumption false.")

class FailureModeItem(BaseModel):
    failure_mode_index: int = Field(description="Enumerate the failure_mode items starting from 1")
    root_cause_assumption_id: str = Field(description="The 'assumption_id' (e.g., 'A1') of the single assumption that is the primary root cause of this failure mode.")
    failure_mode_archetype: str = Field(description="The archetype of failure: 'Process/Financial', 'Technical/Logistical', or 'Market/Human'.")
    failure_mode_title: str = Field(description="A compelling, story-like title (e.g., 'The Gridlock Gamble').")
    risk_analysis: str = Field(
        description="Structured, factual breakdown of causes, contributing factors, and impacts for the failure mode. Use bullet points or short factual sentences. Avoid narratives or fictional elements."
    )
    early_warning_signs: List[str] = Field(
        description="Clear, measurable indicators that this failure mode may occur. Each must be objectively testable."
    )
    owner: Optional[str] = Field(None, description="The single role who owns this risk (e.g., 'Permitting Lead', 'Head of Engineering').")
    likelihood_5: Optional[int] = Field(None, description="Integer from 1 (rare) to 5 (almost certain) of this failure occurring.")
    impact_5: Optional[int] = Field(None, description="Integer from 1 (minor) to 5 (catastrophic) if this failure occurs.")
    tripwires: Optional[List[str]] = Field(None, description="Array of 2-3 short, specific strings with NUMERIC thresholds that signal this failure is imminent (e.g., 'Permit delays exceed 90 days').")
    playbook: Optional[List[str]] = Field(None, description="Array of exactly 3 brief, imperative action steps for the owner to take if a tripwire is hit.")
    stop_rule: Optional[str] = Field(None, description="A single, short, hard stop condition that would trigger project cancellation or a major pivot.")

class PremortemAnalysis(BaseModel):
    assumptions_to_kill: List[AssumptionItem] = Field(description="A list of 3 new, critical, underlying assumptions to test immediately.")
    failure_modes: List[FailureModeItem] = Field(description="A list containing exactly 3 distinct failure failure_modes, one for each archetype.")

PREMORTEM_SYSTEM_PROMPT = """
Persona: You are a senior project analyst. Your primary goal is to write compelling, detailed, and distinct failure stories that are also operationally actionable.

Objective: Imagine the user's project has failed completely. Generate a comprehensive premortem analysis as a single JSON object.

Instructions:
1.  Generate a top-level `assumptions_to_kill` array containing exactly 3 critical assumptions to test, each with an `id`, `statement`, `test_now`, and `falsifier`. An assumption is a belief held without proof (e.g., "The supply chain is stable"), not a project goal.
2.  Generate a top-level `failure_modes` array containing exactly 3 detailed, story-like failure failure_modes, one for each archetype: Process/Financial, Technical/Logistical, and Market/Human.
3.  **CRITICAL LINKING STEP: For each `failure_mode`, you MUST identify its root cause by setting the `root_cause_assumption_id` field to the `assumption_id` of one of the assumptions you created in step 1. ** Each assumption ("A1", "A2", "A3", "A4", etc.) must be used as a root cause exactly once.
4.  Each story in the `failure_modes` array must be a detailed, multi-paragraph story with a clear causal chain. Do not write short summaries.
5.  For each of the 3 failure_modes, you MUST populate all the following fields: `failure_mode_index`, `failure_mode_archetype`, `failure_mode_title`, `risk_analysis`, `early_warning_signs`, `owner`, `likelihood_5`, `impact_5`, `tripwires`, `playbook`, and `stop_rule`.
6.  **CRITICAL:** Each of the 3 failure_modes must be distinct and unique. Do not repeat the same story, phrasing, or playbook actions. Tailor each one specifically to its archetype (e.g., the financial failure should be about money and process, the technical failure about engineering and materials, the market failure about public perception and competition).
7.  Tripwires MUST be objectively measurable (use operators like <=, >=, =, %, days, counts); avoid vague terms like “significant” or “many”.
8.  The `playbook` array MUST contain exactly 3 actions as follows:
    1.  An immediate containment/control action, e.g., 'Contain: Stop the bleeding.'
    2.  An assessment/triage action, e.g., 'Assess: Figure out how bad the damage is.'
    3.  A strategic response action, e.g., 'Respond: Take strategic action based on the assessment.'
9.  The `stop_rule` MUST be a hard, non-negotiable condition for project cancellation or a major pivot.
10. Your entire output must be a single, valid JSON object. For any follow-up requests, you MUST regenerate the full JSON object including all required fields, not just the part being changed. Do not add any text or explanation outside of the JSON structure.

FULL-OBJECT, TWO-KEYS ONLY (HARD RULE)
- The top-level JSON MUST contain exactly two keys: "assumptions_to_kill" and "failure_modes". No other keys are allowed.
- On follow-up requests (even if they ask for “only assumptions”), you MUST return the full JSON object with BOTH keys present and populated. Never omit or leave "failure_modes" empty.
- If asked to start at A4/A7/etc., create exactly 3 new assumptions with those IDs and REBUILD all 3 failure_modes to reference them (each assumption used exactly once).
- The message must END immediately after the closing "}" of the JSON. No markdown or text after it.
- Self-check before sending: output starts with "{" and ends with "}", includes BOTH required keys, has exactly 3 assumptions and exactly 3 failure_modes.
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
        system_prompt = PREMORTEM_SYSTEM_PROMPT.strip()

        accumulated_chat_message_list = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            )
        ]

        user_prompt_list = [
            user_prompt,
            "Generate 3 new assumptions that are thematically different from the previous ones. Start assumption_id at A4.",
            "Generate 3 new assumptions that are thematically different from the previous ones and covers different archetypes. Start assumption_id at A7.",
        ]
        if speed_vs_detail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            user_prompt_list = user_prompt_list[:1]
            logger.info("Running in FAST_BUT_SKIP_DETAILS mode. Omitting some assumptions.")
        else:
            logger.info("Running in ALL_DETAILS_BUT_SLOW mode. Processing all assumptions.")

        responses: list[PremortemAnalysis] = []
        metadata_list: list[dict] = []
        for user_prompt_index, user_prompt_item in enumerate(user_prompt_list):
            logger.info(f"Processing user_prompt_index: {user_prompt_index+1} of {len(user_prompt_list)}")
            chat_message_list = accumulated_chat_message_list.copy()
            chat_message_list.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=user_prompt_item,
                )
            )

            def execute_function(llm: LLM) -> dict:
                sllm = llm.as_structured_llm(PremortemAnalysis)
                start_time = time.perf_counter()
                
                chat_response = sllm.chat(chat_message_list)
                pydantic_response = chat_response.raw
                
                end_time = time.perf_counter()
                duration = int(ceil(end_time - start_time))
                
                metadata = dict(llm.metadata)
                metadata["llm_classname"] = llm.class_name()
                metadata["duration"] = duration
                
                return {
                    "pydantic_response": pydantic_response,
                    "metadata": metadata,
                    "duration": duration
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
                if user_prompt_index == 0:
                    logger.error("The first user prompt failed. This is a critical error. Please check the system prompt and user prompt.")
                    raise llm_error from e
                else:
                    logger.error(f"User prompt {user_prompt_index+1} failed. Continuing with next user prompt.")
                    continue
            
            assistant_content_raw: dict = result["pydantic_response"].model_dump()
            # Compact JSON without newlines and spaces, since it's going to be parsed by the LLM. Pretty printing wastes input tokens for the LLM.
            assistant_content: str = json.dumps(assistant_content_raw, separators=(',', ':'))

            chat_message_list.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                )
            )

            responses.append(result["pydantic_response"])
            metadata_list.append(result["metadata"])
            accumulated_chat_message_list = chat_message_list.copy()

        # Use the last response as the primary result
        assumptions_to_kill: list[AssumptionItem] = []
        failure_modes: list[FailureModeItem] = []
        for response in responses:
            assumptions_to_kill.extend(response.assumptions_to_kill)
            failure_modes.extend(response.failure_modes)

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
    

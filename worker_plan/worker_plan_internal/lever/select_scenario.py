"""
Select the best fitting scenario from the 3 candidate scenarios

PROMPT> python -m worker_plan_internal.lever.select_scenario
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested

logger = logging.getLogger(__name__)

class PlanCharacteristics(BaseModel):
    """A structured analysis of the input plan's core nature."""
    ambition_and_scale: str = Field(
        description="Analysis of the plan's level of ambition and its scale (e.g., personal, local, global, revolutionary)."
    )
    risk_and_novelty: str = Field(
        description="Assessment of the inherent risk and novelty. Is it a proven formula, an experimental pilot, or a groundbreaking endeavor?"
    )
    complexity_and_constraints: str = Field(
        description="Evaluation of the plan's operational complexity and stated constraints (e.g., budget, timeline, technical requirements)."
    )
    domain_and_tone: str = Field(
        description="The plan's subject matter and overall tone (e.g., corporate, scientific, personal, creative)."
    )
    holistic_profile_of_the_plan: str = Field(
        default="",
        description="A concise, holistic summary synthesizing the above characteristics into a single profile of the plan's strategic intent."
    )

class ScenarioFitAssessment(BaseModel):
    """An assessment of how well a single scenario fits the plan."""
    scenario_name: str = Field(description="The name of the scenario being assessed.")
    fit_score: int = Field(
        description="A numerical score from 1 (poor fit) to 10 (perfect fit) indicating how well this scenario aligns with the plan's holistic profile."
    )
    fit_assessment: str = Field(
        description="A brief (1-2 sentences) rationale for the assigned fit score."
    )

class FinalChoice(BaseModel):
    """The final selection and justification."""
    chosen_scenario_name: str = Field(description="The name of the single best-fit scenario.")
    justification: str = Field(
        description="A comprehensive justification (100-150 words) for the chosen scenario. This text MUST explain *why* it's the best fit by explicitly referencing the plan's characteristics (ambition, risk, etc.) and why the other scenarios are less suitable. Use markdown bullet points for clarity."
    )

class ScenarioSelectionResult(BaseModel):
    """The root model for the entire analysis output."""
    plan_characteristics: PlanCharacteristics
    scenario_assessments: List[ScenarioFitAssessment] = Field(
        description="An assessment for every single scenario provided."
    )
    final_choice: FinalChoice

SELECT_SCENARIO_SYSTEM_PROMPT = """
You are a master Strategic Analyst AI. Your task is to perform a final strategic recommendation by analyzing a project plan and selecting the most fitting scenario from a predefined set. You must provide a clear, evidence-based justification for your choice.

**Your process is a three-step analysis:**

1.  **Analyze the Plan's Profile:**
    - Read the user-provided plan.
    - Characterize it across four dimensions: `ambition_and_scale`, `risk_and_novelty`, `complexity_and_constraints`, and `domain_and_tone`.
    - Synthesize these into a `holistic_profile_of_the_plan`.

2.  **Evaluate All Scenarios:**
    - For EACH scenario provided, assess how well its strategic logic fits the plan's profile.
    - Assign a `fit_score` (1-10) and a brief `fit_assessment` rationale for each one.

3.  **Make a Final, Justified Choice:**
    - Based on your evaluations, select the single scenario with the highest fit.
    - Write a comprehensive `justification` for this choice. Your justification is the most important part of your output. It MUST:
      - Clearly state *why* the chosen scenario's philosophy aligns with the plan's ambition, risk, and complexity.
      - Briefly explain *why* the other scenarios are less suitable, creating a strong comparative argument.
      - Use markdown bullet points to structure the key points.

You MUST respond with a single JSON object that strictly adheres to the `ScenarioSelectionResult` schema.
"""

@dataclass
class SelectScenario:
    """Analyze a plan and pick the best-fit scenario."""
    system_prompt: str
    user_prompt: str
    response: ScenarioSelectionResult
    metadata: Dict[str, Any]

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, project_context: str, scenarios: List[Dict[str, Any]]) -> 'SelectScenario':
        if not project_context:
            raise ValueError("Project plan cannot be empty.")
        if not scenarios:
            raise ValueError("Scenarios list cannot be empty.")

        logger.info(f"Analyzing plan and evaluating {len(scenarios)} scenarios.")

        scenarios_json_str = json.dumps(scenarios, indent=2)
        user_prompt = (
            f"**Project Plan:**\n```\n{project_context}\n```\n\n"
            f"**Strategic Scenarios for Evaluation:**\n```json\n{scenarios_json_str}\n```\n\n"
            "Please perform the three-step analysis as instructed and provide the final `ScenarioSelectionResult` JSON."
        )

        system_prompt = SELECT_SCENARIO_SYSTEM_PROMPT.strip()
        chat_message_list = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt)
        ]

        def execute_function(llm: LLM) -> dict:
            sllm = llm.as_structured_llm(ScenarioSelectionResult)
            chat_response = sllm.chat(chat_message_list)
            metadata = dict(llm.metadata)
            metadata["llm_classname"] = llm.class_name()
            return {"chat_response": chat_response, "metadata": metadata}

        try:
            result = llm_executor.run(execute_function)
        except PipelineStopRequested:
            raise
        except Exception as e:
            logger.error("LLM interaction for selecting a scenario failed.", exc_info=True)
            raise ValueError("LLM interaction failed.") from e

        return cls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=result["chat_response"].raw,
            metadata=result["metadata"]
        )

    def to_dict(self, include_responses=True, include_metadata=True, include_system_prompt=True, include_user_prompt=True) -> dict:
        d = {}
        if include_responses:
            d["response"] = self.response.model_dump()
        if include_metadata:
            d['metadata'] = self.metadata
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        if include_user_prompt:
            d['user_prompt'] = self.user_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2))

    def save_clean(self, file_path: str) -> None:
        """Saves the final analysis result to a JSON file."""
        response_dict = self.response.model_dump()
        Path(file_path).write_text(json.dumps(response_dict, indent=2))

if __name__ == "__main__":
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName
    from worker_plan_internal.prompt.prompt_catalog import PromptCatalog

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    prompt_catalog = PromptCatalog()
    prompt_catalog.load_simple_plan_prompts()

    # --- Input data ---
    prompt_id = "19dc0718-3df7-48e3-b06d-e2c664ecc07d"
    # prompt_id = "3ca89453-e65b-4828-994f-dff0b679444a"
    prompt_item = prompt_catalog.find(prompt_id)
    if not prompt_item:
        raise ValueError("Prompt item not found.")
    
    scenarios_file_path = os.path.join(os.path.dirname(__file__), 'test_data', f'candidate_scenarios_{prompt_id}.json')
    output_file = f"select_scenario_{prompt_id}.json"

    if not os.path.exists(scenarios_file_path):
        logger.error(f"Scenarios file not found: {scenarios_file_path}")
        exit(1)

    project_context = prompt_item.prompt

    with open(scenarios_file_path, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    scenarios_list = scenarios_data.get('scenarios', [])

    logger.info(f"Loaded plan {prompt_id!r} and {len(scenarios_list)} scenarios from {scenarios_file_path!r}.")

    # --- Execute the Analysis ---
    model_names = ["ollama-llama3.1"]
    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    selection_result = SelectScenario.execute(
        llm_executor=llm_executor,
        project_context=project_context,
        scenarios=scenarios_list
    )

    # --- Display and Save Results ---
    print("\n--- Final Strategic Recommendation ---")
    result_json = json.dumps(selection_result.to_dict(include_system_prompt=False, include_user_prompt=False), indent=2)
    print(result_json)

    selection_result.save_clean(output_file)
    logger.info(f"Full analysis and recommendation saved to '{output_file}'.")

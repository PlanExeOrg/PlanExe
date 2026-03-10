"""
Strategic Stress Test: A Challenge to the Core Premise.

This module functions as a strategic Go/No-Go gate. It challenges a plan's
foundational assumptions to prevent the perfect execution of a flawed strategy.

It asks: "Should we really be doing this?" and "Is the money better spent elsewhere?"

Workflow Integration:
1. The output of this module generates the "Strategic Stress Test" section.
2. The 'Disconfirming Tests' are used to create a "Phase 0: Strategic Validation"
   in the project's Gantt chart, culminating in a "Premise Gate" milestone.
3. All subsequent operational phases are dependent on this gate.

PROMPT> python -m worker_plan_internal.diagnostics.experimental_premise_attack2
"""
import json
import time
import logging
from datetime import date, timedelta
from math import ceil
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, conint
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)


# --- Pydantic Models ---
class DisconfirmingTest(BaseModel):
    """A cheap, fast, and decisive real-world experiment to prove an objection true."""
    test: str = Field(..., description="A concise description of the test to be run.")
    method: Literal[
        "pilot", "survey", "experiment", "benchmark", "interview",
        "market_test", "legal_review", "lab_test"
    ] = Field(..., description="The methodology for conducting the test.")
    metric: str = Field(..., description="The specific, measurable metric the test will track (e.g., 'Uptake Rate', 'CAC', 'NPS').")
    threshold: str = Field(..., description="The unambiguous quantitative threshold for the metric (must include a comparator like '>= 70%', '< $50').")
    owner: str = Field(..., description="The role or person responsible for executing the test (e.g., 'Legal Counsel', 'Project Manager').")
    deadline: date = Field(..., description="The deadline for completing the test (YYYY-MM-DD).")
    budget: str = Field(..., description="A concise estimated budget for the test (e.g., '$5k', '$25k', '0').")

class Alternative(BaseModel):
    """An alternative strategic path, including 'Do nothing'."""
    name: str = Field(..., description="The name of the alternative strategy.")
    value_note: str = Field(..., description="A brief note on the potential value or upside of this alternative.")
    risk_note: str = Field(..., description="A brief note on the primary risk or downside of this alternative.")

class PremiseAttackModel(BaseModel):
    """The structured output for the Strategic Stress Test."""
    core_premise: str = Field(..., description="A single, concise sentence summarizing the plan's core thesis that is being challenged.")
    objections: List[str] = Field(..., min_length=3, max_length=7, description="3-7 high-leverage, fundamental objections to the core premise.")
    disconfirming_tests: List[DisconfirmingTest] = Field(..., min_length=3, description="At least 3 disconfirming tests, one for each of the primary objections.")
    stop_rules: List[str] = Field(..., min_length=2, description="2-6 crisp, unambiguous Go/No-Go conditions that, if met, would trigger a project halt or pivot.")
    alternatives: List[Alternative] = Field(..., description="A list of strategic alternatives, which must include 'Do nothing'.")
    guardrails_if_proceed: List[str] = Field(..., min_length=3, max_length=5, description="3-5 non-negotiable constraints or changes to implement if the project proceeds despite objections.")
    decision_gate: Literal["Go", "Pivot", "No-Go"] = Field(..., description="The final recommendation based on the analysis.")
    decision_rationale: str = Field(..., description="A 2-6 line rationale linking the decision to the objections and tests.")


# --- **FINAL, PRODUCTION-READY** System Prompt ---
SYSTEM_PROMPT = """
You are an adversarial "Red Team" strategist. Your mission is to challenge a project's foundational premise to prevent the execution of a flawed strategy. Your output must be a valid JSON object adhering to the provided schema.

Your task is to generate a "Strategic Stress Test" by performing the following steps:
1.  **Identify the Core Premise:** Distill the user's plan into a single, concise thesis statement. State the premise *as the user sees it*, even if it appears flawed.
2.  **Formulate Diverse Objections:** Generate 3-5 of the strongest, most fundamental objections to this premise. Ensure the objections are diverse and attack different strategic pillars (e.g., **Ethical Viability, Market/Business Model, Financial Sustainability, Critical Dependencies**). **Each objection must be a full, descriptive sentence or two.** For example: "Ethical Viability: The project's core premise of a 'deadly amusement facility' is fundamentally indefensible and exposes the operators to severe legal and reputational risk."
3.  **Design Actionable Disconfirming Tests:** For each primary objection, devise a **cheap, fast, and decisive** real-world test.
    - `owner` must be a specific **role** (e.g., "Legal Counsel", "Project Manager").
    - `deadline` must be a **realistic date that is on or before the `latest_acceptable_deadline`** provided in the user prompt.
    - `budget` must be a **concise string** representing a monetary value (e.g., "$10k", "$0").
4.  **Define Specific Stop Rules:** Each `stop_rule` must be a direct consequence of a test's outcome. (e.g., "If Legal Review concludes indefensible criminal liability, halt project.").
5.  **Propose Alternatives:** List strategic alternatives, including the mandatory "Do nothing" option.
6.  **Establish 3-5 Concrete Guardrails:** If proceeding, define 3-5 non-negotiable constraints. (e.g., "Mandate non-lethality for all mechanisms, certified by a third party.").
7.  **Make a Decision:** Conclude with "Go," "Pivot," or "No-Go" and a clear rationale.

**Hard Requirements:**
- Your entire output must be a single JSON object.
- Focus exclusively on premise-level flaws. Do NOT list solvable execution risks (like engineering delays or site security).
- The `alternatives` list MUST include an item where `name` is "Do nothing".
- The `owner` field must contain a project role, not a sentence.
- The `budget` field must be a concise monetary string (e.g., "$5k"), not a sentence.
- All `deadline` fields must be realistic dates on or before the provided `latest_acceptable_deadline`.
"""

class PremiseAttack:
    """Executes and validates a Strategic Stress Test on a plan's core premise."""
    def __init__(self, llm: LLM, plan_start_date: date = date.today()):
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        self.llm = llm
        self.plan_start_date = plan_start_date
        self.system_prompt = SYSTEM_PROMPT.strip()
        self.user_prompt: Optional[str] = None
        self.raw_response: Optional[dict] = None
        self.validated_data: Optional[PremiseAttackModel] = None
        self.metadata: dict = {}

    def execute(self, user_prompt: str) -> "PremiseAttack":
        """Generates and validates the premise attack."""
        if not isinstance(user_prompt, str) or not user_prompt:
            raise ValueError("Invalid user_prompt.")
        self.user_prompt = user_prompt
        logger.debug(f"Executing Premise Attack with user prompt:\n{user_prompt}")
        
        latest_acceptable_deadline = self.plan_start_date + timedelta(days=90)
        
        full_user_prompt = (
            f"LATEST ACCEPTABLE DEADLINE: {latest_acceptable_deadline.strftime('%Y-%m-%d')}\n"
            f"PLAN START DATE: {self.plan_start_date.strftime('%Y-%m-%d')}\n\n"
            f"PROJECT PLAN:\n{self.user_prompt}"
        )

        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            ChatMessage(role=MessageRole.USER, content=full_user_prompt),
        ]

        structured_llm = self.llm.as_structured_llm(PremiseAttackModel)
        
        start_time = time.perf_counter()
        try:
            response = structured_llm.chat(chat_messages)
            self.validated_data = response.raw
            self.raw_response = self.validated_data.model_dump(mode='json')
        except Exception as e:
            llm_error = LLMChatError(cause=e)
            logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e
        finally:
            end_time = time.perf_counter()
            duration = int(ceil(end_time - start_time))
            response_bytes = len(json.dumps(self.raw_response).encode("utf-8")) if self.raw_response else 0
            
            self.metadata = {
                "llm_classname": self.llm.class_name(),
                "duration_seconds": duration,
                "response_byte_count": response_bytes,
                "model_name": getattr(self.llm.metadata, "model_name", None),
            }
            logger.info(f"LLM interaction completed in {duration}s. Response size: {response_bytes} bytes.")
        
        self.validate()
        return self

    def validate(self):
        """Performs post-generation validation. Raises ValueError on critical failures."""
        if not self.validated_data:
            raise ValueError("Cannot validate without a successful execution response.")

        errors = []
        warnings = []
        pa = self.validated_data

        near_term_deadline_threshold = self.plan_start_date + timedelta(days=90)
        for i, test in enumerate(pa.disconfirming_tests):
            if test.deadline > near_term_deadline_threshold:
                errors.append(f"Validation failed (Test {i+1}): Deadline {test.deadline.isoformat()} is after the threshold of {near_term_deadline_threshold.isoformat()}.")

        if not any(alt.name.lower().strip() == "do nothing" for alt in pa.alternatives):
             errors.append("Validation failed: The 'alternatives' list must include a 'Do nothing' option.")

        drift_tokens = {"lockdown", "evacuation", "perimeter", "cctv", "badge access", "soc", "firewall", "on-prem"}
        text_blob = " ".join(pa.objections).lower() + " ".join(t.test.lower() for t in pa.disconfirming_tests)
        if any(token in text_blob for token in drift_tokens):
             warnings.append("Validation warning: Detected potential drift into execution/site-security topics. Review objections and tests to ensure they remain at the premise level.")
        
        if warnings:
            logger.warning("Post-generation validation warnings found:\n" + "\n".join(warnings))

        if errors:
            error_message = "\n".join(errors)
            raise ValueError(f"Post-generation validation failed:\n{error_message}")
        
        logger.info("Post-generation validation passed successfully.")
        return True

    def to_dict(self) -> dict:
        """Serializes the validated data and metadata to a dictionary."""
        if not self.validated_data:
            return {}
        
        output = self.raw_response
        output["metadata"] = self.metadata
        return output

    def to_gantt_phase_0(self) -> List[dict]:
        """Transforms disconfirming tests into tasks for a Phase 0 Gantt chart."""
        if not self.validated_data or not self.validated_data.disconfirming_tests:
            return []
            
        tasks = []
        test_ids = []
        for i, test in enumerate(self.validated_data.disconfirming_tests):
            test_id = f"premise_test_{i+1}"
            test_ids.append(test_id)
            duration = (test.deadline - self.plan_start_date).days + 1
            tasks.append({
                "id": test_id,
                "text": f"Stress Test: {test.test}",
                "start_date": self.plan_start_date.strftime('%Y-%m-%d'),
                "duration": duration if duration > 0 else 1,
                "owner": test.owner,
                "custom_tooltip": f"<b>Test:</b> {test.test}<br><b>Method:</b> {test.method}<br><b>Metric:</b> {test.metric}<br><b>Threshold:</b> {test.threshold}<br><b>Budget:</b> {test.budget}"
            })

        latest_deadline = max(test.deadline for test in self.validated_data.disconfirming_tests)
        tasks.append({
            "id": "premise_gate",
            "text": "Premise Gate (Go/No-Go Decision)",
            "start_date": latest_deadline.strftime('%Y-%m-%d'),
            "type": "milestone",
            "dependencies": ",".join(test_ids)
        })
        return tasks


if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    llm = get_llm("ollama-llama3.1")

    # Example using the 'Cube Construction' prompt ID, as it's a great test case
    plan_prompt = find_plan_prompt("5d0dd39d-0047-4473-8096-ea5eac473a57")

    print("--- USER PROMPT ---")
    print(plan_prompt)
    print("-" * 20)

    try:
        test_start_date = date(2025, 8, 14)
        attack = PremiseAttack(llm=llm, plan_start_date=test_start_date).execute(plan_prompt)
        
        print("\n--- VALIDATED JSON RESPONSE ---")
        print(json.dumps(attack.to_dict(), indent=2))

        print("\n--- GENERATED GANTT PHASE 0 TASKS ---")
        gantt_tasks = attack.to_gantt_phase_0()
        print(json.dumps(gantt_tasks, indent=2))

    except ValueError as e:
        print(f"\n--- EXECUTION FAILED ---")
        print(f"Error: {e}")
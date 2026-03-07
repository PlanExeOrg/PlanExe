"""
PROMPT> python -m worker_plan_internal.expert.pre_project_assessment

IDEA: markdown document, that goes into the final report.

Two experts analyze a project plan and provide feedback.

Analysis: Experts assess the plan.

Feedback Generation: Experts create actionable feedback.

Summary & Recommendation: A summary and a "Go/No Go" recommendation is created based on the experts input.

IDEA: Cloud providers often timeouts. 
If a cloud provider is down, it may take hours to get back up. In that case switch to another cloud provider.
It would be useful to cycle through multiple cloud providers to get the best results.
Keeping track of health of each cloud providers would be useful, up/down status, response time, etc.

IDEA: Pool of multiple system prompts. Some of the system prompts yields better results than others.
Run 3 invocation of the LLM, using different system prompts. Select the best result. Or extract the best parts from each result.
Example of where it makes sense to have, a pool of multiple system prompts:
For a simple programming task, one system prompt may focus on the irrelevant things, such as:
- Procure the Python library 'Pygame' to handle graphics and game logic.
It makes no sense to procure a Python library for a programming task. It's a simple pip install command.
- Ensure compatibility with your system architecture (32-bit or 64-bit).
The year is 2025, I haven't dealt with 32 bit issues for the last 20 years.
"""
import json
import time
from datetime import datetime
import logging
from math import ceil
from typing import List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

class FeedbackItem(BaseModel):
    feedback_index: int = Field(description="The index of the feedback item.")
    feedback_title: str = Field(description="What is the feedback about?")
    feedback_description: str = Field(description="Describe the feedback.")

class Expert(BaseModel):
    expert_title: str = Field(description="Job title of the expert.")
    expert_full_name: str = Field(description="First name and last name of the expert.")
    feedback_item_list: List[FeedbackItem] = Field(description="List of feedback items.")

class ExpertDetails(BaseModel):
    expert1: Expert = Field(description="Perspective from expert 1.")
    expert2: Expert = Field(description="Perspective from expert 2.")
    combined_summary: Optional[str] = Field(default="", description="Summary of the feedback from both experts.")
    go_no_go_recommendation: Optional[str] = Field(default="", description="A 'Go' or 'No Go' recommendation, with an explanation.")

# Prompt made with o1mini
EXPERT_BROAD_SYSTEM_PROMPT_1 = """
You are a team of 2 experts providing a critical review of a project with a vague description. Depending on the project type, select appropriate expert roles.

**Requirements:**

1. **Feedback Items:**
   - Each expert must provide exactly **4 feedback items**.
   - Each feedback item must start with **"You must do this:"** and include **3-4 specific reasons or actions**.
   - The **"feedback_title"** should capture the essence of the feedback in **around 7 words**.
   - Use **consistent and professional language** throughout all feedback items.
   - **Avoid redundancy** between experts; ensure each expert addresses distinct aspects of the project.
   - Each "feedback_description" should provide clear, step-by-step actions that are specific and measurable.
   - Avoid vague statements; ensure that each action is actionable and can be directly implemented.
   
   **Focus Areas:**
   - **Expert 1:** Project management, technical feasibility, financial modeling, and stakeholder engagement.
   - **Expert 2:** Environmental impact, regulatory compliance, community engagement, and risk management.
   
2. **Combined Summary and Recommendation:**
   - **"combined_summary":** Summarize the **3 most critical reasons** why the project cannot start tomorrow.
   - **"go_no_go_recommendation":** Provide a clear **"Go"** or **"No Go"** recommendation with a brief explanation.
"""

# Prompt made with gemini
EXPERT_BROAD_SYSTEM_PROMPT_2 = """
Pretend that you are a team of 2 experts providing a critical review of a project with a vague description. You must provide specific, actionable recommendations, including why the project cannot begin tomorrow. Each feedback item must be a specific reason why the project cannot begin tomorrow. Each feedback item must start with 'You must do this:'. Each feedback item should then be broken down into 3-4 specific reasons.

The "feedback_title" must capture the essence of the feedback, use around 7 words.

The "feedback_item_list" must contain 4 items per expert. The response must have consistent language throughout all feedback items.

The "expert_full_name" is a fictional name, that might be plausible for the expert.

You must provide a "Go" or "No Go" recommendation. You must also provide the reasons for that recommendation.

The "combined_summary" must include the 3 most important and critical reasons why the project cannot start tomorrow, and the actions you recommend to address these reasons.

The goal of the experts is to assess the readiness and feasibility of the project, and to identify any risks that would make a 'start tomorrow' plan, unfeasible.
"""

# Prompt made with gemini, by combining the two previous prompts
EXPERT_BROAD_SYSTEM_PROMPT_3 = """
You are a team of 2 experts providing a critical review of a task with a vague description. The task is short-term and requires immediate attention.

Your goal is to assess how to complete the task safely and quickly, providing very specific and actionable steps. Select appropriate expert roles.

**Requirements:**

1.  **Expert Roles:**
  - **Expert 1:** Focus on how to complete the task as quickly and *efficiently* as possible, with very specific, actionable steps.
  - **Expert 2:** Focus on the safety aspects of the task, with very specific, actionable steps to mitigate safety concerns.
  - Each expert must have an appropriate title and a fictional full name, relevant to the chosen roles for the task.

2.  **Feedback Items:**
  - Each expert *MUST* provide exactly **4 feedback items**.
  - Each feedback item must start with **"To execute the task, you must:"** followed by **3-4 *extremely specific, concrete, actionable steps*.** Avoid vague or high-level steps. Each step should include specific details such as measurements, timings, equipment, or precise actions required. For example instead of "handle hot water carefully" use "Wear oven mitts when handling hot water and pour slowly and steadily".
  - The **"feedback_title"** should capture the essence of the feedback in around **7 words**, focusing on the immediate actions to be taken. The title should imply very specific actions.
  - The feedback items MUST NOT be too high level, and MUST be very specific.

3.  **Combined Summary and Recommendation:**
  - **"combined_summary":** Summarize the **3 most critical actions**, using specific examples from the feedback items, needed immediately to enable the task to begin. The summary must reference the `feedback_index` from the experts for each of these three critical actions. Explain *why* those 3 actions are the most critical actions needed.
  - **"go_no_go_recommendation":**
    Provide a clear **"Execute Immediately"** or **"Do Not Execute"** recommendation.
"""

# Prompt made with deepseek, by combining the 3 previous prompts
EXPERT_BROAD_SYSTEM_PROMPT_4 = """
You are a team of 2 experts providing a critical review of a project with a vague description. The project can be short-term, medium-term, or long-term. Your goal is to assess how to complete the project safely, efficiently, and effectively, providing very specific and actionable steps. Select appropriate expert roles based on the project type.

**Requirements:**

1. **Expert Roles:**
   - **Expert 1:** Focus on how to complete the project as quickly and *efficiently* as possible, with very specific, actionable steps. This includes project management, technical feasibility, resource allocation, and timeline optimization.
   - **Expert 2:** Focus on the safety, compliance, and risk mitigation aspects of the project, with very specific, actionable steps to address potential hazards, regulatory requirements, and environmental or community impacts.
   - Each expert must have an appropriate title and a fictional full name, relevant to the chosen roles for the task.

2. **Feedback Items:**
   - Each expert *MUST* provide exactly **4 feedback items**.
   - Each feedback item must start with **"To execute the project, you must:"** followed by **3-4 *extremely specific, concrete, actionable steps*.** Avoid vague or high-level steps. Each step should include specific details such as measurements, timings, equipment, or precise actions required. For example, instead of "handle hot water carefully," use "Wear oven mitts when handling hot water and pour slowly and steadily."
   - The **"feedback_title"** should capture the essence of the feedback in around **7 words**, focusing on immediate actions to be taken. The title should imply very specific actions (e.g., "Assemble Team by [specific date]").
   - The feedback items MUST NOT be too high-level and MUST be very specific.

3. **Combined Summary and Recommendation:**
   - **"combined_summary":** Summarize the **3 most critical actions**, using specific examples from the feedback items, needed immediately to enable the project to begin. The summary must reference the `feedback_index` from the experts for each of these three critical actions. Explain *why* those 3 actions are the most critical actions needed.
   - **"go_no_go_recommendation":** Provide a clear **"Execute Immediately"**, **"Proceed with Caution"**, or **"Do Not Execute"** recommendation, depending on the project's feasibility, risks, and readiness. Include a brief explanation for the recommendation, addressing potential risks and mitigation strategies.

**Focus Areas for Experts:**
- **Expert 1 (Efficiency and Execution):** Prioritize speed, resource optimization, and technical feasibility. Address logistical challenges, stakeholder coordination, and timeline management.
- **Expert 2 (Safety and Compliance):** Prioritize risk mitigation, regulatory compliance, and environmental or community impacts. Address safety protocols, hazard prevention, and legal or ethical considerations.

**Adaptability:**
- For **short-term projects**, emphasize immediate actions, rapid resource allocation, and quick risk assessments.
- For **medium-term projects**, balance efficiency with thorough planning, including phased execution and contingency planning.
- For **long-term projects**, focus on sustainability, regulatory approvals, and long-term risk management.

**Language and Tone:**
- Use **consistent and professional language** throughout all feedback items.
- Avoid redundancy between experts; ensure each expert addresses distinct aspects of the project.
- Ensure all steps are **actionable, measurable, and specific**, avoiding vague or generic advice.
- Use **action-oriented titles** that imply immediate, specific actions (e.g., "Complete Risk Assessment by [specific date]").

**Additional Guidelines:**
- Include **specific deadlines** (e.g., "by year-month-day") in feedback items to emphasize urgency.
- Provide **quantifiable details** (e.g., "500 PPE kits," "10m x 10m command center") to ensure clarity and measurability.
- Highlight **potential risks** and **mitigation strategies** in the combined summary and recommendation.
- Ensure feedback items are **distinct and non-overlapping** between experts, with clear separation of responsibilities.
"""

# Prompt made with gemini, by combining the 4 previous prompts
EXPERT_BROAD_SYSTEM_PROMPT_5 = """
You are a team of 2 experts providing a critical, actionable review of a project given its vague description. Your goal is to rapidly assess the project's feasibility, identify key risks, and provide a clear path forward. The project can be short-term, medium-term, or long-term.

**Overall Requirements:**

1.  **Action-Oriented:** Focus on providing **immediate, concrete steps** that the project team can take to move forward (or decide not to move forward). Avoid analysis or commentary about why those steps need to be done - just list what to do.
2.  **Feasibility and Risk-Based:** Analyze the project for feasibility given the vague description, and highlight any safety, logistical, or ethical concerns.
3.  **Clear Recommendation:** Provide a definitive and clear recommendation on whether the project should proceed *now*, and why.

**Expert Roles:**

   - **Expert 1 (Project Execution & Logistics):** Focus on the **practical steps** required to execute the project efficiently. This includes defining resources, timelines, tasks, and initial goals. Prioritize speed, and assume that no prior work has been done on the project.
   - **Expert 2 (Safety, Compliance & Risk):** Focus on **identifying and mitigating risks** related to the project. This includes health, safety, legal and ethical issues. Assume that the team will not consider these issues unless prompted.

   - Each expert MUST have an appropriate title and a fictional full name relevant to their expertise.

**Feedback Items (for each expert):**

   - Each expert MUST provide exactly **4 feedback items**.
   - Each feedback item must start with **"To initiate this project, you must:"**, followed by **3-4 *extremely specific, concrete, actionable steps*.** Each step should include specific details such as measurements, timings, equipment, personnel requirements or precise actions. *DO NOT* use vague, high-level or generalized statements. The goal is to provide a checklist that can be immediately executed. Use action-oriented language. Use quantifiable details (e.g., "10 meters of rope", "5 sterile collection tubes"). For example, instead of: "procure appropriate gear", instead use: "Procure 10 sets of specialized radiation-resistant suits, including lead-lined inner layers, gloves, and boots rated for 100mSv exposure within 48 hours."
   - The **"feedback_title"** should capture the essence of the feedback in **around 7 words** and should imply very specific actions (e.g., "Procure Safety Gear", "Map the Area"). It should not be a general statement, and it should use an active verb. The 'feedback_title' must NOT include the text 'by Date', as this is unnecessary.
   - The feedback items MUST NOT be too high level, and MUST be very specific. The aim should be to provide a checklist that can be rapidly assessed by any project team. The items must be directly executable as a checklist. When describing quantities, always use phrases such as "at least X" or "no more than X" unless the exact amount is known. All timeframes MUST include a specific date AND time, and the time must be expressed as a 24 hour clock using HH:MM format. If a specific action is likely to be difficult to achieve in the timeframe, the response MUST include an alternative action to mitigate this risk. *Use a bulleted list for all steps, do not include numbered lists.*

**Combined Summary and Recommendation:**

  - **"combined_summary":** Summarize the **3 most critical, immediate actions**, selected from the feedback items across *both* experts, referencing each feedback item using the expert name and `feedback_index`. Explain *why* these actions are the most immediately essential, and how they mitigate the most important risks.
  - **"go_no_go_recommendation":** Provide a clear recommendation of **"Execute Immediately"**, **"Proceed with Caution"**, or **"Do Not Execute"**.  Your recommendation must be based on a balanced assessment of the project's potential risks and feasibility, given the limitations of the provided description and the immediate actions outlined by the experts. Do not default to a single recommendation. The response must show that it has actively considered all three options, and show why it is recommending one option over the other two. If you recommend "Proceed with Caution", include the specific actions required for caution. If you recommend "Do Not Execute" be clear about why that's the best option given the risks. The recommendation should be a reflection of the overall safety and operational concerns, given the described project. Provide a brief, *concrete* explanation supporting this recommendation, highlighting the primary risks or critical actions that have influenced the decision.

**Additional Guidelines:**

  - Use **consistent, professional, action-oriented language** throughout.
  - **Avoid redundancy** between experts; ensure each expert addresses distinct aspects.
  - Include **specific deadlines** (e.g., "by year-month-day") or timings to emphasize urgency.
  - Include **quantifiable details** (e.g., "10 meters of rope", "5 sterile collection tubes") to ensure clarity and measurability.
  - The tone should be that of a professional, who has seen many projects, and therefore immediately recognizes key issues that must be resolved for this project to proceed.
  - Be aware that some timelines may be impossible. If a timeline is unrealistic, the response should provide an alternative approach to obtain those results, rather than just accepting the unrealistic timeframe as given. Do not propose that a government permit can be obtained in a single day.
"""

# Prompt made with gemini, by combining the 5 previous prompts
EXPERT_BROAD_SYSTEM_PROMPT_6 = """
You are a team of 2 experts providing a critical, actionable review of a project given its vague description. Your goal is to rapidly assess the project's feasibility, identify key risks, and provide a clear path forward. The project can be short-term, medium-term, or long-term. The year is CURRENT_YEAR_PLACEHOLDER.

**Overall Requirements:**

1.  **Action-Oriented:** Focus on providing **immediate, concrete steps** that the project team can take to move forward (or decide not to move forward). Avoid analysis or commentary about why those steps need to be done - just list what to do.
2.  **Feasibility and Risk-Based:** Analyze the project for feasibility given the vague description, and highlight any safety, logistical, or ethical concerns.
3.  **Clear Recommendation:** Provide a definitive and clear recommendation on whether the project should proceed *now*, and why.

**Expert Roles:**

   - **Expert 1 (Project Execution & Logistics):** Focus on the **practical steps** required to execute the project efficiently. This includes defining resources, timelines, tasks, and initial goals. Prioritize speed, and assume that no prior work has been done on the project. The feedback *MUST* be specific and derived *ONLY* from the vague description provided. Do *NOT* use generic steps or project management steps. The steps should provide specific details about code, mathematical, and logical details, *if directly implied by the project description*. You MUST explain *why* a specific action is needed.
   - **Expert 2 (Safety, Compliance & Risk):** Focus on **identifying and mitigating risks** related to the project. This includes health, safety, legal and ethical issues, as well as technical risks within the project. Assume that the team will not consider these issues unless prompted. The feedback *MUST* be specific and derived *ONLY* from the vague description provided. Do *NOT* use generic safety steps or general safety advice. If the task is about software, focus on the specific details of *how* to mitigate a risk, and avoid describing the risk itself. You MUST explain *why* a specific action is needed.

   - Each expert MUST have an appropriate title and a fictional full name relevant to their expertise.

**Feedback Items (for each expert):**

   - Each expert MUST provide exactly **4 feedback items**.
   - Each feedback item must start with **"To initiate this project, you must:"**, followed by **3-4 *extremely specific, concrete, actionable steps*.** Each step should include specific details such as measurements, timings, equipment, personnel requirements or precise actions. *DO NOT* use vague, high-level or generalized statements. The goal is to provide a checklist that can be immediately executed. Use action-oriented language. Use quantifiable details (e.g., "10 meters of rope", "5 sterile collection tubes"). For example, instead of: "procure appropriate gear", instead use: "Procure 10 sets of specialized radiation-resistant suits, including lead-lined inner layers, gloves, and boots rated for 100mSv exposure within 48 hours." The feedback items must be derived *ONLY* from the vague project description. If the task is about software, *avoid describing generic steps such as "procure a library" or describing general safety risks*. Instead, focus on calculations, algorithms, or implementation details *if they are directly implied by the project description*. The actions for handling the risks MUST be extremely explicit, and describe *how to handle the risk* rather than *what the risk is*. You MUST explain *why* a specific action is needed.
   - The **"feedback_title"** should capture the essence of the feedback in **around 7 words** and should imply very specific actions (e.g., "Procure Safety Gear", "Map the Area"). It should not be a general statement, and it should use an active verb. The 'feedback_title' must NOT include the text 'by Date', as this is unnecessary.
   - The feedback items MUST NOT be too high level, and MUST be very specific. The aim should be to provide a checklist that can be rapidly assessed by any project team. The items must be directly executable as a checklist. When describing quantities, always use phrases such as "at least X" or "no more than X" unless the exact amount is known. All timeframes MUST include a specific date AND time, and the time must be expressed as a 24 hour clock using HH:MM format. If a specific action is likely to be difficult to achieve in the timeframe, the response MUST include an alternative action to mitigate this risk. *Use a bulleted list for all steps, do not include numbered lists.*

**Combined Summary and Recommendation:**

  - **"combined_summary":** Summarize the **3 most critical, immediate actions**, selected from the feedback items across *both* experts. Explain *why* these actions are the most immediately essential, and how they mitigate the most important risks. Do *not* reference the feedback item using the expert name and `feedback_index`.
  - **"go_no_go_recommendation":** Provide a clear recommendation of **"Execute Immediately"**, **"Proceed with Caution"**, or **"Do Not Execute"**. Your recommendation must be based on a balanced assessment of the project's potential risks and feasibility, given the limitations of the provided description and the immediate actions outlined by the experts. Do not default to a single recommendation. The response must show that it has actively considered all three options, and show why it is recommending one option over the other two. If you recommend "Proceed with Caution", include the specific actions required for caution. *If you recommend "Do Not Execute", the response MUST provide a very clear and detailed justification about why it is not feasible to proceed, given the risks and the nature of the project, and if no reasonable mitigation strategy can be proposed. The response must be derived from the vague project description, with clear and obvious reasons why the project cannot be executed immediately. You must use examples directly from the description to justify your recommendation, and you must explain what part of the description is not feasible or creates a contradiction*. The recommendation should be a reflection of the overall safety and operational concerns, given the described project. Provide a brief, *concrete* explanation supporting this recommendation, highlighting the primary risks or critical actions that have influenced the decision.

**Additional Guidelines:**

  - Use **consistent, professional, action-oriented language** throughout.
  - **Avoid redundancy** between experts; ensure each expert addresses distinct aspects.
  - Include **specific deadlines** (e.g., "by year-month-day") or timings to emphasize urgency, if necessary.
  - Include **quantifiable details** (e.g., "10 meters of rope", "5 sterile collection tubes") to ensure clarity and measurability.
  - The tone should be that of a professional, who has seen many projects, and therefore immediately recognizes key issues that must be resolved for this project to proceed.
  - Be aware that some timelines may be impossible. If a timeline is unrealistic, the response should provide an alternative approach to obtain those results, rather than just accepting the unrealistic timeframe as given. Do not propose that a government permit can be obtained in a single day.
"""

EXPERT_BROAD_SYSTEM_PROMPT = EXPERT_BROAD_SYSTEM_PROMPT_6

@dataclass
class PreProjectAssessment:
    """
    Obtain a broad perspective from 2 experts.
    """
    system_prompt: Optional[str]
    user_prompt: str
    response: dict
    metadata: dict
    preproject_assessment: dict

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str, **kwargs: Any) -> 'PreProjectAssessment':
        """
        Invoke LLM and have 2 experts take a broad look at the initial plan.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid query.")

        # Obtain the current year as a string, eg. "1984"
        current_year_int = datetime.now().year
        current_year = str(current_year_int)

        # Replace the placeholder in the system prompt with the current year
        system_prompt = EXPERT_BROAD_SYSTEM_PROMPT.strip()
        system_prompt = system_prompt.replace("CURRENT_YEAR_PLACEHOLDER", current_year)

        default_args = {
            'system_prompt': system_prompt
        }
        default_args.update(kwargs)

        system_prompt = default_args.get('system_prompt')
        logger.debug(f"System Prompt:\n{system_prompt}")
        if system_prompt and not isinstance(system_prompt, str):
            raise ValueError("Invalid system prompt.")

        chat_message_list1 = []
        if system_prompt:
            chat_message_list1.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=system_prompt,
                )
            )
        
        logger.debug(f"User Prompt:\n{user_prompt}")
        chat_message_user = ChatMessage(
            role=MessageRole.USER,
            content=user_prompt,
        )
        chat_message_list1.append(chat_message_user)

        sllm = llm.as_structured_llm(ExpertDetails)

        logger.debug("Starting LLM chat interaction.")
        start_time = time.perf_counter()
        chat_response1 = sllm.chat(chat_message_list1)
        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))
        response_byte_count = len(chat_response1.message.content.encode('utf-8'))
        logger.info(f"LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        try:
            json_response = json.loads(chat_response1.message.content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON.", exc_info=True)
            raise ValueError("Invalid JSON response from LLM.") from e

        # Cleanup the json response from the LLM model.
        # Discard the name of the experts and the role of the experts, 
        # these are causing confusion downstream when trying to do project planning.
        # Having those names and roles causes the LLM to think they are stakeholders in the project, which they are not.

        # Extract just the feedback items
        flat_feedback_list = []
        for key in ['expert1', 'expert2']:
            expert = json_response.get(key)
            if expert is None:
                logger.error(f"Expert {key} not found in response.")
                continue

            for feedback_item in expert.get('feedback_item_list', []):
                flat_feedback_list.append({
                    "title": feedback_item.get('feedback_title', 'Empty'),
                    "description": feedback_item.get('feedback_description', 'Empty')
                })
        preproject_assessment = {
            'go_no_go_recommendation': json_response.get('go_no_go_recommendation', 'Empty'),
            'combined_summary': json_response.get('combined_summary', 'Empty'),
            'feedback': flat_feedback_list
        }
        logger.info(f"Extracted {len(flat_feedback_list)} feedback items.")

        result = PreProjectAssessment(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            preproject_assessment=preproject_assessment
        )
        logger.debug("CreateProjectPlan instance created successfully.")
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

    def save_preproject_assessment(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.preproject_assessment, indent=2))

if __name__ == "__main__":
    import logging
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    plan_prompt = find_plan_prompt("4dc34d55-0d0d-4e9d-92f4-23765f49dd29")
    query = (
        f"{plan_prompt}\n\n"
        "Today's date:\n2025-Jan-26\n\n"
        "Project start ASAP"
    )

    llm = get_llm("ollama-llama3.1")
    # llm = get_llm("deepseek-chat", max_tokens=8192)

    print(f"Query: {query}")
    result = PreProjectAssessment.execute(llm, query)

    print("\n\nResponse:")
    print(json.dumps(result.to_dict(include_system_prompt=False, include_user_prompt=False), indent=2))

    print("\n\nPreproject assessment:")
    print(json.dumps(result.preproject_assessment, indent=2))

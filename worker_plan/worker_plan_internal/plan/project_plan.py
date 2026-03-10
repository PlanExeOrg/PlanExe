"""
PROMPT> python -m worker_plan_internal.plan.project_plan

Based on a vague description, the creates a rough draft for a project plan.
"""
import json
import time
import logging
from dataclasses import dataclass
from math import ceil
from typing import TypeVar
from pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage, MessageRole
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class SMARTCriteria(BaseModel):
    specific: str = Field(
        description="Clearly defines what is to be accomplished. Provides context and purpose behind the goal. Avoid vagueness, unrealistic targets, broad statements that lack direction."
    )
    measurable: str = Field(
        description="Establish how you will measure success. This could be quantitative (e.g., numbers, percentages) or qualitative (e.g., satisfaction levels). Without clear metrics, it’s difficult to track progress or determine success."
    )
    achievable: str = Field(
        description="Realism: Ensures the goal is attainable with the available resources and within constraints. Feasibility: Considers practical aspects such as time, budget, and expertise."
    )
    relevant: str = Field(
        description="Alignment: Connects the goal to broader objectives or missions. Importance: Highlights the significance and impact of achieving the goal."
    )
    time_bound: str = Field(
        description="Deadline: Specifies a clear timeframe for goal completion. Milestones: Breaks down the timeline into smaller, manageable phases if necessary."
    )

class RiskAssessmentAndMitigationStrategies(BaseModel):
    """
    By systematically identifying, analyzing, and addressing risks, you enhance the project 
    resilience and increase the likelihood of achieving sustainable and long-term success.
    """
    key_risks: list[str] = Field(
        description="Things that can negatively impact the project, such as regulatory changes, financial uncertainties, technological failures, supply chain disruptions and environmental challenges."
    )
    diverse_risks: list[str] = Field(
        description="Things that can negatively impact the project, such as operational risks, systematic risks, business risks."
    )
    mitigation_plans: list[str] = Field(
        description="Develop strategies to minimize or manage these risks, ensuring the project remains on track despite unforeseen obstacles."
    )

class StakeholderAnalysis(BaseModel):
    """
    Understanding the interests, expectations, and concerns of stakeholders is essential for building
    strong relationships, securing support, and ensuring project success.
    """
    primary_stakeholders: list[str] = Field(
        description="List all key stakeholders, including government agencies, local communities, investors, suppliers, and end-users."
    )
    secondary_stakeholders: list[str] = Field(
        description="List all secondary stakeholders, such as regulatory bodies, environmental groups, and suppliers."
    )
    engagement_strategies: list[str] = Field(
        description="Outline how each stakeholder group will be engaged, their roles, and how their interests will be addressed to secure support and collaboration."
    )

class RegulatoryAndComplianceRequirements(BaseModel):
    """
    Detailed overview of the regulatory and compliance requirements necessary for the project.
    Ensures the project operates within legal frameworks and adheres to all necessary standards.
    """
    permits_and_licenses: list[str] = Field(
        description="List of permits and licenses required at local, regional, and national levels. Such as Building Permit, Cave Exploration Permit, Hazardous Materials Handling Permit."
    )
    compliance_standards: list[str] = Field(
        description="Ensure adherence to environmental, safety, and industry-specific standards and regulations."
    )
    regulatory_bodies: list[str] = Field(
        description="List of regulatory bodies and their roles in the project's compliance landscape. Such as International Energy Agency, International Maritime Organization, World Health Organization."
    )
    compliance_actions: list[str] = Field(
        description="List with actions and steps taken to ensure compliance with all relevant regulations and standards. Example: ['Building and Electrical Codes', 'Wildlife Protection', 'Fire safety measures', 'Biosafety Regulations', 'Radiation Safety']."
    )
    
class GoalDefinition(BaseModel):
    """
    A clear, specific, and actionable statement that outlines what you aim to achieve.
    A well-defined goal serves as a foundation for effective planning, problem decomposition, and team assembly.
    """
    goal_statement: str = Field(
        description="Adhering to the SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound)"
    )
    smart_criteria: SMARTCriteria = Field(
        description="Details of the SMART criteria."
    )
    dependencies: list[str] = Field(
        description="Other goals or tasks that must be completed before this goal can be achieved. Example: ['Securing funding for equipment and personnel', 'Obtaining necessary permits for cave exploration']."
    )
    resources_required: list[str] = Field(
        description="Resources necessary to achieve the goal. Example: ['Cave exploration gear', 'Biological sampling tools']."
    )
    related_goals: list[str] = Field(
        description="Other goals that are related or interconnected. Facilitates understanding of goal interdependencies and broader objectives."
    )
    tags: list[str] = Field(
        description="Keywords or labels associated with the goal. Enhances searchability and categorization, making it easier to filter and find goals based on keywords. Example: ['Extremophiles', 'DNA Sequencing', 'Microbial Life']."
    )
    risk_assessment_and_mitigation_strategies: RiskAssessmentAndMitigationStrategies = Field(
        description="Identify potential risks and develop strategies to mitigate them, ensuring the goal remains achievable despite unforeseen challenges."
    )
    stakeholder_analysis: StakeholderAnalysis = Field(
        description="Analyze key stakeholders, their interests, and engagement strategies to secure support and collaboration for goal achievement."
    )
    regulatory_and_compliance_requirements: RegulatoryAndComplianceRequirements = Field(
        description="Ensure compliance with all regulatory and legal requirements, including permits, licenses, and industry-specific standards."
    )

PROJECT_PLAN_SYSTEM_PROMPT_1 = """
You are an expert project planner tasked with creating comprehensive and detailed project plans based on user-provided descriptions. Your output must be a complete JSON object conforming to the provided GoalDefinition schema. Focus on being specific and actionable, generating a plan that is realistic and useful for guiding project development.
Your plans must include:
- A clear goal statement adhering to the SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound). Provide specific metrics and timeframes where possible.
- A breakdown of dependencies and required resources for the project.
- A clear identification of related goals and future applications.
- A detailed risk assessment with specific mitigation strategies. Focus on actionable items to mitigate the risks you have identified.
- A comprehensive stakeholder analysis, identifying primary and secondary stakeholders, and outlining engagement strategies.
- A detailed overview of regulatory and compliance requirements, such as permits and licenses, and how compliance actions are planned.
- Tags or keywords that allow users to easily find and categorize the project.
Prioritize feasibility, practicality, and alignment with the user-provided description. Ensure the plan is actionable, with concrete steps where possible and measurable outcomes.
"""

PROJECT_PLAN_SYSTEM_PROMPT_2 = """
You are an expert project planner tasked with creating comprehensive and detailed project plans based on user-provided descriptions. Your output must be a complete JSON object conforming to the provided GoalDefinition schema. Focus on being specific and actionable, generating a plan that is realistic and useful for guiding project development.

Your plans must include:
- A clear goal statement adhering to the SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound). Provide specific metrics and timeframes where possible.
- A breakdown of dependencies and required resources for the project. Break down dependencies into actionable sub-tasks where applicable.
- A clear identification of related goals and future applications.
- A detailed risk assessment with specific mitigation strategies. Focus on actionable items to mitigate the risks you have identified, ensuring they are tailored to the project's context.
- A comprehensive stakeholder analysis, identifying primary and secondary stakeholders, and outlining engagement strategies.
  - **Primary Stakeholders:** Identify key roles or individuals directly responsible for executing the project. For small-scale or personal projects, this may simply be the person performing the task (e.g., "Coffee Brewer"). For large-scale projects, identify domain-specific roles (e.g., "Construction Manager," "Life Support Systems Engineer").
  - **Secondary Stakeholders:** Identify external parties or collaborators relevant to the project. For small-scale projects, this may include suppliers or individuals indirectly affected by the project (e.g., "Coffee Supplier," "Household Members"). For large-scale projects, include regulatory bodies, material suppliers, or other external entities.
  - **Note:** Do not assume the availability or involvement of any specific individuals unless explicitly stated in the user-provided description.
- A detailed overview of regulatory and compliance requirements, such as permits and licenses, and how compliance actions are planned.
- Tags or keywords that allow users to easily find and categorize the project.

**Adaptive Behavior:**
- Automatically adjust the level of detail and formality based on the scale and complexity of the project. For small-scale or personal projects, keep the plan simple and practical. For large-scale or complex projects, include more detailed and formal elements.
- Infer the appropriate stakeholders, risks, and resources based on the project's domain and context. Avoid overly formal or mismatched roles unless explicitly required by the project's context.

Prioritize feasibility, practicality, and alignment with the user-provided description. Ensure the plan is actionable, with concrete steps where possible and measurable outcomes.
"""

PROJECT_PLAN_SYSTEM_PROMPT_3 = """
You are an expert project planner tasked with creating comprehensive and detailed project plans based on user-provided descriptions. Your output must be a complete JSON object conforming to the provided GoalDefinition schema. Focus on being specific and actionable, generating a plan that is realistic and useful for guiding project development.

Your plans must include:
- A clear goal statement adhering to the SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound). Provide specific metrics and timeframes where possible. For the time-bound, only use "Today" for simple, short duration tasks.
    -Ensure the SMART criteria is high-level, and based directly on the goal statement, and the user description.
        - The **Specific** criteria should clarify what is to be achieved with the goal, and must directly reflect the goal statement, and must not imply any specific actions or processes.
        - The **Measurable** criteria should define how you will know if the goal has been achieved. It should be a metric or some other way of validating that the goal is complete, and must not include implied actions or steps.
        - The **Achievable** criteria should explain why the goal is achievable given the information provided by the user. It should specify any limitations or benefits.
        - The **Relevant** criteria should specify why this goal is necessary, or what value it provides.
        - The **Time-bound** criteria must specify when the goal must be achieved. For small tasks, this will be "Today". For larger tasks, the time-bound should be a general time estimate, and should not specify a specific date or time unless it has been specified by the user.
- A breakdown of dependencies and required resources for the project. Break down dependencies into actionable sub-tasks where applicable. Dependencies should be high-level, and not overly prescriptive, nor should they imply specific actions. Only include dependencies that are explicitly mentioned in the user description or directly implied from it. Do not include any specific timestamps, volumes, quantities or implied resources in the dependencies section, and do not include inferred actions.
- A clear identification of related goals and future applications.
- A detailed risk assessment with specific mitigation strategies. Focus on actionable items to mitigate the risks you have identified, ensuring they are tailored to the project's context.
    - When identifying risks, consider common issues specific to the project's domain (e.g., construction delays, equipment failures, safety hazards, financial issues, security breaches, data losses). For each identified risk, generate a realistic and specific mitigation strategy that is actionable within the project's context. Try to extract risks based on user descriptions. Avoid being too specific, and avoid adding unrealistic risks and mitigation actions. Only include mitigation plans that are explicitly derived from the user description, or are implied from it.
- A comprehensive stakeholder analysis, identifying primary and secondary stakeholders, and outlining engagement strategies.
  - **Primary Stakeholders:** Identify key roles or individuals directly responsible for executing the project. For small-scale or personal projects, this may simply be the person performing the task (e.g., "Coffee Brewer"). For large-scale projects, identify domain-specific roles (e.g., "Construction Manager," "Life Support Systems Engineer").
  - **Secondary Stakeholders:** Identify external parties or collaborators relevant to the project. For small-scale projects, this may include suppliers or individuals indirectly affected by the project (e.g., "Coffee Supplier," "Household Members"). For large-scale projects, include regulatory bodies, material suppliers, or other external entities.
    - When outlining engagement strategies for stakeholders, consider the nature of the project and their roles. Primary stakeholders should have regular updates and progress reports, and requests for information should be answered promptly. Secondary stakeholders may require updates on key milestones, reports for compliance, or timely notification of significant changes to project scope or timeline. For smaller projects, the engagement strategy and stakeholders can be omitted if they are not explicitly mentioned in the user description, or implied from it.
  - **Note:** Do not assume the availability or involvement of any specific individuals beyond those directly mentioned in the user-provided project description. Generate all information independently from the provided description, and do not rely on any previous data or information from prior runs of this tool. Do not include any default information unless explicitly stated.
- A detailed overview of regulatory and compliance requirements, such as permits and licenses, and how compliance actions are planned.
    - When considering regulatory and compliance requirements, identify any specific licenses or permits needed, and include compliance actions in the plan, such as "Apply for permit X", "Schedule compliance audit" and "Implement compliance plan for Y", and ensure compliance actions are included in the project timeline. For smaller projects, the regulatory compliance section can be omitted.
- Tags or keywords that allow users to easily find and categorize the project.
Adaptive Behavior:
- Automatically adjust the level of detail and formality based on the scale and complexity of the project. For small-scale or personal projects, keep the plan simple and avoid formal elements. For massive or complex projects, ensure plans include more formal elements, such as project charters or work breakdown structures, and provide detailed actions for project execution.
- Infer the appropriate stakeholders, risks, and resources based on the project's domain and context. Avoid overly formal or mismatched roles unless explicitly required by the project's context.
- For smaller tasks, only include resources that need to be purchased or otherwise explicitly acquired. Only include resources that are mentioned in the user description, or implied from it. Do not include personnel or stakeholders as a resource.
- Only include dependencies that are explicitly mentioned in the user description, or directly implied from it.
Prioritize feasibility, practicality, and alignment with the user-provided description. Ensure the plan is actionable, with concrete steps where possible and measurable outcomes.
When breaking down dependencies into sub-tasks, specify concrete actions (e.g., "Procure X", "Design Y", "Test Z"), and if possible, include resource requirements (e.g., "Procure 100 Units of X") and estimated timeframes where appropriate. However, for very small, simple tasks, the dependencies do not need a time element, and do not have to be overly specific.

Here's an example of the expected output format for a simple project:
{
  "goal_statement": "Make a cup of coffee.",
  "smart_criteria": {
    "specific": "Prepare a cup of instant coffee, with milk and sugar if available.",
    "measurable": "The completion of the task can be measured by the existence of a prepared cup of coffee.",
    "achievable": "The task is achievable in the user's kitchen.",
    "relevant": "The task will provide the user with a warm drink.",
    "time_bound": "The task should be completed in 5 minutes."
  },
  "dependencies": [],
  "resources_required": [ "instant coffee" ],
  "related_goals": [ "satisfy hunger", "enjoy a drink" ],
  "tags": [ "drink", "coffee", "simple" ]
}
"""

PROJECT_PLAN_SYSTEM_PROMPT = PROJECT_PLAN_SYSTEM_PROMPT_3

T = TypeVar('T', bound=BaseModel)

@dataclass
class ProjectPlan:
    """
    Creating a project plan from a vague description.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'ProjectPlan':
        """
        Invoke LLM to create project plan from a vague description.

        :param llm: An instance of LLM.
        :param user_prompt: A vague description of the project.
        :return: An instance of CreateProjectPlan.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        system_prompt = PROJECT_PLAN_SYSTEM_PROMPT.strip()
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug(f"User Prompt:\n{user_prompt}")

        chat_message_list = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=user_prompt,
            )
        ]
        
        sllm = llm.as_structured_llm(GoalDefinition)

        logger.debug("Starting LLM chat interaction.")
        start_time = time.perf_counter()
        try:
            chat_response = sllm.chat(chat_message_list)
        except Exception as e:
            llm_error = LLMChatError(cause=e)
            logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
            logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e

        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))
        response_byte_count = len(chat_response.message.content.encode('utf-8'))
        logger.info(f"LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")

        json_response = chat_response.raw.model_dump()

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        markdown = cls.convert_to_markdown(chat_response.raw)

        result = ProjectPlan(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            markdown=markdown
        )
        logger.debug("CreateProjectPlan instance created successfully.")
        return result
        
    def to_dict(self, include_metadata=True, include_user_prompt=True, include_system_prompt=True) -> dict:
        d = self.response.copy()
        if include_metadata:
            d['metadata'] = self.metadata
        if include_user_prompt:
            d['user_prompt'] = self.user_prompt
        if include_system_prompt:
            d['system_prompt'] = self.system_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        d = self.to_dict()
        with open(file_path, 'w') as f:
            f.write(json.dumps(d, indent=2))

    @staticmethod
    def convert_to_markdown(document_details: GoalDefinition) -> str:
        """
        Convert the raw document details to markdown.
        """
        rows = []

        rows.append(f"**Goal Statement:** {document_details.goal_statement}")

        rows.append("\n## SMART Criteria\n")
        rows.append(f"- **Specific:** {document_details.smart_criteria.specific}")
        rows.append(f"- **Measurable:** {document_details.smart_criteria.measurable}")
        rows.append(f"- **Achievable:** {document_details.smart_criteria.achievable}")
        rows.append(f"- **Relevant:** {document_details.smart_criteria.relevant}")
        rows.append(f"- **Time-bound:** {document_details.smart_criteria.time_bound}")

        rows.append("\n## Dependencies\n")
        for dep in document_details.dependencies:
            rows.append(f"- {dep}")

        rows.append("\n## Resources Required\n")
        for resource in document_details.resources_required:
            rows.append(f"- {resource}")

        rows.append("\n## Related Goals\n")
        for goal in document_details.related_goals:
            rows.append(f"- {goal}")

        rows.append("\n## Tags\n")
        for tag in document_details.tags:
            rows.append(f"- {tag}")

        rows.append("\n## Risk Assessment and Mitigation Strategies\n")
        rows.append("\n### Key Risks\n")
        for risk in document_details.risk_assessment_and_mitigation_strategies.key_risks:
            rows.append(f"- {risk}")

        rows.append("\n### Diverse Risks\n")
        for risk in document_details.risk_assessment_and_mitigation_strategies.diverse_risks:
            rows.append(f"- {risk}")

        rows.append("\n### Mitigation Plans\n")
        for plan in document_details.risk_assessment_and_mitigation_strategies.mitigation_plans:
            rows.append(f"- {plan}")

        rows.append("\n## Stakeholder Analysis\n")
        rows.append("\n### Primary Stakeholders\n")
        for stakeholder in document_details.stakeholder_analysis.primary_stakeholders:
            rows.append(f"- {stakeholder}")

        rows.append("\n### Secondary Stakeholders\n")
        for stakeholder in document_details.stakeholder_analysis.secondary_stakeholders:
            rows.append(f"- {stakeholder}")

        rows.append("\n### Engagement Strategies\n")
        for strategy in document_details.stakeholder_analysis.engagement_strategies:
            rows.append(f"- {strategy}")

        rows.append("\n## Regulatory and Compliance Requirements\n")
        rows.append("\n### Permits and Licenses\n")
        for permit in document_details.regulatory_and_compliance_requirements.permits_and_licenses:
            rows.append(f"- {permit}")

        rows.append("\n### Compliance Standards\n")
        for standard in document_details.regulatory_and_compliance_requirements.compliance_standards:
            rows.append(f"- {standard}")

        rows.append("\n### Regulatory Bodies\n")
        for body in document_details.regulatory_and_compliance_requirements.regulatory_bodies:
            rows.append(f"- {body}")

        rows.append("\n### Compliance Actions\n")
        for action in document_details.regulatory_and_compliance_requirements.compliance_actions:
            rows.append(f"- {action}")

        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

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

    llm = get_llm("ollama-llama3.1")
    # llm = get_llm("openrouter-paid-gemini-2.0-flash-001")
    # llm = get_llm("deepseek-chat")

    print(f"Query:\n{plan_prompt}\n\n")
    result = ProjectPlan.execute(llm, plan_prompt)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{result.markdown}")

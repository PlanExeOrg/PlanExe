"""
Conduct SWOT analysis of different types: business, personal, or other.

PROMPT> python -m worker_plan_internal.swot.swot_phase2_conduct_analysis
"""
import json
import time
import logging
from math import ceil
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class SWOTAnalysis(BaseModel):
    """
    Identify internal and external factors that are favorable and unfavorable to achieving goals.
    """
    strengths: list[str] = Field(description="List of strengths.")
    weaknesses: list[str] = Field(description="List of weaknesses.")
    opportunities: list[str] = Field(description="List of opportunities.")
    threats: list[str] = Field(description="List of threats.")
    recommendations: list[str] = Field(
        description="A list of specific actions that address weaknesses, mitigate threats, and capitalize on opportunities. Each action should be actionable, time-bound, and assigned to an owner."
    )
    strategic_objectives: list[str] = Field(
        description="Specific, measurable, achievable, relevant, and time-bound (SMART) goals that the project aims to achieve. These objectives should be directly informed by the SWOT analysis and provide a roadmap for action.",
    )
    assumptions: list[str] = Field(
        description="Hypotheses or conditions taken as true for the purpose of this SWOT. If any assumption changes, the analysis may need updating."
    )
    missing_information: list[str] = Field(
        description="Gaps in data, research, or stakeholder input. If these gaps are filled, the accuracy and utility of this SWOT will improve.",
    )
    user_questions: list[str] = Field(
        description="Five crucial, thought-provoking questions to guide the user in identifying and evaluating strengths, weaknesses, opportunities, and threats."
    )

CONDUCT_SWOT_ANALYSIS_BUSINESS_SYSTEM_PROMPT = """
You are a universal strategic consultant with expertise in project management, business analysis, and innovation across various industries.

Create a SWOT analysis for the following topic. 
Highlight the concept of a “killer app” (or “killer application”)—a single highly compelling use-case that can catalyze mainstream adoption. 
Include “killer application” under either Weaknesses (if it's missing) or Opportunities (if it can be developed). 
If relevant, discuss potential obstacles to creating that killer application.

1. Thorough Coverage
   - Capture relevant Strengths, Weaknesses, Opportunities, and Threats.
   - Consider both internal (organizational) and external (market, regulatory, societal, technological) factors.
   - Be specific enough to guide meaningful action.
   - Address the potential for any killer-app or flagship use-case that could significantly accelerate adoption or market penetration, if relevant to the domain.

2. Actionable Recommendations
   - Propose at least three (3) to five (5) concrete actions that address Weaknesses, mitigate Threats, and capitalize on Opportunities.
   - Each recommendation should be time-bound, with clear ownership or stakeholder responsibility where possible.

3. Strategic Objectives
   - Provide three (3) to five (5) SMART (Specific, Measurable, Achievable, Relevant, Time-bound) objectives aligned with the SWOT findings.

4. Assumptions & Missing Information
   - State any assumptions made or conditions presumed.
   - Identify gaps in data or research that, if filled, would lead to a stronger analysis.

5. Critical User Questions
   - Present five (5) thought-provoking questions to help the user or stakeholders delve deeper into the SWOT findings, validating or challenging them as needed.

Approach each analysis as if you were an experienced consultant preparing a structured, concise, and well-reasoned report for decision-makers. 
If any domain-specific details are missing, note them under "Missing Information."

Keep your tone professional, constructive, and user-friendly.
"""

CONDUCT_SWOT_ANALYSIS_PERSONAL_SYSTEM_PROMPT = """
You are a universal strategic consultant with expertise in project management, business analysis, and innovation across various industries, specializing in personal development and growth strategies. Your focus is on creating highly detailed and actionable plans based on thorough self-assessment, with a strong emphasis on achieving transformative personal goals.

Create a highly detailed and comprehensive SWOT analysis for the following personal topic. The analysis MUST be structured around a central "Flagship Goal"—a long-term, overarching aspiration that represents significant personal growth, development, or well-being. This Flagship Goal should be the core focus of the analysis, with EACH SWOT element explicitly and thoroughly explained in relation to its achievement.

**1. Flagship Goal/Transformative Skill (Definition, Explanation, and Potential Obstacles):**

*   **DEFINE:** Clearly and concisely define the Flagship Goal. This should be a long-term, overarching aspiration, similar to a personal vision statement. Be specific.
*   **EXPLAIN:** Explain *in detail* why this goal is considered transformative and its potential impact on your life. What specific positive changes will it bring? How will it enhance your well-being or contribute to your personal fulfillment? Provide concrete examples. Aim for at least 50 words in this explanation.
*   **OBSTACLES:** Identify and describe *in detail* potential obstacles or challenges that might hinder the achievement of this Flagship Goal. Explain *why* these are obstacles and what their potential impact could be. Aim for at least 50 words in this explanation.

**2. SWOT Analysis (Directly and Explicitly Related to the Flagship Goal - CHAIN-OF-THOUGHT REQUIRED):**

For each SWOT element, explicitly explain its connection to the Flagship Goal using a chain-of-thought approach. Explain your reasoning step-by-step. Aim for at least 30 words per SWOT point.

*   **Strengths:** Internal attributes, resources, or advantages that will *support* the achievement of the Flagship Goal. Explain *how* they will provide support.
*   **Weaknesses:** Internal limitations, shortcomings, or disadvantages that will *hinder* the achievement of the Flagship Goal. Explain *how* they will create obstacles.
*   **Opportunities:** External factors, circumstances, or trends that can be *leveraged* to facilitate the achievement of the Flagship Goal. Explain *how* they can be leveraged.
*   **Threats:** External factors, challenges, or obstacles that could *impede* or prevent the achievement of the Flagship Goal. Explain *how* they could pose a threat.

**3. Actionable Recommendations:**

*   Propose at least three (3) to five (5) concrete actions that directly address Weaknesses, mitigate Threats, and capitalize on Opportunities, all in service of achieving the Flagship Goal.
*   Each recommendation should be:
    *   **Action-oriented:** Use strong action verbs (e.g., "I will research," "I will implement," "I will join").
    *   **Time-bound:** Include specific dates, timeframes, or deadlines.
    *   **Assigned to the individual:** Use "I will..." statements to clearly define personal responsibility.
    *   **Specific and Measurable (where possible):** Make the actions as specific as possible and include metrics or indicators of success. Explain *why* you chose these actions. Aim for at least 25 words per recommendation.

**4. Personal Objectives (SMART - Supporting the Flagship Goal):**

*   Provide three (3) to five (5) SMART (Specific, Measurable, Achievable, Relevant, Time-bound) objectives that represent concrete steps towards achieving the Flagship Goal. These objectives should be directly derived from the SWOT analysis and the recommendations. Explain *how* each objective contributes to the Flagship Goal. Aim for at least 25 words per objective.

**5. Assumptions & Missing Information:**

*   State any assumptions made or conditions presumed, explaining *why* they are being made and their potential impact on the analysis and the Flagship Goal.
*   Identify gaps in self-awareness or information that, if filled, would lead to a stronger analysis. Explain *how* this missing information could be obtained and its importance to the Flagship Goal. Aim for at least 25 words per assumption/missing information point.

**6. Critical Reflection Questions (Focused on the Flagship Goal):**

*   Present five (5) thought-provoking questions specifically tailored to the SWOT findings and the Flagship Goal. These questions should:
    *   Challenge assumptions related to the Flagship Goal.
    *   Explore potential consequences of achieving (or not achieving) the Flagship Goal.
    *   Encourage deeper self-reflection on the motivations, values, and beliefs connected to the Flagship Goal. Explain *why* you are asking each question and what insights you hope to gain. Aim for at least 25 words per question.

**7. Contingency/Backup Plan (Addressing Potential Setbacks):**

*   For each major obstacle or threat identified in the SWOT analysis or Flagship Goal section, develop a specific contingency or backup plan.
*   Explain what actions you will take if the original plan encounters difficulties or fails to produce the desired results.
*   Be specific and consider alternative strategies, resources, or approaches. Aim for at least 30 words per contingency plan.

**Structure:**

1.  Flagship Goal/Transformative Skill (Definition, Detailed Explanation, and Detailed Potential Obstacles)
2.  SWOT Analysis (Directly and Explicitly Related to the Flagship Goal - CHAIN-OF-THOUGHT REQUIRED)
    *   Strengths
    *   Weaknesses
    *   Opportunities
    *   Threats
3.  Actionable Recommendations
4.  Personal Objectives (SMART)
5.  Assumptions & Missing Information
6.  Critical Reflection Questions
7.  Contingency/Backup Plan

Approach each analysis as if you were an experienced life coach preparing a structured, concise, and well-reasoned plan for personal development. If any domain-specific details are missing, note them under "Missing Information."

Keep your tone introspective, constructive, and encouraging.
"""

CONDUCT_SWOT_ANALYSIS_OTHER_SYSTEM_PROMPT = """
You are a universal strategic consultant with broad expertise across academic, technical, artistic, or other
general topics that do not fall clearly under personal development or business.

Importantly:
- Avoid business or commercial considerations (e.g., profit margin, competition, marketing) unless the user’s
  request explicitly includes them.
- Focus on the inherent, creative, or technical aspects of the topic.

Create a general (non-business, non-personal) SWOT analysis for the following topic:

INSERT_USER_TOPIC_HERE
INSERT_USER_SWOTTYPEDETAILED_HERE

**Do not** discuss budgets, revenue, profit margins, or marketing strategies. 
Instead, focus on general, academic, educational, creative, or technical factors. 

1. **Strengths & Weaknesses**
   - Provide at least 2–3 bullet points each.
   - Explain the relevance to this specific topic (technical, conceptual, or other).
   - At least two sentences explaining each bullet.

2. **Opportunities & Threats**
   - Consider external factors such as user interest, research trends, or potential community engagement.
   - Avoid referencing commercial viability or market competition unless the user specifically asks.
   - At least two sentences explaining each bullet.

3. **Recommendations**
   - Propose 3–5 actionable steps or suggestions related to improving, expanding, or refining the concept.

4. **Strategic Objectives**
   - Suggest 3–5 goals that are relevant to a general/technical/creative context (e.g., performance milestones,
     educational outcomes, user engagement—only if relevant and **not** financial targets).

5. **Assumptions & Missing Information**
   - List any assumptions about available tools, environments, or resources that are relevant in a general context.
   - Identify any knowledge gaps (e.g., hardware specs, user skill level) that would further refine this analysis.

6. **User Questions**
   - Provide exactly five (5) well-formed questions that encourage deeper reflection on the technical or conceptual
     aspects of the topic.
   - Avoid placeholders. If no further questions are relevant, reframe them to broader conceptual or creative prompts.

Remember:
- Only discuss budgets, commercial viability, or profit motives if explicitly mentioned in the user topic.
- Emphasize creativity, innovation, technical feasibility, or educational value wherever possible.
"""

def swot_phase2_conduct_analysis(llm: LLM, user_prompt: str, system_prompt: str) -> dict:
    """
    Invoke LLM to make a SWOT analysis.
    """
    if not isinstance(llm, LLM):
        raise ValueError("Invalid LLM instance.")
    if not isinstance(user_prompt, str):
        raise ValueError("Invalid user_prompt.")
    if not isinstance(system_prompt, str):
        raise ValueError("Invalid system_prompt.")

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

    sllm = llm.as_structured_llm(SWOTAnalysis)
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
    json_response['metadata'] = metadata
    return json_response

if __name__ == "__main__":
    from worker_plan_internal.prompt.prompt_catalog import PromptCatalog
    from worker_plan_internal.llm_factory import get_llm

    swot_type = 0
    if swot_type == 0:
        system_prompt = CONDUCT_SWOT_ANALYSIS_BUSINESS_SYSTEM_PROMPT
    elif swot_type == 1:
        system_prompt = CONDUCT_SWOT_ANALYSIS_PERSONAL_SYSTEM_PROMPT
    elif swot_type == 2:
        system_prompt = CONDUCT_SWOT_ANALYSIS_OTHER_SYSTEM_PROMPT
    else:
        raise ValueError(f"Invalid SWOT analysis type: {swot_type}")
        
    prompt_catalog = PromptCatalog()
    prompt_catalog.load_example_swot_prompts()
    prompt_item = prompt_catalog.find("427e5163-cefa-46e8-b1d0-eb12be270e19")
    if not prompt_item:
        raise ValueError("Prompt item not found.")
    user_prompt = prompt_item.prompt
    
    llm = get_llm("ollama-llama3.1")

    json_response = swot_phase2_conduct_analysis(llm, user_prompt, system_prompt.strip())
    print(json.dumps(json_response, indent=2))

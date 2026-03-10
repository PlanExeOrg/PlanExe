"""
Narrow down what documents to find by identifying the most relevant documents and removing the rest (duplicates and irrelevant documents).

https://en.wikipedia.org/wiki/Pareto_principle

This module analyzes document lists to identify:
- Duplicate documents (near identical or similar documents)
- Irrelevant documents (documents that don't align with project goals)

The result is a cleaner, more focused list of essential documents.

PROMPT> python -m worker_plan_internal.document.filter_documents_to_find
"""
import os
import json
import time
import logging
from math import ceil
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.assume.identify_purpose import IdentifyPurpose, PlanPurposeInfo, PlanPurpose
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

# The number of documents to keep. It may be less or greater than this number
# Ideally we don't want to throw away any documents.
# There can be +50 documents and then it can be overwhelming to keep an overview.
# Thus only focus a handful of the most important documents.
PREFERRED_DOCUMENT_COUNT = 5

class DocumentImpact(str, Enum):
    """Enum to indicate the assessed impact of a document for the initial project phase."""
    critical = 'Critical' # Absolutely essential for project viability/start, major risk mitigation
    high = 'High'         # Very important for key decisions/planning steps/risk reduction
    medium = 'Medium'     # Useful for context or less critical initial tasks
    low = 'Low'           # Minor relevance for the initial phase or needed much later

class DocumentItem(BaseModel):
    id: int = Field(
        description="The ID of the document being evaluated."
    )
    rationale: str = Field(
        description="The reason justifying the assigned impact rating, linked to the project plan's critical goals, risks, or initial tasks."
    )
    impact_rating: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="The assessed impact level of the document for the initial project phase, based on the 80/20 principle."
    )

class DocumentImpactAssessmentResult(BaseModel):
    """The result of assessing the impact of a list of documents."""
    document_list: List[DocumentItem] = Field(
        description="List of documents with their assessed impact rating for the initial phase."
    )
    summary: str = Field(
        description="A summary highlighting the critical and high impact documents identified as most vital for the project start (80/20)."
    )

FILTER_DOCUMENTS_TO_FIND_BUSINESS_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in project planning documentation prioritization, applying the 80/20 principle (Pareto principle). Your task is to analyze a list of potential documents (from user input) against a provided project plan (also from user input). Evaluate each document's **impact** on the **critical initial phase** of the project.

**Goal:** Identify the vital few documents (the '20%') that will provide the most value (the '80%') right at the project's start. This means focusing on documents essential for:
1.  **Establishing Core Feasibility:** Can the project fundamentally work?
2.  **Defining Core Strategy/Scope:** What are we *actually* doing initially?
3.  **Addressing Major Risks:** Mitigating the highest-priority risks identified *in the plan*.
4.  **Meeting Non-Negotiable Prerequisites:** Fulfilling mandatory requirements to even begin (e.g., foundational compliance, key data for planning).

**Output Format:**
Respond with a JSON object matching the `DocumentImpactAssessmentResult` schema. For each document:
- Provide its original `id`.
- Assign an `impact_rating` using the `DocumentImpact` enum ('Critical', 'High', 'Medium', 'Low').
- Provide a detailed `rationale` explaining *why* that specific impact rating was chosen. **The rationale MUST link the document's content directly to critical project goals, major risks, key decisions, essential analyses, or uncertainties mentioned in the provided project plan for the initial phase.**

**Impact Rating Definitions (Assign ONE per document):**
- **Critical:** Absolutely essential for the initial phase. Project cannot realistically start, core feasibility cannot be assessed, or a top-tier risk (per the plan) cannot be addressed without this. Represents a non-negotiable prerequisite. (This is the core of the 80/20 focus).
- **High:** Very important for the initial phase. Significantly clarifies major uncertainties mentioned in the plan, enables core strategic decisions, provides essential data for key initial analyses, or addresses a significant risk.
- **Medium:** Useful context for the initial phase. Supports secondary planning tasks, provides background information, or addresses lower-priority risks/tasks. Helpful but not strictly required for the *most critical* initial decisions/actions.
- **Low:** Minor relevance for the *initial phase*. Might be useful much later, provides tangential information, or is superseded by higher-impact documents.

**Rationale Requirements (MANDATORY):**
- **MUST** justify the assigned `impact_rating`.
- **MUST** explicitly reference elements from the **user-provided project plan** (e.g., "Needed to address Risk #1 identified in the plan," "Provides data for the market analysis step mentioned," "Required for the 'Regulatory Compliance Assessment' goal").
- **Consider Overlap:** If two documents provide similar high-impact information, assign the highest rating to the most comprehensive or foundational one. Note the overlap in the rationale of the lower-rated document (e.g., "High: Provides important context, though some overlaps with ID [X]'s critical data."). Avoid assigning 'Critical' to multiple highly overlapping documents unless truly distinct aspects are covered.

**Forbidden Rationales:** Single words or generic phrases without linkage to the plan.

**Final Output:**
Produce a single JSON object containing `document_list` (with impact ratings and detailed, plan-linked rationales) and a `summary`.

The `summary` MUST provide a qualitative assessment based on the impact ratings you assigned:
1.  **Relevance Distribution:** Characterize the overall list. Were most documents low impact ('Low'/'Medium'), indicating the initial list was broad or unfocused? Or were many documents assessed as 'High' or 'Critical', suggesting the list was generally relevant to the initial phase?
2.  **Prioritization Clarity:** Comment on how easy it was to apply the 80/20 rule. Was there a clear distinction with only a few 'Critical'/'High' impact documents standing out? Or were there many documents clustered in the 'High'/'Medium' categories, making it difficult to isolate the truly vital few? **Do NOT simply list the documents in the summary.**

Strictly adhere to the schema and instructions, especially for the `rationale` and the new `summary` requirements.
"""

FILTER_DOCUMENTS_TO_FIND_PERSONAL_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in prioritizing information and potential documents for personal projects, applying the 80/20 principle (Pareto principle). Your task is to analyze a list of potential information sources or documents (from user input) against a provided personal project plan (also from user input). Evaluate each item's **impact** on the **critical initial phase** of the personal project.

**Goal:** Identify the vital few pieces of information or documents (the '20%') that will provide the most value (the '80%') right at the project's start. This means focusing on items essential for:
1.  **Establishing Personal Feasibility:** Can *I* realistically do this? Is it achievable given my current situation, resources, health, time, or budget? (e.g., Can I afford the trip? Is this fitness goal safe for me? Is now a feasible time for a major life change?)
2.  **Defining Initial Steps & Strategy:** What are *my* concrete first actions? What's the core approach for the beginning? (e.g., What's the initial diet/exercise plan? What's the travel itinerary for the first week? What's the party theme & initial guest list? What are the first steps for the home project? What key factors need immediate consideration for the life decision?)
3.  **Addressing Major Personal Risks/Obstacles:** Mitigating the highest-priority personal risks or roadblocks identified *in the plan*. (e.g., Risk of injury in fitness plan? Budget overruns for trip/renovation? Emotional impact of a decision? Losing motivation? Conflicting schedules?)
4.  **Meeting Non-Negotiable Personal Prerequisites:** Fulfilling mandatory requirements to even begin. (e.g., Getting a passport/visa? Doctor's check-up? Securing financing? Getting partner agreement? Basic supplies for a hobby? Essential skills check?)

**Output Format:**
Respond with a JSON object matching the `DocumentImpactAssessmentResult` schema. For each document/information source:
- Provide its original `id`.
- Assign an `impact_rating` using the `DocumentImpact` enum ('Critical', 'High', 'Medium', 'Low').
- Provide a detailed `rationale` explaining *why* that specific impact rating was chosen. **The rationale MUST link the item's content directly to critical personal goals, major risks, key decisions, essential preparations, or uncertainties mentioned in the provided project plan for the initial phase.**

**Impact Rating Definitions (Assign ONE per item):**
- **Critical:** Absolutely essential for the initial phase. The project cannot realistically start, core feasibility cannot be assessed, or a top-tier personal risk/obstacle (per the plan) cannot be addressed without this information. Represents a non-negotiable prerequisite. (This is the core of the 80/20 focus). *Example: Visa requirements for an imminent trip, Doctor's fitness clearance before starting exercise.*
- **High:** Very important for the initial phase. Significantly clarifies major uncertainties mentioned in the plan, enables core decisions about the initial approach, provides essential details for key preparations, or addresses a significant personal risk. *Example: Detailed travel guide for initial destination, specific workout routines, contractor quotes for renovation phase 1.*
- **Medium:** Useful context for the initial phase. Supports secondary planning tasks, provides background information, helps explore options, or addresses lower-priority risks/tasks. Helpful but not strictly required for the *most critical* initial decisions/actions. *Example: General travel blogs, nutrition guidelines, inspirational photos for a project.*
- **Low:** Minor relevance for the *initial phase*. Might be useful much later in the project, provides tangential information, or is superseded by higher-impact items. *Example: Information about a destination visited later in a trip, advanced techniques for a skill not yet started, details about finishing touches for a long home project.*

**Rationale Requirements (MANDATORY):**
- **MUST** justify the assigned `impact_rating`.
- **MUST** explicitly reference elements from the **user-provided project plan** (e.g., "Needed to address the 'Risk of Injury' identified in the plan," "Provides cost estimates needed for the 'Budgeting' step," "Required for the 'Passport Application' prerequisite").
- **Consider Overlap:** If two items provide similar high-impact information, assign the highest rating to the most comprehensive or foundational one. Note the overlap in the rationale of the lower-rated item (e.g., "High: Provides useful budget insights, though some overlaps with ID [X]'s critical financial assessment."). Avoid assigning 'Critical' to multiple highly overlapping items unless truly distinct aspects are covered.

**Forbidden Rationales:** Single words or generic phrases without linkage to the plan.

**Final Output:**
Produce a single JSON object containing `document_list` (with impact ratings and detailed, plan-linked rationales) and a `summary`.

The `summary` MUST provide a qualitative assessment based on the impact ratings you assigned:
1.  **Relevance Distribution:** Characterize the overall list. Were most items low impact ('Low'/'Medium'), indicating the initial list was broad or unfocused? Or were many items assessed as 'High' or 'Critical', suggesting the list was generally relevant to the initial phase?
2.  **Prioritization Clarity:** Comment on how easy it was to apply the 80/20 rule. Was there a clear distinction with only a few 'Critical'/'High' impact items standing out? Or were there many items clustered in the 'High'/'Medium' categories, making it difficult to isolate the truly vital few? **Do NOT simply list the items in the summary.**

Strictly adhere to the schema and instructions, especially for the `rationale` and the new `summary` requirements.
"""

FILTER_DOCUMENTS_TO_FIND_OTHER_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in prioritizing information and potential documents for analytical, theoretical, or technical implementation projects, applying the 80/20 principle (Pareto principle). These projects fall into the 'Other' category, distinct from typical business or personal goals. Your task is to analyze a list of potential information sources or documents (from user input) against a provided project plan/description (also from user input). Evaluate each item's **impact** on the **critical initial phase** of this analytical or technical endeavor.

**Goal:** Identify the vital few pieces of information or documents (the '20%') that will provide the most value (the '80%') right at the project's start. This means focusing on items essential for:
1.  **Establishing Analytical/Technical Feasibility:** Can the analysis be performed? Is the theoretical exploration grounded? Is the technical implementation possible with available tools/knowledge? (e.g., Is the required dataset accessible? Does the core theory hold up to initial scrutiny? Are the necessary libraries/APIs available and understood?)
2.  **Defining Core Scope & Methodology:** What is the precise question being answered, concept explored, or function being built? What is the primary method, algorithm, or framework to be used initially? (e.g., What specific variables will the simulation model? What is the core philosophical argument to analyze? What are the input/output specifications for the code?)
3.  **Addressing Foundational Knowledge Gaps & Methodological Risks:** Mitigating risks related to misunderstanding core concepts, using flawed methodology, or lacking essential foundational information identified *in the plan*. (e.g., Risk of using inappropriate statistical methods? Lack of understanding of a key prerequisite theorem? Data interpretation challenges?)
4.  **Meeting Non-Negotiable Technical/Analytical Prerequisites:** Fulfilling mandatory requirements to even begin the analysis or implementation. (e.g., Access to a specific database? Installation of required software? Understanding a specific mathematical notation or programming paradigm? Defining the simulation's boundary conditions?)

**Output Format:**
Respond with a JSON object matching the `DocumentImpactAssessmentResult` schema. For each document/information source:
- Provide its original `id`.
- Assign an `impact_rating` using the `DocumentImpact` enum ('Critical', 'High', 'Medium', 'Low').
- Provide a detailed `rationale` explaining *why* that specific impact rating was chosen. **The rationale MUST link the item's content directly to the plan's core analytical questions, theoretical goals, technical requirements, specified methodology, data needs, or identified knowledge gaps/risks for the initial phase.**

**Impact Rating Definitions (Assign ONE per item):**
- **Critical:** Absolutely essential for the initial phase. The analysis/implementation cannot start, core feasibility cannot be assessed, or a fundamental methodological risk/knowledge gap (per the plan) cannot be addressed without this. Represents a non-negotiable prerequisite for the *specific analytical or technical task*. (This is the core of the 80/20 focus). *Example: The primary dataset for analysis, the seminal paper defining the theory being explored, API documentation for a required library, the formal problem definition.*
- **High:** Very important for the initial phase. Significantly clarifies the chosen methodology, provides essential context for interpreting foundational concepts, defines key parameters for implementation/simulation, or addresses a major risk in the analytical/technical process. *Example: Papers detailing the specific statistical test planned, documentation explaining a core algorithm, sample input/output data for coding, definitions of key terms.*
- **Medium:** Useful context for the initial phase. Supports understanding related concepts, provides background information on alternative methods, helps refine secondary parameters, or addresses lower-priority technical/analytical risks. Helpful but not strictly required for the *most critical* initial analysis/implementation steps. *Example: Survey papers of related fields, documentation for auxiliary tools, historical context of the problem.*
- **Low:** Minor relevance for the *initial phase*. Might be useful for later stages of analysis/implementation, provides tangential information, discusses niche applications, or is superseded by higher-impact items. *Example: Papers on advanced extensions of the core theory, implementation details for optional features, performance comparisons of tools not yet chosen.*

**Rationale Requirements (MANDATORY):**
- **MUST** justify the assigned `impact_rating`.
- **MUST** explicitly reference elements from the **user-provided project plan/description** (e.g., "Needed to define the 'Input Parameters' specified in the plan," "Provides the 'Core Dataset' required for the analysis," "Explains the 'Statistical Method' chosen," "Addresses the risk of 'Misinterpreting Theorem X' mentioned").
- **Consider Overlap:** If two items provide similar high-impact information, assign the highest rating to the most comprehensive or foundational one. Note the overlap in the rationale of the lower-rated item (e.g., "High: Details the algorithm, though ID [X] provides the critical formal specification."). Avoid assigning 'Critical' to multiple highly overlapping items unless truly distinct aspects are covered.

**Forbidden Rationales:** Single words or generic phrases without linkage to the plan.

**Final Output:**
Produce a single JSON object containing `document_list` (with impact ratings and detailed, plan-linked rationales) and a `summary`.

The `summary` MUST provide a qualitative assessment based on the impact ratings you assigned:
1.  **Relevance Distribution:** Characterize the overall list. Were most items low impact ('Low'/'Medium'), indicating the initial list was broad or peripheral to the core analysis/task? Or were many items assessed as 'High' or 'Critical', suggesting the list was generally relevant to the initial analytical/technical phase?
2.  **Prioritization Clarity:** Comment on how easy it was to apply the 80/20 rule. Was there a clear distinction with only a few 'Critical'/'High' impact items standing out as foundational for the analysis/task? Or were there many items clustered in the 'High'/'Medium' categories, making it difficult to isolate the truly vital few? **Do NOT simply list the items in the summary.**

Strictly adhere to the schema and instructions, especially for the `rationale` and the new `summary` requirements.
"""

@dataclass
class FilterDocumentsToFind:
    """
    Analyzes document lists to identify duplicates and irrelevant documents.
    """
    system_prompt: str
    user_prompt: str
    identified_documents_raw_json: list[dict]
    integer_id_to_document_uuid: dict[int, str]
    response: dict
    assessment_result: DocumentImpactAssessmentResult
    metadata: dict
    ids_to_keep: set[int]
    uuids_to_keep: set[str]
    filtered_documents_raw_json: list[dict]

    @staticmethod
    def process_documents_and_integer_ids(identified_documents_raw_json: list[dict]) -> tuple[list[dict], dict[int, str]]:
        """
        Prepare the documents for processing by the LLM.

        Reduce the number of fields in the documents to just the document name and the document description.
        Avoid using the uuid as the id, since it trend to confuses the LLM.
        Instead of uuid, use an integer id.
        """
        if not isinstance(identified_documents_raw_json, list):
            raise ValueError("identified_documents_raw_json is not a list.")

        # Only keep the 'document_name' and 'description' from each document and remove the rest.
        # Enumerate the documents with an integer id.
        process_documents = []
        integer_id_to_document_uuid = {}
        for doc in identified_documents_raw_json:
            if 'document_name' not in doc or 'description' not in doc or 'id' not in doc:
                logger.error(f"Document is missing required keys: {doc}")
                continue

            document_name = doc.get('document_name', '')
            document_description = doc.get('description', '')
            document_id = doc.get('id', '')

            current_index = len(process_documents)

            name = f"{document_name}\n{document_description}"
            dict_item = {
                'id': current_index,
                'name': name
            }
            process_documents.append(dict_item)
            integer_id_to_document_uuid[current_index] = document_id

        return process_documents, integer_id_to_document_uuid

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str, identified_documents_raw_json: list[dict], integer_id_to_document_uuid: dict[int, str], identify_purpose_dict: Optional[dict]) -> 'FilterDocumentsToFind':
        """
        Invoke LLM with the document details to analyze.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")
        if identify_purpose_dict is not None and not isinstance(identify_purpose_dict, dict):
            raise ValueError("Invalid identify_purpose_dict.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        if identify_purpose_dict is None:
            logging.info("No identify_purpose_dict provided, identifying purpose.")
            identify_purpose = IdentifyPurpose.execute(llm, user_prompt)
            identify_purpose_dict = identify_purpose.to_dict()
        else:
            logging.info("identify_purpose_dict provided, using it.")

        # Parse the identify_purpose_dict
        logging.debug(f"IdentifyPurpose json {json.dumps(identify_purpose_dict, indent=2)}")
        try:
            purpose_info = PlanPurposeInfo(**identify_purpose_dict)
        except Exception as e:
            logging.error(f"Error parsing identify_purpose_dict: {e}")
            raise ValueError("Error parsing identify_purpose_dict.") from e

        # Select the appropriate system prompt based on the purpose
        logging.info(f"FilterDocumentsToFind.execute: purpose: {purpose_info.purpose}")
        if purpose_info.purpose == PlanPurpose.business:
            system_prompt = FILTER_DOCUMENTS_TO_FIND_BUSINESS_SYSTEM_PROMPT
        elif purpose_info.purpose == PlanPurpose.personal:
            system_prompt = FILTER_DOCUMENTS_TO_FIND_PERSONAL_SYSTEM_PROMPT
        elif purpose_info.purpose == PlanPurpose.other:
            system_prompt = FILTER_DOCUMENTS_TO_FIND_OTHER_SYSTEM_PROMPT
        else:
            raise ValueError(f"Invalid purpose: {purpose_info.purpose}, must be one of 'business', 'personal', or 'other'. Cannot filter documents.")

        system_prompt = system_prompt.strip()

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

        sllm = llm.as_structured_llm(DocumentImpactAssessmentResult)
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

        assessment_result = chat_response.raw

        ids_to_keep = cls.extract_integer_ids_to_keep(assessment_result)
        uuids_to_keep_list = [integer_id_to_document_uuid[integer_id] for integer_id in ids_to_keep]
        uuids_to_keep = set(uuids_to_keep_list)

        # remove the documents that are not in the uuids_to_keep
        filtered_documents_raw_json = [doc for doc in identified_documents_raw_json if doc['id'] in uuids_to_keep]

        logger.info(f"IDs to keep: {ids_to_keep}")
        logger.info(f"UUIDs to keep: {uuids_to_keep}")
        logger.info(f"Filtered documents raw json length: {len(filtered_documents_raw_json)}")

        if len(filtered_documents_raw_json) != len(ids_to_keep):
            logger.info(f"identified_documents_raw_json: {json.dumps(identified_documents_raw_json, indent=2)}")
            logger.error(f"Filtered documents raw json length ({len(filtered_documents_raw_json)}) does not match ids_to_keep length ({len(ids_to_keep)}).")
            raise ValueError("Filtered documents raw json length does not match ids_to_keep length.")
    
        result = FilterDocumentsToFind(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            identified_documents_raw_json=identified_documents_raw_json,
            integer_id_to_document_uuid=integer_id_to_document_uuid,
            response=json_response,
            assessment_result=assessment_result,
            metadata=metadata,
            ids_to_keep=ids_to_keep,
            uuids_to_keep=uuids_to_keep,
            filtered_documents_raw_json=filtered_documents_raw_json
        )
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

    def save_filtered_documents(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.filtered_documents_raw_json, indent=2))

    @staticmethod
    def extract_integer_ids_to_keep(result: DocumentImpactAssessmentResult) -> set[int]:
        """
        Extract the most important documents from the result.
        """
        ids_to_critical = set()
        ids_to_high = set()
        ids_to_medium = set()
        ids_to_low = set()
        for item in result.document_list:
            if item.impact_rating == DocumentImpact.critical:
                ids_to_critical.add(item.id)
            elif item.impact_rating == DocumentImpact.high:
                ids_to_high.add(item.id)
            elif item.impact_rating == DocumentImpact.medium:
                ids_to_medium.add(item.id)
            elif item.impact_rating == DocumentImpact.low:
                ids_to_low.add(item.id)
            else:
                logger.error(f"Invalid impact_rating value: {item.impact_rating}, document_id: {item.id}. Removing the document.")
        
        ids_to_keep = set()
        ids_to_keep.update(ids_to_critical)
        if len(ids_to_keep) < PREFERRED_DOCUMENT_COUNT:
            ids_to_keep.update(ids_to_high)
        if len(ids_to_keep) < PREFERRED_DOCUMENT_COUNT:
            ids_to_keep.update(ids_to_medium)
        if len(ids_to_keep) < PREFERRED_DOCUMENT_COUNT:
            ids_to_keep.update(ids_to_low)

        if len(ids_to_keep) < PREFERRED_DOCUMENT_COUNT:
            logger.info(f"Fewer documents to keep than the desired count. Only {len(ids_to_keep)} documents found.")

        return ids_to_keep

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    plan_prompt = find_plan_prompt("5c4b4fee-267a-409b-842f-4833d86aa215")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    llm = get_llm("ollama-llama3.1")
    # llm = get_llm("openrouter-paid-gemini-2.0-flash-001")

    path = os.path.join(os.path.dirname(__file__), 'test_data', "eu_prep_identified_documents_to_find.json")
    with open(path, 'r', encoding='utf-8') as f:
        identified_documents_raw_json = json.load(f)

    process_documents, integer_id_to_document_uuid = FilterDocumentsToFind.process_documents_and_integer_ids(identified_documents_raw_json)

    print(f"integer_id_to_document_uuid: {integer_id_to_document_uuid}")

    query = (
        f"File 'plan.txt':\n{plan_prompt}\n\n"
        f"File 'documents.json':\n{process_documents}"
    )
    print(f"Query:\n{query}\n\n")

    result = FilterDocumentsToFind.execute(llm, query, identified_documents_raw_json, integer_id_to_document_uuid, identify_purpose_dict=None)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nIDs to keep:\n{result.ids_to_keep}")
    print(f"\n\nUUIDs to keep:\n{result.uuids_to_keep}")

    print("\n\nFiltered documents:")
    print(json.dumps(result.filtered_documents_raw_json, indent=2))

"""
PROMPT> python -m worker_plan_internal.expert.expert_finder

Find experts that can take a look at the a document, such as a 'SWOT analysis' and provide feedback.

IDEA: Specify a number of experts to be obtained. Currently it's hardcoded 8.
When it's 4 or less, then there is no need to make a second call to the LLM model.
When it's 9 or more, then make multiple calls to the LLM model to get more experts.

Currently the number of experts is hardcoded to 4 per response.
4 experts per response is on the edge of what is doable in reasonable time on my developer computer when using local LLMs.
8 experts and it takes 45 seconds. This makes my feedback cycle painful slow.
The 2 first experts gets enriched with more info.
"""
import json
import time
import logging
from math import ceil
from uuid import uuid4
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage, MessageRole
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class Expert(BaseModel):
    expert_title: str = Field(description="Job title of the expert.")
    expert_knowledge: str = Field(description="Industry Knowledge/Specialization, specific industries or subfields where they have focused their career, such as: tech industry for an IT consultant, healthcare sector for a medical expert. **Must be a brief comma separated list**.")
    expert_why: str = Field(description="Why can this expert be of help. Area of expertise.")
    expert_what: str = Field(description="Describe what area of this document the role should advise about.")
    expert_relevant_skills: str = Field(description="Skills that are relevant to the document.")
    expert_search_query: str = Field(description="What query to use when searching for this expert.")

class ExpertDetails(BaseModel):
    experts: list[Expert] = Field(description="List of experts.")

# Works great with the old LLM "llama3.1:latest".
# Doesn't work with "google/gemini-2.0-flash-001", where the first LLM call works, but the subsequent LLM call fails.
EXPERT_FINDER_SYSTEM_PROMPT_1 = """
Professionals who can offer specialized perspectives and recommendations based on the document.

Ensure that each expert role directly aligns with specific sections or themes within the document.
This could involve linking particular strengths, weaknesses, opportunities, threats, extra sections, to the expertise required.

Diversity in the types of experts suggested by considering interdisciplinary insights that might not be 
immediately obvious but could offer unique perspectives on the document.

Account for geographical and contextual relevance, variations in terminology or regional differences that may affect the search outcome.

The "expert_search_query" field is a human readable text for searching in Google/DuckDuckGo/LinkedIn.

Find exactly 4 experts.
"""

# Works with "google/gemini-2.0-flash-001".
# Doesn't works the old LLM "llama3.1:latest", where the number of experts vary wildly.
EXPERT_FINDER_SYSTEM_PROMPT_2 = """
You produce specialists relevant to a user-provided document or plan.

OUTPUT CONTRACT
- Return ONE value: a valid JSON object only. No markdown, no prose, no backticks, no metadata.
- Root shape exactly:
  {"experts":[
    {"expert_title":"string",
     "expert_knowledge":"string",
     "expert_why":"string",
     "expert_what":"string",
     "expert_relevant_skills":"string",
     "expert_search_query":"string"}
  ]}
- Exactly 4 experts per response. Never more, never less.
- Strings only. No nulls. No "N/A". Use short, specific phrases.
- Keep each string ≤ 160 characters. If token pressure rises, shorten phrasing—never truncate JSON.

FIELD RULES
- expert_title: concise role label. Avoid fluff. No duplicates within this list.
- expert_knowledge: brief, comma-separated nouns/phrases (no sentences).
- expert_why: the unique reason THIS role is needed for THIS input.
- expert_what: the first concrete, high-leverage actions this role would take.
- expert_relevant_skills: brief, comma-separated skills; avoid repeating expert_knowledge verbatim.
- expert_search_query: 3–7 comma-separated search terms; no quotation marks or periods.

CONTEXT & DEDUP
- Maintain an international perspective unless the user input specifies a jurisdiction; then align to it.
- If the conversation already contains an assistant message with a JSON {"experts":[...]} from a previous step, treat those as “already selected” and DO NOT repeat any titles or near-duplicate roles. Produce 4 new, non-overlapping roles.

FORMAT GUARDRAILS
- Output must start with "{" and end with "}".
- No trailing commas anywhere.
- No extra keys beyond the schema.
- No line breaks are required; minified JSON preferred.

SELF-CHECK (silent)
Before emitting, verify: exactly 4 objects under "experts"; all fields present and non-empty; no duplication; JSON is valid and closed.
"""

# Hybrid of the previous two prompts, that seems to work with both "llama3.1:latest" and "google/gemini-2.0-flash-001".
EXPERT_FINDER_SYSTEM_PROMPT_3 = """
You are an expert strategist who identifies key professional roles needed to review and improve a user-provided document or plan.

GUIDING PRINCIPLES
- Direct Alignment: Ensure each expert directly corresponds to specific sections, themes, risks, or opportunities within the user's document (e.g., linking a 'Market Analyst' to the 'Opportunities' section of a SWOT analysis).
- Interdisciplinary Diversity: Suggest a mix of experts, including non-obvious but high-value roles that can offer unique, interdisciplinary insights.
- Contextual Relevance: Consider geographical and regional factors mentioned in the document that might influence the expertise required or how one might search for it.

OUTPUT CONTRACT
- Return ONE value: a valid JSON object only. No markdown, no prose, no backticks, no metadata.
- Root shape exactly:
  {"experts":[
    {"expert_title":"string",
     "expert_knowledge":"string",
     "expert_why":"string",
     "expert_what":"string",
     "expert_relevant_skills":"string",
     "expert_search_query":"string"}
  ]}
- Exactly 4 experts per response. Never more, never less.
- Strings only. No nulls. No "N/A". Use short, specific phrases.
- Keep each string ≤ 160 characters. If token pressure rises, shorten phrasing—never truncate JSON.

FIELD RULES
- expert_title: Concise, professional role label. Avoid fluff. No duplicates within this list.
- expert_knowledge: Brief, comma-separated list of nouns/phrases specifying industry knowledge (e.g., e-commerce logistics, medical device regulation).
- expert_why: The unique reason THIS role is needed for THIS input. **Link their expertise to a specific part of the document.**
- expert_what: The first concrete, high-leverage action this expert would take regarding the document.
- expert_relevant_skills: Brief, comma-separated skills; avoid repeating expert_knowledge verbatim.
- expert_search_query: 3–7 comma-separated search terms for a human to use on Google/LinkedIn. No quotation marks or periods.

CONTEXT & DEDUP
- Maintain an international perspective unless the user input specifies a jurisdiction; then align to it.
- If the conversation already contains an assistant message with a JSON {"experts":[...]} from a previous step, treat those as “already selected” and DO NOT repeat any titles or near-duplicate roles. Produce 4 new, non-overlapping roles.

FORMAT GUARDRAILS
- Output must start with "{" and end with "}".
- No trailing commas anywhere.
- No extra keys beyond the schema.
- No line breaks are required; minified JSON preferred.

SELF-CHECK (silent)
Before emitting, verify: exactly 4 objects under "experts"; all fields present and non-empty; no duplication; JSON is valid and closed.
"""

EXPERT_FINDER_SYSTEM_PROMPT = EXPERT_FINDER_SYSTEM_PROMPT_3

@dataclass
class ExpertFinder:
    """
    Find experts that can advise about the particular domain.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    expert_list: list[dict]

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, user_prompt: str) -> 'ExpertFinder':
        """
        Invoke LLM to find the best suited experts that can advise about attached file.
        """
        if not isinstance(llm_executor, LLMExecutor):
            raise ValueError("Invalid LLMExecutor instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid query.")

        system_prompt = EXPERT_FINDER_SYSTEM_PROMPT.strip()

        chat_message_list1 = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=user_prompt,
            )
        ]

        def execute_function1(llm: LLM) -> dict:
            sllm = llm.as_structured_llm(ExpertDetails)
            logger.debug("Starting LLM chat interaction 1.")
            start_time = time.perf_counter()
            chat_response = sllm.chat(chat_message_list1)
            end_time = time.perf_counter()
            duration = int(ceil(end_time - start_time))
            response_byte_count = len(chat_response.message.content.encode('utf-8'))
            logger.info(f"LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")
            
            metadata = dict(llm.metadata)
            metadata["llm_classname"] = llm.class_name()
            metadata["duration"] = duration
            metadata["response_byte_count"] = response_byte_count
            metadata["expert_count"] = len(chat_response.raw.experts)
            
            return {
                "chat_response": chat_response,
                "metadata": metadata,
            }

        try:
            result1 = llm_executor.run(execute_function1)
        except PipelineStopRequested:
            raise
        except Exception as e:
            llm_error = LLMChatError(cause=e, message="LLM chat interaction 1 failed")
            logger.error(f"{llm_error.message} [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e

        chat_response1 = result1["chat_response"]

        # Do a follow up question, for obtaining more experts.
        chat_message_assistant2 = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=chat_response1.message.content,
        )
        chat_message_user2 = ChatMessage(
            role=MessageRole.USER,
            content="4 more please",
        )
        chat_message_list2 = chat_message_list1.copy()
        chat_message_list2.append(chat_message_assistant2)
        chat_message_list2.append(chat_message_user2)

        def execute_function2(llm: LLM) -> dict:
            sllm = llm.as_structured_llm(ExpertDetails)
            logger.debug("Starting LLM chat interaction 2.")
            start_time = time.perf_counter()
            chat_response = sllm.chat(chat_message_list2)
            end_time = time.perf_counter()
            duration = int(ceil(end_time - start_time))
            response_byte_count = len(chat_response.message.content.encode('utf-8'))
            logger.info(f"LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")
            
            metadata = dict(llm.metadata)
            metadata["llm_classname"] = llm.class_name()
            metadata["duration"] = duration
            metadata["response_byte_count"] = response_byte_count
            metadata["expert_count"] = len(chat_response.raw.experts)
            
            return {
                "chat_response": chat_response,
                "metadata": metadata,
            }

        try:
            result2 = llm_executor.run(execute_function2)
        except PipelineStopRequested:
            raise
        except Exception as e:
            llm_error = LLMChatError(cause=e, message="LLM chat interaction 2 failed")
            logger.error(f"{llm_error.message} [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e

        chat_response2 = result2["chat_response"]

        metadata = {
            "result1": result1["metadata"],
            "result2": result2["metadata"]
        }

        json_response1 = json.loads(chat_response1.message.content)
        json_response2 = json.loads(chat_response2.message.content)

        json_response_merged = {}
        experts1 = json_response1.get('experts', [])
        experts2 = json_response2.get('experts', [])
        json_response_merged['experts'] = experts1 + experts2

        # Cleanup the json response from the LLM model, extract the experts.
        expert_list = []
        for expert in json_response_merged['experts']:
            uuid = str(uuid4())
            expert_dict = {
                "id": uuid,
                "title": expert['expert_title'],
                "knowledge": expert['expert_knowledge'],
                "why": expert['expert_why'],
                "what": expert['expert_what'],
                "skills": expert['expert_relevant_skills'],
                "search_query": expert['expert_search_query'],
            }
            expert_list.append(expert_dict)

        logger.info(f"Found {len(expert_list)} experts.")

        return ExpertFinder(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response_merged,
            metadata=metadata,
            expert_list=expert_list,
        )

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

    def save_cleanedup(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.expert_list, indent=2))
        

if __name__ == "__main__":
    import logging
    from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelFromName
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    path = os.path.join(os.path.dirname(__file__), 'test_data', 'expert_finder_user_prompt1.txt')
    with open(path, 'r', encoding='utf-8') as f:
        user_prompt = f.read()

    # Create LLMExecutor with fallback models
    llm_models = LLMModelFromName.from_names([
        # "openrouter-paid-gemini-2.0-flash-001",
        # "openrouter-paid-openai-gpt-4o-mini",
        "ollama-llama3.1", 
        # "openrouter-paid-qwen3-30b-a3b"
    ])
    llm_executor = LLMExecutor(llm_models=llm_models)

    # print(f"User prompt: {user_prompt}")
    result = ExpertFinder.execute(llm_executor, user_prompt)

    print("\n\nResponse:")
    print(json.dumps(result.to_dict(include_system_prompt=False, include_user_prompt=False), indent=2))

    print("\n\nExperts:")
    print(json.dumps(result.expert_list, indent=2))

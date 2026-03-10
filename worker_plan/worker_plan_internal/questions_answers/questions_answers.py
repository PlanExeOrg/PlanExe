"""
For a reader of the plan that is unfamiliar with the domain, provide a list of Q&A pairs that are relevant to the plan.

PROMPT> python -m worker_plan_internal.questions_answers.questions_answers
"""
import html
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class QuestionAnswerPair(BaseModel):
    item_index: int = Field(
        description="Enumeration of the question answer pairs, starting from 1."
    )
    question: str = Field(
        description="The question."
    )
    answer: str = Field(
        description="The answer to the question."
    )
    rationale: str = Field(
        description="Explain why this particular question answer pair is suggested."
    )

class DocumentDetails(BaseModel):
    question_answer_pairs: list[QuestionAnswerPair] = Field(
        description="List of question answer pairs."
    )
    summary: str = Field(
        description="Providing a high level context."
    )

QUESTION_ANSWER_SYSTEM_PROMPT = """
You are a world-class expert in analyzing project documentation and generating insightful Questions and Answers (Q&A) for a reader who needs clarification on key aspects of the project as presented in the document. Your goal is to analyze the user's provided project description (the plan document itself), identify key concepts, terms, strategies, risks, ethical considerations, and controversial aspects, and generate a JSON response that strictly follows the `DocumentDetails` and `QuestionAnswerPair` models provided below.

Analyze the provided project description thoroughly. Identify the core subject area (the domain), significant terms, concepts, strategies, and importantly, the key risks, ethical considerations, and controversial aspects used within the document. Your task is to generate relevant Q&A that clarify these key aspects for a reader who may have some general business or project knowledge but is not necessarily an expert in this specific domain or the particular challenges highlighted in this project.

For each Question and Answer pair:
1.  Generate a clear `question` about a key concept, term, approach, risk, or controversial aspect presented in the provided document. Frame the question as something a reader of this report might ask for better understanding or clarification, particularly regarding the project's unique challenges or sensitive aspects.
2.  Provide a concise, accurate, and relevant `answer` to the question. The answer should explain the concept, term, or address the risk/controversy as it applies within the context of the project described in the document, using appropriate language (defining necessary terms). Base the answer on the information available in the document or general knowledge required to understand the document's terminology. Explicitly acknowledge and explain the sensitive or controversial points that the document itself raises, while remaining factual and within safety guidelines.
3.  Provide a `rationale` that explains why this specific Q&A helps a reader understand the nuances and challenges presented in this project document. Link the Q&A to the project's specific domain, goals, risks, ethical considerations, or controversial aspects as described in the text.

Generate 5 Question and Answer pairs. Ensure `item_index` starts at 1 and increments for each pair.

Use the following JSON models:

### DocumentDetails
-   **question_answer_pairs** (list of QuestionAnswerPair): A list of Question and Answer pairs generated based on the key concepts, terms, risks, and controversial aspects presented in the project document. **This list must contain 5 items.**
-   **summary** (string): A brief, high-level summary of the generated Q&A section, explaining that it covers key concepts, risks, and terms from the project document to aid understanding.

### QuestionAnswerPair
-   **item_index** (integer): Enumeration of the question answer pairs, starting from 1.
-   **question** (string): A question about a key concept, term, risk, or controversial aspect from the project document.
-   **answer** (string): A clear explanation of the concept, term, risk, or controversial aspect as it relates to the project, based on the document or relevant general knowledge.
-   **rationale** (string): An explanation of why this Q&A helps a reader understand the document's content, particularly its challenges or sensitive points.

## Additional Instructions

1.  **Analyze the Document's Content:** Use the provided text to identify the project's domain, key terms, concepts, strategies, risks, ethical concerns, and controversial aspects as described in the document.
2.  **Generate Q&A from Document Concepts:** Generate Q&A that explain these specific concepts, risks, and controversial points from the document's perspective.
3.  **Target Project-Relevant Level:** Assume the reader can handle some project or business terminology but needs clarification on domain-specific or methodology-specific terms and the specific challenges/controversies highlighted in the document.
4.  **Base Answers on Document/Relevant Knowledge:** Provide answers consistent with the document's content or general knowledge directly relevant to understanding the terms/concepts/risks in that project context. Address the controversial points raised in the document factually and directly.
5.  **Rationale Focus:** The `rationale` must explain the value of the Q&A for understanding this specific document's content, especially its challenging aspects.
6.  **Strict JSON Format:** The final output MUST be a JSON object strictly conforming to the `DocumentDetails` model. Do not include any conversational text or explanations outside the JSON object.
7.  **Language:** Generate the Q&A in the language of the user's text.
8.  **Adhere to Safety Guidelines:** Ensure all responses are within safety guidelines, while still directly addressing the sensitive/controversial topics as they are presented in the document.
"""

SECOND_USER_PROMPT = "Generate 5 additional question and answer pairs from the document, focusing on clarifying the risks, ethical considerations, controversial aspects, or broader implications discussed in the plan."

@dataclass
class QuestionsAnswers:
    """
    Identify what questions and answers are relevant to the plan.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str
    html: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'QuestionsAnswers':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = QUESTION_ANSWER_SYSTEM_PROMPT.strip()

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

        sllm = llm.as_structured_llm(DocumentDetails)
        start_time = time.perf_counter()
        try:
            chat_response1 = sllm.chat(chat_message_list1)
        except Exception as e:
            llm_error = LLMChatError(cause=e)
            logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
            logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e

        end_time = time.perf_counter()
        duration1 = int(ceil(end_time - start_time))
        response_byte_count1 = len(chat_response1.message.content.encode('utf-8'))
        logger.info(f"LLM chat interaction 1 completed in {duration1} seconds. Response byte count: {response_byte_count1}")

        # Do a follow up question, for obtaining more Q&A pairs
        chat_message_assistant2 = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=chat_response1.message.content,
        )
        chat_message_user2 = ChatMessage(
            role=MessageRole.USER,
            content=SECOND_USER_PROMPT.strip()
        )
        chat_message_list2 = chat_message_list1.copy()
        chat_message_list2.append(chat_message_assistant2)
        chat_message_list2.append(chat_message_user2)

        logger.debug("Starting LLM chat interaction 2.")
        start_time = time.perf_counter()
        try:
            chat_response2 = sllm.chat(chat_message_list2)
        except Exception as e:
            llm_error = LLMChatError(cause=e, message="LLM chat interaction 2 failed")
            logger.debug(f"{llm_error.message} [{llm_error.error_id}]: {e}")
            logger.error(f"{llm_error.message} [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e

        end_time = time.perf_counter()
        duration2 = int(ceil(end_time - start_time))
        response_byte_count2 = len(chat_response2.message.content.encode('utf-8'))
        logger.info(f"LLM chat interaction 2 completed in {duration2} seconds. Response byte count: {response_byte_count2}")

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration1"] = duration1
        metadata["duration2"] = duration2
        metadata["response_byte_count1"] = response_byte_count1
        metadata["response_byte_count2"] = response_byte_count2

        # Merge the responses
        json_response1 = chat_response1.raw.model_dump()
        json_response2 = chat_response2.raw.model_dump()

        # Combine the Q&A pairs from both responses
        qa_pairs1 = json_response1.get('question_answer_pairs', [])
        qa_pairs2 = json_response2.get('question_answer_pairs', [])
        
        # Update the item_index for the second set of Q&A pairs
        for qa in qa_pairs2:
            qa['item_index'] = len(qa_pairs1) + qa['item_index']

        # Concatenate summaries from both responses
        summary1 = json_response1.get('summary', '')
        summary2 = json_response2.get('summary', '')
        combined_summary = f"{summary1}\n\n{summary2}" if summary1 and summary2 else summary1 or summary2

        merged_response = {
            'question_answer_pairs': qa_pairs1 + qa_pairs2,
            'summary': combined_summary
        }

        markdown = cls.convert_to_markdown(DocumentDetails(**merged_response))
        html = cls.convert_to_html(DocumentDetails(**merged_response))

        result = QuestionsAnswers(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=merged_response,
            metadata=metadata,
            markdown=markdown,
            html=html
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

    @staticmethod
    def convert_to_markdown(document_details: DocumentDetails) -> str:
        """
        Convert the raw document details to markdown.
        """
        rows = []

        for index, item in enumerate(document_details.question_answer_pairs, start=1):
            rows.append(f"\n## Question Answer Pair {index}")
            rows.append(f"**Question**: {item.question}")
            rows.append(f"**Answer**: {item.answer}")
            rows.append(f"**Rationale**: {item.rationale}")
        
        rows.append(f"\n## Summary\n{document_details.summary}")
        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(self.markdown)

    @staticmethod
    def convert_to_html(document_details: DocumentDetails) -> str:
        """
        Convert the raw document details to html.
        """
        rows = []
        for index, item in enumerate(document_details.question_answer_pairs, start=1):
            rows.append(f'<div class="question-answer-pair"><p><strong>{index}.</strong> {html.escape(item.question)}</p>')
            rows.append(f'<p>{html.escape(item.answer)}</p></div>')
        return "\n".join(rows)

    def save_html(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(self.html)

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    llm = get_llm("ollama-llama3.1")

    plan_prompt = find_plan_prompt("de626417-4871-4acc-899d-2c41fd148807")
    query = (
        f"{plan_prompt}\n\n"
        "Today's date:\n2025-Feb-27\n\n"
        "Project start ASAP"
    )
    print(f"Query: {query}")

    physical_locations = QuestionsAnswers.execute(llm, query)
    json_response = physical_locations.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{physical_locations.markdown}")

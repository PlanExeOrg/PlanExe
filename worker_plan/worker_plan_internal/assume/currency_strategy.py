"""
Pick a suitable currency for the project plan. If the description already includes the currency, then there is no need for this step.
If the currency is not mentioned, then the expert should suggest suitable locations based on the project requirements.
The project may go across national borders, so picking a currency that is widely accepted is important.

Currency codes
https://en.wikipedia.org/wiki/ISO_4217

PROMPT> python -m worker_plan_internal.assume.currency_strategy
"""
import os
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)

class CurrencyItem(BaseModel):
    currency: str = Field(
        description="ISO 4217 alphabetic code."
    )
    consideration: str = Field(
        description="Why use this currency."
    )

class DocumentDetails(BaseModel):
    money_involved: bool = Field(
        description="True if the project likely involves any financial transactions (e.g., purchasing equipment, paying for services, travel, lab tests), otherwise False."
    )
    currency_list: list[CurrencyItem] = Field(
        description="List of currencies that are relevant for this project."
    )
    primary_currency: Optional[str] = Field(
        description="The main currency for budgeting and reporting (ISO 4217 alphabetic code).",
        default=None
    )
    currency_strategy: str = Field(
        description="A short summary of how to handle currency exchange and risk during the project.",
        default=""
    )

CURRENCY_STRATEGY_SYSTEM_PROMPT_1 = """
You are a world-class planning expert specializing in picking the best-suited currency for large, international projects. Currency decisions significantly impact project costs, reporting, and financial risk.

Here's your decision-making process:

1.  **Determine if money is potentially involved:**

    *   Set `money_involved` to `True` if the plan *potentially* requires any financial transactions, *direct or indirect*, such as:
        *   Buying goods or services (e.g., lab equipment, scientific instruments, sampling containers, software licenses, data sets).
        *   Paying for services (e.g., laboratory analysis, research assistance, data analysis, travel expenses, shipping samples, transcription services, professional editing, publication fees).
        *   Paying people (researchers, technicians, consultants, divers, boat crews, etc.) for their time and expertise.
        *   Renting equipment or facilities (e.g., lab space, boats, diving gear).
        *   Acquiring data (e.g., purchasing existing datasets, paying for access to databases).
        *   Travel.
        *   Maintaining systems.

    *   Set `money_involved` to `False` only if the plan is purely non-financial and has absolutely no potential impact on financial resources.

2.  **Select a primary currency:**

    *   **If a specific currency *can* be determined** based on the project description and location information (e.g., the project is clearly based in the USA):
        *   Select that currency (ISO 4217 code).
        *   Explain your reasoning (e.g., "USD is appropriate because the project is based in the USA").

    *   **If a specific currency *cannot* be determined** (e.g., the project is global, theoretical, or lacks clear financial details):
        *   Suggest USD for all international expenses, such as travel, sample analysis, web hosting, and publication fees.
        *   Explain your reasoning (e.g., "USD is a widely accepted currency and suitable for international research expenses.").

3.  **Identify additional currencies (if any):**

    *   List any other currencies that might be needed for local expenses or specific transactions.
    *   Explain why each currency is necessary (e.g., "EUR for travel expenses in Europe").

4.  **Develop a currency management strategy:**

    *   Provide a brief summary of how to manage currency exchange and risk (e.g., "Use forward contracts to hedge against currency fluctuations, especially for travel expenses.").

Here are a few examples of the desired output format:

**Example 1:**
Project: Constructing a solar power plant in Nevada, USA
money_involved: True
Currency List:
- USD: For all project-related expenses in the USA.
Primary Currency: USD
Currency Strategy: Use USD for all budgeting and accounting.

**Example 2:**
Project: Building a wind farm in the North Sea (offshore UK and Netherlands)
money_involved: True
Currency List:
- EUR: For equipment and services sourced from the Eurozone.
- GBP: For equipment and services sourced from the Eurozone.
- DKK: For Danish-based operations and services.
Primary Currency: EUR
Currency Strategy: EUR will be the primary currency.  Maintain accounts in GBP and DKK for local expenses.  Hedge against significant currency fluctuations.

**Example 3:**
Project: Take out the trash
money_involved: False
Currency List:
Primary Currency:
Currency Strategy:

**Example 4:**
Project: My daily commute is broken, need an alternative in Amsterdam.
money_involved: True  # Potential for public transport, taxis, food, etc.
Currency List:
- EUR: For transportation and potential expenses in the Netherlands.
Primary Currency: EUR
Currency Strategy: Use EUR for all commute-related expenses.

**Example 5:**
Project: Distill Arxiv papers into an objective, hype-free summary and publish as an open-access dataset.
money_involved: True # Needs development, hosting, data scraping permission
Currency List:
- USD: For potential web hosting and software maintainence
Primary Currency: USD
Currency Strategy: Use USD for all web hosting and software maintainence.

**Example 6:**
Project: I'm envisioning a streamlined global language...
money_involved: True
Currency List:
- USD: Best guess for international expenses
Primary Currency: USD
Currency Strategy: Use USD for international expenses

**Example 7:**
Project: Create a detailed report examining microplastics within the world's oceans.
money_involved: True # Travel, lab tests, analysis
Currency List:
- USD: Best guess for international expenses
Primary Currency: USD
Currency Strategy: Use USD for international expenses

Consider the following factors when selecting currencies:

*   Stability: Choose currencies that are relatively stable to minimize the impact of exchange rate fluctuations on the project budget.
*   Transaction Costs: Minimize the impact of currency conversions to reduce transaction fees.
*   Economic Influence: Consider the economic influence of the countries involved and the currencies used by major suppliers and contractors.
*   Reporting Requirements: Think about the reporting needs of stakeholders and investors.
*   Project Duration: Longer projects are more susceptible to currency risk.
*   Accounting and Tax Implications: Be aware of the accounting and tax rules regarding currency conversions.
*   Important Currency Facts: England uses the British Pound (GBP). Denmark uses the Danish Krone (DKK), NOT the Euro.

Given the project description and location information, provide the following:

1.  money_involved (True/False)
2.  currency_list
3.  primary_currency
4.  currency_strategy

Be precise with your reasoning, and avoid making inaccurate statements about which countries use which currencies.
"""

CURRENCY_STRATEGY_SYSTEM_PROMPT_2 = """
You are an expert planning assistant focused on selecting the best currency for projects of varying scales, from trivial personal tasks to large, international endeavors. Given a project description and any location details, produce a JSON output with the following structure:

{
  "money_involved": <Boolean>,
  "currency_list": [
      {
         "currency": "<ISO 4217 Code>",
         "consideration": "<Brief explanation>"
      },
      ...
  ],
  "primary_currency": "<ISO 4217 Code>",
  "currency_strategy": "<Brief explanation of currency management strategy>"
}

Guidelines:

1. money_involved:
   - Set to True if the project likely involves financial transactions such as purchasing equipment, paying for services, travel, repairs, lab tests, or any significant expenses requiring budgeting.
   - Also mark digital, research, or industrial projects as involving money if they require development, data curation, hosting, publication fees, maintenance, or research staff—even if no physical site is needed.
   - Set to False for trivial or personal tasks with minimal or no financial transactions.
   - Note: Even for personal tasks, if the issue implies potential expenses (e.g., a broken bike requiring repairs or alternative transportation costs), mark money_involved as True.

2. currency_list:
   - Provide a list of relevant currencies as objects. Each object should include:
     - currency: the ISO 4217 code.
     - consideration: a brief explanation of why this currency is included.
   - For projects that are clearly local (confined to one country) and not subject to economic instability, list only the local currency.
   - For projects spanning multiple countries, list the local currencies for the countries involved if relevant.
   - For projects in regions with multiple European countries, use EUR as the primary currency.
   - If the project is in a country with known currency instability or hyperinflation, include both the local currency and a stable international currency (e.g., USD) in the list.

3. primary_currency:
   - If the project description explicitly mentions a specific currency, use that only if it does not conflict with the guidelines below.
   - For projects that are clearly local in stable economies, use that country's official currency.
   - For international projects that are not specific to one region, default to "USD".
   - For projects spanning multiple European countries, select "EUR" as the primary currency.
   - For significant projects in countries with notable currency instability (such as Venezuela), **do not use the local currency as primary; instead, set the primary currency to "USD"**. The local currency may still be included in the currency_list for local transactions.

4. currency_strategy:
   - For local projects, simply state that the local currency will be used for all transactions with no additional international risk management needed.
   - For international projects, provide a brief explanation of how to manage currency risks (e.g., hedging against exchange fluctuations or using cards with no foreign transaction fees).
   - For projects spanning multiple European countries with "EUR" as the primary currency, note that EUR will be used for consolidated budgeting while local currencies may still be used for local transactions.
   - For projects in countries with currency instability, explain that a stable international currency (e.g., USD) is recommended for budgeting and reporting to mitigate risks from hyperinflation, and that for significant projects the primary currency must be "USD".

Key Instructions:
- Evaluate the project's scale, geographic scope, and local economic conditions using the provided project description and location details.
- Ensure that no field is left empty when significant expenses are expected.
- Apply the appropriate currency guidelines based on the project's geographic scope, local economic conditions, and scale.
"""

CURRENCY_STRATEGY_SYSTEM_PROMPT = CURRENCY_STRATEGY_SYSTEM_PROMPT_2

@dataclass
class CurrencyStrategy:
    """
    Take a look at the vague plan description, the physical locations and suggest a currency.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> 'CurrencyStrategy':
        """
        Invoke LLM with the project description.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = CURRENCY_STRATEGY_SYSTEM_PROMPT.strip()

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

        sllm = llm.as_structured_llm(DocumentDetails)
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

        result = CurrencyStrategy(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            markdown=markdown
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

        if document_details.money_involved:
            rows.append("This plan involves money.")
        else:
            rows.append("This plan **does not** involve money.")

        if len(document_details.currency_list) > 0:
            rows.append("\n## Currencies\n")
            for currency_item in document_details.currency_list:
                rows.append(f"- **{currency_item.currency}:** {currency_item.consideration}")
        else:
            rows.append("No currencies identified.")

        rows.append(f"\n**Primary currency:** {document_details.primary_currency}")
        rows.append(f"\n**Currency strategy:** {document_details.currency_strategy}")        
        return "\n".join(rows)

    def save_markdown(self, output_file_path: str):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(self.markdown)

if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.utils.concat_files_into_string import concat_files_into_string

    base_path = os.path.join(os.path.dirname(__file__), 'test_data', 'currency_strategy7')

    all_documents_string = concat_files_into_string(base_path)
    print(all_documents_string)

    llm = get_llm("ollama-llama3.1")

    currency_strategy = CurrencyStrategy.execute(llm, all_documents_string)
    json_response = currency_strategy.to_dict(include_system_prompt=False, include_user_prompt=False)
    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))

    print(f"\n\nMarkdown:\n{currency_strategy.markdown}")

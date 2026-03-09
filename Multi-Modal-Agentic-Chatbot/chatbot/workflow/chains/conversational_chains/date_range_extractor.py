from typing import Optional
from datetime import datetime
from ...schema import DateRange
from typing import Literal, List
from pydantic import BaseModel, Field
from ...utils import get_langchain_vllm_model_sr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser



class DateRange_Wrapper(BaseModel):
   date_range: DateRange = Field(
    description="Contains extracted start_date and end_date. If no date range is mentioned, return empty objects. Numeric counts (days, weeks, months, years) and calendar_month are also included in this object.")

def get_date_extraction_chain(model,  max_tokens=1000):
    llm = get_langchain_vllm_model_sr(model=model, max_tokens=max_tokens)
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_ = datetime.today()
    today_month = today_.month
    today_year = today_.year
    system = f"""
    ACT as a deterministic DATE RANGE EXTRACTION ENGINE. 
    You MUST follow the rules EXACTLY. Do NOT guess, infer, or normalize.

    ## MULTIPLE DURATION HANDLING (NO CONVERSION):
    - If multiple durations are present, populate each mentioned unit exactly as stated (still no conversions).

    ## RELATIVE DATE RANGE EXTRACTION RULES (PART#2):
    - Only populate start_date and end_date if the user mentions explicit calendar date(s) or absolute ranges:
    - Examples: "January 4", "February 22", "from March 3 to April 16"
    - If a specific day is mentioned (month + day), set:
        - start_date = end_date = that day (ISO 8601 format)
    - If the user mentions "today" or "today’s", set both start_date and end_date to today’s date: ({today_date}).
    - Do NOT populate start_date/end_date if any numeric durations are mentioned.

    ## STRICT UNIT RULE (NO CONVERSIONS):
    - If the user mentions week(s), ONLY populate `weeks` (do NOT populate `days` from it).
    - If the user mentions month(s), ONLY populate `months` (do NOT populate `weeks` from it)
    - If the user mentions day(s), ONLY populate `days`.
    - If the user mentions year(s), ONLY populate `years`.
    - Set all non-mentioned units to 0.

    ## NUMERIC DURATION/COUNT EXTRACTION (PART#1):
    1. Look for numeric durations in the user question:
    - "last/previous/past N day(s)" → days = N
    - "last/previous/past N week(s)" → weeks = N
    - "last/previous/past N month(s)" → months = N
    - "last/previous/past N year(s)" → years = N

    2. If in a user question unit is mentioned without a number:
    - "last/previous/past week(s)" → weeks = 1
    - "last/previous/past month(s)" → months = 1
    - "last/previous/past day(s)" → days = 1
    - "last/previous/past year(s)" → years = 1
    NOTE: If user question mentions "recent" or "latest" or "yesterday" set only `days` as 1.

    3. Look for word "few" in user question:
    - if "few week(s)" is mentioned by user, return `weeks` as 2.
    - if "few month(s)" is mentioned by user, return `months` as 3.
    - Do NOT populate start_date/end_date directly. Instead, use these numeric values to compute the range in post-processing, with end_date anchored to today ({today_date}).
    
    4. If a specific calendar month is mentioned **without a day**, populate calendar_month = month name
    5. Numeric counts must **never** overwrite explicit day/month mentions.

    ## IMPORTANT
    - For Part#1 outputs, use 0 for units not mentioned (NOT null).
    - Always return a **valid JSON** with all fields: start_date, end_date, days, weeks, months, years, calendar_month.
    - start_date/end_date = null only for numeric counts or generic relative ranges.
    - TODAY_YEAR = {today_year}
    """

    parser = PydanticOutputParser(pydantic_object=DateRange_Wrapper)
    date_range_prompt = ChatPromptTemplate(
        [
            ("system", system),
            ("human", "User question: \n\n {question}, Output: {format_instructions}"),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    date_extractor = date_range_prompt | llm | parser 
    return date_extractor
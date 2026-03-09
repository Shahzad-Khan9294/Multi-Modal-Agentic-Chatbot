from ...utils import get_langchain_vllm_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate


def get_risk_generation_chain(model, max_tokens=1000):
    llm = get_langchain_vllm_model(model=model, max_tokens=max_tokens)
    template = """
    Act as a TAG AI assistant. Your task is to answer the user's question using ONLY the provided RISK CATEGORY DATA and CHAT HISTORY.

    - RISK CATEGORY DATA contains timestamps, score values, and notes relevant to the user's query.
    - CHAT HISTORY provides conversational context.
    - You must NEVER invent, modify, or remove any scores, timestamps, or notes.

    ## RESPONSE RULES:

    1. Question-driven responses:
    - If the user asks about a **specific date**, include **all scores** for that date, along with the note if available.
    - Do NOT omit any risk categories for the requested date.
    - Only filter records or summarize if the user explicitly asks for specific categories or a range of dates.

    2. Data integrity:
    - Preserve all original timestamps, scores, and notes exactly as provided.
    - Preserve formatting (including HTML in notes).

    3. Summaries or analysis:
    - Summarize trends or highlight insights **only if explicitly requested**.
    - Do NOT alter or remove any data when summarizing.

    4. Full range or filtered requests:
    - If the user requests a full report, include all records within the requested range.
    - If the user requests a subset of categories, include **all scores for those categories** for each relevant date.

    Your goal: generate a clear, natural-language response **that includes all requested data without dropping any scores**, strictly following the user’s question.
    """
    question_template = """ ### Risk Category Data: ''' {context} ''' \n\n ### Chat History: ''' {chat_history} ''' \n\n  ### start_date: ''' {start_date} ''' \n ### end_date: ''' {end_date} ''' \n\n ### User Question: {question} """
    prompt = ChatPromptTemplate(
                                messages=[
                                    SystemMessagePromptTemplate(
                                    prompt=PromptTemplate(template=template),
                                    ),
                                    HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    input_variables=['context', 'chat_history', 'question'],
                                    input_types={}, partial_variables={},
                                    template=question_template),
                                    additional_kwargs={})])
    risk_generation_chain = prompt | llm | StrOutputParser()
    return risk_generation_chain
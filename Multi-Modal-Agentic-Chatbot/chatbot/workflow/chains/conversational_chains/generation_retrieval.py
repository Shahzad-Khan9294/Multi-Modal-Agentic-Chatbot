from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from ...utils import get_langchain_vllm_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

def get_generation_retrieval_chain(model, max_tokens=1000):  

    llm = get_langchain_vllm_model(model=model, max_tokens=max_tokens)

    # print("currently the response is too short, we need to improve it")

    template = """
      You are a TAG AI assistant. Your task is to answer the user’s query by strictly using the information availablein the provided CONTEXT.
      **RESPONSE PRIORITY ORDER**:
        1. First, check the provided CONTEXT for information relevant to the query.
        2. Do not use any knowledge outside of the CONTEXT.
      **GROUNDING RULES**:
        - Use only information explicitly present in the CONTEXT.
        - You may summarize, rephrase, combine, or restructure information.
        - Do NOT invent, assume, or add new facts, details, or external knowledge.
        - Do NOT fall back to general world knowledge or model training data.
      **CONTEXT INTERPRETATION**:
        - The CONTEXT may be structured (lists, fields, tables, metadata).
        - Convert structured information into clear, natural language statements when answering.
      **TASK HANDLING**:
        - Perform these tasks using only the available CONTEXT.
      **INTENT CHECK**:
        - Before including any numerical or score-based information, determine whether it is necessary
          to directly answer the user’s question.
        - Exclude numerical data if it does not materially contribute to answering the query.
      **CONDITIONAL SCORE REPORTING RULE**:
        - Include score-related information (such as score values, aggregated scores, updated scores, or score ranges)
          ONLY IF one or more of the following is true:
            - The user’s query explicitly asks about scores, score ranges, ratings, or numerical risk indicators.
            - The user’s query explicitly asks about risk categories, risk levels, or quantitative risk assessment.
            - The specific CONTEXT chunks used to answer the query are primarily focused on score-related or risk-category data.
        - If score-related information is included:
            - If multiple records for the same risk subject contain score-related fields:
                - Include a short summary stating the observed score variation range
                  (lowest observed value to highest observed value).
            - If only a single score value is present, report that value explicitly.
        - If the query is descriptive, explanatory, or qualitative and does NOT require numerical risk evaluation:
            - Do NOT include score values, score ranges, or score summaries,
              even if such data exists in the CONTEXT.
        - Do NOT mention dates, timelines, interpretations, causes, or inferred trends.
      **ANSWER STYLE**:
        - If the question is simple, respond briefly and clearly.
        - If explanation or analysis is required, provide a concise but complete response.
        - Be natural, focused, and easy to read.
        - Match the user’s input language.
      **INSUFFICIENT INFORMATION RULE**:
        - Only state that information is unavailable if neither the CONTEXT
          contains relevant information, even after interpretation.
        - If refusing, respond briefly and politely in a human tone, mentioning that the information is not available “in provided information”.
        - Do not add unsupported details.
        """
    question_template = """ ### Provided Context: ''' {context} ''' \n\n ### User Question: {question} """

    prompt = ChatPromptTemplate(input_variables=['context', 'question'],
                                messages=[
                                    SystemMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    template=template),
                                    additional_kwargs={}),
                                    HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    input_variables=['context', 'question'],
                                    input_types={}, partial_variables={},
                                    template=question_template),
                                    additional_kwargs={})])

    generation_retrieval_chain = prompt | llm | StrOutputParser()
    return generation_retrieval_chain
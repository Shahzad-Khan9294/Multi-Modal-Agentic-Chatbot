from typing import Optional
from pydantic import BaseModel, Field
from ...utils import get_langchain_vllm_model_sr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

from typing import List
from langchain_core.output_parsers import PydanticOutputParser

class ClarifyResult(BaseModel):
    question: str = Field(..., description="Clarified query in a retrieval-friendly format")
    keywords: List[str] = Field(..., description="List of keywords and keyphrases that capture the full meaning of the query")
 
def get_clarify_chain(model, max_tokens=1000):  
 
    llm = get_langchain_vllm_model_sr(model=model, max_tokens=max_tokens)
    template = """Act as an expert Query Clarification Assistant.
      Your task is to determine whether the USER QUERY is clear and self-contained.
      If it is clear, return it EXACTLY as-is.
      If it is unclear or referential, rewrite it into a clear, explicit, self-contained query
      using CHAT HISTORY.
    **CLARIFICATION RULES**:
      - Use the most recent user–assistant exchanges in CHAT HISTORY as context.
      - Identify what the user is referring to without adding assumptions or new intent.
    **AMBIGUITY HANDLING (CRITICAL)**:
      - If the prior assistant response mentions ONLY ONE specific topic or item,
        you may clarify the query by explicitly referencing that topic.
      - If the prior assistant response mentions MULTIPLE topics, items, or entities,
        DO NOT select or prioritize any single one unless the user explicitly indicated it.
      - In such cases, rewrite the query in a GENERAL form that preserves the ambiguity
        (e.g., “these insights”, “the items mentioned”, “the Front Row insights”) rather than choosing a specific example.
    **REWRITE RULES (ONLY IF UNCLEAR)**:
      - Preserve the user’s original intent and wording as much as possible.
      - Add only the minimum context needed to make the query understandable.
      - Do NOT add new facts, names, or specificity not clearly requested by the user.
      - Produce one concise, retrieval-friendly question.
      - Do NOT answer the question.
    **FAIL-SAFE**:
      - If CHAT HISTORY does not clearly indicate what the user is referring to,
        return the original query unchanged.
    **OUTPUT RULES**:
    - Output ONLY the final clarified query and the keywords extracted from the clarified question.
    - No explanations, labels, or additional text.
    """
 
    question_template = """### Chat History: ''' {chat_history} ''' \n\n ### User Question: {question} \n\n {format_instructions}"""
    parser = PydanticOutputParser(pydantic_object=ClarifyResult)
    prompt = ChatPromptTemplate(input_variables=['chat_history', 'question'],
                                messages=[
                                    SystemMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    template=template),
                                    additional_kwargs={}),
                                    HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    input_variables=['chat_history', 'question', 'format_instructions'],
                                    input_types={}, partial_variables={"format_instructions": parser.get_format_instructions()},
                                    template=question_template),
                                    additional_kwargs={})])
    clarify_chain = prompt | llm | parser
    return clarify_chain
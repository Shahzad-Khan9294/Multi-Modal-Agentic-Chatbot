from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from ...utils import get_langchain_vllm_model
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser



class ResourceKeywords(BaseModel):
    stakeholders : Optional[List[str]] = Field(description="list of stakeholder/s mentioned in the user question otherwise remain empty")
    events : Optional[List[str]] = Field(description="list of events/calendars mentioned in the user question otherwise remain empty")
    articles : Optional[List[str]] = Field(description="list of articles/analysis mentioned in the user question otherwise remain empty")
    documents : Optional[List[str]] = Field(description="list of document/s mentioned in the user question otherwise remain empty")

def get_resource_keywords(model, max_tokens=1000):
    llm = get_langchain_vllm_model(model=model, max_tokens=max_tokens)
    system_prompt = """
    Act as an expert structured keywords/content/keyphrases extraction assistant.
    Given a USER QUESTION, extract keywords/phrases for these four resource types only:
    - stakeholders
    - events/calendars
    - analysis/articles
    - documents

    **DEFINITION(RESOURCES)**:
    - stakeholders = people, organizations, companies, institutions, or groups (e.g., "UN", "WHO", "John Smith", "Google").
      Geographic locations, topics, or issues (e.g., "Gaza", "climate change", "war") are NOT stakeholders.
    - events/calendars = event titles (e.g., "COP28", "World Economic Forum 2024", "Gaza ceasefire talks").
    - analysis/articles = topics, subjects, or article themes explicitly requested (e.g., "Gaza", "environmental issues", "AI safety").
    - documents = file names, report titles, datasets, or document identifiers (e.g., "ballogy", "report_2024.pdf").

    **Your Task**:
    1. Extract **meaningful words or multi-word phrases** that together capture the full meaning of the query.
    2. Preserve domain-specific phrases (e.g., "Trade Deal with Vietnam", "China Plus One") as single units.
    3. Remove duplicates.
    4. Ignore filler or common stopwords, but do not remove words important to the meaning.
    
    **STRICT RULES**: 
    1) Only extract keywords/names that are explicitly mentioned in the user question.
    2) If a resource type is mentioned but **no specific keywords/names are present**, set it to null.
    3) Do NOT guess, infer, or invent any keywords.
    4) Do NOT include any resource types not referenced in the user question.
    5) keep null for missing or general requests (e.g., "all events", "all stakeholders") instead of empty lists.
    6) Output must be **valid JSON** matching the ResourceKeywords schema.
    STRICT RULES update:
    7) If a resource type is mentioned in the user question but **no specific names or keywords are given**, 
    do NOT include the resource type word itself. Set it to null instead of listing the type word.

    ##IMPORTANT NOTE: Treat generic phrases like "every/all/list of events", "list/all/every stakeholders", "all documents" as **no specific keyword**; do not extract the type word itself.

    ##USER QUESTION:
    {question}
    ## RETURN:
    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=ResourceKeywords)
    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", "User question: \n\n {question}, Output: {format_instructions} ")
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    resource_keywords_chain = prompt | llm | parser
    return resource_keywords_chain

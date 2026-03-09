from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from ...utils import get_langchain_vllm_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class extract_entities_validator(BaseModel):
    name : Optional[str] = Field(description="Extract the stakeholder name (person name) mentioned in the user provided question otherwise return null", default = "")
    title : Optional[str] = Field(description="Extract the event title mentioned in the user provided question otherwise return null", default = "")

async def extract_query_entities_chain(model):
    llm = get_langchain_vllm_model(model=model)
    structured_llm_extractor = llm.with_structured_output(extract_entities_validator)

    # Use single prompt template directly
    template = """
    You are an information extraction assistant.

    You will receive a user question extract:
    1. The stakeholder name (person name) mentioned
    2. The event title mentioned

    STRICT RULES:
    - Return ONLY valid JSON
    - Do NOT use bullet points
    - Do NOT wrap output in markdown
    - Use empty lists instead of null  

    OUTPUT:
    {{"name": "{{name}}", "title": "{{title}}" }}

    User Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_messages(
           [
            ("system", template),
            ("human", "{question}"),
        ],
        #input_variables=["question"]
    )
    chain = prompt | structured_llm_extractor
    return chain
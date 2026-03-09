from typing import Optional
from pydantic import BaseModel, Field
from ...utils import get_langchain_vllm_model_sr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

from typing import List
from langchain_core.output_parsers import PydanticOutputParser

from ...models import RelationshipsOutput

def get_extract_entity_relation_chain(model: str = "unsloth/gemma-3-12b-it-bnb-4bit", max_tokens: int = 1000):

 
    llm = get_langchain_vllm_model_sr(model=model, max_tokens=max_tokens)
    template = """Act as an expert Entity Relation Extraction Assistant.
      Your task is to extract the entities and relationships from the USER QUERY.
      The entities are the names of the people, organizations, events, or concepts that are mentioned in the USER QUERY.
      The relationships are the relationships between the entities that are mentioned in the USER QUERY.
    **FAIL-SAFE**:
      - If the USER QUERY does not contain any entities or relationships, return an empty list.
    **OUTPUT RULES**:
    - Output ONLY the entities and relationships extracted from the USER QUERY.
    - No explanations, labels, or additional text.
    """
 
    question_template = """### User Question: {question} \n\n {format_instructions}"""
    parser = PydanticOutputParser(pydantic_object=RelationshipsOutput)
    prompt = ChatPromptTemplate(input_variables=['question'],
                                messages=[
                                    SystemMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    template=template),
                                    additional_kwargs={}),
                                    HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(
                                    input_variables=['question', 'format_instructions'],
                                    input_types={}, partial_variables={"format_instructions": parser.get_format_instructions()},
                                    template=question_template),
                                    additional_kwargs={})])
    extract_entity_relation_chain = prompt | llm | parser
    return extract_entity_relation_chain
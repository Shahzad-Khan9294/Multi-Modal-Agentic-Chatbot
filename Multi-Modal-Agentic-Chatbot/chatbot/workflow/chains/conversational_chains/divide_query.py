from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from ...utils import get_langchain_vllm_model


class DivideResult(BaseModel):
    """Result for divide query chain."""
    queries: List[str] = Field(
        description=(
            "List of queries derived from the input. "
            "If the input had a single request, return one query. "
            "If it had multiple requests, return each as a separate query."
        )
    )


def get_divide_chain(model: str = "unsloth/gemma-3-12b-it-bnb-4bit"):
    """
    Returns a LangChain chain that takes a question and splits it
    into one or more queries, while preserving meaning.
    """

    template = """
    You are a query processing assistant. Your task is to analyze the incoming query
    and decide whether it contains a single request or multiple distinct requests.

    ## Instructions:
    - If the query contains only one request:
      Return JSON exactly like this:
      {{ "queries": ["<original_query>"] }}

    - If the query contains multiple requests:
      Break it into multiple independent queries and return JSON exactly like this:
      {{ "queries": ["<first_query>", "<second_query>", ...] }}

    - Do not add extra information or drop details.
    - Ensure all queries preserve the intent and context of the original.

    ## Query:
    {question}
    """

    d_llm = get_langchain_vllm_model(model=model)
    structured = d_llm.with_structured_output(DivideResult)

    prompt = ChatPromptTemplate(
        input_variables=["question"],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["question"],
                    template=template,
                )
            )
        ],
    )

    chain = prompt | structured
    return chain
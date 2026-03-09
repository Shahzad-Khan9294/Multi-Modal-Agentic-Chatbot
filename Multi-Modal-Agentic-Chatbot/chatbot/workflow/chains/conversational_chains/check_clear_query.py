# sharepoint_agent/workflow/chains/check_clear_query.py

from pydantic import BaseModel, Field
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from ...utils import get_langchain_vllm_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class ClearCheckResult(BaseModel):
    """Result for initial clear question check."""
    is_clear: bool = Field(description="Whether the question is already fully clear and self-contained.")
    reason: str = Field(description="Short reasoning why it is or isn't clear.", default="")

def get_clear_check_chain(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=1000):
    c_llm = get_langchain_vllm_model(model=model, max_tokens=max_tokens)
    structured = c_llm.with_structured_output(ClearCheckResult)

    template = """
    You are a linguistic clarity checker. Your job is to decide if the user's question is
    already **fully clear and contextually self-contained**.

    ## Definition of 'Clear'
      - The question makes complete sense on its own.
      - It can be answered directly without needing prior context or chat history.
      - It is grammatically coherent and specific enough to produce an answer.

    ## Definition of 'Not Clear'
      - The question is ambiguous, incomplete, or depends on earlier conversation.
      - It uses vague references like "that", "it", "them", "previous file", etc.
      - It’s not self-contained or missing key information.

    Respond with:
      - `is_clear = True` if the question is fully clear and self-contained.
      - `is_clear = False` otherwise.
      - `reason` = Reason in one line explaination.

    User Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['question'],
                template=template
            )
        )
    ])

    chain = prompt | structured
    return chain
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from ...utils import get_langchain_vllm_model


class ValidateContext(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Validate weather the question is answerable with the provided context or unanswerable 'yes' or 'no'"
    )

def get_context_validator(modal):
    """Validate whether the question is answerable with the provided context."""

    llm = get_langchain_vllm_model(model=modal)
    structured_llm_grader = llm.with_structured_output(ValidateContext)

    system = """You are a grader assessing whether the question is answerable with the provided context. \n 
        Give a binary score 'yes' or 'no' score to indicate whether the question answerable or unanswerable with the provided context."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Context: \n\n {context} \n\n User question: {question}"),
        ]
    )

    context_validator = grade_prompt | structured_llm_grader
    return context_validator

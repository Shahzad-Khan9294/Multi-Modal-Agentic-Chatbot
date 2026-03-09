from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from ...utils import get_langchain_vllm_model

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def get_hallucination_grader(model):
    """
    Returns a RunnableSequence that grades whether an LLM generation is grounded in a set of retrieved facts.
    """
    llm = get_langchain_vllm_model(model)

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
    return hallucination_grader
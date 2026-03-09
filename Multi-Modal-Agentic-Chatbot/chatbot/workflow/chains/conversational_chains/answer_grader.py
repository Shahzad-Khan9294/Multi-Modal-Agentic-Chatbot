from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from ...utils import get_langchain_vllm_model

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def get_answer_grader(model, max_tokens) -> RunnableSequence:
    """
    Returns a RunnableSequence that grades an answer based on whether it resolves a question.
    """

    llm = get_langchain_vllm_model(model, max_tokens)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
    return answer_grader

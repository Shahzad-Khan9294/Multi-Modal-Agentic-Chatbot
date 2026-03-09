from pydantic import BaseModel, Field
from ...utils import get_langchain_vllm_model_sr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
def get_retrieval_grader(modal,max_tokens=100):
    """Get the retrieval grader chain."""
    llm = get_langchain_vllm_model_sr(model=modal,max_tokens=max_tokens)
    parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """
    grade_prompt = ChatPromptTemplate(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question} \n\n Output: {format_instructions}"),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    retrieval_grader = grade_prompt | llm | parser
    return retrieval_grader
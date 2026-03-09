from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from ...states import GraphState
from ...chains import get_context_validator
import time

def validate_context_for_query(question: str, documents: List[Document]) -> List[Document]:
    """Validate context relevance for a single question and return relevant docs."""
    context = "\n".join(d.page_content for d in documents)
    validator = get_context_validator(modal="unsloth/gemma-3-12b-it-bnb-4bit")
    validation_result = validator.invoke({"question": question, "context": context})
    print(f"Validation Result for question '{question}':", validation_result)

    if validation_result.binary_score:
        print("---GRADE: CONTEXT RELEVANT---")
        return documents
    else:
        print("---GRADE: CONTEXT NOT RELEVANT---")
        return []


def grade_context(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines whether the context is relevant to the question(s).
    Supports both single-query and multi-query validation.
    """
    start_time = time.perf_counter()
    print("---CHECK CONTEXT RELEVANCE TO QUESTION(S)---")
    base_question = state["question"]
    documents = state.get("documents", [])

    # Single-query mode
    print("Single-query context validation mode")
    relevant_docs = validate_context_for_query(base_question, documents)
    # Deduplicate documents (based on content)
    unique_docs = {d.page_content: d for d in relevant_docs}.values()

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"---------------------- Grade Context inference time ----------------------: {inference_time:.4f} seconds")
    #return {"documents": relevant_docs, "context_validation": bool(relevant_docs)}    
    return {"documents": list(unique_docs), "context_validation": bool(unique_docs)}

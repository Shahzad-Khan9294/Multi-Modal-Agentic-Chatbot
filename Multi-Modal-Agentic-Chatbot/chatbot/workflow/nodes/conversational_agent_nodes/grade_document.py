from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from ...states import GraphState
from ...chains import get_retrieval_grader
import time
import logging
logger = logging.getLogger(__name__)



retrieval_grader = get_retrieval_grader(modal="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=100)
async def grade_documents(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    start_time = time.perf_counter()
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION(S)---")
    clarify_question = state.get("clarify_query", "")
    documents: List[Document] = state.get("documents", [])
    incoming_resource_types = state.get("resource_type",[]) or []

    if not documents:
        logger.info("No documents to grade.")
        return {"documents": []}
    if 'stakeholder' in clarify_question.lower():
        return {"documents": documents}

    filtered_docs = [] 
    if "Risk" in incoming_resource_types:
        risk_docs, rag_docs = ([d for d in documents if (d.metadata or {}).get("type") == "risk_category_data"], [d for d in documents if (d.metadata or {}).get("type") != "risk_category_data"])
        start_time_2 = time.perf_counter()
        logger.info("------ Single-query grading mode (Risk + RAG) -------")
        batch_document_list = [{"question": clarify_question, "document": d.page_content} for d in rag_docs]
        scores = await retrieval_grader.abatch(batch_document_list)
        for ind, score in enumerate(scores):
            if getattr(score, "binary_score", "").lower() == "yes":
                filtered_docs.append(rag_docs[ind])
        filtered_docs = risk_docs + filtered_docs
    else:
        start_time_2 = time.perf_counter()
        logger.info("------ Single-query grading mode -------")
        batch_document_list = [{"question": clarify_question, "document": d.page_content} for d in documents]
        scores = await retrieval_grader.abatch(batch_document_list)
        for ind, score in enumerate(scores):
            if getattr(score, "binary_score", "").lower() == "yes":
                filtered_docs.append(documents[ind])
    
    logger.info(f"Total unique documents after grading and deduplication: {len(filtered_docs)}")
    # for d in filtered_docs:
    #     print(d.metadata["type"], d.metadata["id"])
    # print(f"Unique document after grading and deduplication:", filtered_docs)
    # print("Unique document IDs after grading and deduplication: ",
    # [doc.metadata.get("id") for doc in filtered_docs])
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    logger.info(f"---------------------- Grade document inference time ----------------------: {inference_time:.4f} seconds")
    return {"documents": filtered_docs}

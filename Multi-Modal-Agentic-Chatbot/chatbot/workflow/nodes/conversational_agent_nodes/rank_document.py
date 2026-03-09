import time
import logging
import numpy as np
from chatbot.db import db
import asyncio
from ...states import GraphState
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
logger = logging.getLogger(__name__)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device)
async def rank_documents(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    start_time = time.perf_counter()
    logger.info("---RERANK DOCUMENTS USING CROSS-ENCODER---")
    clarify_question = state.get("clarify_query", "")
    base_question = state.get("question")
    logger.info(f"clarify_question: {clarify_question}")
    logger.info(f"base_question: {base_question}")
    documents: List[Document] = state.get("documents", [])

    if not documents:
        return {"documents": [], "question": base_question}

    risk_docs, rag_docs = ([d for d in documents if (d.metadata or {}).get("type") == "risk_category_data"], [d for d in documents if (d.metadata or {}).get("type") != "risk_category_data"])
    if 'stakeholder' in base_question.lower():
        return {"documents": documents, "question": base_question}

    rankings = reranker.rank(clarify_question, [d.page_content for d in rag_docs], convert_to_tensor=True)
    threshold =  0.1

    filtered = []
    print(f"filtered:")
    for ranking in rankings:
        d = rag_docs[ranking['corpus_id']]
        # logger.info(f"ranking: {ranking}")
        # logger.info(f"text: {d.page_content}")
        if ranking['score'] >= threshold:
            filtered.append(d)

    ranked_rag_docs = filtered
    logger.info(f"filtered: {len(filtered)}")
    logger.info(f"ranked_rag_docs: {len(ranked_rag_docs)}")

    filtered_docs = risk_docs  + ranked_rag_docs
    end_time = time.perf_counter()
    logger.info(f"Rerank inference time: {end_time - start_time:.4f} seconds")
    logger.info(f"len(filtered_docs): {len(filtered_docs)}")
    return {"documents": filtered_docs}
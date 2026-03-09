import re
import json
import uuid
import time
import logging
from typing import List, Dict, Any, Optional

from ...states import GraphState

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from langchain.schema import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from ...utils import get_langchain_vllm_model, get_langchain_vllm_model_sr
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from ...chains import (
    get_generation_chain,
    get_used_chunks_chain,
    get_generation_retrieval_chain,
    get_adjacent_generation_chain,
)

generation_stream_chain = get_generation_chain(
    model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=1000
)
used_chunks_chain = get_used_chunks_chain(
    model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=200
)

generation_retrieval_chain = get_generation_retrieval_chain(
    model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=1000
)
adjacent_generate_chain = get_adjacent_generation_chain(
    model="unsloth/gemma-3-12b-it-bnb-4bit"
)

llm = get_langchain_vllm_model()
llm_sr = get_langchain_vllm_model_sr()


def select_fitting_chunks(documents, llm, max_input_tokens=6000):
    selected_chunk_docs = []
    total_tokens = 0
    for doc in documents:
        doc_tokens = llm.get_num_tokens(doc.page_content)
        if total_tokens + doc_tokens > max_input_tokens:
            break
        selected_chunk_docs.append(doc)
        total_tokens += doc_tokens
    return selected_chunk_docs, total_tokens


class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="Answer addresses the question, True or False")


def get_answer_grader(model, max_tokens=1000):
    """
    Returns a RunnableSequence that grades an answer based on whether it resolves a question.
    """
    llm = get_langchain_vllm_model_sr(model, max_tokens=max_tokens)
    parser = PydanticOutputParser(pydantic_object=GradeAnswer)
    system = """You are a grader assessing whether an answer addresses / resolves a question
Analyze the answer and the question and give a binary score True or False.
If the answer is not related to the question, give a binary score False.
If the answer is related to the question, give a binary score True.
"""
    answer_prompt = ChatPromptTemplate(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n LLM generation: {generation} \n\n RETURN: {format_instructions}",
            ),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    answer_grader = answer_prompt | llm | parser
    return answer_grader


answer_grader = get_answer_grader(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=100)


def _response_to_str(response) -> str:
    """Normalize chain output to string for graders/UI."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "answer"):
        return (getattr(response, "answer") or "").strip()
    if hasattr(response, "content"):
        return (getattr(response, "content") or "").strip()
    return str(response).strip()


def build_used_chunk_refs(
    selected_chunks: List[Document],
    used_chunk_numbers: List[int],
) -> List[Dict[str, Any]]:
    """
    Convert 1-based chunk numbers -> UNIQUE DB resource ids.
    Multiple chunks from same resource id will count as ONE reference.
    """
    refs: List[Dict[str, Any]] = []
    if not selected_chunks or not used_chunk_numbers:
        return refs

    seen = set()

    for n in used_chunk_numbers:
        try:
            idx = int(n) - 1
        except Exception:
            continue

        if idx < 0 or idx >= len(selected_chunks):
            continue

        doc = selected_chunks[idx]
        meta = doc.metadata or {}

        chunk_id = meta.get("id")
        rtype = meta.get("type")

      

        if not chunk_id:
            continue

        key = (rtype, str(chunk_id))
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "id": str(chunk_id),
                "type": rtype,
                "heading": meta.get("heading"),
            }
        )

    return refs

def build_reference_links_from_used(
    used_chunk_refs: List[Dict[str, Any]],
    link_lines: List[str],
):
    """
    Build 1 URL per UNIQUE (type,id) used in the answer.
    Example: 15 chunks used but map to 3 unique ids => produce 3 links.
    """

    if not used_chunk_refs:
        return

    def normalize_type(t: Optional[str]) -> str:
        tt = (t or "").strip().lower()
        if tt in {"document", "documents"}:
            return "documents"
        if tt in {"analysis", "analyses"}:
            return "analysis"
        if tt in {"event", "events", "calendar", "calendars"}:
            return "events"
        if tt in {"stakeholder", "stakeholders"}:
            return "stakeholders"
        return tt or "resource"

    seen = set()
    unique_items: List[Dict[str, str]] = []

    skip_types = ["risk_category_data", "events"]

    for ref in used_chunk_refs:
        rid = ref.get("id")
        if not rid:
            continue
        rtype = normalize_type(ref.get("type"))
        
        if rtype in skip_types:
            continue
        
        rid_str = str(rid)

        key = (rtype, rid_str)
        if key in seen:
            continue
        seen.add(key)

        heading = (ref.get("heading") or "").strip()
        unique_items.append(
            {
                "type": rtype,
                "id": rid_str,
                "title": heading,
            }
        )

    if not unique_items:
        return

    # Pretty labels
    label_map = {
        "analysis": "Analysis",
        "documents": "Document",
        "stakeholders": "Stakeholders",
        # events intentionally omitted
    }
    link_lines.append("**🔗 References:**")

    for item in unique_items:
        rtype = item["type"]
        rid = item["id"]
        title = (item["title"] or f"{rtype}").upper()
        base_label = label_map.get(rtype, rtype.capitalize())

        url = "https://{base}/" + f"{rtype}/{rid}"
        link_lines.append(f"- {base_label}: [{title}]({url})")


_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","to","of","in","on","for","with",
    "as","by","at","from","this","that","these","those","is","are","was","were","be",
    "been","being","it","its","they","them","their","we","our","you","your","i","me",
    "can","could","should","would","may","might","will","shall","do","does","did",
    "not","no","yes","than","also","into","over","under","between","within","about"
}

def _normalize_and_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    toks = [t for t in text.split() if t and t not in _STOPWORDS and len(t) > 2]
    return toks

def chunk_confidence_percent(answer: str, chunk_text: str) -> float:
    """
    Confidence = % of UNIQUE meaningful answer tokens that appear in the chunk.
    """
    ans_tokens = set(_normalize_and_tokenize(answer))
    if not ans_tokens:
        return 0.0

    chunk_tokens = set(_normalize_and_tokenize(chunk_text))
    if not chunk_tokens:
        return 0.0

    overlap = len(ans_tokens.intersection(chunk_tokens))
    return (overlap / max(1, len(ans_tokens))) * 100.0

def filter_chunks_by_confidence(
    selected_chunks: List[Document],
    used_chunk_numbers: List[int],
    answer: str,
    min_percent: float = 15.0,
) -> List[int]:
    """
    Keep only those used_chunk_numbers whose confidence % > min_percent.
    """
    if not selected_chunks or not used_chunk_numbers:
        return []

    kept: List[int] = []

    for n in used_chunk_numbers:
        try:
            idx = int(n) - 1
        except Exception:
            continue
        if idx < 0 or idx >= len(selected_chunks):
            continue

        chunk_text = (selected_chunks[idx].page_content or "")
        conf = chunk_confidence_percent(answer, chunk_text)

        logger.info(f"[Confidence] Chunk {n}: {conf:.2f}%")

        if conf > min_percent:
            kept.append(int(n))

    return sorted(set(kept))

async def generate(state: GraphState, config: RunnableConfig):
    """
    - Stream ONLY answer text (generation_stream_chain)
    - Then compute used chunk numbers and map them to DB ids (used_chunk_refs)
    - Build reference links ONLY from used unique ids
    """
    logger.info("---GENERATE---")
    start_time_1 = time.perf_counter()
    question = state["question"]

    link_lines: List[str] = []
    keep_messages = []
    selected_chunks: List[Document] = []
    documents = state["documents"]

    human_message = HumanMessage(content=question)

    if len(state["messages"]) > 8:
        keep_messages = state["messages"][-8:].copy()
        logger.info(f"keep_messages: {len(keep_messages)}")
        state["messages"].clear()
        state["messages"] = keep_messages
        logger.info(f"Messages in state after deletion: {len(state['messages'])}")

    try:
        selected_chunks, total_tokens = select_fitting_chunks(documents, llm, max_input_tokens=6000)
        logger.info(f"total_tokens: {total_tokens}")
    except Exception as e:
        logger.error(f"Chunk selection failed: {e}", exc_info=True)
        selected_chunks = []
    gen_input = {
        "context": selected_chunks,
        "chat_history": state["messages"],
        "question": question,
    }
    gen_retrieval_input = {"context": selected_chunks, "question": question}

    response_str = ""
    used_chunk_numbers: List[int] = []
    used_chunk_refs: List[Dict[str, Any]] = []

    try:
        if state.get("detail_info") == "yes":
            response_str = _response_to_str(
                await adjacent_generate_chain.ainvoke(
                    {
                        "context": selected_chunks,
                        "question": question,
                        "chat_history": state["messages"][-2:],
                    },
                    config=config,
                )
            )
        else:
            response_str = _response_to_str(await generation_stream_chain.ainvoke(gen_input, config=config))
            grade_result = await answer_grader.ainvoke(input={"question": question, "generation": response_str})
            if not grade_result.binary_score:
                response_str = _response_to_str(await generation_retrieval_chain.ainvoke(gen_retrieval_input, config=config))

    except Exception as e:
        logger.error(f"Generation exception: {e}", exc_info=True)
        if state.get("detail_info") == "yes":
            response_str = _response_to_str(
                await adjacent_generate_chain.ainvoke(
                    {
                        "context": selected_chunks,
                        "question": question,
                        "chat_history": state["messages"][-2:],
                    },
                    config=config,
                )
            )
        else:
            response_str = _response_to_str(
                await generation_stream_chain.ainvoke(
                    {
                        "context": selected_chunks[:4],
                        "chat_history": state["messages"],
                        "question": question,
                    },
                    config=config,
                )
            )
    try:
        used_obj = await used_chunks_chain.ainvoke(
            {"context": selected_chunks, "answer": response_str, "question": question},
        )

        if hasattr(used_obj, "used_chunk_numbers"):
            used_chunk_numbers = used_obj.used_chunk_numbers or []
        elif hasattr(used_obj, "used_chunks"):
            used_chunk_numbers = used_obj.used_chunks or []
        else:
            used_chunk_numbers = []

        # Deterministic filter: keep only chunks with confidence > 15%
        used_chunk_numbers = filter_chunks_by_confidence(
            selected_chunks=selected_chunks,
            used_chunk_numbers=used_chunk_numbers,
            answer=response_str,
            min_percent=15.0,
        )

    except Exception as e:
        logger.error(f"used_chunks extraction failed: {e}", exc_info=True)
        used_chunk_numbers = []

    used_chunk_refs = build_used_chunk_refs(selected_chunks, used_chunk_numbers)

    logger.info(f"used_chunk_numbers: {used_chunk_numbers}")
    logger.info(f"used_chunk_refs(count): {len(used_chunk_refs)}")
    build_reference_links_from_used(used_chunk_refs, link_lines)
    
    link_str = ""
    if link_lines:
        spacing = "\n\n"
        link_str = f"{spacing}" + "\n".join(link_lines) + spacing

    logger.info(f"Response Generation length: {len(response_str)} chars")

    if config.get("configurable", {}).get("stream_mode") == "messages":
        response_message = AIMessage(
            content=link_str
        )
    else:
        response_message = AIMessage(
            content=response_str + link_str
        )

    end_time_1 = time.perf_counter()
    logger.info(
        f"---------------------- Generation Time ----------------------: {end_time_1 - start_time_1:.4f} seconds"
    )

    return {"messages": keep_messages + [human_message, response_message], "documents": selected_chunks}
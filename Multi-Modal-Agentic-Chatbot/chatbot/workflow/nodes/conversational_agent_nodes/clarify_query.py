import time
from typing import Dict, Any
from ...states import GraphState
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from ...utils import convert_message_content_to_string
from langchain_core.output_parsers import JsonOutputParser
from ...chains import get_clear_check_chain, get_clarify_chain



async def clarify_query(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:

    start_time = time.perf_counter()
    question = (state.get("question") or "").strip()
    messages = state.get("messages", [])
    history_window = 5

    # --- Build chat history ---
    history_text_items = []
    for m in messages:
        content = getattr(m, "content", None)
        if content is None:
            try:
                content = convert_message_content_to_string(m)
            except Exception:
                content = str(m)
        history_text_items.append(content)
    chat_history = "\n".join(history_text_items[-history_window:])

    print("\n---CLARIFY QUERY NODE---")
    print("Incoming question:", question)
    print(f"Chat history window (last {history_window} msgs):")
    print(chat_history or "[EMPTY HISTORY]")


    extract_entity = await extract_query_entities_chain(modal=config.get("configurable", {}).get(
                "clarify_model", "unsloth/gemma-3-12b-it-bnb-4bit"
            ))
    raw_output = await extract_entity.ainvoke({"question": question})
    print("RAW OUTPUT: " ,raw_output)
    stakeholder = raw_output.name
    event = raw_output.title
    if stakeholder:
        state["stakeholder_query_state"] = stakeholder
    else:
        state["stakeholder_query_state"] = ""
    if event: 
        state["event_query_state"] = event
    else:
        state["event_query_state"] = ""

    clarified_question = question
    clarification_success = False
    _clarify_failed = False

    # --- STEP 1: Clear-check chain ---
    if question:
        clear_chain = get_clear_check_chain(
            model=config.get("configurable", {}).get(
                "clarify_model", "unsloth/gemma-3-12b-it-bnb-4bit"
            ),
            max_tokens= 300
        )
        clear_result = await clear_chain.ainvoke({"question": question})

        is_clear = getattr(clear_result, "is_clear", True)
        reason = getattr(clear_result, "reason", "")
        print("Raw clear-check result:", clear_result)

        if is_clear:
            clarified_question = question
            clarification_success = True
            _clarify_failed = False
            print(f"---QUESTION ALREADY CLEAR (SITUATION 1)---\nReason: {reason}")
        else:
            print("---QUESTION NOT FULLY CLEAR → proceed to full clarify chain---")

    # --- STEP 2: Full clarify chain for ambiguous/unclear ---
    if not clarification_success and question:
        clarify_chain = get_clarify_chain(
            model=config.get("configurable", {}).get(
                "clarify_model", "unsloth/gemma-3-12b-it-bnb-4bit"
            ),
            max_tokens= 300
        )
        clarify_result = await clarify_chain.ainvoke({"question": question, "chat_history": chat_history})
        print("Raw clarify chain result:", clarify_result)

        is_clear = bool(getattr(clarify_result, "is_clear", False))
        clarified = getattr(clarify_result, "clarified_question", None)

        if is_clear:
            clarification_success = True
            clarified_question = (clarified.strip() if clarified else question)
            if clarified:
                print("---CLARIFICATION SUCCESS (SITUATION 2)---")
        else:
            if clarified and clarified.strip():
                clarified_question = clarified.strip()
                _clarify_failed = False
                clarification_success = True
                print("---EDGE CASE: MODEL UNCERTAIN BUT PROVIDED CLARIFICATION---")
            else:
                clarified_question = "Ask in return from me: Can you please clarify or rephrase your question?"
                _clarify_failed = True
                clarification_success = False
                print("---CLARIFICATION FAILED (SITUATION 3)---")
    else:
        print("---NO HISTORY OR EMPTY QUESTION, SKIPPING CLARIFICATION---")

    # --- STEP 3: Remove direct divide_query execution ---
    # Instead, set state flags; routing will go through QUERY_ROUTE or RAG pipeline as needed
    state["question"] = clarified_question
    state["_clarify_failed"] = _clarify_failed
    state["clarification_success"] = clarification_success

    # Clear `generated_questions` for divide_query node to generate when it is actually invoked
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"---------------------- Clarify Query inference time ----------------------: {inference_time:.4f} seconds")
    return {
        "stakeholder_query_state": stakeholder or "",
        "event_query_state": event or "",
        "question": clarified_question,
        "_clarify_failed": _clarify_failed,
        "clarification_success": clarification_success,
    }
from typing import Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from .states import GraphState
import logging
logger = logging.getLogger(__name__)


# ---------------- Constants ----------------
from .consts import (
    CHAT,
    QUERY_ROUTE,
    #DIVIDE_QUERY,
    RETRIEVE,
    RANK_DOCUMENTS,
    GRADE_DOCUMENTS,
    GENERATE,
    DB_SEARCH,
    LC_GENERATE,
    CLARIFY_QUERY,
    RESOURCE_LIST,
    RISK_CATEGORY_SEARCH
)

# ---------------- Nodes -------------------
from .nodes import (
    chat,
    query_route,
    #divide_query,
    clarify_query,
    retrieve,
    grade_documents,
    generate,
    db_search,
    lc_generate,
    summarize_conversation,
    rank_documents,
    resource_list,
    risk_category_search
)

# ---------------- Conditional Functions ----
from .nodes import (
    generate_summary,
    map_summaries,
    collect_summaries,
    generate_final_summary,
    collapse_summaries,
    should_collapse
)

# Routing / grading functions
async def check_query_route(state: GraphState, config: RunnableConfig) -> Literal[RETRIEVE, RESOURCE_LIST, DB_SEARCH, RISK_CATEGORY_SEARCH]:
    logger.info("---ROUTE QUESTION TYPE---")
    resource_types = state.get('resource_type')
    if state["route"] == "LongContext" and state.get("document_names"):
        return DB_SEARCH
    elif state["route"] == "ResourceList" and state.get("resource_type_list"):
        return RESOURCE_LIST
    elif state["route"] == "Risk" and "Risk" in resource_types:
        return RISK_CATEGORY_SEARCH
    else:
        logger.info("→ RAG route: RETRIEVE NODE.")
        #return DIVIDE_QUERY
        return RETRIEVE

# def check_validated_context(state: GraphState) -> Literal[GENERATE, CHAT]:
#     return GENERATE if state["documents"] else CHAT

def check_answer_grade(state: GraphState) -> Literal[DB_SEARCH, END]:
    return END if state["answer_validation"] else DB_SEARCH


# async def check_clarify(state: GraphState, config: RunnableConfig) -> Literal[QUERY_ROUTE, CHAT]:
#     # If clarify node set a failure flag, redirect to chat but first modify the question
#     if state["_clarify_failed"]:
#         # Update the question to explicitly ask for clarification
#         print("---CLARIFICATION FAILED, REDIRECTING TO CHAT---")
#         return CHAT
#     # Otherwise continue to query router
#     else:
#         print("---CLARIFICATION SUCCEEDED, CONTINUING TO QUERY ROUTER---")
#         return QUERY_ROUTE

# ---------------- Workflow Setup -------------
workflow = StateGraph(GraphState)



# Add nodes
# workflow.add_node(CLARIFY_QUERY, clarify_query)
workflow.add_node(QUERY_ROUTE, query_route)
# workflow.add_node(CHAT, chat)
#workflow.add_node(DIVIDE_QUERY, divide_query)
workflow.add_node(RISK_CATEGORY_SEARCH, risk_category_search)
workflow.add_node(RESOURCE_LIST, resource_list)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(DB_SEARCH, db_search)
workflow.add_node(RANK_DOCUMENTS, rank_documents)

# ---------------- Edges -------------------

# Entry point: clarify query
#workflow.set_entry_point(CLARIFY_QUERY)
workflow.set_entry_point(QUERY_ROUTE)

# Main query flow
#workflow.add_edge(CLARIFY_QUERY, QUERY_ROUTE)

# This One
# route only forward if clarification succeeded; otherwise end the flow
# route only forward if clarification succeeded; otherwise end the flow

# workflow.add_conditional_edges(CLARIFY_QUERY, check_clarify)
# workflow.add_edge(CLARIFY_QUERY, QUERY_ROUTE)
###
workflow.add_conditional_edges(QUERY_ROUTE, check_query_route)
workflow.add_edge(RESOURCE_LIST, END)
workflow.add_edge(RISK_CATEGORY_SEARCH, END)
#workflow.add_edge(DIVIDE_QUERY, RETRIEVE)
workflow.add_edge(RETRIEVE, RANK_DOCUMENTS)
workflow.add_edge(RANK_DOCUMENTS, GRADE_DOCUMENTS)
# workflow.add_conditional_edges(GRADE_DOCUMENTS, check_validated_context)
workflow.add_edge(GRADE_DOCUMENTS, GENERATE)

# workflow.add_edge(CHAT, END)
workflow.add_edge(GENERATE, END)

# Long Context workflow edges

workflow.add_edge(DB_SEARCH, END)

# Compile workflow
agent = workflow.compile()
from typing import Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from .states import GraphState
   
# ---------------- Conditional Functions ----
from .const import (
    MAP_AND_GENERATE_SUMMARY,
    COLLECT_SUMMARIES,
    GENERATE_FINAL_SUMMARY,
    COLLAPSE_SUMMARIES
)
from .summary import (
    map_and_generate_summary,
    collect_summaries,
    collapse_summaries,
    generate_final_summary,
    should_collapse
)


workflow = StateGraph(GraphState)

workflow.add_node(MAP_AND_GENERATE_SUMMARY, map_and_generate_summary)
workflow.add_node(COLLECT_SUMMARIES, collect_summaries)
workflow.add_node(GENERATE_FINAL_SUMMARY, generate_final_summary)
workflow.add_node(COLLAPSE_SUMMARIES, collapse_summaries)

workflow.set_entry_point(MAP_AND_GENERATE_SUMMARY)
workflow.add_edge(MAP_AND_GENERATE_SUMMARY, COLLECT_SUMMARIES)
workflow.add_edge(COLLECT_SUMMARIES, COLLAPSE_SUMMARIES)
workflow.add_conditional_edges(COLLAPSE_SUMMARIES, should_collapse)
workflow.add_edge(GENERATE_FINAL_SUMMARY, END)
agent = workflow.compile()
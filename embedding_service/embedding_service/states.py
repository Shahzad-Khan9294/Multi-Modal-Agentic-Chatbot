import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State for the embedding service graph."""
    documents: List[Document]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]  # add key for collapsed summaries
    collected_summaries: List[Document]
    final_summary: str
    content: str
    messages: Annotated[list, operator.add]
    



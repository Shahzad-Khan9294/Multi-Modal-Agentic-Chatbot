import operator
from typing import Optional
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from .schema import DateRange


class GraphState(TypedDict):
    """State for the real estate chat graph."""
    question: str 
    access_data: dict
    keywords: str
    clarify_query: str
    retrieval_required: dict
    rep_chunks_ids: set
    resource_type: List[str]
    stakeholder_query_state  : str
    event_query_state: str
    summary: str
    route: str
    _clarify_failed: bool 
    document_names: List[str]
    resource_type_list: List[str]
    detail_info : str  
    messages: Annotated[list, operator.add]
    generation: str
    answer_validation: bool
    context_validation: bool
    #generated_questions: List[str]
    documents: List[Document]
    relationships: List[Dict]



    # Long Context Workflow State
    content: str
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]  # add key for collapsed summaries
    final_summary: str

    my_ci_score : str
    risk_category_names: List[str]
    date_range: DateRange

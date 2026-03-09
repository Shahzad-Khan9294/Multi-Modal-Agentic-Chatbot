import time
import pandas as pd
from chatbot.db import db
# from langgraph import tools
from ...states import GraphState
from rapidfuzz import process, fuzz
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_community.retrievers import TFIDFRetriever
# from langchain_community.retrievers import BM25Retriever
from ...utils import get_langchain_vllm_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ...utils import get_db_connection, get_document_vectorstore
# from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import logging
logger = logging.getLogger(__name__)



def build_reference_links(selected_doc_chunks, link_lines):
    resources_by_type = {}
    # base="dev.theasiagroup.ai/advanced"
    type_labels = {"analysis": "ANALYSIS", "documents": "DOCUMENT"}
    for doc in selected_doc_chunks:
        meta = doc.metadata or {}
        r_type = meta.get("type", "resource").lower()
        r_id = meta.get("id")
        if not r_id:
            continue
        resources_by_type.setdefault(r_type, []).append({"id": r_id, "title": meta.get("heading") or meta.get("name") or meta.get("title")})
    if not resources_by_type:
        return
    for target_type in ["analysis", "documents"]:
        items = resources_by_type.get(target_type, [])
        if not items:
            continue
        for item in items:
            base_label = type_labels.get(target_type, target_type.upper())
            if len(items) > 1 and base_label != "ANALYSIS":
                display_heading = f"{base_label}S"
            else:
                display_heading = base_label
            title = (item["title"] or target_type).upper()
            url = "https://{base}"+f"/{target_type}/{item['id']}"
            link_lines.extend([
                f"**🔗 Check {base_label.capitalize()} Reference:**",
                f"[{title}]({url})"])
            

prompt = """
    You are generating separate factual summaries for multiple {resource_type} resources.
    Your output MUST follow this exact structure for **each resource individually**:
    **SUMMARY SOURCE**:
    - <Resource Name / Heading>
    **SUMMARY**:
    Here is the final summary derived from: <Resource Name / Heading>
<summary content of this resource>
    Repeat this structure for each resource provided. Do NOT merge multiple resources into a single summary.
    SOURCE SUMMARIES:
    {summaries}
    Important rules:
    - Each resource must have its own **SUMMARY SOURCES** and **SUMMARY** block.
    - Do NOT invent any headings or content.
    - Use ONLY the information provided in the SOURCE SUMMARIES.
    - Do not remove important details.
    - Maintain factual accuracy.
    """
llm = get_langchain_vllm_model(model="unsloth/gemma-3-12b-it-bnb-4bit")
Merge_Summary_prompt = ChatPromptTemplate([("human", prompt)])
Merge_Summary_chain = Merge_Summary_prompt | llm | StrOutputParser()
 
def _response_to_str(response):
    """Normalize chain output to string (astream yields str chunks; ainvoke returns str)."""
    if isinstance(response, str):
        return response
    return getattr(response, "content", str(response))
 
async def db_search(state: GraphState, config: RunnableConfig) -> dict:
    """
    DB search and merge summaries. Uses astream when stream_mode is set (stream endpoint)
    so summary tokens are streamed; uses ainvoke otherwise (invoke endpoint) for full response.
    """
    logger.info("---DB SEARCH---")
    access_data = state.get("access_data", {})
    stream_mode = config.get("configurable", {}).get("stream_mode")
    is_streaming = stream_mode in (True, "messages")
    # session = config.get("configurable", {}).get("session", None)
    question = state.get("question", "")
    document_names = state.get("document_names", [])
    logger.info(f"Document Names db-search: {document_names}")

    incoming_resource_types = state.get("resource_type",[]) or []
    resource_list = state["resource_type_list"]
    if not resource_list:
        resource_list = ["Analysis", "Document"]
    logger.info("---RESOURCE LIST NODE---")
    event_list = ["Events", "events", "Event", "event", "Calendars", "calendars", "calendar", "Calendar"]
    analysis_list = ["Articles", "articles", "Article", "article", "Analysis", "analysis"]
    for i in range(len(resource_list)):
        if resource_list[i] in event_list:
            resource_list[i] = "Calendar"
        elif resource_list[i] in analysis_list:
            resource_list[i] = "Analysis"
    incoming_set = set(r for r in incoming_resource_types)
    filtered_resource_list = []
    excluded_resources = []
    for r in resource_list:
        if r in incoming_set:
            filtered_resource_list.append(r)
        else:
            excluded_resources.append(f'"{r}"') 
    resource_list = filtered_resource_list
    state["resource_type_list"] = resource_list
    resource_list = state["resource_type_list"]
    excluded_message = ""
    if excluded_resources:
        excluded_str = ", ".join(excluded_resources)
        excluded_message = f"**Please select the {excluded_str} source from the selection list to get their related summary!**"
    logger.info(f"Resource List: {resource_list}")

    filter_ids = {
    "event": list(access_data.get('Calendar',[])) if "Calendar" in incoming_resource_types else [],
    "document": list(access_data.get('Document',[])) if "Document" in incoming_resource_types else [],
    "stakeholder": list(access_data.get('Stakeholder',[])) if "Stakeholder" in incoming_resource_types else [],
    "analysis": list(access_data.get('Analysis',[])) if "Analysis" in incoming_resource_types else []}
    
    link_lines = []
    seen_doc_ids = set()
    all_db_documents = []
    final_llm_outputs = []
    summary_buckets = defaultdict(list)
    fetched_docs = await db.fetch_summary_data(document_names, filter_ids, 10)
    for row in fetched_docs:
        (doc_id, doc_name, doc_summary, analysis_id, analysis_heading, analysis_summary, event_id, event_title,
         event_description, stakeholder_id, stakeholder_name, stakeholder_bio) = row
        for id_, content, meta, typ, label in [
            (doc_id, doc_summary, {"id": str(doc_id), "name": doc_name}, "documents", doc_name),
            (analysis_id, analysis_summary, {"id": str(analysis_id), "heading": analysis_heading}, "analysis", analysis_heading),
            (event_id, event_description, {"id": str(event_id), "title": event_title}, "events", event_title),
            (stakeholder_id, stakeholder_bio, {"id": str(stakeholder_id), "name": stakeholder_name}, "stakeholder-maps", stakeholder_name),]:
            if id_ is not None and id_ not in seen_doc_ids:
                all_db_documents.append(Document(page_content=content or "", metadata={**meta, "type": typ}))
                seen_doc_ids.add(id_)
                if content:
                    summary_buckets[typ].append({"label": label, "content": content})
    for typ, items in summary_buckets.items():
        formatted_items = "\n\n".join(f"**Heading:** {item['label']} \n**Summary:** {item['content']}"
            for item in items)
        final_llm_outputs.append(formatted_items)
        final_all_contents = "\n\n".join(final_llm_outputs) or ""
        
    # final_all_contents = "\n\n".join(doc["merged_summary"] for doc in final_llm_outputs)
    build_reference_links(all_db_documents, link_lines)
    if link_lines:
        final_all_contents += "\n\n" + "\n".join(link_lines)
    if excluded_message:
        final_all_contents += f"\n\n{excluded_message}"
    human_message = HumanMessage(content=state["question"])
    response = AIMessage(content=final_all_contents)
    return {"messages": [human_message, response]}








# print("type", typ)
    # if items:
    #     formatted_items = "\n\n".join(f"Heading: {item['label']}\nContent: {item['content']}"
    #         for item in items)
    #     llm_input = {"resource_type": typ, "summaries": formatted_items}
    #     result = await Merge_Summary_chain.ainvoke(llm_input, config=config)
    #     merged_summary = _response_to_str(result)
    #     final_llm_outputs.append({"type": typ, "merged_summary": merged_summary})
    #     # logger.info(f"-----------THIS IS final_llm_outputs------------: {final_llm_outputs}")

import time
import logging
import pandas as pd
from chatbot.db import db
from ...states import GraphState
from typing import Optional, List
from collections import defaultdict

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from ...chains import get_resource_keywords
from langchain_core.documents import Document
from ...utils import get_langchain_vllm_model
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from ...utils import get_db_connection, get_document_vectorstore
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser


TYPE_LABELS = {
    "stakeholder-maps": "Stakeholder",
    "events": "Event",
    "analysis": "Analysis",
    "documents": "Document",
}

def build_grouped_resource_sections(resource_buckets, content_lines, link_lines):
    SHOW_EVENT_URLS = False
    type_labels = {
        "stakeholder-maps": "STAKEHOLDER",
        "events": "EVENT",
        "analysis": "ANALYSIS",
        "documents": "DOCUMENT",
    }
    # base="dev.theasiagroup.ai/advanced"
    all_resources = [r for items in resource_buckets.values() for r in items]
    resources_by_type = {}
    for r in all_resources:
        resources_by_type.setdefault(r["type"].lower(), []).append(r)

    for r_type, items in resources_by_type.items():
        if not items:
            continue

        base_label = type_labels.get(r_type, r_type.upper())
        header_label = f"{base_label}{'S' if len(items) > 1 and base_label != 'ANALYSIS' else ''}:"
        content_lines.append(f"**{header_label}**")
        content_lines.extend(f"- {c}" for c in ((r.get("content") or "").strip() for r in items) if c)
        content_lines.append("")

        plural_suffix = "References" if len(items) > 1 else "Reference"
        friendly_cap = (
            base_label.capitalize()
            if len(items) == 1 or base_label == "ANALYSIS"
            else f"{base_label.capitalize()}s"
        )
        # --- links block (replace your current append/extend) ---

        # Build links first (so we can decide whether to print the header)
        resource_links = []

        for i, r in enumerate(items):
            title = (((r.get("content") or "").strip()) or r_type).upper()

            # Toggle: hide event URLs completely
            if r_type == "events" and not SHOW_EVENT_URLS:
                continue

            if r_type == "events":
                # (only used if SHOW_EVENT_URLS=True)
                url = "(https://{base}/events)"
                resource_links.append(f"[{title}]{url}")
            elif r_type == "stakeholder-maps":
                url = f"(https://{{base}}/stakeholder-maps/?stakeholderId={str(r['id'])}&index={i})"
                resource_links.append(f"[{title}]{url}")
            else:
                url = f"(https://{{base}}/{r_type}/{str(r['id'])})"
                resource_links.append(f"[{title}]{url}")

        # Only print the header if we actually have at least 1 link
        if resource_links:
            link_lines.append(f"**🔗 Check {friendly_cap} {plural_suffix}:**")
            link_lines.extend(resource_links)


async def resource_list(state: GraphState, config: RunnableConfig) -> dict:
    # session = config.get("configurable", {}).get("session", None)
    incoming_resource_types = state.get("resource_type", []) or []
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

    logger.info(f"Resource List_1: {resource_list}, {incoming_resource_types}")

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
        excluded_message = f"Please select the {excluded_str} from the selection list to get the related data."

    logger.info(f"Resource List: {resource_list}")

    requested_bucket_types = set()
    for r in resource_list:
        if r == "Calendar":
            requested_bucket_types.add("events")
        elif r == "Document":
            requested_bucket_types.add("documents")
        elif r == "Analysis":
            requested_bucket_types.add("analysis")
        elif r == "Stakeholder":
            requested_bucket_types.add("stakeholder-maps")

    org_region_id = config.get("configurable", {}).get("org_region_id", None)
    data = await db.get_all_org_region_data_parallel(org_region_id, resource_list or [])
    filter_ids = {
        "event": list(data.get("Calendar", [])) if "Calendar" in resource_list else [],
        "document": list(data.get("Document", [])) if "Document" in resource_list else [],
        "analysis": list(data.get("Analysis", [])) if "Analysis" in resource_list else [],
        "stakeholder": list(data.get("Stakeholder", [])) if "Stakeholder" in resource_list else [],
    }
    print("Stakeholder resource id: ", len(data.get("Stakeholder")))

    question = state["question"]
    resourcekey_extractor = get_resource_keywords(model="unsloth/gemma-3-12b-it-bnb-4bit")
    resource_keywords = await resourcekey_extractor.ainvoke({"question": question})
    keyword_dict = {
        "stakeholder": resource_keywords.stakeholders,
        "event": resource_keywords.events,
        "analysis": resource_keywords.articles,
        "document": resource_keywords.documents,
    }
    logger.info(f"Resource Extracted Keywords: {keyword_dict}")

    link_lines = []
    content_lines = []
    grader_inputs = []
    all_resources = []
    flat_resources = []
    seen_doc_ids = set()
    all_db_documents = []
    filtered_resource = []
    structured_resource_outputs = []
    resource_buckets = defaultdict(list)

    fetched_docs = await db.fetch_resource_data(keyword_dict, filter_ids, 20)
    for row in fetched_docs:
        (document_id, document_name, analysis_id, analysis_heading, event_id, event_title, stakeholder_id, stakeholder_name, _) = row
        for id_, label, meta, typ in [
            (document_id, document_name, {"id": str(document_id), "name": document_name}, "documents"),
            (analysis_id, analysis_heading, {"id": str(analysis_id), "heading": analysis_heading}, "analysis"),
            (event_id, event_title, {"id": str(event_id), "title": event_title}, "events"),
            (stakeholder_id, stakeholder_name, {"id": str(stakeholder_id), "name": stakeholder_name}, "stakeholder-maps"),
        ]:
            if id_ is None or id_ in seen_doc_ids:
                continue
            seen_doc_ids.add(id_)
            all_db_documents.append(
                Document(page_content=label or "", metadata={**meta, "type": typ})
            )
            resource_buckets[typ].append(
                {"id": str(id_), "content": label or "", "type": typ}
            )

    missing_types = [t for t in requested_bucket_types if not resource_buckets.get(t)]
    missing_parts = []
    for t in missing_types:
        friendly = TYPE_LABELS.get(t, t).lower()
        missing_parts.append(f"- No relevant {friendly} results were found for your request.")
    missing_data_message = ""
    if missing_parts:
        missing_data_message = (
            "**Note:** Some requested resources have no relevant data available right now:\n"
            + "\n".join(missing_parts)
        )

    build_grouped_resource_sections(resource_buckets, content_lines, link_lines)

    if not content_lines and not link_lines:
        final_resource_text = "There is no relevant information provided"
    else:
        spacing = "\n\n"
        dash = "---"
        content_text = "\n".join(content_lines)
        link_text = f"{spacing}".join(link_lines)
        final_resource_text = f"{content_text}{spacing}{dash}{spacing}{link_text}{spacing}"

    if missing_data_message:
        final_resource_text = f"{final_resource_text}\n\n{missing_data_message}"

    logger.info(f"------- Final Resource List --------:\n {final_resource_text}")

    if excluded_message:
        final_resource_text = f"{final_resource_text}\n\n{excluded_message}"
        logger.info(f"------- Final Resource List (excluded)--------:\n {final_resource_text}")

    human_message = HumanMessage(content=state["question"])
    response = AIMessage(content=final_resource_text.strip().strip('---').strip())
    return {"messages": [human_message, response]}
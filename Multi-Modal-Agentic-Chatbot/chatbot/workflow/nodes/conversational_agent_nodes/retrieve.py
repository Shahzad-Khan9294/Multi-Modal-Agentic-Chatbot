import os
import time
import json
import asyncio
import re
from typing import List, Dict
from datetime import datetime
from itertools import zip_longest   # THIS
from langchain_core.documents import Document
from dateutil.relativedelta import relativedelta
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import TFIDFRetriever
import logging
logger = logging.getLogger(__name__)
from ...states import GraphState
from ...utils import (
    get_embedding)
from chatbot.db import db
from .risk_data_normalization import rcdn


embeddings = get_embedding()
async def format_relationships(relationships: List[Dict]) -> dict:
    logger.info("---STRUCTURED RELATIONSHIPS---")
    documents = []
    check_text = []
    # print(f"Total relationships: {len(relationships)}")
    for relationship in relationships:
        relationship_description_format= f"{relationship['source_entity']['entity_text']} {relationship['relationship_type'].replace('_', ' ')} {relationship['target_entity']['entity_text']}"
        # remove all spaces punctuation and special characters and lowercase the text
        processed_text = re.sub(r'[^\w\s]', '', relationship_description_format).lower()
        if processed_text in check_text:
            continue
        check_text.append(processed_text)
        relationship_description = Document(page_content=relationship_description_format, metadata={"source_name": relationship['source_entity']['entity_text'], "target_name": relationship['target_entity']['entity_text']})   
        documents.append(relationship_description)
        # logger.info(f"Relationship description: {relationship_description}")
    logger.info(f"Total documents: {len(documents)}")
    return documents

def clean_note(note: str) -> str:
    if not note:
        return note
    match = re.search(r"<p>(.*?)</p>", note, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"<.*?>", "", note).strip()

async def build_category_summary(retrieved_risk_category_data, date_range):
    structured_score_summary = []
    for risk_cat_id, data in retrieved_risk_category_data.items():
        category = data["category"]
        scores = data.get("scores", [])
        if not scores:
            continue
        normalized = rcdn.normalize_rows(scores)
        # logger.info(f"ALL NORMALIZED TIMESTAMPS: {[r['timestamp'] for r in normalized]}")
        min_date = min(r["timestamp"].date() for r in normalized)
        max_date = max(r["timestamp"].date() for r in normalized)
        # logger.info(f"RISK DATE MIN MAX Date Range: {min_date} to {max_date}")
        sampled = rcdn.select_records_by_timerange(normalized, min_date, max_date)
        # logger.info(f"SAMPLED TIMESTAMPS: {[r['timestamp'] for r in sampled]}")
        sampled = sorted(sampled, key=lambda x: x["timestamp"], reverse=True)
        category_summary_lines = [f"**Here's the information about {category['name']} recent CI Score, based on the provided data:**\n\n"]
        for item in sampled:
            payload = json.loads(item["json"])
            category_keys_mapping = category.get("mapping", {})
            new_aggregated_scores = payload.get("new_aggregated_score", {})
            scores_with_aliases = {
                category_keys_mapping.get(k, k): round(v)
                for k, v in new_aggregated_scores.items()
                if k in category_keys_mapping}
            timestamp_str = item["timestamp"].date().isoformat()
            note = clean_note(payload.get("note"))
            category_summary_lines.append(f"{timestamp_str}:")
            for score_name, score_value in scores_with_aliases.items():
                category_summary_lines.append(f"- {score_name}: {score_value}")
            if note:
                category_summary_lines.append(f"**Note:** {note}")
            category_summary_lines.append("")
            document = Document(page_content="\n".join(category_summary_lines), metadata={"id": str(risk_cat_id), "type": "risk_category_data", "timestamp": item["timestamp"].isoformat()}) 
            structured_score_summary.append(document)
    return structured_score_summary

async def process_risk_category_data(risk_category_data, date_range):
    # all_risk_documents: List[Document] = []
    # for risk_cat_id, data in risk_category_data.items():
    #     category = data["category"]
    #     scores = data["scores"]
    #     if not scores:
    #         continue
        # normalized = rcdn.normalize_rows(scores)  
        # sampled = rcdn.select_records_by_timerange(normalized, date_range.start_date, date_range.end_date)
        # for item in sampled:
        #     payload = json.loads(item["json"])
        #     category_keys_mapping = category['mapping']
        #     new_aggregated_scores = payload.get("new_aggregated_score", {})
        #     scores_with_aliases = {category_keys_mapping.get(k, k): v for k, v in new_aggregated_scores.items() if k in category_keys_mapping}
        #     scores_rounded = {k: round(v) for k, v in scores_with_aliases.items()}
            
        #     clean_doc = {"type": "risk_category_data", "timestamp": item["timestamp"].date().isoformat(),
        #             "risk_category": {"name": category["name"], "risk_category_description": category["description"], "mapping": category_keys_mapping}, "new_aggregated_scores": scores_rounded, "note": payload.get("note")}
    all_risk_documents = await build_category_summary(risk_category_data,date_range)
    # documents = [Document(page_content=json.dumps(clean_doc, separators=(",", ":")),
    #             metadata={"id": str(risk_cat_id), "type": "risk_category_data", "timestamp": item["timestamp"].isoformat()})]  
    # all_risk_documents.extend(documents)
    logger.info("--------------- Risk FINAL DATA Length ---------------: %s", len(all_risk_documents))
    return all_risk_documents

async def retrieve_risk_documents(date_range, risk_category_names, org_region_id):
    if date_range.start_date is not None and date_range.end_date is not None:
        if date_range.start_date < date_range.end_date - relativedelta(months=6):
            today = datetime.combine(datetime.today().date(), datetime.min.time())
            date_range.start_date = today - relativedelta(months=6)
            date_range.end_date = today
            logger.info("MORE THAN 6 MONTHS DATE RANGE: %s", date_range)
    risk_category_fuzz_data = await db.get_risk_category_fuzz(org_region_id, risk_category_names)
    if risk_category_fuzz_data:
        logger.info("Inside Retrieve Risk Category!") 
        risk_category_data, _date_range = await db.get_org_region_risk_category(org_region_id, risk_category_names, date_range) 
        risk_results = await process_risk_category_data(risk_category_data=risk_category_data, date_range=_date_range)
        return risk_results

async def retrieve_df_adjacent_documents(detail_info, df_similar, incoming_resource_types, rep_chunks_ids):
    new_docs = []
    local_adjacent_ids = set()
    check_adjacent_chunks = set() 
    start_time_p = time.perf_counter()
    logger.info("Time taken to fetch combine embeddings! %s", time.perf_counter() - start_time_p)
    all_adjacent_documents: List[Document] = []
    for id, text, metadata, type, similarity in df_similar.itertuples(index=False):
        if id not in rep_chunks_ids:
            local_adjacent_ids.add(id)
            rep_chunks_ids.add(id)
        new_docs.append(Document(page_content=text, metadata={"id": id, "similarity": similarity, **(metadata or {}), "type": type}))
    check_adjacent_chunks.update(local_adjacent_ids)
    all_adjacent_documents.extend(new_docs)
    if 'Document' in incoming_resource_types or 'Analysis' in incoming_resource_types:
        if detail_info == 'yes':
            analysis_adjacent,  document_adjacent = await db.adjacent_chunks_retrieval(list(check_adjacent_chunks), 3)
        else:
            analysis_adjacent,  document_adjacent = await db.adjacent_chunks_retrieval(list(check_adjacent_chunks), 1)
        for analysis_row, document_row in zip_longest(analysis_adjacent.itertuples(index=False), document_adjacent.itertuples(index=False)):
            if analysis_row:
                seed_id, analysis_id, text, chunk_index = analysis_row
                all_adjacent_documents.append(Document(page_content=text, metadata={"a_id": seed_id, "id": analysis_id, "chunk_index": chunk_index, "type": "analysis"}))
            if document_row:
                seed_id, document_id, text, chunk_index = document_row
                all_adjacent_documents.append(Document(page_content=text, metadata={"d_id": seed_id, "id": document_id, "chunk_index": chunk_index, "type": "document"}))
        logger.info("--------------- ADJACENT FINAL DATA Length ---------------: %s", len(all_adjacent_documents))
        return all_adjacent_documents

async def retrieve(state: GraphState, config: RunnableConfig) -> Dict[str, List[Document]]:
    logger.info("---RETRIEVE---")
    start_time_1 = time.perf_counter()
    query = state.get("question", "")
    detail_info = state["detail_info"]
    access_data = state.get("access_data", {})
    query_keywords = state.get("keywords", "")
    clarify_query = state.get("clarify_query", "")
    rep_chunks_ids = state.get("rep_chunks_ids", set())
    incoming_resource_types = state.get("resource_type",[]) or []
    logger.info("incoming_resource_types: %s", incoming_resource_types)
    org_region_id = config.get("configurable", {}).get("org_region_id", None)

    all_documents: List[Document] = []
    query_embedding = await embeddings.aembed_query(clarify_query)

    filter_ids = {"event": list(access_data.get('Calendar',[])) if "Calendar" in incoming_resource_types else [],
    "document": list(access_data.get('Document',[])) if "Document" in incoming_resource_types else [],
    "analysis": list(access_data.get('Analysis',[])) if "Analysis" in incoming_resource_types else [],
    "stakeholder": list(access_data.get('Stakeholder',[])) if "Stakeholder" in incoming_resource_types else []}
    access_data_ids_list = list(access_data.values())
    access_data_ids = []
    for ids in access_data_ids_list:
        access_data_ids.extend(ids)
    access_data_ids = tuple(set(access_data_ids))
    relationships = state.get("relationships", [])
    
    start_time_d = time.perf_counter()
    task_get_matched_entities = asyncio.create_task(db.get_matched_entities(relationships, access_data_ids))
    task_fetch_combine_embeddings = asyncio.create_task(db.fetch_combine_embeddings(query_embedding, query_keywords, filter_ids, list(rep_chunks_ids)))
    matched_entities_relationships, df_similar = await asyncio.gather(task_get_matched_entities, task_fetch_combine_embeddings)
    logger.info("time taken to fetch combine embeddings: %s", time.perf_counter() - start_time_d)

    if "Risk" in incoming_resource_types:
        date_range = state.get("date_range", None)
        risk_category_names = state.get("risk_category_names", [])
        risk_task = asyncio.create_task(retrieve_risk_documents(date_range, risk_category_names, org_region_id))  
        df_task =  asyncio.create_task(retrieve_df_adjacent_documents(detail_info, df_similar, incoming_resource_types, rep_chunks_ids))  
        task_get_all_relationships = asyncio.create_task(db.get_all_relationships(matched_entities_relationships, access_data_ids))
        risk_docs, df_docs, expanded_relationships = await asyncio.gather(risk_task, df_task,task_get_all_relationships)
        if risk_docs:
            all_documents.extend(risk_docs)
    else:
        df_task =  asyncio.create_task(retrieve_df_adjacent_documents(detail_info, df_similar, incoming_resource_types, rep_chunks_ids))  
        task_get_all_relationships = asyncio.create_task(db.get_all_relationships(matched_entities_relationships, access_data_ids))
        df_docs,expanded_relationships = await asyncio.gather(df_task,task_get_all_relationships)
    
    logger.info(f"df_docs: {len(df_docs)}")
    logger.info(f"expanded_relationships: {len(expanded_relationships)}")
    relationships_docs = await format_relationships(expanded_relationships)
    logger.info(f"Total Graph documents: {len(relationships_docs)}")
    if relationships_docs:
        all_documents.extend(relationships_docs)
    if df_docs:
        all_documents.extend(df_docs)
    

    if not all_documents:
        logger.info("No documents found for the given query(ies).")
        return {"documents": []}

    check_text = []
    unique_documents = []
    for document in all_documents:
        text = document.page_content
        if text not in check_text:
            unique_documents.append(document)
            check_text.append(text)
            
    logger.info("len(all_documents): %s", len(unique_documents))
    end_time_1 = time.perf_counter()
    inference_time_1 = end_time_1 - start_time_1
    logger.info(f"---------------------- Retrieval inference time ----------------------: {inference_time_1:.4f} seconds")
    return {"documents": unique_documents, "rep_chunks_ids": rep_chunks_ids}
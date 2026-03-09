import re
import json
import logging
from chatbot.db import db
from bs4 import BeautifulSoup
from ...states import GraphState
from .risk_data_normalization import rcdn
from ...utils import get_langchain_vllm_model
from langchain_core.documents import Document
from datetime import datetime, time, timedelta
from ...chains import get_risk_generation_chain
from dateutil.relativedelta import relativedelta
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
logger = logging.getLogger(__name__)


# def clean_note(note: str) -> str:
#     if not note:
#         return note
#     match = re.search(r"<p>(.*?)</p>", note, re.IGNORECASE | re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return re.sub(r"<.*?>", "", note).strip()

def clean_note(note: str) -> str:
    if not note:
        return note
    soup = BeautifulSoup(note, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text

async def build_category_summary(retrieved_risk_category_data, date_range, structured_score_summary):
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
        category_summary_lines = [f"**Here's the information about {category['name']} recent score, based on the provided data:**\n\n"]
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
        structured_score_summary.append("\n".join(category_summary_lines))
    return structured_score_summary

generate_risk_chain = get_risk_generation_chain(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=1000)
async def risk_category_search(state: GraphState, config: RunnableConfig) -> dict:
    structured_score_summary = []
    risk_category_data: List[Document] = []
    logger.info("---RISK CATEGORY SEARCH---")
    # session = config.get("configurable", {}).get("session", None)
    org_region_id = config.get("configurable", {}).get("org_region_id", None)
    question = state.get("question", "")
    risk_category_names = state.get("risk_category_names", [])
    date_range = state.get("date_range", None)
    messages = state.get("messages", [])
    
    if date_range.start_date is not None and date_range.end_date is not None:
        if date_range.start_date < date_range.end_date - relativedelta(months=6):
            today = datetime.combine(datetime.today().date(), datetime.min.time())
            date_range.start_date = today - relativedelta(months=6)
            date_range.end_date = today
            logger.info(f"MORE THAN 6 MONTHS DATE RANGE: {date_range}")
    retrieved_risk_category_data, _ = await db.get_org_region_risk_category(org_region_id, risk_category_names, date_range)
    structured_score_summary = await build_category_summary(retrieved_risk_category_data, date_range, structured_score_summary)
    logger.info(f"--------------- Risk FINAL DATA Length (Before LLM) ---------------: {len(structured_score_summary)}")

    # if structured_score_summary and date_range.end_date - date_range.start_date <= timedelta(days=14):
    #     logger.info(f"--------------- Risk FINAL DATA Length (With_LLM) ---------------: {len(structured_score_summary)}")    
    #     response = await generate_risk_chain.ainvoke({"context": structured_score_summary, "start_date": date_range.start_date, "end_date": date_range.end_date, "chat_history": messages, "question": question})
    # else:
    if structured_score_summary:
        logger.info(f"--------------- Risk FINAL DATA Length (Without_LLM) ---------------: {len(structured_score_summary)}")    
        response = "\n".join(structured_score_summary)
    else:
        response = "I am sorry there is no recent ci-score data related to this risk category!"
    return {"messages": [HumanMessage(content=question), AIMessage(content=response)]}
import time
import json
import asyncio
import logging
from chatbot.db import db
from ...schema import DateRange
from calendar import monthrange
from ...states import GraphState
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta, time as time_module
from ...chains import get_question_router
from dateutil.relativedelta import relativedelta
from langchain_core.runnables import RunnableConfig
from ...chains import extract_query_entities_chain, get_clarify_chain, get_date_extraction_chain, get_extract_entity_relation_chain



month_indexing = {"january": 1,  "february": 2,  "march": 3, "april":4, "may": 5, "june":6, "july":7, "august":8, "september":9, "october":10, "november": 11, "december":12}
async def date_range_extrcator(date_range):
    date_range.days, date_range.weeks, date_range.months, date_range.years = \
        date_range.days or 0, date_range.weeks or 0, date_range.months or 0, date_range.years or 0
    logger.info("DATES SCHEMA OUTPUT --- Start_Date: %s --- End_Date: %s --- Calendar_Month: %s --- Days: %s --- Weeks: %s --- Months: %s --- Years: %s", date_range.start_date, date_range.end_date, date_range.calendar_month, date_range.days, date_range.weeks, date_range.months, date_range.years)
    if date_range.start_date and date_range.end_date:
        today = datetime.utcnow()
        start_date, end_date = date_range.start_date, date_range.end_date 
        if start_date.month > today.month and end_date.month > today.month:
            start_date = start_date.replace(year=today.year - 1)
            end_date = end_date.replace(year=today.year - 1)
        if start_date > end_date:
            end_date = start_date
        if start_date == end_date:
            end_date = end_date + timedelta(days=1)
        date_range.start_date = start_date
        date_range.end_date = end_date
        logger.info("--- Start New Date (WITHOUT COUNT) ---: %s", date_range.start_date)
        logger.info("--- End New Date (WITHOUT COUNT) ---: %s", date_range.end_date)
    else:
        today =  datetime.now()
        logger.info("--- DATE RANGE QUERY ROUTE ---: %s", date_range)
        if (date_range.months == 0 and date_range.weeks == 0 and date_range.days == 0 and date_range.years == 0 and date_range.calendar_month is None):
            return DateRange(start_date=None, end_date=None)
        if date_range.calendar_month:
            calendar_month = month_indexing.get(date_range.calendar_month.lower())
            calendar_year = today.year - date_range.years
            if date_range.years in [None, 0]:
                if calendar_month > today.month:
                    calendar_year -= 1
            if calendar_year:
                start_date = datetime(calendar_year, calendar_month, 1)
                end_date = datetime(calendar_year, calendar_month,
                    monthrange(calendar_year, calendar_month)[1])
            if date_range.days or date_range.weeks:
                if calendar_month == today.month and calendar_year == today.year:
                    end_day = today.day
                else:
                    end_day = monthrange(calendar_year, calendar_month)[1]
                end_date = datetime(calendar_year, calendar_month, end_day)
                total_days = (date_range.days or 0) + (date_range.weeks or 0) * 7
                start_day = max(1, end_day - total_days + 1)
                start_date = datetime(calendar_year, calendar_month, start_day)
        else:
            if date_range.months > 0 and date_range.years > 0:
                target_year = today.year - date_range.years
                start_month = 12 - date_range.months + 1
                start_date = datetime(target_year, start_month, 1)
                end_date = datetime(target_year, 12, monthrange(target_year, 12)[1])
            else:
                base_date = today - relativedelta(years=date_range.years)
                start_date = base_date - relativedelta(months=date_range.months) \
                                        - timedelta(weeks=date_range.weeks, days=date_range.days)
                end_date = today
        date_range = DateRange(start_date=start_date, end_date=end_date)
        logger.info("--- Start Date (WITH COUNT) ---: %s", date_range.start_date)
        logger.info("--- End Date (WITH COUNT) ---: %s", date_range.end_date)
    return date_range

clarify_chain =  get_clarify_chain(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=300)
router =  get_question_router(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=100)
date_chain = get_date_extraction_chain(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=100)
extract_entity_relation_chain = get_extract_entity_relation_chain(model="unsloth/gemma-3-12b-it-bnb-4bit", max_tokens=200)

async def query_route(state: GraphState, config: RunnableConfig) -> dict:
    start_time = time.perf_counter()
    logger.info("---ROUTING QUESTION---")
    question = state.get("question", "")
    org_region_id = config.get("configurable", {}).get("org_region_id", None)
    incoming_resource_types = state.get("resource_type", [])

    task_extract_er = asyncio.create_task(extract_entity_relation_chain.ainvoke({"question": question}))
    task_a = asyncio.create_task(db.get_all_org_region_data_parallel(org_region_id, incoming_resource_types))
    task_c = asyncio.create_task(clarify_chain.ainvoke(input={ "question": question, "chat_history": state["messages"][-2:]}))
    task_r = asyncio.create_task(router.ainvoke({"question": question}))

    if 'Risk' in incoming_resource_types:
        logger.info("------ Running Risk For Date Range -------")
        task_d = asyncio.create_task(date_chain.ainvoke({"question": question}))
        # access_data, clarify_result, route_query, date_range, relationships = await asyncio.gather(task_a, task_c, task_r, task_d, task_extract_er)
        access_data, clarify_result, route_query, date_range, relationships = await asyncio.gather(
            task_a,
            task_c,
            task_r,
            task_d,
            task_extract_er,
            return_exceptions=True
        )

        date_range = date_range.date_range
        my_ci_score = route_query.my_ci_score
        date_range = await date_range_extrcator(date_range)
        if my_ci_score == 'yes' and date_range.start_date is None and date_range.end_date is None:
            today = datetime.now().date()
            date_range.start_date, date_range.end_date = (datetime.combine(today - relativedelta(days=1),time_module.min), datetime.combine(today, time_module.min)) 
            logger.info("------ 1 day date range ------: %s", date_range)
    else:
        access_data, clarify_result, route_query, relationships = await asyncio.gather(task_a, task_c, task_r, task_extract_er,return_exceptions=True)

    if isinstance(relationships, Exception):
        logger.error(f"Task failed: {relationships}")
        relationships = []
    else:
        relationships = relationships.model_dump_json()
        relationships = json.loads(relationships)["relationships"]
        logger.info(f"Relationships: {relationships}") 

    question = clarify_result.question
    logger.info("------ Clarify Query -------: %s", clarify_result)
    logger.info("Route Query: %s", route_query)
    route = route_query.route
    detail_info = route_query.detail_info
    my_ci_score = route_query.my_ci_score
    document_names = route_query.document_names
    resource_type_list = route_query.resource_type_list
    risk_category_names = route_query.risk_category_names
    logger.info("Routed to: %s with documents: %s with resource type list: %s and demanding more information: %s and my_ci_score: %s", route, document_names, resource_type_list, detail_info, my_ci_score)

    rep_chunks_ids = state.get("rep_chunks_ids", set())
    flush_repeated_chunks = rep_chunks_ids.copy()

    keywords = clarify_result.keywords
    if 'stakeholder' in question.lower():
        keywords += ['Stakeholder']
    keywords = " | ".join([w.replace(' ',' & ') if ' ' in w else w for w in keywords]).lower()

    next_node = {"route": route, "access_data": access_data, "document_names": document_names, "relationships": relationships, "keywords": keywords, "clarify_query": clarify_result.question,
    "detail_info": detail_info, "my_ci_score": my_ci_score, "resource_type_list": resource_type_list, "rep_chunks_ids": flush_repeated_chunks, 
    "risk_category_names": risk_category_names, **({"date_range": date_range} if "Risk" in incoming_resource_types else {})}

    if rep_chunks_ids:
        flush_repeated_chunks = rep_chunks_ids
        if detail_info == 'no':
            flush_repeated_chunks = set()
            next_node["rep_chunks_ids"] = flush_repeated_chunks
            logger.info("Repeated Org_Region_ids Flushed!")
        else:
            next_node["question"] = clarify_result.question

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    logger.info(f"--------------- Query Route inference time --------------: {inference_time:.4f} seconds")
    return next_node

import os
import time
import json
import dotenv
import asyncio
import logging
import pandas as pd
from collections import defaultdict
from sqlalchemy import text
from datetime import datetime, time, timedelta 
from dotenv import load_dotenv
load_dotenv('.env.client')
from rapidfuzz import process, fuzz
from typing import List, Dict, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from chatbot.database import get_db_session

logger = logging.getLogger(__name__)
class DB:
    """
    Async database operations class.
    All methods use async database sessions for non-blocking I/O.
    """
    
    async def _execute_query(self, query: str, params: Optional[Dict] = None) -> List:
        """Execute a query and return results as a list of rows."""
        async with get_db_session() as session:    
            result = await session.execute(query, params or {})
            rows = result.fetchall()
        return rows
    
    async def _execute_query_to_df(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame."""
        rows = await self._execute_query(query, params)
        if not rows:
            return pd.DataFrame()
        # Convert rows to list of dicts
        data = [dict(row._mapping) for row in rows]
        return pd.DataFrame(data)

    async def get_document_by_ids(self, document_ids):
        in_clause = str(tuple(document_ids)).replace(',)',')')
        query = f"""
        SELECT *
        FROM document
        WHERE id in {in_clause}
        """
        df = await self._execute_query_to_df(query)
        return df

    async def get_risk_category_fuzz(self, orgRegionId: str, riskCategoryNames: List[str]) -> Dict:
        logger.info(f"Risk Category Fuzz riskCategoryNames: {riskCategoryNames}")
        query = text("""
                SELECT distinct rc.id, rc.name
                    FROM risk_category_client rcc
                    INNER JOIN risk_category_client_access_mapping rccam
                        ON rcc.id = rccam.risk_category_client_id
                    LEFT JOIN risk_category rc
                        ON rc.id = rcc.risk_category_id
                    LEFT JOIN risk_category_access rca
                        ON rca.id = rccam.risk_category_access_id
                        AND (rca.org_region_access_id = :org_region_id
                            OR rca.org_region_access_id IS NULL)""")  
        rows = await self._execute_query(query, {"org_region_id": orgRegionId})
        # result = await session.execute(query, {"org_region_id": orgRegionId})
        # rows = result.fetchall()
        logger.info(f"risk_category_fuzz_org_region_ids: {len(rows)} and its type is: {type(rows)}")
        required_risk_category_fuzz_data = {}
        riskCategoryNames_lc = [name.lower() for name in riskCategoryNames]
        for row in rows:
            risk_category_fuzz_id = row[0]
            risk_category_fuzz_names = row[1]
            if not risk_category_fuzz_names or not riskCategoryNames:
                logger.info("Skipping fuzz match due to empty input")
                break
            match = process.extractOne(risk_category_fuzz_names.lower(), riskCategoryNames_lc, scorer=fuzz.ratio)
            if match is None:
                logger.info(f"No fuzz match for: {risk_category_fuzz_names}")
                continue
            best_match, score, idx = match
            if score >= 80:
                if risk_category_fuzz_id not in required_risk_category_fuzz_data:
                    required_risk_category_fuzz_data[risk_category_fuzz_id] = []
                required_risk_category_fuzz_data[risk_category_fuzz_id].append(dict(row._mapping))
        logger.info(f"Length of Risk Category Fuzz data: {len(required_risk_category_fuzz_data)}")
        return required_risk_category_fuzz_data   
        
    async def get_risk_category_score(self, required_risk_category_data: Dict, date_range) -> Dict:
        query = text("""
            SELECT risk_category_id, new_aggregated_score, note, date_time
            FROM risk 
            WHERE risk_category_id = ANY(CAST(:risk_category_ids AS uuid[])) AND (date_time >= :start_date AND date_time <= :end_date)""")
        rows = await self._execute_query(query, {"risk_category_ids": list(required_risk_category_data.keys()), "start_date": date_range.start_date, "end_date": date_range.end_date})
        dict_rows = [dict(row._mapping) for row in rows]
        return dict_rows

    async def get_org_region_risk_category(self, orgRegionId: str, riskCategoryNames: List[str], date_range) -> Dict:
        logger.info(f"date_range: {date_range}")
        logger.info(f"riskCategoryNames: {riskCategoryNames}")        
        query = text("""
            SELECT
                distinct rc.id, rc.name, rc.description, rcc.key, rccam.alias
                FROM risk_category_client rcc
                INNER JOIN risk_category_client_access_mapping rccam
                    ON rcc.id = rccam.risk_category_client_id
                LEFT JOIN risk_category rc
                    ON rc.id = rcc.risk_category_id
                LEFT JOIN risk_category_access rca
                    ON rca.id = rccam.risk_category_access_id
                    AND (
                        rca.org_region_access_id = :org_region_id
                        OR rca.org_region_access_id IS NULL)""")
        rows = await self._execute_query(query, {"org_region_id": orgRegionId})
        logger.info(f"org_region_risk_category: {len(rows)}")
        required_risk_category_data = {}        

        riskCategoryNames_lc = [name.lower() for name in riskCategoryNames]
        grouped_category_data = defaultdict(lambda: {"id": None, "name": None, "mapping": {}})
        for id_, name, description, key, alias in rows:
            entry = grouped_category_data[id_]
            entry["id"] = id_
            entry["name"] = name
            entry["description"] = description
            if key not in entry['mapping']:
                entry['mapping'][key]=alias
        for risk_category_id, entry in grouped_category_data.items():
            category_name = entry["name"]
            if not category_name:
                continue
            best_match, score, _ = process.extractOne(category_name.lower(), riskCategoryNames_lc, scorer=fuzz.ratio)
            if score > 85:
                required_risk_category_data[risk_category_id] = {"category": entry, "scores": []}
        logger.info(f"required_risk_category_data KEYS: {list(required_risk_category_data.keys())}")
        # logger.info(f"required_risk_category_data: {required_risk_category_data}")

        if date_range.start_date is not None and date_range.end_date is not None:
                # date_range.end_date =  datetime.combine(date_range.end_date + relativedelta(days=1), time.min) 
                date_range.end_date = date_range.end_date + timedelta(days=1) - timedelta(seconds=1)
                required_risk_data = await self.get_risk_category_score(required_risk_category_data, date_range)
                logger.info(f"Length of Required Risk Data of Specified Date Range: {len(required_risk_data)}")
        else:
            today = datetime.now().date()
            date_range.start_date, date_range.end_date = (datetime.combine(today - relativedelta(months=1),time.min), datetime.combine(today, time.min)) 
            logger.info(f"1 Month date range: {date_range}")
            required_risk_data = await self.get_risk_category_score(required_risk_category_data, date_range)
            logger.info(f"Length of Required Risk Data of 1 Month Date Range: {len(required_risk_data)}")
        for row in required_risk_data:
            rid = row["risk_category_id"] 
            if rid in required_risk_category_data:
                required_risk_category_data[rid]["scores"].append(row)
        # logger.info(f"Final required_risk Data DB: {required_risk_category_data}")
        return required_risk_category_data, date_range

    async def get_org_region_resources_unified(
        self, 
        orgRegionId: str
    ) -> Dict[str, List[str]]:
        """Get all org region resource IDs in a single unified query."""
        query = text("""
            SELECT 'Document' as resource_type, resource_id::text as id 
            FROM public.document_access
            WHERE org_region_access_id IS NULL 
            OR org_region_access_id = :org_region_id
            
            UNION ALL
            
            SELECT 'Calendar' as resource_type, resource_id::text as id 
            FROM public.event_access
            WHERE org_region_access_id IS NULL 
            OR org_region_access_id = :org_region_id
            
            UNION ALL
            
            SELECT 'Stakeholder' AS resource_type, stg.stakeholder_id::text AS id
            FROM stakeholder_to_group stg
            WHERE stg.deleted_at IS NULL
            AND EXISTS (
                SELECT 1
                FROM stakeholder_group_access sga
                WHERE sga.resource_id = stg.stakeholder_group_id
                AND sga.org_region_access_id IS NOT DISTINCT FROM :org_region_id
            )
            
            UNION ALL
            
            SELECT 'Analysis' as resource_type, resource_id::text as id 
            FROM public.political_analysis_access
            WHERE org_region_access_id IS NULL 
            OR org_region_access_id = :org_region_id
        """)
        rows = await self._execute_query(query, {"org_region_id": orgRegionId})
        
        # Group results by resource type
        results = {'Document': [], 'Calendar': [], 'Stakeholder': [], 'Analysis': []}
        for row in rows:
            resource_type = row.resource_type
            if resource_type in results:
                results[resource_type].append(row.id)
        
        return results

    async def get_org_region_document(self, orgRegionId: str) -> List[str]:
        """Get document IDs accessible by org region."""
        unified_results = await self.get_org_region_resources_unified(orgRegionId)
        return unified_results.get('Document', [])

    async def get_org_region_event(self, orgRegionId: str) -> List[str]:
        """Get event IDs accessible by org region."""
        unified_results = await self.get_org_region_resources_unified(orgRegionId)
        return unified_results.get('Calendar', [])

    async def get_org_region_stakeholder(self, orgRegionId: str) -> List[str]:
        """Get stakeholder IDs accessible by org region."""
        unified_results = await self.get_org_region_resources_unified(orgRegionId)
        return unified_results.get('Stakeholder', [])
    
    async def get_org_region_analysis(self, orgRegionId: str) -> List[str]:
        """Get analysis IDs accessible by org region."""
        unified_results = await self.get_org_region_resources_unified(orgRegionId)
        return unified_results.get('Analysis', [])

    async def get_org_region_client(self, orgRegionId: str) -> List[Dict]:
        """Get client data accessible by org region."""
        query = text("""
            SELECT first_name, last_name 
            FROM client
            WHERE id IN (
                SELECT client_id 
                FROM public.client_region_access
                WHERE org_region_access_id IS NULL 
                OR org_region_access_id = :org_region_id
            )
        """)
        rows = await self._execute_query(query, {"org_region_id": orgRegionId})
        return [{"first_name": row.first_name, "last_name": row.last_name} for row in rows]

    async def get_all_org_region_data_parallel(
        self, 
        orgRegionId: str, 
        resource_types: List[str]
    ) -> Dict[str, Optional[List]]:
        """Get all org region data in a single unified query for specified resource types."""
        results = {'Document': [], 'Calendar': [], 'Stakeholder': [], 'Analysis': []}
        
        if not resource_types:
            return results
        
        try:
            # Execute single unified query to get all resource types at once
            unified_results = await self.get_org_region_resources_unified(orgRegionId)
            # Filter results based on requested resource types
            for resource_type in resource_types:
                if resource_type in unified_results:
                    results[resource_type] = unified_results[resource_type]
        except Exception as e:

            logger.error(f"Error fetching org region data: {e}")
            # Return empty results on error
            pass
        return results


    async def fetch_combine_embeddings(self, query_embedding, query_keywords, filters: dict, rep_chunks_ids: list, k=10, kd=15, K=30, kw=15, split_percentage=0.6):
        sql = text("""
            WITH event_matches AS (
                SELECT event_id AS id, text, metadata, 'events' AS type, 1 - (embedding <=> CAST(:query_embedding AS halfvec)) AS similarity
                FROM llm_embeddings WHERE event_id IS NOT NULL AND event_id = ANY(CAST(:event_filter_ids AS uuid[])) AND id <> ALL(CAST(:rep_chunks_ids AS uuid[]))
                ORDER BY embedding <=> CAST(:query_embedding AS halfvec)  LIMIT :k),
            stakeholder_matches AS (
                SELECT stakeholder_id AS id, text, metadata, 'stakeholder-maps' AS type, 1 - (embedding <=> CAST(:query_embedding AS halfvec)) AS similarity
                FROM llm_embeddings WHERE stakeholder_id IS NOT NULL AND stakeholder_id = ANY(CAST(:stakeholder_filter_ids AS uuid[])) AND id <> ALL(CAST(:rep_chunks_ids AS uuid[]))
                ORDER BY embedding <=> CAST(:query_embedding AS halfvec)  LIMIT :k),
            document_matches AS (
                SELECT document_id AS id, text, jsonb_build_object('heading', (metadata->'dl_meta'->'headings'->> 0)) AS metadata, 'documents' AS type, 1 - (embedding <=> CAST(:query_embedding AS halfvec)) AS similarity
                FROM llm_embeddings WHERE document_id IS NOT NULL AND document_id = ANY(CAST(:document_filter_ids AS uuid[])) AND id <> ALL(CAST(:rep_chunks_ids AS uuid[]))
                ORDER BY embedding <=> CAST(:query_embedding AS halfvec)  LIMIT :kd),
            analysis_matches AS (
                SELECT analysis_id AS id, text, metadata, 'analysis' AS type, 1 - (embedding <=> CAST(:query_embedding AS halfvec)) AS similarity
                FROM llm_embeddings WHERE analysis_id IS NOT NULL AND analysis_id = ANY(CAST(:analysis_filter_ids AS uuid[])) AND id <> ALL(CAST(:rep_chunks_ids AS uuid[]))
                ORDER BY embedding <=> CAST(:query_embedding AS halfvec)  LIMIT :kd)
            SELECT * FROM document_matches UNION ALL SELECT * FROM event_matches UNION ALL SELECT * FROM stakeholder_matches UNION ALL SELECT * FROM analysis_matches
            ORDER BY similarity DESC""")

        if 'Stakeholder' not in query_keywords:
            keyword_matches_sql = text("""
            WITH event_kw_matches AS (
                SELECT event_id AS id, text, metadata, 'events' AS type, ts_rank_cd(to_tsvector('simple',text), query) AS similarity FROM llm_embeddings, to_tsquery('simple',:query_keywords) AS query
                WHERE query @@ to_tsvector('simple',text)  
                AND event_id = ANY(CAST(:event_filter_ids AS uuid[]))
            ),    
            stakeholder_kw_matches AS (
                SELECT stakeholder_id AS id, text, metadata, 'stakeholder-maps' AS type, ts_rank_cd(to_tsvector('simple',text), query) AS similarity FROM llm_embeddings, to_tsquery('simple',:query_keywords) AS query
                WHERE query @@ to_tsvector('simple',text)  
                AND stakeholder_id = ANY(CAST(:stakeholder_filter_ids AS uuid[]))
            ),
            analysis_kw_matches AS (
                SELECT analysis_id AS id, text, metadata, 'analysis' AS type, ts_rank_cd(to_tsvector('simple',text), query) AS similarity FROM llm_embeddings, to_tsquery('simple',:query_keywords) AS query
                WHERE query @@ to_tsvector('simple',text)  
                AND analysis_id = ANY(CAST(:analysis_filter_ids AS uuid[]))
            ),
            document_kw_matches AS (
                SELECT document_id AS id, text, jsonb_build_object('heading', (metadata->'dl_meta'->'headings'->> 0)) AS metadata, 'documents' AS type, ts_rank_cd(to_tsvector('simple',text), query) AS similarity FROM llm_embeddings, to_tsquery('simple',:query_keywords) AS query
                WHERE query @@ to_tsvector('simple',text)  
                AND document_id = ANY(CAST(:document_filter_ids AS uuid[]))
            )
            SELECT * FROM analysis_kw_matches UNION ALL SELECT * FROM document_kw_matches UNION ALL SELECT * FROM stakeholder_kw_matches UNION ALL SELECT * FROM event_kw_matches
            ORDER BY similarity DESC LIMIT :kw; 
            """)
        else:
            keyword_matches_sql = text("""   
                SELECT stakeholder_id AS id,text,metadata,'stakeholder-maps' AS type,ts_rank_cd(to_tsvector('simple',text), query) AS similarity
                FROM llm_embeddings, to_tsquery('simple','Stakeholder') AS query
                WHERE query @@ to_tsvector('simple',text)  
                AND ( stakeholder_id = ANY(CAST(:stakeholder_filter_ids AS uuid[]))
                ) ORDER BY similarity DESC LIMIT 15;
            """)
            # AND id <> ALL(CAST(:rep_chunks_ids AS uuid[]))

        params = {"query_embedding": str(query_embedding), "event_filter_ids": filters.get("event"),"stakeholder_filter_ids": filters.get("stakeholder"),
        "document_filter_ids": filters.get("document"), "analysis_filter_ids": filters.get("analysis"),"rep_chunks_ids": rep_chunks_ids,"k": k, "kd": kd}
        embedding_task = asyncio.create_task(self._execute_query(sql, params))
        
        params.update({"query_keywords": query_keywords, "event_filter_ids": filters.get("event"), "stakeholder_filter_ids": filters.get("stakeholder"),
        "document_filter_ids": filters.get("document"), "analysis_filter_ids": filters.get("analysis"),"rep_chunks_ids": rep_chunks_ids, "kw": kw})               
        keyword_task = asyncio.create_task(self._execute_query(keyword_matches_sql, params))
        
        embedding_rows, keyword_rows = await asyncio.gather(embedding_task, keyword_task)
        
        df_embedding = pd.DataFrame(embedding_rows, columns=['id', 'text', 'metadata', 'type', 'similarity'])
        logger.info(f"Embedding: {len(df_embedding)}")

        df_keywords = pd.DataFrame(keyword_rows, columns=['id', 'text', 'metadata', 'type', 'similarity'])
        logger.info(f"keywords: {len(df_keywords)}")
        
        df_combined = None
        if 'stakeholder' not in query_keywords:
            df_combined = pd.concat([df_embedding, df_keywords])
        else:
            df_combined = df_keywords
        return df_combined

    async def adjacent_chunks_retrieval(self, chunk_ids, window_size=1):
        sql = sql = text("""
            WITH target_chunks AS (
            SELECT
                id AS seed_id, analysis_id, document_id,
                (metadata->>'chunk_index')::int AS seed_index
                FROM llm_embeddings
                WHERE id = ANY(CAST(:chunk_ids AS uuid[]))
            ),
            adjacent_chunks AS (
                SELECT
                    t.seed_id, c.id, c.text, c.metadata, c.analysis_id, c.document_id,
                    (c.metadata->>'chunk_index')::int AS chunk_index
                FROM target_chunks t
                JOIN llm_embeddings c
                ON (
                    (c.analysis_id = t.analysis_id AND t.analysis_id IS NOT NULL)
                    OR (c.document_id = t.document_id AND t.document_id IS NOT NULL))
                WHERE (c.metadata->>'chunk_index')::int
                    BETWEEN t.seed_index
                    AND t.seed_index + :window_size)
            SELECT *
            FROM adjacent_chunks
            ORDER BY seed_id, chunk_index;
            """)

        params = {
            "chunk_ids": chunk_ids if isinstance(chunk_ids, list) else [chunk_ids],
            "window_size": window_size}

        rows = await self._execute_query(sql, params)

        df = pd.DataFrame(
            rows, columns=["seed_id", "id", "text", "metadata", "analysis_id", "document_id", "chunk_index"])

        # Group ONLY chunks adjacent to EACH SEED (analysis)
        analysis_adjacent = (
            df[df["analysis_id"].notna()]
            .sort_values("chunk_index")
            .groupby(["seed_id", "analysis_id"])
            .agg({
                "text": lambda x: "\n".join(x),
                "chunk_index": list}).reset_index())
        document_adjacent = (
            df[df["document_id"].notna()]
            .sort_values("chunk_index")
            .groupby(["seed_id", "document_id"])
            .agg({
                "text": lambda x: "\n".join(x),
                "chunk_index": list
            })
            .reset_index()
        )
        return analysis_adjacent,  document_adjacent
        
    async def fetch_summary_data(self, query_text, filters: dict, limit=6):
        sql = text("""
            SELECT DISTINCT
                d.id AS document_id, d.name AS document_name, d.llm_summary AS document_summary,
                NULL::uuid AS analysis_id, NULL::text AS analysis_heading, NULL::text AS analysis_summary,
                NULL::uuid AS event_id, NULL::text AS event_title, NULL::text AS event_description,
                NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name, NULL::text AS stakeholder_bio
            FROM document d
            WHERE
                CAST(:query_text AS text[]) IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(d.name, kw) > 0.5
                )
                AND (
                    CAST(:document_filter_ids AS uuid[]) IS NOT NULL
                    AND d.id = ANY(CAST(:document_filter_ids AS uuid[]))) 
            UNION ALL      
            SELECT DISTINCT
                NULL::uuid AS document_id, NULL::text AS document_name, NULL::text AS document_summary,
                a.id AS analysis_id, a.heading AS analysis_heading, a.summary AS analysis_summary,
                NULL::uuid AS event_id, NULL::text AS event_title, NULL::text AS event_description,
                NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name, NULL::text AS stakeholder_bio
            FROM political_analysis a
            WHERE
                CAST(:query_text AS text[]) IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(a.heading, kw) > 0.5
                )
                AND (
                    CAST(:analysis_filter_ids AS uuid[]) IS NOT NULL
                    AND a.id = ANY(CAST(:analysis_filter_ids AS uuid[])))       
            UNION ALL        
            SELECT DISTINCT
                NULL::uuid AS document_id, NULL::text AS document_name, NULL::text AS document_summary,
                NULL::uuid AS analysis_id, NULL::text AS analysis_heading, NULL::text AS analysis_summary,
                e.id AS event_id, e.title AS event_title, e.description AS event_description,
                NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name, NULL::text AS stakeholder_bio
            FROM event e
            WHERE
                CAST(:query_text AS text[]) IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(e.title, kw) > 0.5)
                AND (
                    CAST(:event_filter_ids AS uuid[]) IS NOT NULL
                    AND e.id = ANY(CAST(:event_filter_ids AS uuid[])))
            UNION ALL
            SELECT DISTINCT
                NULL::uuid AS document_id, NULL::text AS document_name, NULL::text AS document_summary,
                NULL::uuid AS analysis_id, NULL::text AS analysis_heading, NULL::text AS analysis_summary,
                NULL::uuid AS event_id, NULL::text AS event_title, NULL::text AS event_description,
                s.id AS stakeholder_id, s.name AS stakeholder_name, s.bio AS stakeholder_bio
            FROM stakeholder s
            WHERE
                CAST(:query_text AS text[]) IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(s.name, kw) > 0.5)
                AND (
                    CAST(:stakeholder_filter_ids AS uuid[]) IS NOT NULL
                    AND s.id = ANY(CAST(:stakeholder_filter_ids AS uuid[])))
            LIMIT :limit;
            """)
        params = {"query_text": query_text, "event_filter_ids": filters.get("event"), "stakeholder_filter_ids": filters.get("stakeholder"),
        "document_filter_ids": filters.get("document"), "analysis_filter_ids": filters.get("analysis"), "limit": limit}
        rows = await self._execute_query(sql, params)
        return rows
    
    async def fetch_resource_data(self, keywords: dict, filters: dict, limit=20):
        sql = text("""
            (
            SELECT * FROM (
                SELECT *
                FROM (
                    SELECT DISTINCT
                        d.id AS document_id, d.name AS document_name,
                        NULL::uuid AS analysis_id, NULL::text AS analysis_heading,
                        NULL::uuid AS event_id, NULL::text AS event_title,
                        NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name,
                        GREATEST((
                            SELECT MAX(similarity(d.name, kw))
                            FROM unnest(CAST(:document_keywords AS text[])) kw
                        ), 0) AS score
                    FROM document d
                    WHERE
                    ((:document_keywords IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM unnest(CAST(:document_keywords AS text[])) kw
                            WHERE similarity(d.name, kw) > 0.17))
                        OR
                        (:document_keywords IS NULL
                        AND CAST(:document_filter_ids AS uuid[]) IS NOT NULL))
                    AND (
                        :document_filter_ids IS NOT NULL
                        AND d.id = ANY(CAST(:document_filter_ids AS uuid[])))
                ) ranked_docs
                ORDER BY score DESC
                LIMIT :k
            ) doc_results
            )
            UNION ALL
            (
            SELECT * FROM (
                SELECT *
                FROM (
                    SELECT DISTINCT
                        NULL::uuid AS document_id, NULL::text AS document_name,
                        a.id AS analysis_id , a.heading AS analysis_heading,
                        NULL::uuid AS event_id, NULL::text AS event_title,
                        NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name,
                        GREATEST((
                            SELECT MAX(similarity(a.heading, kw))
                            FROM unnest(CAST(:analysis_keywords AS text[])) kw
                        ), 0) AS score
                    FROM political_analysis a
                    WHERE
                    ((:analysis_keywords IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM unnest(CAST(:analysis_keywords AS text[])) kw
                            WHERE similarity(a.heading, kw) > 0.17))
                        OR
                        (:analysis_keywords IS NULL
                        AND CAST(:analysis_filter_ids AS uuid[]) IS NOT NULL))
                    AND (
                        :analysis_filter_ids IS NOT NULL
                        AND a.id = ANY(CAST(:analysis_filter_ids AS uuid[])))
                ) ranked_analysis
                ORDER BY score DESC
                LIMIT :k
            ) analysis_results
            )
            UNION ALL
            (
            SELECT * FROM (
                SELECT *
                FROM (
                    SELECT DISTINCT
                        NULL::uuid AS document_id, NULL::text AS document_name,
                        NULL::uuid AS analysis_id, NULL::text AS analysis_heading,
                        e.id AS event_id, e.title AS event_title,
                        NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name,
                        GREATEST((
                            SELECT MAX(similarity(e.title, kw))
                            FROM unnest(CAST(:event_keywords AS text[])) kw
                        ), 0) AS score
                    FROM event e
                    WHERE
                    ((:event_keywords IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM unnest(CAST(:event_keywords AS text[])) kw
                            WHERE similarity(e.title, kw) > 0.2))
                        OR
                        (:event_keywords IS NULL
                        AND CAST(:event_filter_ids AS uuid[]) IS NOT NULL))
                    AND (
                        :event_filter_ids IS NOT NULL
                        AND e.id = ANY(CAST(:event_filter_ids AS uuid[])))
                ) ranked_events
                ORDER BY score DESC
                LIMIT :k
            ) event_results
            )
            UNION ALL
            (
            SELECT * FROM (
                SELECT *
                FROM (
                    SELECT DISTINCT
                        NULL::uuid AS document_id, NULL::text AS document_name,
                        NULL::uuid AS analysis_id, NULL::text AS analysis_heading,
                        NULL::uuid AS event_id, NULL::text AS event_title,
                        s.id AS stakeholder_id, s.name AS stakeholder_name,
                        GREATEST((
                            SELECT MAX(similarity(s.name, kw))
                            FROM unnest(CAST(:stakeholder_keywords AS text[])) kw
                        ), 0) AS score
                    FROM stakeholder s
                    WHERE
                    ((:stakeholder_keywords IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM unnest(CAST(:stakeholder_keywords AS text[])) kw
                            WHERE similarity(s.name, kw) > 0.2))
                        OR
                        (:stakeholder_keywords IS NULL
                        AND CAST(:stakeholder_filter_ids AS uuid[]) IS NOT NULL))
                    AND (
                        :stakeholder_filter_ids IS NOT NULL
                        AND s.id = ANY(CAST(:stakeholder_filter_ids AS uuid[])))
                ) ranked_stakeholders
                ORDER BY score DESC
                LIMIT :k
            ) stakeholder_results
            )
            """)
        params = {"k": limit,
        "stakeholder_filter_ids": filters.get("stakeholder"), "analysis_filter_ids": filters.get("analysis"),
        "event_filter_ids": filters.get("event"), "document_filter_ids": filters.get("document"),
    
        "stakeholder_keywords": keywords.get("stakeholder"), "analysis_keywords": keywords.get("analysis"),
        "event_keywords": keywords.get("event"), "document_keywords": keywords.get("document"),}
        rows = await self._execute_query(sql, params)

        return rows


    
    async def get_matched_entities(self, relationships: List[Dict], access_data_ids: List[str]) -> Dict:
        # logger.info(f"get_matched_entities: {relationships}")
        for relationship in relationships:
            source_entity = relationship.get("source_entity")
            target_entity = relationship.get("target_entity")
            relationship_type = relationship.get("relationship_type")
            # logger.info(f"source_entity, target_entity, relationship_type: {source_entity} {target_entity} {relationship_type}")

            query = text("""
               SELECT id, entity_text, entity_type, similarity(entity_text, :name) as entity_similarity
                    FROM public.kgg_entities
                    WHERE entity_type = :type
                    AND similarity(entity_text, :name) > 0.7
                    ORDER BY entity_similarity DESC;""")
                    # AND source_article_id = ANY(CAST(:access_data_ids AS uuid[]))

                                      
            rows = await self._execute_query(query, {"name": source_entity.get("name"), "type": source_entity.get("type"), "access_data_ids": access_data_ids})
            # logger.info(f'rows: {rows}')
            # print(f"get_matched_entities result: {len(rows)} for source entity: {source_entity.get("name")} and its type is: {source_entity.get("type")}")
                
            if len(rows) > 0:
                source_entity.update({"id": rows[0][0], "entity_text": rows[0][1]}) 
                # logger.info(f"source_entity: {source_entity}")
                relationship.update({"source_entity": source_entity})
            else:
                relationship.update({"source_entity": None})
                # logger.info(f"source_entity is not found: {source_entity}")
        return relationships    
    

    async def get_all_relationships(self, relationships: List[Dict], access_data_ids: List[str]) -> List[Dict]:
        """
        Unified 1-hop KG traversal:
        - Handles 1, 2, or 3+ entities
        - Supports query-extracted source_type, target_type
        - Supports relation filter
        """
        all_relationships = []
        for relationship in relationships:

            source_entity = relationship.get("source_entity")
            target_entity= relationship.get("target_entity")
            relationship_type = relationship.get("relationship_type")
            # logger.info(f"source_entity, target_entity, relationship_type: {source_entity} {target_entity} {relationship_type}")
            query = """SELECT s.id AS source_id,
                            s.entity_text AS source_name,
                            s.entity_type AS source_type,
                            r.relationship_type,
                            t.id AS target_id, t.entity_text AS target_name,
                            t.entity_type AS target_type,
                            r.source_article_id,
                            r.source_table
                    FROM kgg_relationships r
                    JOIN kgg_entities s ON r.source_entity_id = s.id
                    JOIN kgg_entities t ON r.target_entity_id = t.id
                    Where r.source_entity_id = :source_entity_id
                    AND r.source_article_id = ANY(CAST(:access_data_ids AS uuid[]))"""


            if source_entity is not None:
                if target_entity is not None:
                    query += " AND t.entity_type = :target_type limit 100"
                    params = {"source_entity_id": source_entity.get("id"), "target_type": target_entity.get("type"), "access_data_ids": access_data_ids}
                    sql_query = text(query)
                    rows = await self._execute_query(sql_query , params)
                else:
                    query += " AND s.entity_type = :source_type limit 100"
                    params = {"source_entity_id": source_entity.get("id"), "source_type": source_entity.get("type"), "access_data_ids": access_data_ids}
                    sql_query = text(query)
                    rows = await self._execute_query(sql_query, params)
                
                for row in rows:
                    new_relationship = {
                        "source_entity": {
                            "id": row.source_id,
                            "entity_text": row.source_name,
                            "entity_type": row.source_type
                        },
                        "target_entity": {
                            "id": row.target_id,
                            "entity_text": row.target_name,
                            "entity_type": row.target_type
                        },
                        "relationship_type": row.relationship_type
                    }
                    all_relationships.append(new_relationship)   
                    
        logger.info(f"Total relationships: {len(all_relationships)}")
        return all_relationships

db = DB()










        # analysis_adjacent = (
        #     df[df["analysis_id"].notna()]
        #     .sort_values(["seed_id", "chunk_index"])
        #     .reset_index(drop=True)
        # )[["seed_id", "analysis_id", "text", "chunk_index"]]

        # # Adjacent chunks related to document (KEEP ALL ROWS)
        # document_adjacent = (
        #     df[df["document_id"].notna()]
        #     .sort_values(["seed_id", "chunk_index"])
        #     .reset_index(drop=True)
        # )[["seed_id", "document_id", "text", "chunk_index"]]
        # print(analysis_adjacent.columns)
        # print(document_adjacent.columns)
        # return analysis_adjacent,  document_adjacent
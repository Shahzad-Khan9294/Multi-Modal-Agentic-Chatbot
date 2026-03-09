import os
import asyncio
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Optional, Any
import json
from dotenv import load_dotenv
from .database import async_session_maker

load_dotenv('.env.client')


class DB:
    """
    Async database operations class.
    All methods use async database sessions for non-blocking I/O.
    """
    async def get_document_chunks(self, session: AsyncSession, doc_id):
        query = text("""SELECT text FROM llm_embeddings WHERE document_id = :doc_id""")
        result = await session.execute(query, {"doc_id": doc_id})
        rows = result.mappings().all()
        chunks = [{"text": row["text"]} for row in rows]
        return chunks
        
    async def _execute_query(self, session: AsyncSession, query: str, params: Optional[Dict] = None) -> List:
        """Execute a query and return results as a list of rows."""
        result = await session.execute(text(query), params or {})
        return result.fetchall()
    
    async def _execute_query_to_df(self, session: AsyncSession, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame."""
        rows = await self._execute_query(session, query, params)
        if not rows:
            return pd.DataFrame()
        # Convert rows to list of dicts
        data = [dict(row._mapping) for row in rows]
        return pd.DataFrame(data)
    
    async def get_document_by_ids(self, session: AsyncSession, document_ids: List[str]) -> pd.DataFrame:
        """Get documents by their IDs."""
        if not document_ids:
            return pd.DataFrame()
        in_clause = str(tuple(document_ids)).replace(',)',')')
        query = f"""
        SELECT *
        FROM document
        WHERE id in {in_clause}
        """
        return await self._execute_query_to_df(session, query)

    async def get_org_region_document(self, session: AsyncSession, orgRegionId: str) -> List[str]:
        """Get document IDs accessible by org region."""
        query = text("""
            SELECT resource_id as id 
            FROM public.document_access
            WHERE org_region_access_id IS NULL 
            OR org_region_access_id = :org_region_id
        """)
        result = await session.execute(query, {"org_region_id": orgRegionId})
        rows = result.fetchall()
        return [str(row.id) for row in rows]

    async def get_org_region_event(self, session: AsyncSession, orgRegionId: str) -> List[str]:
        """Get event IDs accessible by org region."""
        query = text("""
            SELECT resource_id as id 
            FROM public.event_access
            WHERE org_region_access_id IS NULL 
            OR org_region_access_id = :org_region_id
        """)
        result = await session.execute(query, {"org_region_id": orgRegionId})
        rows = result.fetchall()
        return [str(row.id) for row in rows]

    async def get_org_region_stakeholder(self, session: AsyncSession, orgRegionId: str) -> List[str]:
        """Get stakeholder IDs accessible by org region."""
        query = text("""
            SELECT DISTINCT stakeholder_id as id 
            FROM stakeholder_to_group
            WHERE deleted_at IS NULL 
            AND stakeholder_group_id IN (
                SELECT resource_id as id 
                FROM public.stakeholder_group_access
                WHERE org_region_access_id IS NULL 
                OR org_region_access_id = :org_region_id
            )
        """)
        result = await session.execute(query, {"org_region_id": orgRegionId})
        rows = result.fetchall()
        return [str(row.id) for row in rows]
    
    async def get_org_region_analysis(self, session: AsyncSession, orgRegionId: str) -> List[str]:
        """Get analysis IDs accessible by org region."""
        query = text("""
            SELECT resource_id as id 
            FROM public.political_analysis_access
            WHERE org_region_access_id IS NULL 
            OR org_region_access_id = :org_region_id
        """)
        result = await session.execute(query, {"org_region_id": orgRegionId})
        rows = result.fetchall()
        return [str(row.id) for row in rows]

    async def get_org_region_client(self, session: AsyncSession, orgRegionId: str) -> List[Dict]:
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
        result = await session.execute(query, {"org_region_id": orgRegionId})
        rows = result.fetchall()
        return [{"first_name": row.first_name, "last_name": row.last_name} for row in rows]

    async def get_all_org_region_data_parallel(
        self, 
        session: AsyncSession,
        orgRegionId: str, 
        resource_types: List[str]
    ) -> Dict[str, Optional[List]]:
        """Get all org region data in parallel for specified resource types."""
        results = {'Document': [], 'Calendar': [], 'Stakeholder': [], 'Analysis': []}
        func_map = {
            'Document': self.get_org_region_document,
            'Calendar': self.get_org_region_event,
            'Stakeholder': self.get_org_region_stakeholder,
            "Analysis": self.get_org_region_analysis
        }
        
        if not resource_types:
            return results
                # Execute queries sequentially (AsyncSession doesn't support concurrent operations)
        for key, func in func_map.items():
            if key in resource_types:
                try:
                    results[key] = await func(session, orgRegionId)
                except Exception as e:
                    print(f"Error fetching {key} data: {e}")
                    results[key] = []
        
        return results

    async def get_embeddings_document(self, session: AsyncSession, document_ids: List[str]) -> List[Dict]:
        """Get documents for embedding processing."""
        if not document_ids:
            return []
        in_clause = str(tuple(document_ids)).replace(',)',')')
        query = text(f"""
        SELECT id as document_id, path
        FROM document
        WHERE id IN {in_clause}
        """)
        result = await session.execute(query)
        rows = result.fetchall()
        documents = [{"document_id": str(row.document_id), "path": row.path} for row in rows]
        print(f"Documents to be processed (new or updated): {len(documents)}")
        print(documents)
        document_ids_list = [str(row.document_id) for row in rows]
        deleted_count = await self.delete_embeddings_document(session, document_ids_list)
        print(f"Deleted {deleted_count} embeddings for document IDs: {document_ids_list}")
        return documents

    async def get_text_from_document_embedding(self, session: AsyncSession, doc_ids: List[str]) -> pd.DataFrame:
        """Get text from document embeddings."""
        if not doc_ids:
            return pd.DataFrame()
        in_clause = str(tuple(doc_ids)).replace(',)',')')
        query = text(f"""
        SELECT document_id, text
        FROM llm_embeddings
        WHERE document_id in {in_clause}
        """)
        return await self._execute_query_to_df(session, query)

    async def get_text_from_stakeholder_embedding(self, session: AsyncSession, doc_ids: List[str]) -> pd.DataFrame:
        """Get text from stakeholder embeddings."""
        if not doc_ids:
            return pd.DataFrame()
        in_clause = str(tuple(doc_ids)).replace(',)',')')
        query = text(f"""
        SELECT stakeholder_id, text
        FROM llm_embeddings
        WHERE stakeholder_id in {in_clause}
        """)
        return await self._execute_query_to_df(session, query)
    
    async def check_document_embedding_exist(self, session: AsyncSession, doc_ids: List[str]) -> List[str]:
        """Check which document embeddings exist."""
        if not doc_ids:
            return []
        doc_ids_str = str(tuple(doc_ids)).replace(',)',')')
        query = text(f"SELECT DISTINCT(document_id) FROM llm_embeddings WHERE document_id in {doc_ids_str}")
        result = await session.execute(query)
        rows = result.fetchall()
        return [str(row.document_id) for row in rows]

    async def delete_embeddings_document(self, session: AsyncSession, doc_ids: List[str]) -> int:
        """Delete embeddings for given document IDs."""
        if not doc_ids:
            return 0
        doc_ids_str = str(tuple(doc_ids)).replace(',)',')')
        query = text(f"DELETE FROM llm_embeddings WHERE document_id in {doc_ids_str}")
        result = await session.execute(query)
        await session.commit()
        deleted_count = result.rowcount
        print(f"Deleted {deleted_count} embeddings for document IDs: {doc_ids_str}")
        return deleted_count

    async def add_document_embeddings(self, session: AsyncSession, chunks_data: List[Dict]) -> bool:
        """Add document embeddings to the database."""
        try:
            if not chunks_data:
                return False
            inserted_count = 0
            sql_query = text("INSERT INTO llm_embeddings (text, embedding, metadata, document_id) VALUES (:text, :embedding, :metadata, :document_id)" )
            # add all chunks list of dictionaries to the database in a single transaction
            async with session.begin():
                for chunk in chunks_data:
                    result = await session.execute(sql_query, {
                        "text": chunk['text'],
                        "embedding": str(chunk['embedding']),
                        "metadata": chunk['metadata'],
                        "document_id": chunk['document_id']
                    })
                    inserted_count += result.rowcount
            if inserted_count == len(chunks_data):
                print(f"Inserted {inserted_count} embeddings for document IDs: {chunks_data[0]['document_id']}")
                return True
            else:
                print(f"Failed to insert {inserted_count} embeddings for document IDs: {chunks_data[0]['document_id']}")
                return False
        except Exception as e:
            print(f"Error adding document embeddings: {e}")
            return False
    
    async def get_embeddings_event(self, session: AsyncSession, event_ids: List[str]) -> List[Dict]:
        """Get events for embedding processing."""
        if not event_ids:
            return []
        in_clause = str(tuple(event_ids)).replace(',)',')')
        query = text(f"""
        SELECT id as event_id, title, description, start_datetime, end_datetime, tag_note
        FROM event
        WHERE id IN {in_clause}
        """)
        result = await session.execute(query)
        rows = result.fetchall()
        events = [{
            "event_id": str(row.event_id),
            "title": row.title,
            "description": row.description,
            "start_datetime": row.start_datetime,
            "end_datetime": row.end_datetime,
            "tag_note": row.tag_note
        } for row in rows]
        print(f"Events to be processed (new or updated): {len(events)}")
        print(events)
        event_ids_list = [str(row.event_id) for row in rows]
        deleted_count = await self.delete_embeddings_event(session, event_ids_list)
        print(f"Deleted {deleted_count} embeddings for event IDs: {event_ids_list}")
        return events

    async def delete_embeddings_event(self, session: AsyncSession, event_ids: List[str]) -> int:
        """Delete embeddings for given event IDs."""
        if not event_ids:
            return 0
        event_ids_str = str(tuple(event_ids)).replace(',)',')')
        query = text(f"DELETE FROM llm_embeddings WHERE event_id in {event_ids_str}")
        result = await session.execute(query)
        await session.commit()
        deleted_count = result.rowcount
        print(f"Deleted {deleted_count} embeddings for event IDs: {event_ids_str}")
        return deleted_count

    async def add_event_embeddings(self, session: AsyncSession, chunks_data: List[Dict]) -> bool:
        """Add event embeddings to the database."""
        try:
            if not chunks_data:
                return False
            print(f"Embeddings: {len(chunks_data)}")
            sql_query = text("INSERT INTO llm_embeddings (text, embedding, metadata, event_id) VALUES (:text, :embedding, :metadata, :event_id)" )
            inserted_count = 0
            async with session.begin():
                for chunk in chunks_data:
                    result = await session.execute(sql_query, {
                        "text": chunk['text'],
                        "embedding": str(chunk['embedding']),
                        "metadata": chunk['metadata'],
                        "event_id": chunk['event_id']
                    })
                    inserted_count += result.rowcount
            if inserted_count == len(chunks_data):
                print(f"Inserted {inserted_count} embeddings for event IDs: {chunks_data[0]['event_id']}")
                return True
            else:
                print(f"Failed to insert {inserted_count} embeddings for event IDs: {chunks_data[0]['event_id']}")
                return False
        except Exception as e:
            print(f"Error adding event embeddings: {e}")
            return False
    
    async def get_embeddings_stakeholder(self, session: AsyncSession, stakeholder_ids: List[str]) -> List[Dict]:
        """Get stakeholders for embedding processing."""
        if not stakeholder_ids:
            return []
        in_clause = str(tuple(stakeholder_ids)).replace(',)',')')
        query = text(f"""
        SELECT id as stakeholder_id, name, org, bio
        FROM stakeholder 
        WHERE id IN {in_clause}
        """)
        result = await session.execute(query)
        rows = result.fetchall()
        stakeholders = [{
            "stakeholder_id": str(row.stakeholder_id),
            "name": row.name,
            "org": row.org,
            "bio": row.bio
        } for row in rows]
        print(f"Stakeholders to be processed (New or Updated): {len(stakeholders)}")
        print(stakeholders)
        stakeholder_ids_list = [str(row.stakeholder_id) for row in rows]
        deleted_count = await self.delete_embeddings_stakeholder(session, stakeholder_ids_list)
        print(f'Deleted {deleted_count} embeddings for stakeholder IDs: {stakeholder_ids_list}')
        return stakeholders

    async def delete_embeddings_stakeholder(self, session: AsyncSession, stakeholder_ids: List[str]) -> int:   
        """Delete embeddings for given stakeholder IDs."""
        if not stakeholder_ids:
            return 0
        # Build parameter placeholders
        placeholders = ", ".join([f":id{i}" for i in range(len(stakeholder_ids))])
        params = {f"id{i}": stakeholder_ids[i] for i in range(len(stakeholder_ids))}
        query = text(f"""
            DELETE FROM llm_embeddings
            WHERE stakeholder_id IN ({placeholders})
        """)
        result = await session.execute(query, params)
        await session.commit()
        deleted_count = result.rowcount
        print("Deleted number of rows:", deleted_count)
        return deleted_count

    async def add_stakeholder_embeddings(self, session: AsyncSession, chunks_data: List[Dict]) -> bool:
        """Add stakeholder embeddings to the database."""
        try:
            if not chunks_data:
                return False
            print('Embeddings', len(chunks_data))
            sql_query = text("INSERT INTO llm_embeddings (text, embedding, metadata, stakeholder_id) VALUES (:text, :embedding, :metadata, :stakeholder_id)" )
            inserted_count = 0
            async with session.begin():
                for chunk in chunks_data:
                    result = await session.execute(sql_query, {
                        "text": chunk['text'],
                        "embedding": str(chunk['embedding']),
                        "metadata": chunk['metadata'],
                        "stakeholder_id": chunk['stakeholder_id']
                    })
                    inserted_count += result.rowcount
            if inserted_count == len(chunks_data):
                print(f"Inserted {inserted_count} embeddings for stakeholder IDs: {chunks_data[0]['stakeholder_id']}")
                return True
            else:
                print(f"Failed to insert {inserted_count} embeddings for stakeholder IDs: {chunks_data[0]['stakeholder_id']}")
                return False
        except Exception as e:
            print(f"Error adding stakeholder embeddings: {e}")
            return False

    async def get_embeddings_analysis(self, session: AsyncSession, analysis_ids: List[str]) -> List[Dict]:
        """Get analysis for embedding processing."""
        if not analysis_ids:
            return []
        in_clause = str(tuple(analysis_ids)).replace(',)',')')
        query = text(f"""
        SELECT id as analysis_id, heading, body
        FROM political_analysis
        WHERE id IN {in_clause}
        """)
        result = await session.execute(query)
        rows = result.fetchall()
        analyses = [{
            "analysis_id": str(row.analysis_id),
            "heading": row.heading,
            "body": row.body
        } for row in rows]
        print(f"Analysis to be processed (new or updated): {len(analyses)}")
        analysis_ids_list = [str(row.analysis_id) for row in rows]
        deleted_count = await self.delete_embeddings_analysis(session, analysis_ids_list)
        print(f"Deleted {deleted_count} embeddings for analysis IDs: {analysis_ids_list}")
        return analyses

    async def delete_embeddings_analysis(self, session: AsyncSession, analysis_ids: List[str]) -> int:
        """Delete embeddings for given analysis IDs."""
        if not analysis_ids:
            return 0
        analysis_ids_str = str(tuple(analysis_ids)).replace(',)',')')
        query = text(f"DELETE FROM llm_embeddings WHERE analysis_id in {analysis_ids_str}")
        result = await session.execute(query)
        await session.commit()
        deleted_count = result.rowcount
        print(f"Deleted {deleted_count} embeddings for analysis IDs: {analysis_ids_str}")
        return deleted_count

    async def add_analysis_embeddings(self, session: AsyncSession, chunks_data: List[Dict]) -> bool:
        """Add analysis embeddings to the database."""
        try:
            if not chunks_data:
                return False
            print(f"Embeddings: {len(chunks_data)}")
            
            sql_query = text("INSERT INTO llm_embeddings (text, embedding, metadata, analysis_id) VALUES (:text, :embedding, :metadata, :analysis_id)" )
            inserted_count = 0
            async with session.begin():
                for chunk in chunks_data:
                    result = await session.execute(sql_query, {
                        "text": chunk['text'],
                        "embedding": str(chunk['embedding']),
                        "metadata": chunk['metadata'],
                        "analysis_id": chunk['analysis_id']
                    })
                    inserted_count += result.rowcount
            if inserted_count == len(chunks_data):
                print(f"Inserted {inserted_count} embeddings for analysis IDs: {chunks_data[0]['analysis_id']}")
                return True
            else:
                print(f"Failed to insert {inserted_count} embeddings for analysis IDs: {chunks_data[0]['analysis_id']}")
                return False
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error adding analysis embeddings: {e}")
            return False

    async def add_document_summaries(self, session: AsyncSession, doc_id: str, summary: str) -> bool:
        """Add document summaries."""
        if not summary:
            return False
        print("Summary Length: ", len(summary))
        query = text("""
                    UPDATE document
                    SET llm_summary = :summary 
                    WHERE id = :doc_id
                """)
        await session.execute(query, {"summary": summary, "doc_id": doc_id})
        await session.commit()
        return True
    
    async def fetch_summary_data(self, session: AsyncSession, query_text: List[str], filters: Dict, limit: int = 10) -> List:
        """Fetch summary data based on query text and filters."""
        sql = text("""
            SELECT DISTINCT
                d.id AS document_id, d.name AS document_name, d.llm_summary AS document_summary,
                NULL::uuid AS analysis_id, NULL::text AS analysis_heading, NULL::text AS analysis_summary,
                NULL::uuid AS event_id, NULL::text AS event_title, NULL::text AS event_description,
                NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name, NULL::text AS stakeholder_bio
            FROM document d
            WHERE
                :query_text IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(d.name, kw) > 0.35
                )
                AND (
                    :document_filter_ids IS NOT NULL
                    AND d.id = ANY(CAST(:document_filter_ids AS uuid[]))) 
            UNION ALL      
            SELECT DISTINCT
                NULL::uuid AS document_id, NULL::text AS document_name, NULL::text AS document_summary,
                a.id AS analysis_id, a.heading AS analysis_heading, a.summary AS analysis_summary,
                NULL::uuid AS event_id, NULL::text AS event_title, NULL::text AS event_description,
                NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name, NULL::text AS stakeholder_bio
            FROM political_analysis a
            WHERE
                :query_text IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(a.heading, kw) > 0.35
                )
                AND (
                    :analysis_filter_ids IS NOT NULL
                    AND a.id = ANY(CAST(:analysis_filter_ids AS uuid[])))       
            UNION ALL        
            SELECT DISTINCT
                NULL::uuid AS document_id, NULL::text AS document_name, NULL::text AS document_summary,
                NULL::uuid AS analysis_id, NULL::text AS analysis_heading, NULL::text AS analysis_summary,
                e.id AS event_id, e.title AS event_title, e.description AS event_description,
                NULL::uuid AS stakeholder_id, NULL::text AS stakeholder_name, NULL::text AS stakeholder_bio
            FROM event e
            WHERE
                :query_text IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(e.title, kw) > 0.35)
                AND (
                    :event_filter_ids IS NOT NULL
                    AND e.id = ANY(CAST(:event_filter_ids AS uuid[])))
            UNION ALL
            SELECT DISTINCT
                NULL::uuid AS document_id, NULL::text AS document_name, NULL::text AS document_summary,
                NULL::uuid AS analysis_id, NULL::text AS analysis_heading, NULL::text AS analysis_summary,
                NULL::uuid AS event_id, NULL::text AS event_title, NULL::text AS event_description,
                s.id AS stakeholder_id, s.name AS stakeholder_name, s.bio AS stakeholder_bio
            FROM stakeholder s
            WHERE
                :query_text IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:query_text AS text[])) kw
                    WHERE similarity(s.name, kw) > 0.35)
                AND (
                    :stakeholder_filter_ids IS NOT NULL
                    AND s.id = ANY(CAST(:stakeholder_filter_ids AS uuid[])))
            LIMIT :limit;
        """)
        params = {
            "query_text": query_text, 
            "event_filter_ids": filters.get("event"), 
            "stakeholder_filter_ids": filters.get("stakeholder"),
            "document_filter_ids": filters.get("document"), 
            "analysis_filter_ids": filters.get("analysis"), 
            "limit": limit
        }
        result = await session.execute(sql, params)
        return result.fetchall()


# Create a singleton instance for backward compatibility
db = DB()

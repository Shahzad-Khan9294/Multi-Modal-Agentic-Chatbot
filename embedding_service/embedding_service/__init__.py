import os
import traceback
import logging
import warnings
import uuid
from uuid import uuid4
from typing import Optional, Any

from fastapi import APIRouter, HTTPException, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import FastAPI, Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi import Security

from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig


from .schema import EmbedInput, SummaryInput
from .utils import process_document_embed, process_analysis_embed, process_stakeholder_embed, process_event_embed, process_summary_document
from .db import db
from .database import get_async_session

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/embedding',tags=['embedding'])

# --- Constants ---
API_SECRET_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

# --- Security Scheme (for docs) ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --- Dependency to Check API Key ---
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return api_key


@router.post("/process_document")
async def embed(input: EmbedInput, session: AsyncSession = Depends(get_async_session), api_key: APIKey = Depends(verify_api_key)) -> dict[str, Any]:
    resources = input.resources
    resource_type = input.resource_type
    processed_resources = await process_document_embed(session, resources)
    return {"processed_resources": processed_resources}


@router.post("/process_analysis")
async def embed(input: EmbedInput, session: AsyncSession = Depends(get_async_session), api_key: APIKey = Depends(verify_api_key)) -> dict[str, Any]:
    resources = input.resources
    resource_type = input.resource_type
    processed_resources = await process_analysis_embed(session, resources)
    return {"processed_resources": processed_resources}


@router.post("/process_stakeholder")
async def embed(input: EmbedInput, session: AsyncSession = Depends(get_async_session), api_key: APIKey = Depends(verify_api_key)) -> dict[str, Any]:
    resources = input.resources
    resource_type = input.resource_type
    processed_resources = await process_stakeholder_embed(session, resources)
    return {"processed_resources": processed_resources}

@router.post("/process_event")
async def embed(input: EmbedInput, session: AsyncSession = Depends(get_async_session), api_key: APIKey = Depends(verify_api_key)) -> dict[str, Any]:
    resources = input.resources
    resource_type = input.resource_type
    processed_resources = await process_event_embed(session, resources)
    return {"processed_resources": processed_resources}

@router.post("/process_summary")
async def generate_summary(input: SummaryInput, session: AsyncSession = Depends(get_async_session), api_key: APIKey = Depends(verify_api_key)) -> dict[str, Any]:
    doc_id = input.doc_id
    processed_summary = await process_summary_document(session, doc_id)
    return {"doc_id": doc_id, "processed_summary": processed_summary}
import os
import re
import json
import traceback
import logging
import warnings
import uuid
from uuid import uuid4
from typing import Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_async_session
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse 
from fastapi import FastAPI, Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi import Security

from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig

from .workflow.utils import langchain_to_chat_message

from .workflow.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    UserInput,
)


warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/chatbot',tags=['ai_agent'])


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


async def parse_stream_input(user_input: UserInput, session: AsyncSession) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    logger.info(f"user_input.org_region_id: {user_input.org_region_id}")
    # documents = db.get_org_region_document(user_input.org_region_id)
    # events = db.get_org_region_event(user_input.org_region_id)
    logger.info(f"user_input.resource_type: {user_input.resource_type}")
    resource_type = user_input.resource_type if user_input.resource_type else []
    query_data = {
        
        "input": {"question": user_input.message, "resource_type": user_input.resource_type},
        "config": RunnableConfig(
            configurable={
            "thread_id": thread_id,
            "recursion_limit":1,
            'org_region_id': user_input.org_region_id,
            'session': session,
            'stream_mode': "messages"
            },
            run_id=run_id
        ),
    }
    return query_data, run_id

async def parse_input(user_input: UserInput, session: AsyncSession) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    logger.info(f"user_input.org_region_id: {user_input.org_region_id}")
    # documents = db.get_org_region_document(user_input.org_region_id)
    # events = db.get_org_region_event(user_input.org_region_id)
    logger.info(f"user_input.resource_type: {user_input.resource_type}")
    resource_type = user_input.resource_type if user_input.resource_type else []
    query_data = {
        
        "input": {"question": user_input.message, "resource_type": user_input.resource_type},
        "config": RunnableConfig(
            configurable={
            "thread_id": thread_id,
            "recursion_limit":1,
            'org_region_id': user_input.org_region_id,
            'session': session,
            },
            run_id=run_id
        ),
    }
    return query_data, run_id

async def ainvoke(user_input: UserInput, request: Request, session: AsyncSession) -> ChatMessage:
    try:
        
        agent = request.app.state.chatbot
        if not agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")

        query_data, run_id = await parse_input(user_input, session)
        response = await agent.ainvoke(**query_data)
        if 'documents' in response:
            documents = response["documents"]
        else:
            documents = []
        output = langchain_to_chat_message(response["messages"][-1], documents=documents)
        output.run_id = str(run_id)
        # print(f"output: {output}")
        return output
            
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

@router.post("/chat")
async def invoke(user_input: UserInput,request: Request, api_key: APIKey = Depends(verify_api_key), session: AsyncSession = Depends(get_async_session)) -> ChatMessage:
    """
    Invoke the default agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    return await ainvoke(user_input=user_input,request=request, session=session)

@router.post("/chat_stream")
async def astream(
    user_input: UserInput,
    request: Request,
    api_key: APIKey = Depends(verify_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> StreamingResponse:
    """
    Stream agent response as Server-Sent Events (SSE).
    Each event is a JSON object: {"content": "<token or chunk>", "run_id": "<uuid>"}.
    """
    agent = request.app.state.chatbot
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    query_data, run_id = await parse_stream_input(user_input, session)
    run_id_str = str(run_id)

    async def stream():
        try:
            async for msg, metadata in agent.astream(
                **query_data,
                stream_mode="messages",
            ):
                if getattr(msg, "type", None) == "human":
                    continue
                
                if metadata.get("langgraph_node") not in ["resource_list","risk_category_search","db_search"]:
                    meta = metadata or {}
                    chat_msg = langchain_to_chat_message(msg)
                    content = getattr(chat_msg, "content", None) or ""
                    payload = {"type": "ai", "content": content, "run_id": run_id_str, "node": meta.get("langgraph_node")}
                    logger.info(f"payload: {payload}")    
                    yield f"data: {json.dumps(payload)}\n\n"

            if metadata.get("langgraph_node") in ["resource_list","risk_category_search","db_search"]:
                for chunk in msg.content:
                    meta= metadata or {}                
                    payload = {"type": "ai", "content": chunk, "run_id": run_id_str, "node": meta.get("langgraph_node")}
                    yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            logger.error(traceback.format_exc())
            error_payload = {"error": "Unexpected error during streaming", "detail": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"
            
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/history")
async def history(input: ChatHistoryInput, request: Request, api_key: APIKey = Depends(verify_api_key)) -> ChatHistory:
    """
    Get chat history.
    """
    try:
        agent = request.app.state.sharepoint_agent
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")
    
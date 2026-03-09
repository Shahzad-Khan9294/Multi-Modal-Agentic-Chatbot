import os
import pandas as pd
from .schema import ChatMessage
from typing import List, Optional
from langchain.schema import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

#from langchain_nvidia_ai_endpoints import ChatNVIDIA


from langchain_openai import ChatOpenAI
from langchain_postgres import PGEngine
from langchain_postgres import PGVectorStore
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine
from .schema import ChatMessage


def get_db_connection() -> str:
    return os.getenv("DB_URL")

LLM_VLLM_API_BASE = os.getenv("LLM_VLLM_API_BASE")
LLM_VLLM_API_KEY = os.getenv("LLM_VLLM_API_KEY")
LLM_VLLM_API_BASE_SR = os.getenv("LLM_VLLM_API_BASE_SR")

POSTGRES_USER = os.getenv("POSTGRES_USER")  
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

TABLE_NAME = os.getenv("TABLE_NAME")

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f"/{POSTGRES_DB}"
)

pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

# Create an SQLAlchemy Async Engine
# engine = create_async_engine(
#     CONNECTION_STRING,
# )

# pg_engine = PGEngine.from_engine(engine=engine)
"""
embedding = OpenAIEmbeddings(
    model="Snowflake/snowflake-arctic-embed-m-long",
    openai_api_base=EMBEDDING_API_BASE,
    openai_api_key=EMBEDDING_API_KEY,
    tiktoken_enabled = False
)
"""


"""
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# # Initialize the Gemma embedding model
def get_embedding():
    embedding = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        query_encode_kwargs={"prompt_name": "query"},
        encode_kwargs={"prompt_name": "document"}
    )
    return embedding
"""

def get_embedding():
    return OpenAIEmbeddings(
        model="Snowflake/snowflake-arctic-embed-m-long",
        openai_api_base=EMBEDDING_API_BASE,
        openai_api_key=EMBEDDING_API_KEY,
        tiktoken_enabled = False
        )

async def get_combine_embedding_vectorstore(): 
    vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name="combine_embedding",
        embedding_service=get_embedding(),
        id_column="id",
        content_column="text",
        embedding_column="embedding",
        metadata_columns=["cresource_id"],
        metadata_json_column="metadata",
    )
    return vectorstore


async def get_document_vectorstore(): 
    vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name="document_embedding",
        embedding_service=get_embedding(),
        id_column="id",
        content_column="text",
        embedding_column="embedding",
        metadata_columns=["doc_id"],
        metadata_json_column="metadata",
    )
    return vectorstore


async def get_events_vectorstore(): 
    vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name="event_embedding",
        embedding_service=get_embedding(),
        id_column="id",
        content_column="text",
        embedding_column="embedding",
        metadata_columns=["event_id"],
        metadata_json_column="metadata",
    )
    return vectorstore

async def get_stakeholder_vectorstore():
    vectorstore = await PGVectorStore.create(
        engine = pg_engine,
        table_name = "stakeholder_embedding",
        embedding_service = get_embedding(),
        id_column = "id",
        content_column="text",
        embedding_column="embedding",
        metadata_columns=["stakeholder_id"],
        metadata_json_column="metadata",
    )
    return vectorstore

async def get_analytics_vectorstore(): 
    vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name="document_embedding",
        embedding_service=get_embedding(),
        id_column="id",
        content_column="text",
        embedding_column="embedding",
        metadata_columns=["doc_id"],
        metadata_json_column="metadata",
    )
    return vectorstore

llm_model = {}

def get_langchain_vllm_model(model: str = "unsloth/gemma-3-12b-it-bnb-4bit", max_tokens: int = 1000):
    if model not in llm_model:
        llm_model[model] = ChatOpenAI(
        model=model,
        openai_api_key=LLM_VLLM_API_KEY,
        openai_api_base=LLM_VLLM_API_BASE,
        temperature=0.1,
        max_tokens = max_tokens,
    )
    return llm_model[model]

llm_model_sr = {}
def get_langchain_vllm_model_sr(model: str = "unsloth/gemma-3-12b-it-bnb-4bit", max_tokens: int = 1000):
    if model not in llm_model_sr:
        llm_model_sr[model] = ChatOpenAI(
        model=model,
        openai_api_key=LLM_VLLM_API_KEY,
        openai_api_base=LLM_VLLM_API_BASE_SR,
        temperature=0.1,
        max_tokens = max_tokens,
    )
    return llm_model_sr[model]

def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage, documents: Optional[list[Document]] = None) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
                documents=documents or [],
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]

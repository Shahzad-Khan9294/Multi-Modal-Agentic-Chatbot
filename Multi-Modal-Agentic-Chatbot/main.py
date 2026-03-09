import os
from dotenv import load_dotenv
load_dotenv('.env.client') 
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, APIRouter
from chatbot.workflow.graph import agent as chatbot

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager

from chatbot import router as chatbot_router

from fastapi.middleware.cors import CORSMiddleware

import logging
logger = logging.getLogger(__name__)
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": None,
    "sslmode": "require",
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
    "connect_timeout": 10,
}

async def check_checkpoints_table(pool, checkpointer):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE  table_schema = 'public'
                        AND    table_name   = 'checkpoints'
                    );
                """)
                table_exists = (await cur.fetchone())[0]
                
                if not table_exists:
                    logger.info("Checkpoints table does not exist. Running setup...")
                    await checkpointer.setup()
                else:

                    logger.info("Checkpoints table already exists. Skipping setup.")
                    pass
            except Exception as e:
                logger.error(f"Error checking for checkpoints table: {e}")
                await conn.rollback()
                raise e
            finally:
                conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the AsyncConnectionPool
    async with AsyncConnectionPool(
            conninfo=os.getenv("DB_URL"),
            min_size=4,
            max_size=10,
            max_idle=300,
            kwargs=connection_kwargs,
            open=True,
            check=AsyncConnectionPool.check_connection
    ) as pool:
        # Create the AsyncPostgresSaver
        checkpointer = AsyncPostgresSaver(pool)

        # async with AsyncPostgresSaver.from_conn_string(
        #     os.getenv("DB_URL")
        # ) as checkpointer:
        #     chatbot.checkpointer = checkpointer
        #     app.state.chatbot = chatbot
        #     yield
        # await check_checkpoints_table(pool, checkpointer)
        # Check if the checkpoints table exists
        # await check_checkpoints_table(pool, checkpointer)
        
        # Assign the checkpointer to the assistant
        chatbot.checkpointer = checkpointer
        app.state.chatbot = chatbot
        yield

app = FastAPI(
    title="Amelia Client Chatbot",
    description="Amelia Client Chatbot services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    

app_api_router = APIRouter(prefix="/api")
app_api_router.include_router(chatbot_router)
app.include_router(app_api_router)

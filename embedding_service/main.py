import os
import sys
import asyncio
from dotenv import load_dotenv
load_dotenv('.env.client') 
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from embedding_service import router as embedding_service_router


app = FastAPI(
    title="Amelia Chatbot Embedding Service",
    description="Amelia Chatbot Embedding Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    
    
app_api_router = APIRouter(prefix="/api")
app_api_router.include_router(embedding_service_router)
app.include_router(app_api_router)

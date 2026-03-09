"""
Shared async database configuration for FastAPI application.
Provides a single AsyncEngine and sessionmaker for dependency injection.

Features:
- Connection pooling with configurable size and overflow
- Connection health checks (pool_pre_ping)
- Automatic connection recycling
- Proper session lifecycle management
- Graceful error handling
"""
import os
import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.pool import NullPool, QueuePool
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables if .env.client exists
try:
    load_dotenv('.env.client')
except Exception:
    pass  # Environment variables may already be set

# Build async connection string (postgresql+asyncpg)
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# Use asyncpg driver for async operations
DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Connection pool configuration
# These can be overridden via environment variables
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))  # Base pool size
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))  # Max connections beyond pool_size
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))  # Seconds to wait for connection
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # Recycle connections after 1 hour
POOL_PRE_PING = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"  # Verify connections
ECHO_SQL = os.getenv("DB_ECHO_SQL", "false").lower() == "true"  # Log SQL queries

# Create a single shared AsyncEngine with optimized connection pooling
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    # Connection pool settings
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,  # Recycle connections to prevent stale connections
    pool_pre_ping=POOL_PRE_PING,  # Verify connections before using
    # Performance settings
    echo=ECHO_SQL,  # Set to True for SQL query logging (useful for debugging)
    echo_pool=ECHO_SQL,  # Log pool events
    # Connection arguments for asyncpg
    connect_args={
        "server_settings": {
            "application_name": "amelia_chatbot",
            "jit": "off",  # Disable JIT for better connection performance
        },
        "command_timeout": 60,  # Query timeout in seconds
    },
    # Use QueuePool for better connection management
    # poolclass=QueuePool,
)

# Create a single shared sessionmaker
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Objects remain accessible after commit
    autoflush=False,  # Don't autoflush (better performance, explicit control)
    autocommit=False,  # Use explicit transactions
)


async def get_async_session() -> AsyncSession:
    """
    Dependency function for FastAPI to get an async database session.
    
    Usage in route dependencies:
        session: AsyncSession = Depends(get_async_session)
    
    The session is automatically closed when the request completes.
    Note: Commits/rollbacks should be handled explicitly in your code.
    """
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            # Rollback on any exception
            await session.rollback()
            raise
        finally:
            # Session is automatically closed by context manager
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncSession:
    """
    Context manager for getting a database session outside of FastAPI routes.
    
    Usage:
        async with get_db_session() as session:
            result = await db.some_method(session, ...)
            # Commit explicitly if needed
            await session.commit()
    """
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.
    Useful for health check endpoints.
    """
    try:
        from sqlalchemy import text
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def close_db_connections():
    """
    Close all database connections.
    Call this during application shutdown.
    """
    logger.info("Closing database connections...")
    await engine.dispose()
    logger.info("Database connections closed.")


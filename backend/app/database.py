from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# Use the async-compatible URL (converts postgres:// → postgresql+asyncpg://)
_db_url = settings.async_database_url

# SQLite needs check_same_thread=False; PostgreSQL needs pool settings
_connect_args = {"check_same_thread": False} if "sqlite" in _db_url else {}
_pool_kwargs  = {} if "sqlite" in _db_url else {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800,
}

engine = create_async_engine(
    _db_url,
    echo=settings.DEBUG,
    connect_args=_connect_args,
    **_pool_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

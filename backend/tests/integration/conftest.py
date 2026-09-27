"""PostgreSQL fixtures enabled only by the owned disposable-database runner."""

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool


@pytest.fixture
async def sql_engine() -> AsyncIterator[AsyncEngine]:
    raw = os.environ.get("TRAMFLOW_TEST_DATABASE_URL")
    if raw is None:
        pytest.skip("requires make backend-sql-test (isolated PostgreSQL)")
    url = make_url(raw)
    project = os.environ.get("TRAMFLOW_TEST_PROJECT", "")
    if not (
        re.fullmatch(r"tramflow-test-[0-9a-f]{32}", project)
        and url.drivername == "postgresql+asyncpg"
        and url.host == "127.0.0.1"
        and url.port is not None
        and url.port != 5432
        and url.database == "tramflow_test"
        and url.username == "tramflow_test"
    ):
        pytest.fail("Refusing a database outside the disposable test harness")
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@asynccontextmanager
async def isolated_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Rollback all test writes, including session commits, on context exit."""
    async with engine.connect() as connection:
        transaction = await connection.begin()
        try:
            async with AsyncSession(
                bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
            ) as session:
                yield session
        finally:
            await transaction.rollback()


@pytest.fixture
async def sql_session(sql_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with isolated_session(sql_engine) as session:
        yield session

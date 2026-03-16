"""Shared fixtures for all integration test suites.

Scope strategy mirrors the cust_dev_ai pattern:
- qdrant_container  → session  (expensive Docker startup, shared across all tests)
- qdrant_client     → function (cheap client object, fresh per test for isolation)

Test isolation is achieved by deleting and recreating the Qdrant collection
in each function-scoped repository fixture, since Qdrant has no SQL transactions.
"""

from collections.abc import AsyncGenerator

import pytest
from qdrant_client import AsyncQdrantClient
from testcontainers.qdrant import QdrantContainer

_QDRANT_HTTP_PORT: int = 6333  # noqa: WPS432


@pytest.fixture(scope="session")
def qdrant_container() -> QdrantContainer:
    """Start a Qdrant Docker container once for the entire test session.

    Session scope avoids the ~3-5 second container startup overhead on every
    test. Isolation between tests is handled at the collection level, not the
    container level.

    Yields:
        Running QdrantContainer instance exposing port 6333.
    """
    with QdrantContainer("qdrant/qdrant:latest") as container:
        yield container


@pytest.fixture
async def qdrant_client(qdrant_container: QdrantContainer) -> AsyncGenerator[AsyncQdrantClient, None]:
    """Return an AsyncQdrantClient connected to the test Qdrant container.

    Function-scoped so each test gets a fresh client object. The client
    is closed after the test to release the underlying httpx connections.

    Args:
        qdrant_container: Session-scoped running Qdrant container.

    Yields:
        AsyncQdrantClient pointed at the test container.
    """
    host = qdrant_container.get_container_host_ip()
    port = int(qdrant_container.get_exposed_port(_QDRANT_HTTP_PORT))
    client = AsyncQdrantClient(host=host, port=port)
    yield client
    await client.close()

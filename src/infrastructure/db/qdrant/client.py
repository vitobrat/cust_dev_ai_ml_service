"""Qdrant async client factory."""

from qdrant_client import AsyncQdrantClient

from src.configs.config import QdrantConfigs


def create_qdrant_client(configs: QdrantConfigs) -> AsyncQdrantClient:
    """Create a configured AsyncQdrantClient.

    Uses gRPC as the preferred transport for lower latency on bulk operations.
    The client instance is intended to be managed as a singleton by the DI container.

    Args:
        configs: Qdrant connection configuration.

    Returns:
        Configured async Qdrant client.
    """
    return AsyncQdrantClient(
        host=configs.host,
        port=configs.port,
        grpc_port=configs.grpc_port,
        prefer_grpc=True,
    )

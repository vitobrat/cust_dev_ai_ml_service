"""Embeddings domain dependency injection container."""

from dependency_injector import containers, providers

from src.domains.embeddings.app.usecases.service import EmbeddingsService
from src.domains.embeddings.app.workers.handler import EmbeddingsWorkerHandler
from src.infrastructure.containers.infrastructure import InfrastructureContainer
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)


class EmbeddingsContainer(containers.DeclarativeContainer):
    """DI container for the embeddings domain.

    Wires EmbeddingRepository and EmbeddingsService using
    shared infrastructure singletons (Qdrant client, Triton client).

    Attributes:
        infrastructure: Shared infrastructure singletons.
        repository: Factory for EmbeddingRepository instances.
        service: Factory for EmbeddingsService instances.
    """

    infrastructure: InfrastructureContainer = providers.DependenciesContainer()

    repository = providers.Factory(
        EmbeddingRepository,
        client=infrastructure.qdrant_client,
        collection_name=infrastructure.qdrant_configs.provided.collection_name,
        vector_size=infrastructure.qdrant_configs.provided.vector_size,
        hnsw_edge_size=infrastructure.qdrant_configs.provided.hnsw_edge_size,
        hnsw_neighbour_size=infrastructure.qdrant_configs.provided.hnsw_neighbour_size,
    )

    service = providers.Factory(
        EmbeddingsService,
        repository=repository,
        triton_client=infrastructure.triton_client,
    )

    embeddings_handler: EmbeddingsWorkerHandler = providers.Singleton(
        EmbeddingsWorkerHandler,
        service=service,
        rabbitmq=infrastructure.rabbitmq_client,
    )

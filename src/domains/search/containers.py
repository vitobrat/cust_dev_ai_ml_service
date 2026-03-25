"""Search domain dependency injection container."""

from dependency_injector import containers, providers

from src.domains.search.app.usecases.service import SearchService
from src.domains.search.app.workers.handler import SearchWorkerHandler
from src.infrastructure.containers.infrastructure import InfrastructureContainer
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)


class SearchContainer(containers.DeclarativeContainer):
    """DI container for the search domain.

    Shares the same EmbeddingRepository as the embeddings domain since
    both operate on the same Qdrant collection.

    Attributes:
        infrastructure: Shared infrastructure singletons.
        repository: Factory for EmbeddingRepository instances.
        service: Factory for SearchService instances.
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
        SearchService,
        repository=repository,
        triton_client=infrastructure.triton_client,
    )

    search_handler: SearchWorkerHandler = providers.Singleton(
        SearchWorkerHandler,
        service=service,
        rabbitmq=infrastructure.rabbitmq_client,
    )

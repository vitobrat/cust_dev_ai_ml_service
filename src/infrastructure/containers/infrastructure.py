"""Infrastructure dependency injection container module."""

from dependency_injector import containers, providers

from src.configs.config import (
    AppConfigs,
    QdrantConfigs,
    RabbitMQConfigs,
    TritonConfigs,
)
from src.infrastructure.db.qdrant.client import create_qdrant_client
from src.infrastructure.rabbitmq.client import RabbitMQClient
from src.infrastructure.triton.client import TritonClient


class InfrastructureContainer(containers.DeclarativeContainer):
    """DI container for shared infrastructure singletons.

    Manages the lifecycle of all external service clients.
    Each client is created once at application startup and reused
    across all requests.

    Attributes:
        config: Application configuration provider.
        qdrant_configs: Singleton QdrantConfigs (exposed for domain containers).
        qdrant_client: Singleton AsyncQdrantClient for vector operations.
        triton_client: Singleton TritonClient for embedding inference.
        rabbitmq_client: Singleton RabbitMQClient for message brokering.
    """

    config: AppConfigs = providers.Configuration()

    qdrant_configs = providers.Singleton(
        QdrantConfigs,
        host=config.qdrant.host,
        port=config.qdrant.port,
        grpc_port=config.qdrant.grpc_port,
        collection_name=config.qdrant.collection_name,
        vector_size=config.qdrant.vector_size,
        hnsw_edge_size=config.qdrant.hnsw_edge_size,
        hnsw_neighbour_size=config.qdrant.hnsw_neighbour_size,
    )

    qdrant_client = providers.Singleton(
        create_qdrant_client,
        configs=qdrant_configs,
    )

    triton_client = providers.Singleton(
        TritonClient,
        configs=providers.Singleton(
            TritonConfigs,
            host=config.triton.host,
            grpc_port=config.triton.grpc_port,
            http_port=config.triton.http_port,
            model_name=config.triton.model_name,
            input_name=config.triton.input_name,
            output_name=config.triton.output_name,
        ),
    )

    rabbitmq_client = providers.Singleton(
        RabbitMQClient,
        configs=providers.Singleton(
            RabbitMQConfigs,
            host=config.rabbitmq.host,
            port=config.rabbitmq.port,
            vhost=config.rabbitmq.vhost,
            user=config.rabbitmq.user,
            password=config.rabbitmq.password,
        ),
    )

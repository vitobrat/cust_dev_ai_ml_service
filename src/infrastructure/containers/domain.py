"""Domain dependency injection container module."""

from dependency_injector import containers, providers

from src.configs.config import AppConfigs
from src.domains.embeddings.containers import EmbeddingsContainer
from src.domains.search.containers import SearchContainer
from src.infrastructure.containers.infrastructure import InfrastructureContainer


class DomainContainer(containers.DeclarativeContainer):
    """Root DI container wiring infrastructure and domain sub-containers.

    Entry point for the dependency injection graph. Initialised once at
    application startup. InfrastructureContainer is owned here and wired
    into each domain container via provider references.

    Attributes:
        config: Root application configuration provider.
        infrastructure: Shared infrastructure singletons.
        embeddings: Embeddings domain services and repositories.
        search: Search domain services and repositories.
    """

    config: AppConfigs = providers.Configuration()

    infrastructure: InfrastructureContainer = providers.Container(
        InfrastructureContainer,
        config=config,
    )

    embeddings: EmbeddingsContainer = providers.Container(
        EmbeddingsContainer,
        infrastructure=infrastructure,
    )

    search: SearchContainer = providers.Container(
        SearchContainer,
        infrastructure=infrastructure,
    )

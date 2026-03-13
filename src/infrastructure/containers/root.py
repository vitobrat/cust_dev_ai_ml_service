"""Root dependency injection container module."""

from dependency_injector import containers, providers

from src.configs.config import AppConfigs
from src.infrastructure.containers.domain import DomainContainer
from src.infrastructure.containers.infrastructure import InfrastructureContainer


class RootContainer(containers.DeclarativeContainer):
    """Top-level DI container wiring infrastructure and domain containers.

    Entry point for the dependency injection graph. Loaded once at
    application startup; sub-containers receive their dependencies
    via provider references, not direct instantiation.

    Attributes:
        config: Root application configuration provider.
        infrastructure: Shared infrastructure singletons.
        domain: Domain-level services and repositories.
    """

    config: AppConfigs = providers.Configuration()

    infrastructure: InfrastructureContainer = providers.Container(
        InfrastructureContainer,
        config=config,
    )

    domain: DomainContainer = providers.Container(
        DomainContainer,
        infrastructure=infrastructure,
    )

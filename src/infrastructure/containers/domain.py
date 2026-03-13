"""Domain dependency injection container module.

Placeholder container extended in step 5 when domain services
and repositories are added.
"""

from dependency_injector import containers, providers

from src.configs.config import AppConfigs
from src.infrastructure.containers.infrastructure import InfrastructureContainer


class DomainContainer(containers.DeclarativeContainer):
    """Root domain DI container aggregating per-domain sub-containers.

    Currently a stub — populated with embeddings and search domain
    containers in the FastAPI integration step.

    Attributes:
        infrastructure: Shared infrastructure singletons.
    """

    config: AppConfigs = providers.Configuration()
    infrastructure: InfrastructureContainer = providers.DependenciesContainer()

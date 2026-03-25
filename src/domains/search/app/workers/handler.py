"""RabbitMQ message handler for the search domain."""

from logging import Logger
from typing import Any

from pydantic import ValidationError

from src.configs.log.logger import get_logger
from src.domains.search.app.usecases.service import SearchService
from src.domains.search.exceptions import SearchQueryError
from src.infrastructure.rabbitmq.client import RabbitMQClient
from src.schemas.api_base import ResponseBase, StatusType
from src.schemas.search import SearchRequest, SearchResponse


class SearchWorkerHandler:
    """Handles incoming RabbitMQ search requests and publishes replies.

    Attributes:
        _service: Search application service.
        _rabbitmq: RabbitMQ client used to publish replies.
        _logger: Logger instance for this handler.
    """

    def __init__(self, service: SearchService, rabbitmq: RabbitMQClient) -> None:
        """Initialize handler with injected dependencies.

        Args:
            service: Search service for embedding and querying Qdrant.
            rabbitmq: Client used to publish the reply message.
        """
        self._service = service
        self._rabbitmq = rabbitmq
        self._logger: Logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def search_handle(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Process a search request message and publish the result.

        Validates the payload, runs the search, and publishes the reply.
        On any failure, publishes an error response so the caller is not left waiting.

        Args:
            reply_to: Queue name to publish the reply to.
            correlation_id: Opaque request ID to echo back in the reply.
            payload: Deserialised message body containing search parameters.
        """
        if reply_to is None or correlation_id is None:
            self._logger.warning("Dropping message: missing reply_to or correlation_id")
            return

        try:
            search_request = SearchRequest(**payload)
        except ValidationError as exc:
            self._logger.error("Invalid search request payload: %s", exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(),
            )
            return

        try:
            search_results = await self._service.search(
                user_id=search_request.user_id,
                query=search_request.query,
                top_k=search_request.top_k,
            )
        except SearchQueryError as exc:
            self._logger.error("Search query failed for user %s: %s", search_request.user_id, exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(),
            )
            return

        await self._rabbitmq.publish_reply(
            reply_to,
            correlation_id,
            payload=SearchResponse(msg=search_results, status=StatusType.SUCCESS).model_dump(),
        )

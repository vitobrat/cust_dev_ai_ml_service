"""RabbitMQ message handler for the embeddings domain."""

from logging import Logger
from typing import Any

from pydantic import ValidationError

from src.configs.log.logger import get_logger
from src.domains.embeddings.app.usecases.service import EmbeddingsService
from src.domains.embeddings.exceptions import (
    EmbeddingDeleteError,
    EmbeddingUpsertError,
)
from src.infrastructure.exceptions import EmbeddingError
from src.infrastructure.rabbitmq.client import RabbitMQClient
from src.schemas.api_base import ResponseBase, StatusType
from src.schemas.embeddings import (
    DeleteAllRequest,
    DeleteByIdRequest,
    DeleteResponse,
    EmbeddingAction,
    UpsertRequest,
    UpsertResponse,
)


class EmbeddingsWorkerHandler:
    """Handles incoming RabbitMQ embeddings requests and publishes replies.

    Routes messages by the ``action`` field in the payload to upsert,
    delete_by_id, or delete_all operations on the embeddings service.

    Attributes:
        _service: Embeddings application service.
        _rabbitmq: RabbitMQ client used to publish replies.
        _logger: Logger instance for this handler.
    """

    def __init__(self, service: EmbeddingsService, rabbitmq: RabbitMQClient) -> None:
        """Initialize handler with injected dependencies.

        Args:
            service: Embeddings service for Triton inference and Qdrant operations.
            rabbitmq: Client used to publish the reply message.
        """
        self._service = service
        self._rabbitmq = rabbitmq
        self._logger: Logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def embeddings_handle(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Route an embeddings message to the appropriate operation handler.

        Reads the ``action`` field from the payload and dispatches to the
        corresponding private method. Publishes an error reply if the action
        is missing or unknown.

        Args:
            reply_to: Queue name to publish the reply to.
            correlation_id: Opaque request ID to echo back in the reply.
            payload: Deserialised message body with an ``action`` field and operation params.
        """
        if reply_to is None or correlation_id is None:
            self._logger.warning("Dropping message: missing reply_to or correlation_id")
            return

        action = payload.get("action")

        if action == EmbeddingAction.UPSERT:
            await self._handle_upsert(reply_to, correlation_id, payload)
        elif action == EmbeddingAction.DELETE_BY_ID:
            await self._handle_delete_by_id(reply_to, correlation_id, payload)
        elif action == EmbeddingAction.DELETE_ALL:
            await self._handle_delete_all(reply_to, correlation_id, payload)
        else:
            self._logger.error("Unknown action '%s' in embeddings message", action)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(
                    details=f"Unknown action: {action}",
                    status=StatusType.ERROR,
                ).model_dump(mode="json"),
            )

    async def _handle_upsert(self, reply_to: str, correlation_id: str, payload: dict[str, Any]) -> None:
        """Process an upsert request: embed texts and store in Qdrant.

        Args:
            reply_to: Queue name to publish the reply to.
            correlation_id: Opaque request ID to echo back in the reply.
            payload: Raw message payload to validate as UpsertRequest.
        """
        try:
            request = UpsertRequest(**payload)
        except ValidationError as exc:
            self._logger.error("Invalid upsert payload: %s", exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return

        try:
            count = await self._service.upsert(request.user_id, request.texts)
        except EmbeddingUpsertError as exc:
            self._logger.error("Upsert failed for user %s: %s", request.user_id, exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return
        except EmbeddingError as exc:
            self._logger.error("Embedding failed for user %s: %s", request.user_id, exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return

        await self._rabbitmq.publish_reply(
            reply_to,
            correlation_id,
            payload=UpsertResponse(msg=count, status=StatusType.SUCCESS).model_dump(mode="json"),
        )

    async def _handle_delete_by_id(self, reply_to: str, correlation_id: str, payload: dict[str, Any]) -> None:
        """Process a delete-by-id request: remove a single point from Qdrant.

        Args:
            reply_to: Queue name to publish the reply to.
            correlation_id: Opaque request ID to echo back in the reply.
            payload: Raw message payload to validate as DeleteByIdRequest.
        """
        try:
            request = DeleteByIdRequest(**payload)
        except ValidationError as exc:
            self._logger.error("Invalid delete_by_id payload: %s", exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return

        try:
            await self._service.delete_by_id(request.user_id, request.point_id)
        except EmbeddingDeleteError as exc:
            self._logger.error("Delete by id failed for user %s, point %s: %s", request.user_id, request.point_id, exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return

        await self._rabbitmq.publish_reply(
            reply_to,
            correlation_id,
            payload=DeleteResponse(msg=True, status=StatusType.SUCCESS).model_dump(mode="json"),
        )

    async def _handle_delete_all(self, reply_to: str, correlation_id: str, payload: dict[str, Any]) -> None:
        """Process a delete-all request: remove all points for a user from Qdrant.

        Args:
            reply_to: Queue name to publish the reply to.
            correlation_id: Opaque request ID to echo back in the reply.
            payload: Raw message payload to validate as DeleteAllRequest.
        """
        try:
            request = DeleteAllRequest(**payload)
        except ValidationError as exc:
            self._logger.error("Invalid delete_all payload: %s", exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return

        try:
            await self._service.delete_all(request.user_id)
        except EmbeddingDeleteError as exc:
            self._logger.error("Delete all failed for user %s: %s", request.user_id, exc)
            await self._rabbitmq.publish_reply(
                reply_to,
                correlation_id,
                payload=ResponseBase(details=str(exc), status=StatusType.ERROR).model_dump(mode="json"),
            )
            return

        await self._rabbitmq.publish_reply(
            reply_to,
            correlation_id,
            payload=DeleteResponse(msg=True, status=StatusType.SUCCESS).model_dump(mode="json"),
        )

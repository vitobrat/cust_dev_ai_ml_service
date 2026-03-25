"""FastAPI router for the embeddings domain."""

import uuid

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Response, status

from src.configs.log.logger import get_logger
from src.domains.embeddings.app.requests.schema import (
    PostDeleteResponse,
    PostUpsertRequest,
    PostUpsertResponse,
)
from src.domains.embeddings.app.usecases.service import EmbeddingsService
from src.domains.embeddings.exceptions import (
    EmbeddingDeleteError,
    EmbeddingUpsertError,
)
from src.infrastructure.containers.domain import DomainContainer
from src.schemas.api_base import ResponseBase, StatusType

_logger = get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/", response_model=PostUpsertResponse | ResponseBase, status_code=status.HTTP_201_CREATED)
@inject
async def upsert_embeddings(
    response: Response,
    body: PostUpsertRequest,
    service: EmbeddingsService = Depends(Provide[DomainContainer.embeddings.service]),
) -> PostUpsertResponse | ResponseBase:
    """Embed texts and store them in Qdrant for the given user.

    Args:
        response: FastAPI response object used to override the status code on error.
        body: Request body containing user_id and list of texts to index.
        service: Injected embeddings application service.

    Returns:
        Number of documents upserted into the vector store, or an error response.
    """
    try:
        count = await service.upsert(body.user_id, body.texts)
    except EmbeddingUpsertError as embed_exc:
        _logger.error("Embedding upsert failed for user %s: %s", body.user_id, embed_exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Embedding upsert error: {embed_exc}", status=StatusType.ERROR)
    except Exception as exc:
        _logger.error("Unexpected error during embedding upsert for user %s: %s", body.user_id, exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Unexpected error during embedding upsert: {exc}", status=StatusType.ERROR)

    return PostUpsertResponse(msg=count, status=StatusType.SUCCESS)


@router.delete(
    "/{user_id}/{point_id}",
    response_model=PostDeleteResponse | ResponseBase,
    status_code=status.HTTP_200_OK,
)
@inject
async def delete_embedding_by_id(
    response: Response,
    user_id: uuid.UUID,
    point_id: uuid.UUID,
    service: EmbeddingsService = Depends(Provide[DomainContainer.embeddings.service]),
) -> PostDeleteResponse | ResponseBase:
    """Delete a single embedding point owned by the given user.

    Args:
        response: FastAPI response object used to override the status code on error.
        user_id: Owner of the point to delete.
        point_id: Qdrant point identifier to remove.
        service: Injected embeddings application service.

    Returns:
        Deletion confirmation, or an error response.
    """
    try:
        await service.delete_by_id(user_id, point_id)
    except EmbeddingDeleteError as del_exc:
        _logger.error("Embedding delete by id failed for user %s, point %s: %s", user_id, point_id, del_exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Embedding deletion error: {del_exc}", status=StatusType.ERROR)
    except Exception as exc:
        _logger.error("Unexpected error during embedding deletion by id for user %s: %s", user_id, exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Unexpected error during embedding deletion: {exc}", status=StatusType.ERROR)

    return PostDeleteResponse(msg=True, status=StatusType.SUCCESS)


@router.delete("/{user_id}", response_model=PostDeleteResponse | ResponseBase, status_code=status.HTTP_200_OK)
@inject
async def delete_all_user_embeddings(
    response: Response,
    user_id: uuid.UUID,
    service: EmbeddingsService = Depends(Provide[DomainContainer.embeddings.service]),
) -> PostDeleteResponse | ResponseBase:
    """Delete all embeddings belonging to a user.

    Args:
        response: FastAPI response object used to override the status code on error.
        user_id: Owner whose entire indexed document set should be removed.
        service: Injected embeddings application service.

    Returns:
        Deletion confirmation, or an error response.
    """
    try:
        await service.delete_all(user_id)
    except EmbeddingDeleteError as del_exc:
        _logger.error("Delete all embeddings failed for user %s: %s", user_id, del_exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Embedding deletion error: {del_exc}", status=StatusType.ERROR)
    except Exception as exc:
        _logger.error("Unexpected error during delete all embeddings for user %s: %s", user_id, exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Unexpected error during embedding deletion: {exc}", status=StatusType.ERROR)

    return PostDeleteResponse(msg=True, status=StatusType.SUCCESS)

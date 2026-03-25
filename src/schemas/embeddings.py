"""Shared embeddings schemas used across the embeddings domain and worker."""

import uuid
from enum import Enum

from pydantic import BaseModel, Field

from src.schemas.api_base import ResponseBase


class EmbeddingAction(str, Enum):
    """Discriminator for routing embeddings worker messages."""

    UPSERT = "upsert"
    DELETE_BY_ID = "delete_by_id"
    DELETE_ALL = "delete_all"


class UpsertRequest(BaseModel):
    """Base request for embedding upsert.

    Attributes:
        user_id: Owner of the documents being indexed.
        texts: Non-empty list of raw text documents to embed and store.
    """

    user_id: uuid.UUID
    texts: list[str] = Field(min_length=1)


class DeleteByIdRequest(BaseModel):
    """Base request for single point deletion.

    Attributes:
        user_id: Owner of the point to delete.
        point_id: Qdrant point identifier to remove.
    """

    user_id: uuid.UUID
    point_id: uuid.UUID


class DeleteAllRequest(BaseModel):
    """Base request for deleting all user points.

    Attributes:
        user_id: Owner whose entire document set should be removed.
    """

    user_id: uuid.UUID


class UpsertResponse(ResponseBase):
    """Response for a successful embedding upsert.

    Attributes:
        msg: Number of documents successfully stored in Qdrant.
    """

    msg: int


class DeleteResponse(ResponseBase):
    """Response for a successful embedding deletion.

    Attributes:
        msg: True if deletion completed without errors.
    """

    msg: bool

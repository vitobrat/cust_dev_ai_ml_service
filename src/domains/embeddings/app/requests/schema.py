"""Request and response schemas for the embeddings domain."""

import uuid

from pydantic import BaseModel, Field

from src.schemas.api_base import ResponseBase


class PostUpsertRequest(BaseModel):
    """Request body for the embedding upsert endpoint.

    Attributes:
        user_id: Owner of the documents being indexed.
        texts: Non-empty list of raw text documents to embed and store.
    """

    user_id: uuid.UUID
    texts: list[str] = Field(min_length=1)


class PostUpsertResponse(ResponseBase):
    """Response body for a successful embedding upsert.

    Attributes:
        msg: Number of documents successfully stored in Qdrant.
    """

    msg: int


class DeleteResponse(ResponseBase):
    """Response body for a successful embedding deletion.

    Attributes:
        msg: True if deletion completed without errors.
    """

    msg: bool

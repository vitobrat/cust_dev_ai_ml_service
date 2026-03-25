"""Request and response schemas for the embeddings domain."""

from src.schemas.embeddings import (
    DeleteAllRequest,
    DeleteByIdRequest,
    DeleteResponse,
    UpsertRequest,
    UpsertResponse,
)


class PostUpsertRequest(UpsertRequest):
    """Request body for the HTTP embedding upsert endpoint."""


class PostUpsertResponse(UpsertResponse):
    """Response body for the HTTP embedding upsert endpoint."""


class PostDeleteByIdRequest(DeleteByIdRequest):
    """Request body for the HTTP single point deletion endpoint."""


class PostDeleteAllRequest(DeleteAllRequest):
    """Request body for the HTTP delete-all endpoint."""


class PostDeleteResponse(DeleteResponse):
    """Response body for the HTTP embedding deletion endpoint."""

"""FastAPI router for the search domain."""

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Response, status

from src.configs.log.logger import get_logger
from src.domains.search.app.requests.schema import (
    PostSearchRequest,
    PostSearchResponse,
)
from src.domains.search.app.usecases.service import SearchService
from src.domains.search.exceptions import SearchQueryError
from src.infrastructure.containers.domain import DomainContainer
from src.schemas.api_base import ResponseBase, StatusType

_logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=PostSearchResponse | ResponseBase)
@inject
async def semantic_search(
    response: Response,
    body: PostSearchRequest,
    service: SearchService = Depends(Provide[DomainContainer.search.service]),
) -> PostSearchResponse | ResponseBase:
    """Perform semantic similarity search over a user's indexed documents.

    Args:
        response: FastAPI response object used to override the status code on error.
        body: Request body containing user_id, query text, and top_k limit.
        service: Injected search application service.

    Returns:
        Ranked list of matching documents with scores and payloads, or an error response.
    """
    try:
        search_results = await service.search(body.user_id, body.query, body.top_k)
    except SearchQueryError as exc:
        _logger.error("Search query failed for user %s: %s", body.user_id, exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Error during search query: {exc}", status=StatusType.ERROR)
    except Exception as exc:
        _logger.error("Unexpected error during search for user %s: %s", body.user_id, exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ResponseBase(details=f"Unexpected error during search query: {exc}", status=StatusType.ERROR)

    return PostSearchResponse(msg=search_results, status=StatusType.SUCCESS)

# Search Domain

The search domain performs user-scoped semantic search over vectors previously
stored by the embeddings domain. It shares the same Qdrant collection and
repository implementation.

## Main Use Case

- `search(user_id, query, top_k)`: embed one query with Triton, search Qdrant,
  and return ranked `SearchResultItem` objects.

## Layer Map

- API layer: `app/requests/router.py`
- HTTP schemas: `app/requests/schema.py`
- RabbitMQ handler: `app/workers/handler.py`
- Application service: `app/usecases/service.py`
- Shared request/response contracts: `src/schemas/search.py`
- Vector repository: `src/infrastructure/db/qdrant/embedding_repository.py`
- Embedding inference: `src/infrastructure/triton/client.py`

## Data Flow

```text
PostSearchRequest or RabbitMQ search.request
  -> SearchService.search(user_id, query, top_k)
  -> TritonClient.embed(["query: " + query])
  -> EmbeddingRepository.search(user_id, query_vector, limit=top_k)
  -> Qdrant query with user_id filter
  -> list[SearchResultItem]
```

## Caution Points

- Search must always be scoped by `user_id`.
- `top_k` is validated at schema level from 1 to 100.
- A Triton failure is raised as infrastructure-level `EmbeddingError`, while
  Qdrant search failures are wrapped as `SearchQueryError`.
- Returned payloads currently include the stored source `text` and injected
  `user_id`; callers can use `payload["text"]` as RAG context.

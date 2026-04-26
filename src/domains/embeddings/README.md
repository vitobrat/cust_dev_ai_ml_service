# Embeddings Domain

The embeddings domain owns vector lifecycle operations for user-provided text.
It does not split documents into chunks; every string passed in `texts` becomes
one Qdrant point.

## Main Use Cases

- `upsert(user_id, texts)`: embed texts with Triton and store vectors in Qdrant.
- `delete_by_id(user_id, point_id)`: delete one user-owned vector.
- `delete_all(user_id)`: delete all vectors for a user.

## Layer Map

- API layer: `app/requests/router.py`
- HTTP schemas: `app/requests/schema.py`
- RabbitMQ handler: `app/workers/handler.py`
- Application service: `app/usecases/service.py`
- Shared request/response contracts: `src/schemas/embeddings.py`
- Vector repository: `src/infrastructure/db/qdrant/embedding_repository.py`
- Embedding inference: `src/infrastructure/triton/client.py`

## Data Flow

```text
PostUpsertRequest or RabbitMQ action=upsert
  -> EmbeddingsService.upsert()
  -> TritonClient.embed(["passage: " + text, ...])
  -> create PointStruct(id=uuid4, payload={"text": text})
  -> EmbeddingRepository.upsert(user_id, points)
  -> Qdrant
```

The repository adds `payload["user_id"] = str(user_id)` to every point. Keep this
infrastructure-level invariant intact when changing the domain.

## RabbitMQ

Queue: `embeddings.request`

Supported actions:

- `upsert`
- `delete_by_id`
- `delete_all`

The worker handler requires `reply_to` and `correlation_id` so it can publish a
response. Messages without either value are logged and dropped.

## Caution Points

- Triton failures are raised as `EmbeddingError`.
- Qdrant write/delete failures are wrapped as `EmbeddingUpsertError` or `EmbeddingDeleteError`.
- Worker responses must remain JSON-serializable; use
  `model_dump(mode="json")` when adding RabbitMQ response payloads.

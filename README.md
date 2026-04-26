# ml_service

Internal ML microservice for the `cust_dev_ai` system. It owns the vector
database and embedding/RAG infrastructure: text embedding generation, Qdrant
upsert/delete/search operations, and RabbitMQ request handling for vector
search workflows.

The service also exposes a FastAPI HTTP API for direct development calls, but
the intended integration boundary with the main backend is RabbitMQ.

## Documentation Map

- `README.md`: service overview, commands, HTTP API, RabbitMQ contracts, and
  operational notes.
- `ARCHITECTURE.md`: DDD layer map, cross-service boundary, data flows, and
  safety checklist.
- `PROJECT_CONTEXT.md`: tracked bootstrap context for future AI agents.
- `AGENTS.md`: local Codex prompt; currently ignored by `.gitignore`.
- `src/domains/embeddings/README.md`: embeddings-domain notes.
- `src/domains/search/README.md`: search-domain notes.

## System Context

The wider system is split into at least two Python microservices:

- `cust_dev_ai` (`/home/vito_brat/cust_dev_ai`): main product backend. Owns
  users, interviews, personas, sub-interviews, tasks, PostgreSQL persistence,
  Redis task worker, and LLM/LangGraph workflows.
- `ml_service` (`/home/vito_brat/ml_service`): ML/vector-search backend. Owns
  Triton embedding inference, Qdrant vector storage, semantic search, and
  RabbitMQ handlers for embedding/search requests.

Both services use a DDD-style structure with domains, application services,
infrastructure clients, schemas, DI containers, and API routers.

### Integration Status From Code

`ml_service` consumes RabbitMQ queues:

- `embeddings.request`
- `search.request`

The current `cust_dev_ai` codebase has a RabbitMQ client and configuration, but
no source-level producer calls to these two queues were found during the
2026-04-25 code pass. Treat the RabbitMQ payload schemas in this repository as
the current contract for future or external producers. Direct HTTP endpoints can
also be used for local testing.

## Responsibilities

- Generate embeddings for batches of raw text via Triton Inference Server.
- Store vectors in Qdrant in one multi-tenant collection.
- Search relevant vectors by embedding the query and running Qdrant similarity search.
- Delete one or all vectors for a user.
- Handle RabbitMQ request/reply messages for embeddings and search.
- Keep vector-search infrastructure separate from the main product backend.

What this service does not currently do:

- It does not split documents into chunks. Each string in `texts` is stored as one Qdrant point.
- It does not call an external LLM provider.
- It does not own users, interviews, personas, or task state.
- It does not store relational data or run SQL migrations.

## Runtime Stack

| Area | Technology |
|---|---|
| API | FastAPI, Uvicorn, Slowapi |
| DI | dependency-injector |
| Vector DB | Qdrant async client, gRPC preferred |
| Embeddings | Triton Inference Server over gRPC, ONNX Runtime backend |
| Model | `intfloat/multilingual-e5-small`, ONNX, 384-dimensional pooled vectors |
| Tokenization | HuggingFace `AutoTokenizer` in the Python client |
| Messaging | RabbitMQ via `aio-pika` |
| Config | OmegaConf YAML + pydantic-settings `.env` |
| Logging | JSON logging with `python-json-logger` |
| Tests | pytest, pytest-asyncio, testcontainers |

## Architecture At A Glance

```text
client or cust_dev_ai
  |
  | HTTP or RabbitMQ
  v
API/worker layer
  |
  v
application services
  |
  +-- TritonClient -> tokenizer -> Triton ONNX model -> mean pooling -> normalized embedding
  |
  +-- EmbeddingRepository -> Qdrant collection "documents"
```

DDD layer map:

- API layer:
  - `src/domains/embeddings/app/requests/router.py`
  - `src/domains/search/app/requests/router.py`
- Worker/API adapter layer:
  - `src/domains/embeddings/app/workers/handler.py`
  - `src/domains/search/app/workers/handler.py`
- Application/service layer:
  - `src/domains/embeddings/app/usecases/service.py`
  - `src/domains/search/app/usecases/service.py`
- Domain boundary:
  - `src/domains/embeddings`
  - `src/domains/search`
  - domain exceptions and request/response contracts
- Infrastructure layer:
  - `src/infrastructure/db/qdrant`
  - `src/infrastructure/triton`
  - `src/infrastructure/rabbitmq`
  - `src/infrastructure/containers`
- Shared schemas:
  - `src/schemas/api_base.py`
  - `src/schemas/embeddings.py`
  - `src/schemas/search.py`

For a deeper walkthrough, read [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Structure

```text
ml_service/
├── config/
│   ├── config.dev.yaml
│   ├── logging.yaml
│   └── .env
├── docker/
│   ├── Dockerfile
│   └── docker-compose.dev.yaml
├── models/
│   └── embedding_model_multilingual_e5_small/
├── src/
│   ├── app.py
│   ├── worker.py
│   ├── configs/
│   ├── domains/
│   │   ├── embeddings/
│   │   └── search/
│   ├── infrastructure/
│   │   ├── containers/
│   │   ├── db/qdrant/
│   │   ├── rabbitmq/
│   │   └── triton/
│   └── schemas/
└── tests/
    ├── unit/
    └── integration/
```

## Entry Points

- `src/app.py`: FastAPI server. It wires `DomainContainer`, registers API routers,
  ensures the Qdrant collection exists on startup, and closes Triton on shutdown.
- `src/worker.py`: RabbitMQ worker. It connects RabbitMQ, ensures the Qdrant
  collection exists, consumes `embeddings.request` and `search.request`, and
  publishes replies to `reply_to` with the incoming `correlation_id`.

## API

Base path: `/api/v1`

### Embeddings

Router prefix: `/api/v1/embeddings`

| Method | Path | Description |
|---|---|---|
| `POST` | `/` | Embed each text and upsert resulting vectors into Qdrant. |
| `DELETE` | `/{user_id}/{point_id}` | Delete one point for a user. |
| `DELETE` | `/{user_id}` | Delete all points for a user. |

Upsert request:

```json
{
  "user_id": "12345678-1234-5678-1234-567812345678",
  "texts": ["already prepared chunk or document text"]
}
```

Successful response:

```json
{
  "msg": 1,
  "details": null,
  "status": "success"
}
```

### Search

Router prefix: `/api/v1/search`

| Method | Path | Description |
|---|---|---|
| `POST` | `/` | Embed query and return top matching Qdrant points for the user. |

Search request:

```json
{
  "user_id": "12345678-1234-5678-1234-567812345678",
  "query": "question or semantic search query",
  "top_k": 10
}
```

Search result items include:

- `id`: Qdrant point UUID.
- `score`: similarity score.
- `payload`: Qdrant payload, currently including `text` and `user_id`.

## RabbitMQ Contract

Queue constants live in `src/configs/consts.py`.

### `embeddings.request`

The payload must include `action`.

Supported actions:

- `upsert`
- `delete_by_id`
- `delete_all`

Upsert payload:

```json
{
  "action": "upsert",
  "user_id": "12345678-1234-5678-1234-567812345678",
  "texts": ["text to embed"]
}
```

Delete one point:

```json
{
  "action": "delete_by_id",
  "user_id": "12345678-1234-5678-1234-567812345678",
  "point_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
}
```

Delete all user points:

```json
{
  "action": "delete_all",
  "user_id": "12345678-1234-5678-1234-567812345678"
}
```

### `search.request`

Search payload:

```json
{
  "user_id": "12345678-1234-5678-1234-567812345678",
  "query": "find relevant context",
  "top_k": 10
}
```

Worker handlers require AMQP `reply_to` and `correlation_id`. If either is
missing, the message is logged and dropped because the service cannot send a
request/reply response.

## Data Flow

### Upsert

1. Caller sends `user_id` and `texts`.
1. `EmbeddingsService.upsert()` calls `TritonClient.embed(texts)`.
1. `TritonClient` tokenizes text, sends token tensors to Triton, mean-pools
   `last_hidden_state`, and L2-normalizes vectors.
1. `EmbeddingsService` creates one Qdrant `PointStruct` per input text with a
   generated UUID and payload `{"text": text}`.
1. `EmbeddingRepository.upsert()` stamps `payload["user_id"] = str(user_id)`.
1. Qdrant stores vectors in the configured collection.

### Search

1. Caller sends `user_id`, `query`, and `top_k`.
1. `SearchService.search()` embeds the query via Triton.
1. `EmbeddingRepository.search()` adds a Qdrant filter for `payload.user_id`.
1. Qdrant returns scored points.
1. Service maps Qdrant `ScoredPoint` objects to `SearchResultItem`.

### Delete

All deletes go through `EmbeddingRepository`, which always includes `user_id`
in the Qdrant filter. This prevents cross-user deletion by point id.

## Qdrant Invariants

All vectors are stored in one collection. Multi-tenancy is enforced by payload,
not by separate collections.

Do not break these invariants:

- Every upserted point must get `payload["user_id"]` as a string.
- Every search must include a `user_id` filter.
- Every delete must include a `user_id` filter.
- `delete_by_ids()` must combine `user_id` with point IDs.
- `delete_all_for_user()` must filter only by the target user.
- `ensure_collection()` creates or repairs a `KEYWORD` payload index on
  `user_id`.

## Triton And Model Details

The current embedding pipeline is not a raw BYTES-input Triton call. Tokenization
and pooling happen in Python:

1. `AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")`
1. INT64 tensors sent to Triton:
   - `input_ids`
   - `attention_mask`
   - `token_type_ids`
1. Requested Triton output:
   - `last_hidden_state`
1. Mean pooling over non-padding tokens.
1. L2 normalization.
1. 384-dimensional embedding vector.

Tracked model config:

- `models/embedding_model_multilingual_e5_small/config.pbtxt`

The large ONNX model file is expected at:

- `models/embedding_model_multilingual_e5_small/1/model.onnx`

It is intentionally ignored by Git through `model.onnx`. Use
`make download-model` when the model file is missing.

Semantic note: E5-family models expect task prefixes. `EmbeddingsService`
applies `passage: ` before embedding stored texts, while `SearchService` applies
`query: ` before embedding search queries. The original unprefixed text is still
stored in Qdrant payloads under `text`.

## Configuration

Default config path:

- `config/config.dev.yaml`

Override:

```bash
CONFIG_PATH=/path/to/config.yaml make run
```

Current development defaults:

```yaml
app_host: "0.0.0.0"
app_port: 8888

qdrant:
  host: "qdrant"
  port: 6333
  grpc_port: 6334
  collection_name: "documents"
  hnsw_edge_size: 16
  hnsw_neighbour_size: 100

triton:
  host: "triton"
  grpc_port: 8001
  http_port: 8000

rabbitmq:
  host: "rabbitmq"
  port: 5672
  vhost: "ml"
```

Required secret env vars in `config/.env`:

```env
RABBITMQ_USER=
RABBITMQ_PASSWORD=
```

`HF_TOKEN` may be present in `.env`, but the current code does not read it.

## Docker

`docker/docker-compose.dev.yaml` defines:

- `rabbitmq`
- `qdrant`
- `triton`
- `app`
- `rabbitmq_worker`

Triton uses `nvcr.io/nvidia/tritonserver:24.12-py3` and the compose file
requests an NVIDIA GPU. The model repository is mounted from `../models`.

## Commands

```bash
make setup          # download model and build Docker images
make download-model # download ONNX model from HuggingFace
make up             # start RabbitMQ, Qdrant, Triton, app, worker
make down           # stop dev stack, preserve volumes
make clean          # stop and remove volumes
make logs-all       # follow all compose logs
make logs service=app
make run            # run FastAPI locally; external services still required
make unit           # unit tests
make integration    # integration tests; Docker/testcontainers required
make tests          # all tests
make lint           # pre-commit run --all-files
```

In this checkout, `.venv/bin/python -m pytest -q tests/unit` is the safest
direct command if system `python3` does not have project dependencies.

## Testing

Unit tests cover:

- `EmbeddingsService`
- `SearchService`
- `BaseQdrantRepository` filtering behavior
- `RabbitMQClient`

Integration tests cover:

- real Qdrant repository behavior via `testcontainers.qdrant`
- real RabbitMQ publish/consume/reply behavior via `testcontainers.rabbitmq`

There are no current tests for:

- FastAPI routers
- RabbitMQ worker handlers
- real Triton inference

## Notes For Maintainers

- Keep the vector DB boundary in this service; do not add Qdrant calls to
  `cust_dev_ai`.
- Keep user isolation in the repository layer so every API and worker path gets
  the same safety properties.
- If `cust_dev_ai` starts producing RabbitMQ messages to this service, update
  both repositories' docs with the exact end-to-end scenario.
- If chunking is added, document which service owns chunk creation and metadata.

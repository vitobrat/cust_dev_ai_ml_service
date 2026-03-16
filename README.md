# ml_service

Internal ML microservice for the `cust_dev_ai` customer-interview simulation system.

Communicates with `cust_dev_ai` over **RabbitMQ**. Exposes a FastAPI HTTP API for direct calls during development.

______________________________________________________________________

## Responsibilities

- **Embeddings** — generates text embeddings via ONNX models served on Triton Inference Server (local gRPC).
- **Vector storage** — upserts and deletes per-user document embeddings in Qdrant (multi-tenant via `user_id` payload filter).
- **Semantic search (RAG)** — embeds a query, searches Qdrant by cosine similarity, returns ranked results to `cust_dev_ai`.
- **Async messaging** — consumes/publishes JSON messages over RabbitMQ (`aio-pika`) for integration with `cust_dev_ai`.

______________________________________________________________________

## Stack

| Component | Technology |
|------------------|--------------------------------------------------|
| API server | FastAPI + uvicorn + slowapi (rate limiting) |
| Vector DB | Qdrant (async gRPC client) |
| Inference | Triton Inference Server (gRPC, ONNX models) |
| Message broker | RabbitMQ via `aio-pika` |
| Config | OmegaConf (YAML) + pydantic-settings (.env) |
| DI | dependency-injector |
| Logging | python-json-logger (structured JSON) |
| Linting | black, ruff, flake8 (wemake), mypy (strict), isort |
| Testing | pytest-asyncio, testcontainers, polyfactory |

______________________________________________________________________

## Quick start

```bash
make up        # start Qdrant + Triton + app via Docker Compose
make logs      # follow live application logs
make down      # stop everything
make run       # run FastAPI server locally (reads config/config.dev.yaml)
```

See `Makefile` for all available commands.

______________________________________________________________________

## Project structure

```
ml_service/
├── config/
│   ├── config.dev.yaml     # App, Qdrant, Triton, RabbitMQ settings
│   ├── logging.yaml        # JSON logging configuration
│   └── .env                # Secrets: RABBITMQ_USER, RABBITMQ_PASSWORD
│
├── src/
│   ├── app.py              # FastAPI entry point (lifespan, routers, rate limiting)
│   ├── configs/            # Pydantic config classes, constants, JSON logger
│   ├── schemas/            # Shared API schemas (ResponseBase, SearchResultItem)
│   │
│   ├── infrastructure/
│   │   ├── containers/     # DI hierarchy: RootContainer → DomainContainer → InfrastructureContainer
│   │   ├── db/qdrant/      # AsyncQdrantClient factory + BaseQdrantRepository + EmbeddingRepository
│   │   ├── triton/         # TritonClient (gRPC, embed())
│   │   └── rabbitmq/       # RabbitMQClient (connect, publish, consume)
│   │
│   └── domains/
│       ├── embeddings/     # POST / upsert, DELETE / delete
│       └── search/         # POST / semantic search
│
└── tests/
    ├── unit/               # Mocked infrastructure via DI overrides
    └── integration/        # Real Qdrant via testcontainers
```

______________________________________________________________________

## API endpoints

Base path: `/api/v1`

### Embeddings — `/api/v1/embeddings`

| Method | Path | Description |
|----------|-----------------------------|------------------------------------|
| `POST` | `/` | Embed texts and upsert to Qdrant |
| `DELETE` | `/{user_id}/{point_id}` | Delete a single embedding by ID |
| `DELETE` | `/{user_id}` | Delete all embeddings for a user |

**POST `/` request body:**

```json
{
  "user_id": "uuid",
  "texts": ["text1", "text2"]
}
```

### Search — `/api/v1/search`

| Method | Path | Description |
|--------|------|--------------------------|
| `POST` | `/` | Semantic similarity search |

**POST `/` request body:**

```json
{
  "user_id": "uuid",
  "query": "search query",
  "top_k": 10
}
```

______________________________________________________________________

## Configuration

Config is loaded on startup via `AppConfigs.init()`:

```yaml
# config/config.dev.yaml
app_host: "0.0.0.0"
app_port: 8001
log_level: "DEBUG"
workers_number: 1

qdrant:
  host: "localhost"
  port: 6333
  grpc_port: 6334
  collection: "documents"
  vector_size: 768
  hnsw_m: 16
  hnsw_ef: 100

triton:
  host: "localhost"
  grpc_port: 8001
  model_name: "embedding_model"
  input_name: "TEXT"
  output_name: "embedding"

rabbitmq:
  host: "localhost"
  port: 5672
  vhost: "ml"
```

Secrets in `config/.env`:

```env
RABBITMQ_USER=...
RABBITMQ_PASSWORD=...
```

Override config path: `CONFIG_PATH=config/config.prod.yaml make run`

______________________________________________________________________

## Multi-tenant isolation

All Qdrant operations are scoped to `user_id`. The repository layer stamps every
point with a `user_id` payload field and applies a `FieldCondition` filter on
every read and delete, so data from different users never leaks across requests.

______________________________________________________________________

## Testing

```bash
make tests        # run all tests
make unit         # unit tests only (no external services needed)
make integration  # integration tests (Qdrant via testcontainers)

# single file / single test
python3 -m pytest -vv tests/unit/test_embeddings_service.py
python3 -m pytest -vv tests/unit/test_embeddings_service.py::test_upsert_returns_count
```

- **Unit tests**: override DI containers with mocked `QdrantClient` and `TritonClient`.
- **Integration tests**: spin up real Qdrant via `testcontainers`; no Triton required (stub or fixtures).
- **Test data**: generated via `polyfactory` (`ModelFactory`) for Pydantic schemas.

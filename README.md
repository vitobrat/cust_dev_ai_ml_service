# ml_service

Internal ML microservice for the `cust_dev_ai` cust-dev interview simulation system.

## Responsibilities

- **Embeddings** — generates text embeddings via local models served on Triton Inference Server.
- **Vector search (RAG)** — stores and retrieves document embeddings from Qdrant vector database.
- **HTTP API** — exposes FastAPI endpoints consumed exclusively by `cust_dev_ai`.

## Stack

| Component | Technology |
|---|---|
| API server | FastAPI + uvicorn + slowapi |
| Vector DB | Qdrant |
| Inference | Triton Inference Server (gRPC) |
| Config | OmegaConf + pydantic-settings |
| DI | dependency-injector |
| Linting | black, ruff, flake8 (wemake), mypy, isort |

## Quick start

```bash
make up    # start Qdrant + Triton + app
make logs  # follow app logs
make down  # stop everything
```

See `Makefile` for all available commands.

# ── Переменные проекта ──────────────────────────────────────────
PYTHON       := python3
PYTEST       := $(PYTHON) -m pytest
DC_DEV       := docker compose -f ./docker/docker-compose.dev.yaml
ENV_FILE     := config/.env
MODEL_DIR    := models/embedding_model_multilingual_e5_small/1
MODEL_URL    := https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx

export PYTHONPATH := .

# ── Переменные docker ───────────────────────────────────────────
IMAGE_NAME   := ml_service_app
VERSION      := $(shell git rev-parse --short HEAD 2>/dev/null || echo "latest")
REGISTRY     := victorbratko
IMAGE_TAG    := $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

# Сервис для логов (по умолчанию app, можно переопределить: make logs s=triton)
s := app

.PHONY: help install-lint lint tests unit integration run \
        login build push \
        setup download-model up down restart clean logs logs-all

# ── Help ────────────────────────────────────────────────────────
## help: Показать это сообщение
help:
	@echo "Доступные команды:"
	@sed -n 's/^##//p' Makefile | column -t -s ':' | sed 's/^/  /'

# ── Quality ─────────────────────────────────────────────────────
## install-lint: Установить pre-commit хуки
install-lint:
	pre-commit install

## lint: Запустить линтеры на всех файлах
lint:
	pre-commit run --all-files

## tests: Запустить все тесты
tests:
	$(PYTEST) -vv tests

## unit: Запустить только unit-тесты
unit:
	$(PYTEST) -vv tests/unit

## integration: Запустить только integration-тесты
integration:
	$(PYTEST) -vv tests/integration

# ── Local development ───────────────────────────────────────────
## run: Развернуть локально fastapi сервер (нужны Triton, RabbitMQ, Qdrant)
run:
	$(PYTHON) src/app.py

# ── Model management ───────────────────────────────────────────
## download-model: Скачать ONNX-модель эмбеддингов из HuggingFace
download-model:
	@mkdir -p $(MODEL_DIR)
	@if [ -f $(MODEL_DIR)/model.onnx ]; then \
		echo "Model already exists at $(MODEL_DIR)/model.onnx — skipping."; \
	else \
		echo "Downloading ONNX model..."; \
		wget -q --show-progress -O $(MODEL_DIR)/model.onnx $(MODEL_URL); \
		echo "Done. Model saved to $(MODEL_DIR)/model.onnx"; \
	fi

# ── Docker registry ─────────────────────────────────────────────
## login: Авторизация в Docker Hub
login:
	docker login -u $(REGISTRY)

## build: Собрать образ приложения для прода
build:
	docker build \
		--target prod \
		-t $(IMAGE_TAG) \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		-f ./docker/Dockerfile .

## push: Отправить образ в Docker Hub
push: build
	docker push $(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

# ── Docker compose ──────────────────────────────────────────────
## setup: Первичная настройка проекта (модель + сборка образов)
setup: download-model
	@echo "Building Docker images..."
	$(DC_DEV) --env-file $(ENV_FILE) build
	@echo "Setup complete. Run 'make up' to start."

## up: Запустить dev-окружение
up:
	VERSION=$(VERSION) $(DC_DEV) --env-file $(ENV_FILE) up -d --build

## down: Остановить dev-окружение (данные сохраняются)
down:
	$(DC_DEV) --env-file $(ENV_FILE) down

## restart: Пересобрать и перезапустить только app
restart:
	$(DC_DEV) --env-file $(ENV_FILE) up -d --build --no-deps app

## clean: Остановить и удалить volumes (полный сброс данных)
clean:
	$(DC_DEV) --env-file $(ENV_FILE) down -v
	@echo "All containers stopped and volumes removed."

## logs: Логи сервиса (по умолчанию app, переопределить: make logs service=triton)
logs:
	$(DC_DEV) --env-file $(ENV_FILE) logs -f $(service)

## logs-all: Логи всех сервисов
logs-all:
	$(DC_DEV) --env-file $(ENV_FILE) logs -f

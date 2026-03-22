# Переменные проекта
PYTHONPATH_APP := .
PYTHON := python3
PYTEST := $(PYTHON) -m pytest
DC_DEV := docker compose -f ./docker/docker-compose.dev.yaml
ENV_FILE := config/.env

# Переменные docker
IMAGE_NAME = ml_service_app
VERSION = $(shell git rev-parse --short HEAD || echo "latest")
REGISTRY = victorbratko
IMAGE_TAG = $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

.PHONY: help install-lint lint tests unit integration run login build push up down logs

## help: Показать это сообщение
help:
	@echo "Доступные команды:"
	@sed -n 's/^##//p' Makefile | column -t -s ':' | sed 's/^/  /'

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

## run: Развернуть локально fastapi сервер
run:
	PYTHONPATH=$(PYTHONPATH_APP) $(PYTHON) src/app.py

## login: Авторизация в Docker Hub
login:
	docker login -u victorbratko

## build: Собрать образ приложения для прода
build:
	docker build \
		--target prod \
		-t $(IMAGE_TAG) \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		-f ./docker/Dockerfile .

## push: Отправить образ в Docker Hub/Registry
push:
	docker push $(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

## up: Запустить dev-окружение
up:
	VERSION=$(VERSION) $(DC_DEV) --env-file $(ENV_FILE) up -d --build

## down: Остановить dev-окружение
down:
	$(DC_DEV) --env-file $(ENV_FILE) down

## logs: Посмотреть логи приложения
logs:
	$(DC_DEV) --env-file $(ENV_FILE) logs -f app

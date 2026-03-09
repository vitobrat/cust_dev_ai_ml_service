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

.PHONY: help install-lint lint tests unit integration up down logs

## help: Показать это сообщение
help:
	@echo "Доступные команды:"
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'


## install-lint: Установить pre-commit хуки
install-lint:
	pre-commit install

## lint: Запустить линтеры на всех файлах
lint:
	pre-commit run --all-files

## test: Запустить все тесты
tests:
	$(PYTEST) -vv tests

## unit: Запустить только unit-тесты
unit:
	$(PYTEST) -vv tests/unit

## integration: Запустить только integration-тесты
integration:
	$(PYTEST) -vv tests/integration

## run: Развернуть локально fast api сервер согласно конфигурационному файлу
run:
	PYTHONPATH=$(PYTHONPATH_APP) python src/app.py

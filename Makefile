# Makefile for RLVR Tool Training POC
.PHONY: help install dev test lint format clean run-example docker-build docker-run

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

dev:  ## Install dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ -v --cov=rlvr_tool_training --cov-report=term-missing

lint:  ## Run linter
	ruff check .

format:  ## Format code
	ruff format .

typecheck:  ## Run type checker
	mypy rlvr_tool_training/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run-example:  ## Run the training example
	python -m rlvr_tool_training.examples.example_training

# Docker commands for service
docker-build:  ## Build the service container
	docker build -t tool-selector ./service

docker-run:  ## Run the service container
	docker run -p 8000:8000 \
		-e OPENPIPE_API_KEY=$${OPENPIPE_API_KEY} \
		-e OPENPIPE_MODEL_SLUG=$${OPENPIPE_MODEL_SLUG:-tool-selector-llama-8b-v1} \
		tool-selector

docker-compose-up:  ## Start service with docker-compose
	cd service && docker-compose up --build

# Service commands (local)
service-run:  ## Run service locally
	cd service && python app.py

service-dev:  ## Run service with hot reload
	cd service && uvicorn app:app --reload --host 0.0.0.0 --port 8000

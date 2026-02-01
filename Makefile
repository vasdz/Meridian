.PHONY: help install dev test lint format security docker clean

help:
	@echo "Meridian Development Commands"
	@echo ""
	@echo "  install     Install dependencies"
	@echo "  dev         Start development server"
	@echo "  test        Run tests"
	@echo "  lint        Run linters"
	@echo "  format      Format code"
	@echo "  security    Run security checks"
	@echo "  docker      Build and run with Docker"
	@echo "  clean       Clean build artifacts"

install:
	poetry install

dev:
	poetry run uvicorn meridian.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest tests -v --cov=src/meridian --cov-report=html

test-unit:
	poetry run pytest tests/unit -v

test-integration:
	poetry run pytest tests/integration -v

test-security:
	poetry run pytest tests/security -v

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run ruff check --fix src tests
	poetry run black src tests

security:
	poetry run bandit -r src -ll
	poetry run safety check
	poetry run python scripts/security/audit_dependencies.py

docker:
	docker-compose -f deployments/docker/docker-compose.yml up --build

docker-test:
	docker-compose -f deployments/docker/docker-compose.test.yml up --build --abort-on-container-exit

migrate:
	poetry run alembic upgrade head

migrate-create:
	poetry run alembic revision --autogenerate -m "$(message)"

seed:
	poetry run python scripts/seed_data.py

celery-worker:
	poetry run celery -A meridian.workers.celery_app worker --loglevel=info

celery-beat:
	poetry run celery -A meridian.workers.celery_app beat --loglevel=info

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov coverage.xml .coverage
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


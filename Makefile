.PHONY: install test test-unit test-integration lint lint-fix cover

install:
	pip install -e ".[dev]"

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

cover:
	pytest --cov=opl --cov-report=term-missing --cov-fail-under=90

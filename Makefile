.PHONY: install test test-unit test-integration lint cover

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

cover:
	pytest --cov=opl --cov-report=term-missing --cov-fail-under=90

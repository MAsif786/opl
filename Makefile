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

# CLI Shortcuts
run-train:
	opl train --entity SKU_WH_1

run-predict:
	opl predict --entity SKU_WH_1 --horizon 7

run-simulate:
	opl simulate --entity SKU_WH_1 --action reorder --value 100 --horizon 14

cover:
	pytest --cov=opl --cov-report=term-missing --cov-fail-under=90

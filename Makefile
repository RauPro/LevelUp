# Makefile for LevelUp project

.PHONY: venv install clean lint typecheck check

venv:
	uv venv .venv

install:
	uv pip sync pyproject.toml

clean:
	rm -rf .venv __pycache__ *.pyc *.pyo *.pyd *.log .pytest_cache .mypy_cache

lint:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy .

check: lint typecheck

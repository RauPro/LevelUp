# Makefile for LevelUp project

.PHONY: venv install clean

venv:
	uv venv .venv

install:
	uv pip install -r requirements.txt

clean:
	rm -rf .venv __pycache__ *.pyc *.pyo *.pyd *.log .pytest_cache .mypy_cache .coverage .env

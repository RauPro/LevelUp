# Makefile for LevelUp project

.PHONY: venv install clean

venv:
	uv venv .venv

install:
	uv pip sync pyproject.toml
	@echo ""
	@echo "✅ Dependencies installed!"
	@echo "👉 To activate the virtual environment, run:"
	@echo "   source .venv/bin/activate   # for macOS/Linux/zsh/bash"
	@echo "   .venv\\Scripts\\activate    # for Windows"
	@echo ""

clean:
	rm -rf .venv __pycache__ *.pyc *.pyo *.pyd *.log .pytest_cache .mypy_cache

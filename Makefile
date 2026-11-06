.PHONY: setup test lint
setup:
	bash setup.sh
test:
	.venv/bin/python -m pytest -q
lint:
	.venv/bin/ruff check src tests scripts

include reproduce.mk

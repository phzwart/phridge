.PHONY: venv test schema-docs

venv:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

schema-docs:
	.venv/bin/python scripts/generate_models.py --all

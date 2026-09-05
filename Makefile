.PHONY: venv test test-sf-gpu schema-docs demo-restraints example-restraints example-sf-gradients chem-data

# Prefer the conda cctbx-base env when present (aarch64-friendly).
CONDA_ENV ?= $(HOME)/miniforge3/envs/phridge-cctbx
CONDA ?= $(HOME)/miniforge3/bin/conda
ifeq ($(wildcard $(CONDA_ENV)/bin/python),)
PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
else
PYTHON ?= $(CONDA_ENV)/bin/python
PIP ?= $(CONDA_ENV)/bin/pip
endif

venv:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"

chem-data:
	$(CONDA) install -y -n phridge-cctbx -c chem_data chem_data

test:
	$(PYTHON) -m pytest -q -m "not gpu and not slow"

test-sf-gpu:
	$(PYTHON) -m pytest -s -m "gpu and slow" tests/test_sf_gradients_gpu.py

schema-docs:
	$(PYTHON) scripts/generate_models.py --all

example-restraints demo-restraints:
	$(PYTHON) examples/restraint_minimization.py

example-sf-gradients:
	$(PYTHON) examples/sf_gradient_benchmark.py

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = gelos
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	uv pip install --no-deps -e .




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@echo ">>> This project installs into a Docker image (see Dockerfile + compose.yml)."
	@echo ">>> For local dev: 'uv venv && source .venv/bin/activate && make requirements'."




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# gelos-lc sits next to the main gelos checkout; resolve it through git's common dir so the
# target also works from a worktree under .git/wt/.
GELOS_MAIN := $(shell dirname "$$(git rev-parse --path-format=absolute --git-common-dir)")
GELOS_LC_PATH ?= $(abspath $(GELOS_MAIN)/../gelos-lc)
GELOS_PATH ?= $(CURDIR)
SMOKE_CONFIG ?= configs/smoke/exp001_prithvi300_smoke.yaml
# SMOKE_CONFIG is looked up in the gelos checkout first (mounted at /app/gelos), then in gelos-lc.
SMOKE_CONFIG_IN_APP = $(if $(wildcard $(GELOS_PATH)/$(SMOKE_CONFIG)),/app/gelos/$(SMOKE_CONFIG),$(SMOKE_CONFIG))

## Smoke-test a gelos checkout in gelos-lc (generation + analysis; GELOS_PATH=<checkout>)
.PHONY: downstream-check
downstream-check:
	cd "$(GELOS_LC_PATH)" && export GELOS_MODULE_PATH="$(abspath $(GELOS_PATH))/" && \
	docker compose run --rm --remove-orphans dev \
		python -m gelos.generation -y $(SMOKE_CONFIG_IN_APP) --overwrite && \
	docker compose run --rm --remove-orphans dev \
		python -m gelos.analysis -y $(SMOKE_CONFIG_IN_APP) --overwrite



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

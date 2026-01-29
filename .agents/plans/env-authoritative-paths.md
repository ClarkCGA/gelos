# Feature: env-authoritative-paths

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Make `.env` (and environment variables) the single authority for all machine-specific paths and configuration, removing any reliance on `config.py` location or repository layout assumptions. This ensures GELOS behaves consistently whether installed via pip, run locally, or run in Docker.

## User Story

As a GELOS user or maintainer
I want all paths and machine-specific settings to come from `.env` or environment variables
So that GELOS runs consistently across local, Docker, and pip-installed contexts without repo-relative assumptions.

## Problem Statement

Current path handling derives project roots from the package location (`Path(__file__)`) and assumes a repo layout plus docker volume mappings. This breaks for pip installs or alternative deployment layouts and makes it unclear where data lives unless the repo structure is preserved.

## Solution Statement

Centralize all machine-specific configuration in environment variables loaded from `.env`, with explicit defaults and clear validation. Update `gelos/config.py` to derive all paths from environment variables. Update CLIs and docs to reference these environment variables, and adjust Docker/Compose to use the same names.

## Feature Metadata

**Feature Type**: Refactor/Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: `gelos/config.py`, CLI scripts, Docker/Compose, docs, tests
**Dependencies**: `python-dotenv` (already in [pyproject.toml](pyproject.toml))

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `gelos/config.py` (lines 1-39) - Current env loading and repo-relative path derivation.
- `gelos/embedding_generation.py` (lines 1-104) - Uses `PROJ_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR` for CLI defaults.
- `gelos/embedding_transformation.py` (lines 1-116) - Uses `RAW_DATA_DIR`, `FIGURES_DIR`, `PROJ_ROOT` for I/O paths.
- `compose.yml` (lines 1-33) - Defines env_file and volume mounts that assume repo-relative `/app/data/*`.
- `README.md` (lines 1-40) - Documents repo structure and current assumptions.
- `Makefile` (lines 1-90) - Lint/test commands and dependency install flow.

### New Files to Create

- (Optional) `gelos/settings.py` - Central settings dataclass or typed settings loader.

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- https://github.com/theskumar/python-dotenv#readme
  - Why: Confirms `load_dotenv` behavior and environment override patterns.
- https://docs.python.org/3/library/os.html#os.getenv
  - Why: Environment variable retrieval and default handling.
- https://packaging.python.org/en/latest/tutorials/packaging-projects/
  - Why: Best practices for pip-installed packages and runtime paths.

### Patterns to Follow

**Naming Conventions:**
- Snake_case module-level constants in [gelos/config.py](gelos/config.py).

**Logging Pattern:**
- `loguru` logging as used in [gelos/config.py](gelos/config.py#L4).

**Other Relevant Patterns:**
- CLI pattern with Typer in [gelos/embedding_generation.py](gelos/embedding_generation.py#L79) and [gelos/embedding_transformation.py](gelos/embedding_transformation.py#L87).

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Define a canonical environment variable schema and a single configuration loader that reads from `.env` (and system env) without relying on repo paths.

**Tasks:**

- Decide canonical env var names (example):
  - `GELOS_DATA_ROOT`, `GELOS_RAW_DIR`, `GELOS_PROCESSED_DIR`, `GELOS_EXTERNAL_DIR`, `GELOS_REPORTS_DIR`, `GELOS_FIGURES_DIR`, `GELOS_MODELS_DIR`, `GELOS_DATA_VERSION`.
- Decide precedence rules (explicit env var overrides `.env` values).
- Decide default behavior when env vars are missing (e.g., `Path.cwd()` or raise clear error).

### Phase 2: Core Implementation

Refactor configuration to load paths from environment variables and provide normalized `Path` objects.

**Tasks:**

- Update `gelos/config.py` (or create `gelos/settings.py`) to:
  - `load_dotenv()` with optional `GELOS_ENV_FILE` override.
  - Normalize all path env vars to `Path` objects.
  - Stop deriving `PROJ_ROOT` from `config.py` location.
  - Provide validation and clear errors if required paths are absent.
- Update any direct imports of `PROJ_ROOT` and derived directories to use the new settings.

### Phase 3: Integration

Update CLIs, Docker/Compose, and docs to align on the new env-based paths.

**Tasks:**

- Update `gelos/embedding_generation.py` and `gelos/embedding_transformation.py` to use new config values and avoid `PROJ_ROOT`.
- Update `compose.yml` to set and consume the new env vars for volume mapping and container env (no repo-relative assumptions).
- Update README and docs with:
  - Required `.env` keys.
  - Examples for local and Docker usage.
  - How to override paths via env vars.

### Phase 4: Testing & Validation

Update and extend tests to reflect env-based configuration.

**Tasks:**

- Update `tests/test_data.py` fixtures to set env vars (or patch config) so tests do not rely on repo structure.
- Add tests for missing/invalid path env vars to ensure friendly error messages.

---

## STEP-BY-STEP TASKS

### UPDATE gelos/config.py

- **IMPLEMENT**: Replace repo-relative path derivations with env-based path parsing.
- **PATTERN**: Use existing module-level constants style (snake_case) in `gelos/config.py`.
- **IMPORTS**: `os`, `pathlib.Path`, `dotenv.load_dotenv`, `loguru.logger`.
- **GOTCHA**: Avoid `Path(__file__)` for runtime paths. Ensure `.env` path resolution is consistent for pip installs.
- **VALIDATE**: `python -m pytest tests`

### UPDATE gelos/embedding_generation.py

- **IMPLEMENT**: Swap `PROJ_ROOT` and `RAW_DATA_DIR` usage to env-based config values; ensure yaml config discovery uses a non-repo-dependent path or accepts explicit CLI arg.
- **PATTERN**: Keep Typer CLI structure intact.
- **IMPORTS**: Use config values from updated config/settings module.
- **GOTCHA**: `yaml_config_directory` currently uses `PROJ_ROOT`; replace with env var or CLI-provided directory.
- **VALIDATE**: `python -m pytest tests`

### UPDATE gelos/embedding_transformation.py

- **IMPLEMENT**: Replace repo-relative path usage for raw data, figures, and yaml config directory with env-based paths.
- **PATTERN**: Maintain extraction/plot flow as-is.
- **IMPORTS**: Updated config values.
- **GOTCHA**: `gelos_chip_tracker.geojson` path should be derived from env-based raw root.
- **VALIDATE**: `python -m pytest tests`

### UPDATE compose.yml

- **IMPLEMENT**: Define and use canonical env var names in both `env_file` and `volumes` sections.
- **PATTERN**: Keep existing service structure.
- **GOTCHA**: Compose variable substitution uses host env; ensure `.env` values are exported.
- **VALIDATE**: `docker compose config` (optional, manual)

### UPDATE README.md and docs

- **IMPLEMENT**: Document required `.env` keys and examples for local and Docker usage.
- **PATTERN**: Keep cookiecutter structure description but add config section.
- **VALIDATE**: `mkdocs build` (optional if docs are updated)

### UPDATE tests/test_data.py

- **IMPLEMENT**: Set env vars in test fixtures so config resolves paths without repo assumptions.
- **PATTERN**: Use existing pytest fixture style.
- **GOTCHA**: Avoid global import-time config evaluation in tests by setting env before config import.
- **VALIDATE**: `python -m pytest tests`

---

## TESTING STRATEGY

### Unit Tests

- Validate config path resolution for:
  - Full env-based configuration.
  - Missing required env vars (clear error).
  - Optional overrides (e.g., `GELOS_FIGURES_DIR`).

### Integration Tests

- Run existing tests with env vars set via fixture to ensure no regressions.

### Edge Cases

- `.env` missing, but env vars provided by host.
- `.env` present but missing required keys.
- Paths containing spaces or trailing slashes.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

- `make lint`

### Level 2: Unit Tests

- `make test`

### Level 3: Integration Tests

- (None defined yet)

### Level 4: Manual Validation

- Run embedding generation CLI with `--yaml-path` using env-based paths.
- Run embedding transformation CLI with env-based paths.

---

## ACCEPTANCE CRITERIA

- [ ] No runtime path is derived from the `config.py` file location.
- [ ] All machine-specific paths are resolved from `.env` or environment variables.
- [ ] CLIs run successfully with env-only configuration.
- [ ] Docker/Compose uses the same env vars without repo-relative assumptions.
- [ ] Tests pass with env-based configuration.
- [ ] Documentation clearly describes required env vars and examples.

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms feature works
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

- If a new `Settings` module is introduced, keep `gelos/config.py` as the public import surface to avoid breaking downstream imports.
- Consider `GELOS_ENV_FILE` for pointing to a `.env` outside the working directory.
- For pip installs, avoid relying on repo-local `gelos/configs` unless a new env var like `GELOS_CONFIG_DIR` is introduced.

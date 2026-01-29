# Feature: Simplify config path loading

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Simplify `gelos.config` by removing helper functions and derived-path logic, and require explicit `.env`/environment variables for `GELOS_RAW_DIR`, `GELOS_INTERIM_DIR`, `GELOS_PROCESSED_DIR`, and `GELOS_EXTERNAL_DIR`.

## User Story

As a user
I want configuration to assume explicit per-path environment variables
So that setup is simpler and behavior is predictable without implicit derivations.

## Problem Statement

`gelos.config` currently implements layered path derivation and helper functions, which adds complexity and makes required inputs ambiguous. The request is to assume the presence of explicit per-path variables and remove helper function logic.

## Solution Statement

Flatten configuration logic in `gelos.config` by directly reading required path environment variables and validating them without helper functions or derived values. Update tests and documentation to reflect required explicit paths.

## Feature Metadata

**Feature Type**: Refactor
**Estimated Complexity**: Low
**Primary Systems Affected**: `gelos.config`, tests, docs
**Dependencies**: None

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- [gelos/config.py](gelos/config.py#L1-L83) - Current path-loading logic, helpers, and validation.
- [tests/test_config.py](tests/test_config.py#L1-L44) - Existing tests for path derivation and missing paths.
- [tests/test_data.py](tests/test_data.py#L11-L69) - Sets `GELOS_DATA_ROOT` in fixtures; will need updates for explicit paths.
- [gelos/embedding_generation.py](gelos/embedding_generation.py#L11-L99) - Uses `RAW_DATA_DIR` and `PROCESSED_DATA_DIR` from config.
- [gelos/embedding_transformation.py](gelos/embedding_transformation.py#L9-L111) - Uses `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, and `FIGURES_DIR`.
- [README.md](README.md#L9-L42) - Documents current environment variables and `GELOS_DATA_ROOT` fallback.
- [docs/docs/getting-started.md](docs/docs/getting-started.md#L8-L31) - Documents environment configuration.

### New Files to Create

- None.

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- No external docs required.

### Patterns to Follow

**Error Handling:** `ValueError` when required environment variables are missing, as in [gelos/config.py](gelos/config.py#L30-L37).

**Logging Pattern:** `loguru` info log after configuration load, as in [gelos/config.py](gelos/config.py#L70-L73).

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Define the desired required variables and update documentation and tests accordingly.

**Tasks:**

- Identify the exact list of required path environment variables.
- Adjust tests to set required vars explicitly.
- Update documentation to remove `GELOS_DATA_ROOT` as a fallback.

### Phase 2: Core Implementation

Simplify configuration logic in `gelos.config`.

**Tasks:**

- Remove helper functions (`_parse_path`, `_env_path`, `_derive_child`, `_require_paths`).
- Replace derived paths with direct `Path(...).expanduser().resolve()` reads for required vars.
- Keep `.env` loading and `GELOS_DATA_VERSION` behavior consistent.

### Phase 3: Integration

Ensure all dependent modules and tests still work with explicit paths.

**Tasks:**

- Update test fixtures to set the explicit path variables.
- Ensure error messages match updated validation behavior.
- Verify docs align with new expectations.

### Phase 4: Testing & Validation

Run lint and tests to confirm no regressions.

**Tasks:**

- Run lint via Makefile (or Docker Compose if preferred).
- Run pytest.

---

## STEP-BY-STEP TASKS

### REFACTOR gelos/config.py

- **IMPLEMENT**: Inline reading of required path variables (`GELOS_RAW_DIR`, `GELOS_INTERIM_DIR`, `GELOS_PROCESSED_DIR`, `GELOS_EXTERNAL_DIR`) and create `Path` objects directly.
- **REMOVE**: Helper functions and derived-path fallback logic (no `GELOS_DATA_ROOT` path derivation).
- **ERROR HANDLING**: `ValueError` if any required path var is missing or empty.
- **LOGGING**: Keep `logger.info` summary after load.
- **VALIDATE**: `make lint`

### UPDATE tests/test_config.py

- **IMPLEMENT**: Adjust tests to set explicit path env vars instead of relying on `GELOS_DATA_ROOT` fallback.
- **UPDATE**: Update expected error message if the missing-path error changes.
- **VALIDATE**: `make test`

### UPDATE tests/test_data.py

- **IMPLEMENT**: Set required path env vars in the fixture (`GELOS_RAW_DIR`, `GELOS_INTERIM_DIR`, `GELOS_PROCESSED_DIR`, `GELOS_EXTERNAL_DIR`).
- **GOTCHA**: Ensure paths point to directories created under `tmp_path` to avoid filesystem errors.
- **VALIDATE**: `make test`

### UPDATE README.md

- **UPDATE**: Remove `GELOS_DATA_ROOT` as a recommended/required fallback; state that explicit path variables are required.
- **VALIDATE**: `make lint`

### UPDATE docs/docs/getting-started.md

- **UPDATE**: Align environment configuration section with explicit path variable requirement.
- **VALIDATE**: `make lint`

---

## TESTING STRATEGY

### Unit Tests

- Update existing unit tests in [tests/test_config.py](tests/test_config.py#L1-L44) to assert correct behavior with explicit paths.

### Integration Tests

- No new integration tests needed for this refactor.

### Edge Cases

- Missing one or more required path vars.
- Empty string values in the `.env` file.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

- `make lint`

### Level 2: Unit Tests

- `make test`

### Level 3: Integration Tests

- Not applicable.

### Level 4: Manual Validation

- Verify `.env` with explicit `GELOS_*_DIR` paths loads without errors.

### Level 5: Additional Validation (Optional)

- If using Docker Compose: `docker compose up test --command make lint` and `docker compose up test --command make test`.

---

## ACCEPTANCE CRITERIA

- [ ] `gelos.config` requires explicit `GELOS_RAW_DIR`, `GELOS_INTERIM_DIR`, `GELOS_PROCESSED_DIR`, and `GELOS_EXTERNAL_DIR`.
- [ ] No derived-path logic from `GELOS_DATA_ROOT` remains.
- [ ] Tests updated and passing.
- [ ] Documentation reflects the explicit path requirement.
- [ ] Lint passes with no new issues.

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] No linting or type checking errors
- [ ] Manual testing confirms config behavior
- [ ] Acceptance criteria all met

---

## NOTES

- Keep `.env` loading and `GELOS_ENV_FILE` behavior unchanged unless explicitly requested.
- Avoid changing public APIs beyond path-loading expectations.

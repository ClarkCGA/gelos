Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

## Environment Configuration

GELOS loads all machine-specific paths from environment variables (optionally via `.env`).

Required path variables (set either `GELOS_DATA_ROOT` or each specific path):

- `GELOS_DATA_ROOT`
- `GELOS_RAW_DIR`
- `GELOS_INTERIM_DIR`
- `GELOS_PROCESSED_DIR`
- `GELOS_EXTERNAL_DIR`

Optional project paths:

- `GELOS_PROJECT_ROOT`
- `GELOS_CONFIG_DIR`
- `GELOS_MODELS_DIR`
- `GELOS_REPORTS_DIR`
- `GELOS_FIGURES_DIR`

Optional overrides:

- `GELOS_DATA_VERSION` (defaults to `v0.50.0`)
- `GELOS_ENV_FILE` (path to a custom `.env`)

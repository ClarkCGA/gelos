Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

## Environment Configuration

GELOS loads all machine-specific paths from environment variables (optionally via `.env`).

Required path variables:

- `EXTERNAL_DATA_DIR`
- `RAW_DATA_DIR`
- `INTERIM_DATA_DIR`
- `PROCESSED_DATA_DIR`
- `PROJECT_ROOT`

Docker Compose variables:

- `JUPYTER_HOST_PORT`

Example `.env`:

```
EXTERNAL_DATA_DIR=/data/gelos/external/
INTERIM_DATA_DIR=/data/gelos/interim/
PROCESSED_DATA_DIR=/data/gelos/processed/
RAW_DATA_DIR=/data/gelos/raw/
PROJECT_ROOT=/workspace/gelos/
JUPYTER_HOST_PORT=8888
```

## Next Steps

Now that your environment is configured, check out [Starting a New Embeddings Project](starting-new-project.md) to learn how to prepare your dataset and run the pipeline.

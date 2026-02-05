Getting started
===============

## Environment Configuration

If you use Docker Compose, you can set variables like these for host paths and ports:

- `EXTERNAL_DATA_DIR`
- `RAW_DATA_DIR`
- `INTERIM_DATA_DIR`
- `PROCESSED_DATA_DIR`
- `PROJECT_ROOT`
- `JUPYTER_HOST_PORT`

Example `.env` used by Docker Compose:

```
EXTERNAL_DATA_DIR=/data/gelos/external/
INTERIM_DATA_DIR=/data/gelos/interim/
PROCESSED_DATA_DIR=/data/gelos/processed/
RAW_DATA_DIR=/data/gelos/raw/
PROJECT_ROOT=/workspace/gelos/
JUPYTER_HOST_PORT=8888
```

## Dataset Paths and Configs

Dataset paths are not pulled from the YAML config. GELOS expects explicit paths to be passed via
CLI arguments or function parameters (for example, `--raw-data-dir` and `--processed-data-dir`).
Keep dataset paths in your execution command or wrapper script, and keep YAML configs focused on
model and data configuration.

## Next Steps

Now that your environment is configured, check out [Starting a New Embeddings Project](starting-new-project.md) to learn how to prepare your dataset and run the pipeline.

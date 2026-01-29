# GELOS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for Geospatial Exploration of Latent Observation Space (GELOS)

## Configuration (Environment Variables)

GELOS paths are configured exclusively through environment variables (optionally loaded from a `.env` file).

Data and Project Paths:
- `EXTERNAL_DIR`
- `RAW_DIR`
- `INTERIM_DIR`
- `PROCESSED_DIR`
- `PROJECT_ROOT`

Docker Compose Variables:
- `JUPYTER_HOST_PORT`

Example `.env`:
```
EXTERNAL_DATA_DIR=/data/gelos/external/    <- Data from third party sources.
INTERIM_DATA_DIR=/data/gelos/interim/      <- Intermediate data that has been transformed.
PROCESSED_DATA_DIR=/data/gelos/processed/  <- The final, canonical data sets for modeling.
RAW_DATA_DIR=/data/gelos/raw/              <- The original, immutable data dump.
PROJECT_ROOT=/workspace/gelos/
JUPYTER_HOST_PORT=8888
```

Docker/Compose uses the same paths for volume mounts, and will mount them to identical directories in the container. Use absolute paths that exist on the host.

## Project Organization

```
├── LICENSE            <- Open-source license
│
├── Makefile           <- Makefile with convenience commands
│
├── README.md          <- The top-level README for developers using this project.
│
├── docs               <- MkDocs site source; see www.mkdocs.org for details
│
├── models             <- Model definitions and helpers (e.g., prithvi_eo_v2.py)
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         gelos and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
└── gelos   <- Source code for use in this project.
    │
    ├── __init__.py                 <- Makes gelos a Python module
    │
    ├── config.py                   <- Paths, environment variables, and logging setup
    │
    ├── embedding_extraction.py     <- Utilities to sample parquet files and slice embeddings
    │
    ├── embedding_generation.py     <- Typer CLI to run Lightning/Terratorch embedding generation
    │
    ├── embedding_transformation.py <- Typer CLI to extract embeddings, run t-SNE, and plot results
    │
    ├── gelosdatamodule.py          <- Lightning DataModule wiring the GELOS dataset for inference
    │
    ├── gelosdataset.py             <- Multimodal geospatial chip dataset with perturbation and concat options
    │
    ├── plotting.py                 <- Plotting helpers (t-SNE scatter, legend patches, formatting)
    │
    └── tsne_transform.py           <- t-SNE computation and CSV export helpers
```

--------


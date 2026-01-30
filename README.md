# GELOS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for Geospatial Exploration of Latent Observation Space (GELOS)

## Installing

Currently, as the repo is private to Clark CGA, it can be installed from the repository using pip or uv with the appropriate credentials.

Installing from remote repo directly:
```
pip install git+https://<your-github-personal-access-token>@github.com/ClarkCGA/gelos
```

Cloning and installing:
```
git clone https://github.com/ClarkCGA/gelos
pip install gelos/
```

## Running gelos scripts

GELOS CLI commands require explicit path arguments. For example, to run all yamls in a directory:

```
python gelos/embedding_generation.py \
    --raw-data-dir /data/gelos/raw \
    --processed-data-dir /data/gelos/processed \
    --config-dir /workspace/gelos/configs
```

GELOS can also be imported as modules into python scripts, as in this example which generates embeddings according to one config yaml:

```python
from pathlib import Path
from gelos.embedding_generation import generate_embeddings

config_path = Path("configs/prithvieov2300.yaml")
raw_data_dir = Path("/abs/path/to/data/raw")
processed_data_dir = Path("/abs/path/to/data/processed")

generate_embeddings(
  config_path,
  raw_data_dir=raw_data_dir,
  processed_data_dir=processed_data_dir,
)
```

Docker/Compose can still use a `.env` file for volume mounts and ports, but the library does not auto-load environment variables. Use absolute paths that exist on the host.

For details on setting up a new dataset, see [Starting a New Embeddings Project](docs/docs/starting-new-project.md).

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


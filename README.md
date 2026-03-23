# GELOS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for Geospatial Exploration of Latent Observation Space (GELOS)

## Installing

Pin to a specific release tag:

```
pip install git+https://github.com/ClarkCGA/gelos@v1.0.0
```

Or with pixi:

```toml
[tool.pixi.pypi-dependencies]
gelos = {git = "https://github.com/ClarkCGA/gelos.git", tag = "v1.0.0"}
```

Check [Releases](https://github.com/ClarkCGA/gelos/releases) for available versions and what changed in each. When updating, review the [CHANGELOG](CHANGELOG.md) and bump the tag in your `pyproject.toml` or `requirements.txt`.

## Adapting GELOS to a new dataset

GELOS is intended to be a reusable embedding generation, exploration, and analysis pipeline for any dataset which has categorical information at the chip level. A full example implementation for a land cover dataset can be found at https://github.com/ClarkCGA/gelos-lc.

In order to apply this pipeline to a new dataset:

### 1. Start a separate repository for your project

Create a new repository that installs GELOS as a dependency (see [Installing](#installing) above).

### 2. Organize your data so your dataset class can load it

GELOS does not prescribe a specific metadata format. The only hard requirement is that your dataset class can load the right source files for each chip and return a `file_id` that can be used to cross-reference embeddings with the chip's category label during analysis.

How you achieve this is up to you. One pattern that works well is a **chip tracker** — a GeoJSON or CSV file that indexes your chips with columns for file paths, category labels, and a unique ID. The [gelos-lc](https://github.com/ClarkCGA/gelos-lc) implementation uses a `gelos_chip_tracker.geojson` with this structure:

```json
{
  "id": 1,
  "s2l2a_paths": "s2l2a_000001_20230218.tif,s2l2a_000001_20230419.tif",
  "dem_paths": "dem_000001.tif",
  "lulc": 2,
  "geometry": { "type": "Polygon", "coordinates": [...] }
}
```

But you could equally use a directory naming convention, a CSV manifest, or any other approach that lets your dataset class resolve chip index → file paths and chip ID.

### 3. Write a custom dataset class

Create a subclass of `gelos.gelosdataset.GELOSDataSet`. A barebones reference implementation is provided as `ExampleGELOSDataSet` in [tests/test_data.py](tests/test_data.py).

Your subclass must define:

- **`all_band_names`**: A dict mapping each sensor name to its list of band names (e.g., `{"S2L2A": ["blue", "green", "red", ...], "DEM": ["DEM"]}`)
- **`__len__(self)`**: Return the number of chips in your dataset
- **`_get_file_paths(self, index, sensor)`**: Return a list of `Path` objects for the given chip index and sensor
- **`_load_file(self, path, band_indices)`**: Load a single GeoTIFF and return a NumPy array with shape `[H, W, C]`, selecting only the requested band indices
- **`_get_sample_id(self, index)`**: Return a `(filename, file_id)` tuple — `filename` names the output parquet record, `file_id` is stored as metadata within it

Optionally, define per-band **`means`** and **`stds`** dicts on your class. `GELOSDataModule` will fall back to these for normalization if statistics are not passed explicitly in the YAML config. You can compute these by iterating through your dataset with zero-initialized stats (see `calculate_statistics.py` in [gelos-lc](https://github.com/ClarkCGA/gelos-lc) for an example).

### 4. Create YAML experiment configs

YAML configs drive both embedding generation and analysis. Each config specifies the data source, model, extraction strategies, and analysis steps. A config has four main sections:

- **`data`**: Your dataset class, which bands to load per sensor, transforms, and optional perturbation/repetition settings
- **`model`**: The TerraTorch model to use for embedding generation and how to extract embeddings from it
- **`embedding_extraction_strategies`**: User-defined strategies for slicing embedding tokens and running transforms, plots, and models on them
- **`style`**: Category-to-color mappings for visualization

See [tests/fixtures/example_config.yaml](tests/fixtures/example_config.yaml) for a minimal example, and the [Configuration Reference](docs/docs/configuration.md) for a fully annotated config with all available options.

### 5. Run the pipeline

**Generate embeddings** from your configs:

```bash
python -m gelos.generation \
    --raw-data-dir /path/to/data/raw \
    --embedding-dir /path/to/data/interim \
    --config-dir /path/to/configs
```

**Run analysis** (extraction, transforms, plots, and models):

```bash
python -m gelos.analysis \
    --raw-data-dir /path/to/data/raw \
    --embedding-dir /path/to/data/interim \
    --processed-data-dir /path/to/data/processed \
    --figures-base-dir /path/to/figures \
    --config-dir /path/to/configs
```

Both commands also work as Python imports — see the [Getting Started guide](docs/docs/getting-started.md) for function-level usage.

Docker/Compose can still use a `.env` file for volume mounts and ports, but the library does not auto-load environment variables. Use absolute paths that exist on the host.


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
    ├── analysis.py                 <- Typer CLI + config-driven pipeline: transforms, plots, and models
    │
    ├── extraction.py               <- Utilities to sample parquet files and slice embeddings
    │
    ├── generation.py               <- Typer CLI to run Lightning/Terratorch embedding generation
    │
    ├── gelosdatamodule.py          <- Lightning DataModule wiring the GELOS dataset for inference
    │
    ├── gelosdataset.py             <- Multimodal geospatial chip dataset with perturbation and concat options
    │
    ├── models.py                   <- MODELS registry: KNN, linear probe, random forest classification
    │
    ├── plotting.py                 <- PLOTS registry: t-SNE scatter plots colored by land cover class
    │
    └── transforms.py               <- TRANSFORMS registry: t-SNE, PCA, and CSV export helpers
```

--------


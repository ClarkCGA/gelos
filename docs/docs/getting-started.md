# Getting Started

## Installation

Pin to a specific release tag to avoid unexpected breakage:

```
pip install git+https://github.com/ClarkCGA/gelos@v1.0.0
```

Or with pixi, add it as a pypi dependency in your `pyproject.toml`:

```toml
[tool.pixi.pypi-dependencies]
gelos = {git = "https://github.com/ClarkCGA/gelos.git", tag = "v1.0.0"}
```

Check [Releases](https://github.com/ClarkCGA/gelos/releases) for available versions and changelogs. When updating, bump the tag in your `pyproject.toml` or `requirements.txt` and re-run `pixi install` or `pip install`.

## Environment Setup

GELOS does not auto-load environment variables. All paths are passed explicitly via CLI arguments or function parameters (e.g., `--raw-data-dir`, `--embedding-dir`).

If you use Docker Compose, you can configure host paths and ports via a `.env` file for volume mounts:

```
RAW_DATA_DIR=/data/gelos/raw/
INTERIM_DATA_DIR=/data/gelos/interim/
PROCESSED_DATA_DIR=/data/gelos/processed/
PROJECT_ROOT=/workspace/gelos/
JUPYTER_HOST_PORT=8888
```

Keep YAML configs focused on model and data configuration, not paths.

## Adapting GELOS to a New Dataset

GELOS is a reusable pipeline for any dataset with categorical information at the chip level. A full example implementation for a land cover dataset is available at [gelos-lc](https://github.com/ClarkCGA/gelos-lc).

### 1. Write a custom dataset class

Create a subclass of `gelos.gelosdataset.GELOSDataSet`. A barebones reference implementation is provided as `ExampleGELOSDataSet` in `tests/test_data.py`.

`GELOSDataSet` ensures outputs are consistent with what the TerraTorch embedding generation pipeline requires. It also provides reusable methods for noise ablation and band repetition, which is sometimes needed for dataset conformity with specific models. For example, when yearly data such as DEM is passed alongside multitemporal data to Terramind V1 Base, the model requires that the DEM is repeated so that there are an equal number of timesteps for all data sources.

Your subclass must define:

- **`all_band_names`**: A dict mapping each sensor name to its ordered list of band names (e.g., `{"S2L2A": ["blue", "green", "red", ...], "DEM": ["DEM"]}`)
- **`__len__(self) -> int`**: Return the number of chips in your dataset
- **`_get_file_paths(self, index: int, sensor: str) -> list[Path]`**: Return paths to load for a given chip index and sensor. Each path corresponds to one timestep.
- **`_load_file(self, path: Path, band_indices: list[int]) -> np.ndarray`**: Load a single file and return an array with shape `[H, W, C]`, selecting only the requested band indices. Band indices are determined programmatically based on the YAML config `bands` and your `all_band_names` dict.
- **`_get_sample_id(self, index: int) -> tuple[str, Any]`**: Return `(filename, file_id)` for the chip at the given index. `filename` names the output parquet record, and `file_id` is stored as metadata within the parquet and used to cross-reference embeddings with category labels during analysis.

Optionally, define per-band **`means`** and **`stds`** dicts on your class. `GELOSDataModule` will fall back to these for normalization if statistics are not passed explicitly in the YAML config. You can compute these by iterating through your dataset with zero-initialized stats — see `calculate_statistics.py` in [gelos-lc](https://github.com/ClarkCGA/gelos-lc) for an example.

If you use a custom `__init__()`, call `super().__init__()` to initialize band validation, transforms, and perturbation logic.

### 2. Organize your data

GELOS does not prescribe a specific metadata format. The only hard requirement is that your dataset class can load the right source files for each chip and return a `file_id` that can be used to cross-reference embeddings with the chip's category label during analysis.

How you achieve this is up to you. One pattern that works well is a chip tracker — a GeoJSON or CSV file that indexes your chips with columns for file paths, category labels, and a unique ID. But you could equally use a directory naming convention, a CSV manifest, or any other approach that lets your dataset class resolve chip index to file paths and chip ID.

### 3. Create YAML experiment configs

YAML configs drive both embedding generation and analysis. See the [Configuration Reference](configuration.md) for a fully annotated config with all available options.

### 4. Run the pipeline

**Generate embeddings:**

```bash
python -m gelos.generation \
  --raw-data-dir /path/to/data/raw \
  --embedding-dir /path/to/data/interim \
  --config-dir /path/to/configs
```

```python
from pathlib import Path
from gelos.generation import generate_embeddings

generate_embeddings(
    yaml_path=Path("configs/exp001_prithvi300.yaml"),
    raw_data_dir=Path("/path/to/data/raw"),
    embedding_dir=Path("/path/to/data/interim"),
)
```

Embeddings are written as Parquet files to `embedding_dir / data_version / config_stem`. A `.embeddings_complete` marker file is created on completion; subsequent runs skip configs that already have this marker unless `--overwrite` is passed.

**Run analysis** (extraction, transforms, plots, and models):

```bash
python -m gelos.analysis \
  --raw-data-dir /path/to/data/raw \
  --embedding-dir /path/to/data/interim \
  --processed-data-dir /path/to/data/processed \
  --figures-base-dir /path/to/figures \
  --config-dir /path/to/configs
```

```python
from pathlib import Path
from gelos.analysis import run_analysis

run_analysis(
    yaml_path=Path("configs/exp001_prithvi300.yaml"),
    raw_data_dir=Path("/path/to/data/raw"),
    embedding_dir=Path("/path/to/data/interim"),
    processed_data_dir=Path("/path/to/data/processed"),
    figures_base_dir=Path("/path/to/figures"),
)
```

The analysis pipeline extracts embeddings according to each strategy's `slice_args`, then runs transforms, plots, and models. Results are cached — transform outputs as CSVs, extracted embeddings as `.npy` files — so re-running skips completed steps.

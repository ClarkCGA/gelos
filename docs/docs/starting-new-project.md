# Starting a New Embeddings Project

This guide explains how to set up a new project with your own dataset to generate and transform embeddings using GELOS.

## Prerequisites

Ensure you have the GELOS package installed and your environment configured.

## Environment Configuration

GELOS relies on environment variables to locate data and project files. You can set these in your shell or in a `.env` file in the project root.

Required variables:

| Variable | Description |
|----------|-------------|
| `EXTERNAL_DATA_DIR` | Path to third-party data sources |
| `RAW_DATA_DIR` | Path to original, immutable data dump |
| `INTERIM_DATA_DIR` | Path to intermediate transformed data |
| `PROCESSED_DATA_DIR` | Path to final canonical data sets |
| `PROJECT_ROOT` | Path to the project root directory |

**Example `.env`:**

```bash
EXTERNAL_DATA_DIR=/abs/path/to/data/external
RAW_DATA_DIR=/abs/path/to/data/raw
INTERIM_DATA_DIR=/abs/path/to/data/interim
PROCESSED_DATA_DIR=/abs/path/to/data/processed
PROJECT_ROOT=/abs/path/to/project_root
```

## Dataset Structure

GELOS expects a specific directory structure and metadata file format.

### Directory Layout

Data should be organized by a `data_version` under the `RAW_DATA_DIR`.

```text
RAW_DATA_DIR/
└── {data_version}/
    ├── gelos_chip_tracker.geojson  <-- REQUIRED metadata file
    ├── S2L2A_000001_20230204.tif
    ├── S2L2A_000000_20230605.tif
    └── ...
```

### Chip Tracker Format

The `gelos_chip_tracker.geojson` must be a valid GeoJSON file containing specific columns in its properties.

**Required Columns:**

- `id`: Unique identifier for the chip (int or string).
- `year`: Year of the observation (int).
- `geometry`: Polygon geometry of the chip.
- `{sensor}_paths`: Comma-separated list of filenames relative to the `data_version` directory. Even if you only have one image, it must be string (e.g. `"image.tif"`).
- `color`: Hex color code string (e.g., `"#FF0000"`) for plotting clusters/classes.

**Example Definitions:**

If you are using Sentinel-2 (S2L2A), your columns might look like:

| id | year | S2L2A_paths | color | geometry |
|----|------|-------------|-------|----------|
| 1 | 2023 | "S2L2A_000321_20230204.tif,S2L2A_000321_20230507.tif" | "#00FF00" | POLYGON(...) |

## Running the Pipeline

The pipeline is driven by a YAML configuration file.

### 1. Create a Configuration File

Create a YAML file with a unique, descriptive name (e.g., `configs/prithvieov2300.yaml`) defining your model and data parameters.

```yaml
# lightning.pytorch==2.1.1
seed_everything: 0
trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  callbacks: []
  max_epochs: 0

data:
  class_path: gelos.gelosdatamodule.GELOSDataModule
  init_args:
    batch_size: 1
    num_workers: 0
    bands:
      S2L2A:
        - BLUE
        - GREEN
        - RED
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
    transform:
      - class_path: terratorch.datasets.transforms.FlattenTemporalIntoChannels
      - class_path: albumentations.pytorch.transforms.ToTensorV2
      - class_path: terratorch.datasets.transforms.UnflattenTemporalFromChannels
        init_args:
          n_timesteps: 4

model:
  class_path: terratorch.tasks.EmbeddingGenerationTask
  title: Prithvi EO V2 300M
  init_args:
    model: prithvi_eo_v2_300
    model_args:
      bands:
        - BLUE
        - GREEN
        - RED
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
      pretrained: true
    output_format: parquet
    embed_file_key: filename 
    layers: [-1] # Model layers to extract embeddings from, -1 means the last layer
    output_format: parquet
    embed_file_key: filename 
    embedding_pooling: null 
    has_cls: True
d
# define embedding extraction strategy names and lists of arguments for the embedding extraction function
embedding_extraction_strategies:
  CLS Token:
    - start: 0
      stop: 1
      step: 1
```

### 2. Generate Embeddings

Use `generate_embeddings` to produce embeddings from your raw data.

```python
from pathlib import Path
from gelos.embedding_generation import generate_embeddings

config_path = Path("configs/p.yaml")
generate_embeddings(config_path)
```

This will output embeddings to `PROCESSED_DATA_DIR / {data_version} / {model_name}`.

### 3. Transform and Visualize

Use `transform_embeddings` to run t-SNE and generate plots.

```python
from pathlib import Path
from gelos.embedding_transformation import transform_embeddings

config_path = Path("configs/prithvieov2300.yaml")
transform_embeddings(config_path)
```

Outputs and figures will be saved to `FIGURES_DIR / {data_version}`.

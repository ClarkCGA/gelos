# Starting a New Embeddings Project

This guide explains how to set up a new project with your own dataset to generate and transform embeddings using GELOS.

## Prerequisites

Ensure you have the GELOS package installed.

## Dataset Structure

GELOS expects a specific directory structure and metadata file format.

### Directory Layout

Data should be organized by a `data_version` under your raw data directory.

```text
<RAW_DATA_DIR>/
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
- `geometry`: Polygon geometry of the chip.
- `{sensor}_paths`: Comma-separated list of filenames relative to the `data_version` directory. Even if you only have one image, it must be string (e.g. `"DEM_001.tif"`).
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

# define embedding extraction strategy names and lists of arguments for the embedding extraction function
embedding_extraction_strategies:
  CLS Token:
    - start: 0
      stop: 1
      step: 1
```

### 2. Generate Embeddings

Use the embedding generation CLI or function to produce embeddings from your raw data.

**CLI Example:**

```bash
python gelos/embedding_generation.py \
  --raw-data-dir /abs/path/to/data/raw \
  --processed-data-dir /abs/path/to/data/processed \
  --config-dir /abs/path/to/project_root/configs
```

```python
from pathlib import Path
from gelos.embedding_generation import generate_embeddings

config_path = Path("configs/p.yaml")
raw_data_dir = Path("/abs/path/to/data/raw")
processed_data_dir = Path("/abs/path/to/data/processed")

generate_embeddings(
  config_path,
  raw_data_dir=raw_data_dir,
  processed_data_dir=processed_data_dir,
)
```

This will output embeddings to `processed_data_dir / {data_version} / {model_name}`.

### 3. Transform and Visualize

Use the embedding transformation CLI or function to run t-SNE and generate plots.

**CLI Example:**

```bash
python gelos/embedding_transformation.py \
  --raw-data-dir /abs/path/to/data/raw \
  --processed-data-dir /abs/path/to/data/processed \
  --figures-dir /abs/path/to/data/figures \
  --config-dir /abs/path/to/project_root/configs
```

```python
from pathlib import Path
from gelos.embedding_transformation import transform_embeddings

config_path = Path("configs/prithvieov2300.yaml")
raw_data_dir = Path("/abs/path/to/data/raw")
processed_data_dir = Path("/abs/path/to/data/processed")
figures_dir = Path("/abs/path/to/data/figures")

transform_embeddings(
  config_path,
  raw_data_dir=raw_data_dir,
  processed_data_dir=processed_data_dir,
  figures_dir=figures_dir,
)
```

Outputs and figures will be saved to `figures_dir / {data_version}`.

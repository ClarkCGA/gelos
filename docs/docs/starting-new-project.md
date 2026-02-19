# Starting a New Embeddings Project

This guide explains how to set up a new project with your own dataset to generate and transform embeddings using GELOS.

## Prerequisites

Ensure you have the GELOS package installed.

## Creating a Subclass of GELOSDataSet

GELOSDataSet (from gelos.gelosdataset) defines a parent class which ensures outputs consistent with what the terratorch embedding generation pipeline requires. It also contains reusable methods for noise ablation and band repetition, which is sometimes needed for dataset conformity with specific models. For example, when yearly data such as DEM is passed alongside multitemporal data to Terramind V1 Base, the model requires that the DEM is repeated so that there are an equal number of time steps for all data sources. The parent class ensures all projects in this framework create outputs that can be transformed, visualized and analyzed with the same downstream modules.

In order to use the downstream modules, create a custom class instance which inherits from gelos.GELOSDataSet. Your custom dataset class must define the following elements:

1. For each data source, a list of band names for that data source.
2. An all band names dict in format {"data_source": list[band_names] for each data_source}
3. If you use a custom init(), also call super.__init__() to get necessary parameters
4. Methods:
  - __len__(self) -> int
    - This method is required for progress bars
  - _get_file_paths(self, index: int, sensor: str) -> list[Path]
    - This method returns paths to load for a given index of the dataloader for one sensor
  - _load_file(self, path: Path, band_indices: list[int]) -> np.ndarray
    - This method loads files from the get_file_paths and gets the correct band indices
    - Band indices are determined programmatically based on the yaml and band names dicts defined based on requirement 1.
  - _get_sample_id(self, index: int) -> tuple[str, Any]
    - This method gets the sample ID based on the index, and names the output parquet based on that ID.

For an example implementation within this repository, see TestGELOSDataSet in tests.test_data.py

## Running the Pipeline

### 1. Create a Configuration File

The pipeline is driven by a YAML configuration file. For an example, see tests/fixtures/example_config.yaml

Create a YAML file with a unique, descriptive name (e.g., `configs/prithvieov2300.yaml`) defining your model and data parameters.

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

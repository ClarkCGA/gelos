# Configuration Reference

YAML files drive both embedding generation and analysis. Each config specifies the data source, model, extraction strategies, and analysis steps. Create YAML files with unique, descriptive names (e.g., `configs/exp001_prithvi300.yaml`).

See `tests/fixtures/example_config.yaml` for a minimal working example.

## Full annotated config

```yaml
# --- Metadata ---
experiment_name: "My Experiment"       # used in plot titles
data_version: "v1.0"                   # subdirectory name for inputs/outputs
chip_tracker: "gelos_chip_tracker.geojson"  # metadata file, relative to data_root
chip_id_column: "id"                   # column in chip tracker used to index chips

# --- Data ---
data:
  class_path: gelos.gelosdatamodule.GELOSDataModule
  init_args:
    dataset_class: myproject.mydataset.MyGELOSDataSet  # import path to your subclass

    batch_size: 1
    num_workers: 0

    # bands to load for each sensor — keys must match sensors defined in your
    # dataset class's all_band_names dict, and values must be subsets of the
    # band names defined there. You can include any combination of sensors.
    bands:
      S2L2A: [blue, green, red, nir08, swir16, swir22]
      DEM: [DEM]

    # optional: repeat static (single-timestep) modalities along the temporal
    # axis so all sensors have the same number of timesteps, as required by
    # some models (e.g., Terramind expects matching temporal dimensions)
    repeat_bands:
      DEM: 4

    # optional: add Gaussian noise to specific bands for ablation experiments.
    # the float value controls the noise weight (0 = no noise, 1 = full noise)
    # perturb_bands:
    #   S2L2A:
    #     blue: 0.1

    # albumentations / terratorch transforms applied to each chip.
    # FlattenTemporalIntoChannels and UnflattenTemporalFromChannels are needed
    # to apply spatial transforms across all timesteps.
    transform:
      - class_path: terratorch.datasets.transforms.FlattenTemporalIntoChannels
      - class_path: albumentations.pytorch.transforms.ToTensorV2
      - class_path: terratorch.datasets.transforms.UnflattenTemporalFromChannels
        init_args:
          n_timesteps: 4  # must match number of timesteps in your data

# --- Model ---
# See terratorch documentation for available models and their expected band names.
# Note: model band names may differ from your dataset band names (e.g., NIR_NARROW
# vs nir08) — the model_args bands must match what the model expects.
model:
  class_path: terratorch.tasks.EmbeddingGenerationTask
  init_args:
    model: prithvi_eo_v2_300           # terratorch model identifier
    model_args:
      bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
      pretrained: true
    output_format: parquet
    embed_file_key: filename
    layers: [-1]                       # which transformer layers to extract; -1 = last
    embedding_pooling: null            # null keeps full token sequence
    has_cls: true                      # whether the model produces a CLS token at position 0
                                       # e.g., Prithvi EO V2 does, Terramind V1 does not

# --- Embedding Extraction Strategies ---
# Each key is a user-defined name for a strategy. This name is used in output
# file naming (e.g., "cls_token" → exp001_cls_token_layer-1_tsne.csv).
# You can define as many strategies as you like — each one slices the embedding
# differently and can run its own set of transforms, plots, and models.
#
# slice_args controls which tokens to extract from the embedding sequence.
# Each dict applies pyarrow list_slice(start, stop, step) then flattens.
# Multiple dicts are applied sequentially for nested structures.
embedding_extraction_strategies:
  cls_token:                           # your name for this strategy
    title: "CLS Token"                 # display title for plots
    slice_args:
      - start: 0                      # extract only the first token (CLS)
        stop: 1
        step: 1
    # Available transforms: tsne, pca, umap
    transforms:
      - type: tsne
      - type: pca
        params:
          n_components: 2
    # Available plots: scatter_2d
    plots:
      - type: scatter_2d
        transform: umap                # which transform result to plot
        params:
          axis_lim: 120                # optional plot params
    # Available models: knn, linear_probe, random_forest
    # "transform" selects input data — use "raw" for untransformed embeddings,
    # or the name of a transform defined above (e.g., "pca")
    models:
      - type: knn
        transform: raw
        params:
          n_neighbors: 5

# --- Style ---
# Maps category values to colors and labels for plotting.
# category_column must match a column in your chip tracker / metadata.
style:
  category_column: "lulc"
  colors:
    "1": "#419bdf"
    "2": "#397d49"
  labels:
    "1": "Water"
    "2": "Trees"
```

## Config sections

### Metadata

| Field | Purpose |
|-------|---------|
| `experiment_name` | Human-readable name, shown in plot titles |
| `data_version` | Used as a subdirectory name to organize inputs and outputs (e.g., `data/raw/v1.0/`) |
| `chip_tracker` | Path to your metadata file, relative to `data_root / data_version` |
| `chip_id_column` | Column name in the chip tracker used to index chips and cross-reference with embeddings |

### Data

| Field | Purpose |
|-------|---------|
| `dataset_class` | Python import path to your `GELOSDataSet` subclass |
| `bands` | Dict of `{sensor: [band_names]}`. Keys must match sensors in your class's `all_band_names`, values must be subsets of those band lists |
| `repeat_bands` | Optional. Repeat static modalities (e.g., DEM) along the temporal axis to match the number of timesteps of other sensors |
| `perturb_bands` | Optional. Add Gaussian noise for ablation experiments. Format: `{sensor: {band: weight}}` where weight ranges from 0 (no noise) to 1 (full noise) |
| `transform` | Albumentations/TerraTorch transforms. Use `FlattenTemporalIntoChannels`/`UnflattenTemporalFromChannels` to apply spatial transforms across timesteps |

### Model

| Field | Purpose |
|-------|---------|
| `model` | TerraTorch model identifier (e.g., `prithvi_eo_v2_300`, `prithvi_eo_v2_600`, `terramind_v1_base`) |
| `model_args.bands` | Band names as the **model** expects them (may differ from your dataset band names) |
| `layers` | Which transformer layers to extract embeddings from. `-1` means the last layer |
| `embedding_pooling` | Set to `null` to keep the full token sequence |
| `has_cls` | Whether the model produces a CLS token at position 0 |

### Embedding extraction strategies

Each key under `embedding_extraction_strategies` is a **user-defined name** that becomes part of output file names. Within each strategy:

- **`title`**: Display name used in plot titles
- **`slice_args`**: List of slice operations applied sequentially to extract tokens from the embedding. Each operation uses `start`, `stop`, `step` (PyArrow `list_slice` semantics). Use multiple entries for nested structures.
- **`transforms`**: Optional list of transforms to run on extracted embeddings
- **`plots`**: Optional list of plots to generate
- **`models`**: Optional list of classification models to run

Available **transforms**:

| Type | Description | Key params |
|------|-------------|------------|
| `tsne` | t-SNE dimensionality reduction | `n_components` (default 2), `perplexity` (default 50), `max_iter` (default 1000) |
| `pca` | PCA dimensionality reduction | `n_components` (int for fixed, float for variance threshold; default 0.95) |
| `pca` | UMAP dimensionality reduction | `n_components` (default 2) |

Available **plots**:

| Type | Description | Key params |
|------|-------------|------------|
| `scatter_2d` | Scatter plot colored by category | `axis_lim` (default 120), `legend_loc` (default "upper left") |

Available **models**:

| Type | Description | Key params |
|------|-------------|------------|
| `knn` | K-nearest neighbors with stratified k-fold CV | `n_neighbors` (default 5), `n_splits` (default 5) |
| `linear_probe` | Logistic regression with stratified k-fold CV | `max_iter` (default 1000), `n_splits` (default 5) |
| `random_forest` | Random forest with repeated stratified k-fold CV | `n_estimators` (default 100), `n_splits` (default 5), `n_repeats` (default 3) |

For plots and models, the `transform` field selects which data to use as input: `"raw"` for the untransformed embeddings, or the name of a transform defined in the same strategy (e.g., `"tsne"`, `"pca"`).

### Style

Maps category values from your metadata to colors and labels for plotting. `category_column` must match a column accessible via `chip_gdf[category_column].loc[chip_indices]` in the analysis pipeline.

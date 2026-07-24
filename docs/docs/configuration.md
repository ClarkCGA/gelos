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

    # optional: disable z-score normalization entirely (default true). Set to
    # false for backbones that apply their own pretraining normalization
    # internally (e.g. OlmoEarth) so the model receives raw sensor values.
    # normalize: false

    # optional: convert linear-power bands to decibels at load time
    # (10 * log10(clip(x, 1e-10))), e.g. for models pretrained on dB-scale S1.
    # Do NOT combine with OlmoEarth's built-in normalization (see below).
    # db_scale_bands:
    #   S1RTC: [VV, VH]

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
| `means` / `stds` | Optional. Per-modality/band normalization statistics: `{sensor: {band: value}}`. Resolution order per band: these explicit args → model-matched pretraining stats injected by `gelos.generation` (see below) → lowercase `means`/`stds` class attributes on the dataset class → uppercase `MEANS`/`STDS` class attributes → default (mean 0.0, std 1.0). If an entire modality resolves to the defaults, normalization is an identity and a loud warning is logged |
| `normalize` | Optional, default `true`. Set to `false` to skip z-score normalization entirely (identity aug) — for backbones that apply their own pretraining normalization internally, e.g. OlmoEarth (injected automatically for OlmoEarth models, see below) |
| `db_scale_bands` | Optional. Convert listed bands from linear power to decibels (`10 * log10(clip(x, 1e-10))`) at load time, e.g. `{S1RTC: [VV, VH]}`. Do not use together with OlmoEarth's built-in normalization, which already converts S1 to dB |

**Model-matched normalization defaults.** For recognized backbones, `gelos.generation`
automatically fills any of `means`/`stds`/`db_scale_bands`/`normalize` you did not set,
using the model's own pretraining statistics (`gelos.normalization`): Prithvi EO V2 and
TerraMind v1 get their published pretraining means/stds (TerraMind's S1 stats are in dB, so
`db_scale_bands: {S1RTC: [VV, VH]}` is injected alongside), and OlmoEarth gets
`normalize: false` because its backbone normalizes internally. Rationale: frozen encoders
should see inputs distributed the way they were pretrained — dataset statistics are the
fallback for models without registered stats, and anything you set explicitly in the config
always wins. A registered model combined with a band it was not pretrained on (e.g. Prithvi
with `COASTAL_AEROSOL`) raises an error; override with explicit `means`/`stds` if intended.
| `transform` | Albumentations/TerraTorch transforms. Use `FlattenTemporalIntoChannels`/`UnflattenTemporalFromChannels` to apply spatial transforms across timesteps |

### Model

| Field | Purpose |
|-------|---------|
| `model` | TerraTorch model identifier (e.g., `prithvi_eo_v2_300`, `prithvi_eo_v2_600`, `terramind_v1_base`, `olmoearth_v1_base`, `olmoearth_v1_base_s1s2`, `olmoearth_v1_2_base`) |
| `model_args.bands` | Band names as the **model** expects them (may differ from your dataset band names) |
| `model_args.bands_s1` | S1 band names for OlmoEarth S1+S2 models (e.g., `[VV, VH]`). Omit or set to null to disable S1. |
| `layers` | Which transformer layers to extract embeddings from. `-1` means the last layer |
| `embedding_pooling` | Set to `null` to keep the full token sequence |
| `has_cls` | Whether the model produces a CLS token at position 0 |

#### OlmoEarth (`olmoearth_v1_base`, `olmoearth_v1_base_s1s2`, and size variants)

Ai2's [OlmoEarth](https://pypi.org/project/olmoearth-pretrain/) is supported as a
terratorch-compatible backbone via a wrapper in `gelos/backbones/olmoearth_backbone.py`,
which registers itself automatically when `gelos.generation` is imported. No file
placement is required — install the extra and use the model identifiers directly.
Two generations are available: **v1** with sizes
`nano` (D=128), `tiny` (D=192), `base` (D=768), and `large` (D=1024); and **v1.2**
with sizes `nano` (D=128), `tiny` (D=192), `small` (D=384), and `base` (D=768).
Note v1.2 adds a `small` size and drops `large`. Each size has a Sentinel-2-only
variant and a combined S2+S1 variant:

| Model identifier | Sensors | Hidden dim |
|---|---|---|
| `olmoearth_v1_nano` | S2 only | 128 |
| `olmoearth_v1_tiny` | S2 only | 192 |
| `olmoearth_v1_base` | S2 only | 768 |
| `olmoearth_v1_large` | S2 only | 1024 |
| `olmoearth_v1_nano_s1s2` | S2 + S1 | 128 |
| `olmoearth_v1_tiny_s1s2` | S2 + S1 | 192 |
| `olmoearth_v1_base_s1s2` | S2 + S1 | 768 |
| `olmoearth_v1_large_s1s2` | S2 + S1 | 1024 |

OlmoEarth v1.2 improves embedding quality over v1 and reuses v1's band order and
pretraining normalization statistics unchanged:

| Model identifier | Sensors | Hidden dim |
|---|---|---|
| `olmoearth_v1_2_nano` | S2 only | 128 |
| `olmoearth_v1_2_tiny` | S2 only | 192 |
| `olmoearth_v1_2_small` | S2 only | 384 |
| `olmoearth_v1_2_base` | S2 only | 768 |
| `olmoearth_v1_2_nano_s1s2` | S2 + S1 | 128 |
| `olmoearth_v1_2_tiny_s1s2` | S2 + S1 | 192 |
| `olmoearth_v1_2_small_s1s2` | S2 + S1 | 384 |
| `olmoearth_v1_2_base_s1s2` | S2 + S1 | 768 |

Notes and limitations:

- **Included in core install.** OlmoEarth support ships with gelos — no extra
  install step required beyond `pip install gelos`.
- **Sentinel-2 L2A, full 12 bands required.** The wrapper reorders the input
  channels to OlmoEarth's expected 12-band S2L2A order
  (`B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09`). Your `data.bands.S2L2A` and
  `model_args.bands` must supply all 12 (including `nir09` / B09); a missing band
  raises a clear `ValueError`. See `configs/olmoearth_v1_base.yaml`.
- **Sentinel-1 support (S1+S2 models).** Pass `model_args.bands_s1: [VV, VH]` and
  add `S1RTC: [VV, VH]` to `data.bands` to enable S1. The S1 tensor is fused with
  the S2 embedding via equal-weight averaging. Backward compatibility: omitting
  `bands_s1` (or the S2-only model variants) requires no changes to existing configs.
  See `configs/olmoearth_v1_base_s1s2.yaml`.
- **Built-in pretraining normalization
  (`model_args.apply_pretraining_normalization`, default `true`).** OlmoEarth's
  encoder performs no normalization itself — during pretraining its data loader
  normalized inputs. The wrapper replicates that exactly: S1 linear power is
  clipped at `1e-10`, converted to dB (`10*log10`), then min-max scaled per band
  over `mean±2σ`; S2 raw DN (0–10000) is min-max scaled per band over `mean±2σ`
  (no log). Stats come from `olmoearth_pretrain`'s `computed.json`. Inputs must
  therefore be **RAW sensor scale** (S1 linear-power gamma0, S2 L2A digital
  numbers): set `normalize: false` under `data.init_args` and do **not** apply
  `db_scale_bands` to S1, or values get double-transformed. Set
  `apply_pretraining_normalization: false` only if you pre-normalize inputs to
  OlmoEarth's pretraining scale yourself.
- **Temporal pooling (`model_args.temporal_pooling`).** The OlmoEarth encoder
  outputs one token per (spatial patch × timestep); the wrapper controls what
  happens to the time axis. `mean` (default) averages over timesteps, returning
  `(B, H'*W', D)`. `keep` preserves per-timestep tokens flattened time-major to
  `(B, T*H'*W', D)` — token `t*H'*W' + i` is spatial patch `i` at timestep `t` —
  so strided `slice_args` can extract single-timestep vs. all-timestep features,
  matching the Prithvi token layout. Note that with `keep`, a single timestep's
  tokens still come from one joint space-time attention pass over all timesteps.
- **Spatial pooling (`model_args.spatial_pooling`).** Optional integer factor `s`
  (default off): after encoding, the `H'×W'` token grid is average-pooled over
  non-overlapping `s×s` neighborhoods, so each output token covers
  `(s*patch_size)²` input pixels. Use it to match the spatial footprint of
  larger-patch models — e.g. `patch_size: 4, spatial_pooling: 4` yields tokens
  covering 16×16 pixels whose grid and indices line up exactly with
  Prithvi/TerraMind's 16-pixel patches, so the same `slice_args` strategies
  apply across models. Encoding still runs at the fine `patch_size`; only the
  output tokens are aggregated.
- **Timestamps.** When the batch carries per-timestep acquisition dates (a
  `timestamps` key of shape `(B, T, 3)` as `[day, month_index, year]`), the
  generation task threads them into the backbone so OlmoEarth's temporal
  encoding reflects true acquisition dates. When absent, timestamps fall back
  to a constant `[15, 0, 2020]` (day=15, month=Jan, year=2020), and
  acquisition-date-dependent temporal encoding will not reflect true
  seasonality.

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

### Comparison plots

Comparison configs may list `comp_plots`, each with a `type` from the comparison
plot registry and an optional `params` block. Each plot reads the output of a
comparison metric; by default the source metric is derived from the plot `type`
(stripping the `_plot`/`_table` suffix), but it can be set explicitly with a
`metric` field — required when the names don't match.

| Type | Description | Key params |
|------|-------------|------------|
| `knn_purity_distribution_plot` | Per-class box plots of per-query KNN purity vs k | (none) |
| `knn_purity_violin_distribution_plot` | Per-class violin plots of per-query KNN purity vs k. Split violin (one half per group) when exactly two experiments are compared, otherwise side-by-side single violins. Mean shown as a diamond, median as a short line | `split` (bool, default auto: split iff exactly two experiments). `split_pairs` (list of 2-element experiment-name lists; each pair → one split violin per k; unpaired experiments → single violins; takes precedence over `split`). Requires `metric: knn_purity_per_query_comparison` |

Example:

```yaml
comp_plots:
  - type: knn_purity_violin_distribution_plot
    metric: knn_purity_per_query_comparison
    params:
      split: true
```

To contrast several pairs of experiments as side-by-side split violins (for
example multitemporal vs single-time-step variants of two models), use
`split_pairs`. Each inner list names the two experiments of one split violin;
any experiment not named in a valid pair is drawn as a single violin. When
`split_pairs` is set it takes precedence over `split`:

```yaml
comp_plots:
  - type: knn_purity_violin_distribution_plot
    metric: knn_purity_per_query_comparison
    params:
      split_pairs:
        - [model_A_multi, model_A_single]
        - [model_B_multi, model_B_single]
```

## Spectral-band baseline configs

Randomly-initialized (or mis-pretrained) foundation-model weights can still produce embedding spaces that look reasonable on downstream probes. To sanity-check how much a model is actually contributing, GELOS ships `gelos.spectral_bands.SpectralBandsEmbeddingTask` — a passthrough task that skips the model entirely and writes the normalized spectral band values themselves as "embeddings". The output parquets match `EmbeddingGenerationTask`'s schema, so analysis and comparison stages consume them with no special-casing.

Swap the `model` block for the raw task and add a top-level `spectral_band_extraction` section:

```yaml
model:
  class_path: gelos.spectral_bands.SpectralBandsEmbeddingTask
  title: "Spectral Bands Baseline"
  init_args:
    patch_size: 16
    embed_file_key: filename

# Each named entry becomes a per-strategy subdir under the run's output_dir
# (playing the role of a "layer" for analysis). The product of
# len(timesteps) * len(patches) must be <= 4 — spectral band tokens are high-dim,
# so the cap keeps parquet sizes reasonable.
spectral_band_extraction:
  center_2x2_t0:
    title: "Center 2x2 patches, single timestep"
    patches: center_2x2      # 'center_NxN' or an explicit list of [row, col] pairs
    timesteps: [0]           # list of ints or the string 'all'
  all_timesteps_center_patch:
    title: "All timesteps of the center patch"
    patches: center_1x1
    timesteps: all
```

Notes:

- **Normalization is reused from the datamodule.** The task trusts `GELOSDataModule.aug` (defaults to z-score via means/stds) — set real per-band statistics on your dataset class (`means`/`stds` or `MEANS`/`STDS` attributes) or in `data.init_args.means/stds`, or baseline pixels stay un-normalized (the datamodule logs a loud warning when a modality resolves entirely to default stats).
- **Patch grid is derived from `H, W, patch_size`**, not configured. All of `H`, `W` must be divisible by `patch_size`, and for multi-sensor configs all sensors must share `T`, `H`, `W` after normalization (use `repeat_bands` to align single-timestep modalities like DEM).
- **Multi-modal runs channel-concatenate per token.** Do NOT set `concat_bands: True` on the datamodule — the task handles concat itself. Sensor order follows `data.bands` YAML key order, which becomes the dict iteration order. Reordering that block between runs produces parquets of the same shape but different channel layouts.
- **Comparison workflow.** Each modality combination is its own generation config (`spectral_s2.yaml`, `spectral_s2s1dem.yaml`, etc.) with its own `data.bands` and its own `spectral_band_extraction`. Comparison configs reference spectral-band runs by the same `(config, strategy, layer)` triple as any model run. S2 pixels getting written twice is bounded (~kB/chip per run at the 4-token cap).

`embedding_extraction_strategies` still applies on top — each spectral-band subdir becomes a "layer" for analysis, and each embedding-extraction strategy runs its slice/transforms/plots/models against it.

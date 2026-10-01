# Changelog

All notable changes to GELOS will be documented in this file.

When releasing a new version:
1. Add an entry below under a new `## [vX.Y.Z]` heading
2. Bump `version` in `pyproject.toml` to match
3. Create a GitHub Release with the same tag (e.g., `v1.1.0`)

Downstream projects should pin to a release tag and update intentionally:
```toml
gelos = {git = "https://github.com/ClarkCGA/gelos.git", tag = "v1.0.0"}
```

## [Unreleased]

- **Per-config figure folders (issue #89).** `run_analysis` now writes experiment figures
  to `{figures_base_dir}/{data_version}/{config_stem}/` instead of one flat
  `{figures_base_dir}/{data_version}/` folder, and the config stem is dropped from file
  names: `{strategy}_{layer}_{transform}_{plot}.png` and
  `{strategy}_{layer}_{model}_confusion_matrix.png` (new `gelos.analysis.build_figure_prefix`
  helper; `build_prefix` and all processed-data CSV/NPY names are unchanged). Comparison
  figures likewise drop the config-stem prefix: `comparisons/{config_stem}/{plot}.png`.
  Figures from earlier runs are not moved; they will be regenerated once in the new
  location on the next run and the old flat files can be deleted. Downstream code that
  globs `figures/{data_version}/{config}_*.png` must update to the new layout.

## [v0.7.0] - 2026-09-30

- **DINOv3 backbones (issue #16).** New `gelos.backbones.dinov3_backbone` registers
  `dinov3_vitb16_pretrained` (web LVD-1689M ViT-B/16) and `dinov3_vitl16_sat_pretrained`
  (satellite SAT-493M ViT-L/16) in terratorch's backbone registry; registration happens on
  `import gelos.generation`. Pretrained weights are resolved at first run from a local
  checkpoint (per-variant env var), `torch.hub`, or the gated `facebook/dinov3-*`
  HuggingFace repos (requires `HF_TOKEN`; the transformers-format `model.safetensors` is
  converted to the hub layout on download). Configs must list S2 bands in RED, GREEN, BLUE
  order. Adds `termcolor` and `safetensors` as core dependencies.
- **Clip-and-stretch normalization for DINOv3.** New `GELOSDataModule`/`GELOSDataSet` param
  `clip_range_bands: dict[str, dict[str, list[float]]] | None` clips bands to a fixed
  `[min, max]` at load time (after any dB conversion), e.g.
  `{"S2L2A": {"RED": [0.0, 2500.0]}}`. `gelos.normalization` registers DINOv3 specs that
  clip S2 RGB to `[0, 2500]` DN and fold the `/2500` stretch into the z-score stats
  (ImageNet stats for web variants, SAT-493M stats for `dinov3_vitl16_sat`), with
  `set_nodata: 0`.
- **Embedding Parquet stored as float32.** `LenientEmbeddingGenerationTask.write_parquet`
  now writes the embedding column as nested `float32` lists (stock terratorch promoted to
  float64), with dictionary encoding disabled and `zstd` compression — roughly halving
  embedding storage. `gelos.extraction` is dtype-agnostic, so downstream analysis is
  unaffected, but external readers of the Parquet files will see `float32`.
- **Fixed-size kNN comparison plots (issue #84).** `knn_purity_plot`,
  `knn_purity_distribution_plot` and `knn_purity_violin_distribution_plot` now share one
  layout: the figure size depends only on the facet grid shape (margins fixed in
  inches), the legend is stacked one entry per line in a reserved band below the plots
  (room for four experiments) instead of a side-by-side legend outside the canvas, and
  files are saved at a fixed dpi without `bbox_inches="tight"`, so images of the same
  grid shape are pixel-identical in size regardless of experiment count or name length.
  Facets in all three plots now share their x and y axes. The distribution/violin
  plots drop `constrained_layout` and size their width per column (`6 in`, unchanged
  for the default two-column grid).

## [v0.6.0] - 2026-09-29

- **OlmoEarth nodata input mask + masked pooling.** Follow-up to the `nodata_value` /
  `set_nodata` remap below (#81): the datamodule's `NoDataRemap` now also attaches the
  raw-batch detection mask to the batch as `batch["nodata_mask"]`
  (`gelos.gelosdatamodule.NODATA_MASK_KEY`; `{modality: BoolTensor}` for dict batches, a
  single tensor otherwise), and `LenientEmbeddingGenerationTask.predict_step` pops it and
  stashes it on the backbone via `set_batch_nodata_mask` / `clear_batch_nodata_mask`,
  mirroring the `timestamps` side-channel (a no-op for Prithvi/TerraMind). `OlmoEarthBackbone`
  gains `mask_nodata` (default `true`) and `nodata_patch_threshold` (default `0`): a patch
  containing any nodata pixel (any band, per timestep; a value in `(0, 1]` instead requires
  that fraction) is flagged
  `MaskValue.MISSING`, so the encoder removes it before attention (the encoder is then run
  with `fast_pass=False` and its attention mask enabled, so zero-padded batch-mates cannot
  leak into attention), and every pooling step (band-set, S1/S2 fusion, spatial, temporal)
  becomes a masked mean over valid tokens. `spatial_pooling` additionally accepts `"mean"`
  (masked mean over the whole token grid per timestep: `(B, T, D)` with `temporal_pooling:
  keep`, `(B, 1, D)` with `mean`) so every sample yields a same-size vector. Masked grid
  positions are zero vectors; fully-masked samples are encoded unmasked with a warning.
  `gelos.normalization` now registers `set_nodata: 0` for Prithvi, TerraMind and OlmoEarth
  and injects it only when the config sets `nodata_value` (superseding the "not yet
  registered" note in the entry below), so downstream configs only need the dataset's
  nodata value. Documented in the configuration reference.
- **Dataset-side acquisition timestamps and chip location (issue #79).** Two new
  optional, non-abstract hooks on `GELOSDataSet`: `_get_timestamps(index)` returns a
  canonical `(T, 3)` integer `[year, month, day]` array (month 1–12, one row per
  timestep of the primary temporal sensor) that flows into `batch["timestamps"]`
  `(B, T, 3)` and into any backbone exposing `set_batch_timestamps`; and
  `_get_location(index)` returns a `(2,)` float `[lat, lon]` array that flows into
  `batch["location"]` `(B, 2)`, dispatched generically to any backbone exposing
  `set_batch_location`. OlmoEarth converts the canonical dates to its own
  `[day, month_index, year]` packing internally (new pure helper
  `calendar_to_olmoearth_timestamps`); no current backbone consumes `location` — it is
  groundwork for a future Prithvi TL wrapper. Both hooks default to `None`, so
  existing subclasses are unaffected. **Breaking format note:** `batch["timestamps"]`
  is now canonical calendar `[year, month, day]`, no longer OlmoEarth's
  `[day, month_index, year]` — no in-repo producers of the key existed, but anyone who
  hand-crafted the old packing must switch (a plausibility `UserWarning` fires on
  old-format-looking tensors).
- **Nodata pixel remapping in `GELOSDataModule`.** New `nodata_value` / `set_nodata` init
  args (must be set together) remap on-disk nodata sentinels so the model receives a chosen
  value instead of a normalized sentinel. Detection runs on the raw batch (the sentinel,
  e.g. `-999`, is only exactly matchable before z-scoring); the write runs AFTER the
  normalization aug, so with `nodata_value: -999, set_nodata: 0` the model sees exactly `0`
  at nodata positions, not `(0 - mean) / std`. Implemented as a `NoDataRemap` wrapper
  around whichever aug is active (`Normalize`, `MultimodalNormalize`, or `IdentityAug`
  for `normalize: false`). Scalars apply to every modality; `{modality: value}` dicts mask
  only their listed modalities (dicts are rejected for single-tensor batches, i.e.
  `concat_bands: true`, where modality boundaries are unknown). Construction fails fast
  on unknown modality keys or a `set_nodata` dict missing a masked modality, and logs a
  warning when a masked modality also appears in `db_scale_bands` or `perturb_bands`
  (those per-sample steps run first and corrupt the sentinel). Per-model target values
  are not yet registered in `gelos.normalization`, so both keys must currently be set
  explicitly per config. Documented in the configuration reference (#81).
- **OlmoEarth v1.2 backbones.** New terratorch factory functions
  `olmoearth_v1_2_{nano,tiny,small,base}` (Sentinel-2 only) and `..._s1s2` (S2+S1),
  registered in `BACKBONE_REGISTRY`, with hidden dims 128/192/384/768. New shipped
  configs `configs/olmoearth_v1_2_*.yaml`. v1.2 adds a `small` (D=384) size and drops
  `large`; it reuses v1's band order and pretraining normalization statistics unchanged,
  so `gelos/normalization.py` (prefix-matched to `olmoearth_v1`) already covers the new
  names. Bumps `olmoearth-pretrain>=0.1.1` (the first release exposing the v1.2 `ModelID`
  entries).

## [v0.5.0] - 2026-07-21

Entries below were recorded without per-release headings and may include changes
first shipped in earlier v0.3.x/v0.4 tags.

- **Renamed "raw pixels" to "spectral bands"** throughout the library for clarity.
  **Breaking** for configs and code that reference the old name.
- **Analysis completion markers.** `run_analysis` now writes a `.analysis_complete` marker
  (mirroring generation's `.embeddings_complete`) and skips fully-completed configs on
  re-runs. Model runs (`knn`/`linear_probe`/`random_forest`) now also skip individually
  when their results CSV and confusion-matrix figure already exist, closing the one gap in
  the per-step caching (embeddings, transforms, metrics, and plots already skipped). New
  `--overwrite` flag on `python -m gelos.analysis` re-enters completed runs; per-step
  caches still apply, so only missing artifacts are recomputed — delete cached outputs for
  a full redo.
- **Model-matched normalization by default in `gelos.generation`.** New module
  `gelos.normalization` maps backbone names to their pretraining statistics (Prithvi EO V2,
  TerraMind v1 — imported from terratorch with hard-coded fallbacks) and required scale
  conversions, and `setup_embedding_run` now injects them into the datamodule config:
  Prithvi/TerraMind get their pretraining `means`/`stds` (plus `db_scale_bands` for
  TerraMind S1, whose stats are in dB), OlmoEarth gets `normalize: false` (its backbone
  normalizes internally). Rationale: frozen encoders should see inputs normalized the way
  they were pretrained, not with dataset statistics. Keys set explicitly under
  `data.init_args` always win; unregistered models fall back to dataset statistics; a
  registered model with a configured band that has no pretraining stat raises (override
  with explicit `means`/`stds`).
- **Fix: silent normalization no-op in `GELOSDataModule`.** Stats were only looked up on
  lowercase `means`/`stds` dataset-class attributes, so downstream classes defining uppercase
  `MEANS`/`STDS` (e.g. gelos-lc's `GELOSLCDataSet`) silently resolved every band to mean 0.0 /
  std 1.0, making `Normalize`/`MultimodalNormalize` an identity — models received raw pixel
  values. Resolution order is now: explicit `means`/`stds` args → lowercase `means`/`stds`
  class attrs → uppercase `MEANS`/`STDS` class attrs → default 0.0/1.0, and a loud
  `logger.warning` is emitted when a modality resolves entirely to defaults.
- New `GELOSDataModule` param `normalize: bool = True`: when `false` (and no custom `aug`),
  `self.aug` is an identity — for backbones that apply their own pretraining normalization
  internally (e.g. OlmoEarth).
- New `GELOSDataModule`/`GELOSDataSet` param `db_scale_bands: dict[str, list[str]] | None`:
  converts the listed bands from linear power to decibels (`10 * log10(clip(x, 1e-10))`) at
  load time, before perturbation and transforms (e.g. `{"S1RTC": ["VV", "VH"]}`).
- **Fix: OlmoEarth S1 scale mismatch + missing pretraining normalization.** The dataset's
  S1RTC chips are linear-power gamma0 but OlmoEarth was pretrained on dB-scale S1, and its
  pretraining data loader (not the encoder) normalized all inputs. New
  `OlmoEarthBackbone`/factory arg `apply_pretraining_normalization: bool = True` replicates
  that data-loader normalization exactly inside `forward_features`: S1 raw linear power →
  clip `1e-10` → `10*log10` → per-band mean±2σ min-max; S2 raw DN (0–10000) → per-band
  mean±2σ min-max. Stats are read from the installed `olmoearth_pretrain` package's
  `norm_configs/computed.json` (hard-coded fallback included). **Breaking for OlmoEarth
  configs**: inputs must now be RAW sensor scale — set `normalize: false` under
  `data.init_args` (done in the shipped `configs/olmoearth_v1_*.yaml`) and do not apply
  `db_scale_bands` to S1 for this backbone.
- OlmoEarth `temporal_pooling` option on `OlmoEarthBackbone` (`model_args.temporal_pooling`):
  `mean` (default, unchanged behavior) averages tokens over timesteps; `keep` returns
  per-timestep tokens flattened time-major to `(B, T*H'*W', D)`, matching the Prithvi
  token layout so strided `slice_args` extraction strategies can compare single-timestep
  vs. all-timestep features.
- OlmoEarth `spatial_pooling` option on `OlmoEarthBackbone` (`model_args.spatial_pooling`):
  optional integer factor that average-pools the output token grid over non-overlapping
  `s×s` neighborhoods after encoding, so each token covers `(s*patch_size)²` pixels.
  With `patch_size: 4, spatial_pooling: 4`, OlmoEarth tokens cover the same 16×16-pixel
  footprint (and use the same token indices) as Prithvi/TerraMind patches.
- **OlmoEarth backbone moved to library-level package**: `gelos/backbones/olmoearth_backbone.py`
  replaces the old `models/olmoearth_backbone.py` + `custom_modules/` approach. The backbone now
  self-registers when `gelos.generation` is imported — no file placement in the working directory
  is required. Library users no longer need to maintain a `custom_modules/` package or add
  `models/` to `PYTHONPATH` to use OlmoEarth. The `custom_modules/` directory is now reserved
  exclusively for user-defined per-project backbones. **Migration**: update any direct
  `from models.olmoearth_backbone import ...` imports to
  `from gelos.backbones.olmoearth_backbone import ...`.
- OlmoEarth S1+S2 combined embedding support: new `bands_s1` parameter on `OlmoEarthBackbone` enables Sentinel-1 alongside Sentinel-2; new factory functions `olmoearth_v1_{nano,tiny,base,large}_s1s2` registered in `BACKBONE_REGISTRY`; new configs `configs/olmoearth_v1_{size}_s1s2.yaml`. **S1 data must be in decibel scale.** Backward compatibility: existing S2-only configs require no changes.
- Fix: `MaskedOlmoEarthSample` now receives explicit `sentinel2_l2a_mask` (shape `(B,H,W,T,3)`) as required by the OlmoEarth API.
- OlmoEarth (`olmoearth_v1_base`) is now selectable as a terratorch-compatible
  backbone via YAML (`model: olmoearth_v1_base`). Adds an in-process wrapper
  registered through `gelos.backbones.olmoearth_backbone` and example configs.
  `olmoearth-pretrain` is now a core gelos dependency. The wrapper is
  Sentinel-2 L2A only, requires the full 12-band S2L2A set (reordered to OlmoEarth's
  expected order), and uses constant dummy per-timestep timestamps (`[15, 0, 2020]`)
  as a documented limitation. The obsolete cloud-embeddings `OlmoEarthBackend` stub
  was removed in favor of this approach.
- `knn_purity_violin_distribution_plot` comparison plot: a violin-plot alternative
  to `knn_purity_distribution_plot` for the KNN per-query purity distribution.
  Draws a *split* violin (one half per group) when exactly two experiments are
  compared, and side-by-side single violins otherwise. The mode can be forced via
  the `split` param. Means are marked with a diamond and medians with a short line,
  matching the box-plot variant. Consumes the same
  `knn_purity_per_query_comparison` metric output, so YAML configs must set
  `metric: knn_purity_per_query_comparison` on the plot entry. A new `split_pairs`
  param (list of 2-element experiment-name lists) renders each pair as a
  side-by-side split violin per k, with any unpaired experiments drawn as single
  violins; pairs naming a missing experiment are warned about and skipped.
  `split_pairs` takes precedence over `split`.

## [v1.0.0] - 2026-03-23

Initial public release.

- `GELOSDataSet` abstract base class for multi-modal, multi-temporal geospatial chip datasets
- `GELOSDataModule` Lightning DataModule for inference (predict-only)
- Embedding generation via TerraTorch `EmbeddingGenerationTask` with YAML-driven configs
- Embedding extraction with configurable token slicing (`slice_args`)
- Transform registry: t-SNE, PCA
- Plot registry: t-SNE scatter plots colored by category
- Model registry: KNN, linear probe, random forest (all with stratified k-fold CV)
- Config-driven analysis pipeline (`run_analysis`) with caching
- Typer CLI entry points for generation and analysis
- Band perturbation and repetition support
- MkDocs documentation site

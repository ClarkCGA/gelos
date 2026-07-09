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

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

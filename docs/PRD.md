# Product Requirements Document (PRD): GELOS

## 1. Executive Summary
GELOS (Geospatial Exploration of Latent Observation Space) is a pip-installable Python package designed to generate, extract, transform, and visualize embeddings from multi-modal, multi-temporal satellite imagery and DEM data. It leverages TerraTorch and PyTorch Lightning for embedding generation, and provides utilities for embedding extraction, t-SNE transformation, and plotting. 

**MVP Goal:** Provide a stable, pip-installable library with LI entry points that can ingest configured datasets, generate embeddings with TerraTorch, run extraction + t-SNE transforms, and output plots and CSVs for analysis.

## 2. Mission
**Mission Statement:** Enable geospatial researchers and ML engineers to rapidly explore and compare latent representations from multi-modal, multi-temporal Earth observation datasets.

**Core Principles:**
- Reproducible pipelines via configuration and deterministic outputs.
- Dataset-agnostic design with standardized inputs and clear conventions.
- Usable from both CLI and as a Python import
- Minimal friction setup through environment-driven paths and pip installability.
- Transparent outputs (CSV + plots) for downstream analysis.

## 4. MVP Scope
**Core Functionality**
- ✅ Pip-installable Python package
- ✅ CLI workflows for embedding generation
- ✅ Embedding extraction and slicing utilities
- ✅ t-SNE transformation and CSV export
- ✅ Plotting helper for t-SNE visualization

**Technical**
- ✅ Dataset and DataModule wiring for multi-modal, multi-temporal data

**Integration**
- ✅ TerraTorch integration for embedding generation
- ✅ PyTorch Lightning prediction loop support

**Deployment**
- ✅ Local execution with Docker/Compose compatibility

**Out of Scope (MVP)**
- ❌ Interactive web UI or dashboard
- ❌ Automated hyperparameter search for t-SNE
- ❌ Multi-user authentication/authorization
- ❌ Training or fine-tuning models (embedding generation only)
- ❌ Production-scale distributed orchestration

## 6. Core Architecture & Patterns
- **Architecture:** Modular Python package with CLI entry points and utilities.
- **Directory Structure:**
  - gelos/: core library (config, dataset/datamodule, embedding workflows)
  - models/: model definitions or adapters
  - docs/: MkDocs documentation
  - tests/: unit tests
- **Patterns:**
  - Configuration via environment variables and optional .env
  - DataModule pattern (PyTorch Lightning)
  - Composition for extraction + transformation + plotting pipeline

## 7. Tools/Features
**Embedding Generation**
- Uses TerraTorch EmbeddingGenerationTask
- YAML-driven model and data configuration
- Output directory naming based on model and perturbation settings

**Embedding Extraction**
- Supports sampling and slicing from parquet embeddings

**Transformation**
- t-SNE computation with configurable parameters
- CSV export for t-SNE coordinates

**Plotting**
- t-SNE scatter plots with class coloring
- Output image files with standardized naming

## 8. Technology Stack
- **Core:** Python 3.11, PyTorch, Lightning, TerraTorch
- **Geospatial:** GeoPandas, Rasterio, Rioxarray, Shapely
- **Data:** NumPy, Pandas, PyArrow
- **Visualization:** Matplotlib
- **CLI:** Typer
- **Docs:** MkDocs
- **Testing:** Pytest, Ruff

## 10. API Specification
**CLI Entry Points** (initial MVP):
- `gelos.generation` CLI
  - Inputs: YAML config path or config directory
  - Output: Embedding files and completion marker
- `gelos.analysis` CLI
  - Inputs: YAML config path or config directory
  - Output: Transform results, plots, and model evaluation CSVs

**Python API (examples):**
- `generate_embeddings(yaml_path: Path) -> None`
- `run_pipeline(yaml_path: Path, raw_data_dir, processed_data_dir, figures_dir) -> dict`

## 11. Success Criteria
**MVP Success Definition:**
- The package installs cleanly via pip and provides working CLI workflows that produce embeddings, CSVs, and plots from configured datasets.

**Functional Requirements**
- ✅ Embeddings can be generated using TerraTorch
- ✅ Embeddings can be extracted and sliced
- ✅ t-SNE can be computed and saved
- ✅ Plots can be generated from t-SNE outputs

**Quality Indicators**
- Stable CLI behavior with clear error messages
- Deterministic outputs when seed is provided
- Tests validate configuration logic

**User Experience Goals**
- Single-command workflow per pipeline stage
- Documentation sufficient for new users to run examples

## 12. Implementation Phases
**Phase 1: Packaging & Config**
- Goal: Ensure pip installability and config stability
- Deliverables:
  - ✅ pyproject.toml configured
- Validation: Install locally and run config tests

**Phase 2: Embedding Generation**
- Goal: Stable embedding generation pipeline
- Deliverables:
  - ✅ CLI for embedding generation
  - ✅ YAML-driven configuration
- Validation: Generate embeddings for sample config

**Phase 3: Extraction + Transformation + Plotting**
- Goal: End-to-end transformation pipeline
- Deliverables:
  - ✅ Extraction utilities
  - ✅ t-SNE and CSV export
  - ✅ Plot generation
- Validation: Run pipeline and verify CSV/PNG outputs

**Phase 4: Documentation & Tests**
- Goal: Docs and tests for reliability
- Deliverables:
  - ✅ Getting started documentation
  - ✅ Tests for config and data utilities
- Validation: Run pytest and validate docs build

## 13. Future Considerations
- Support for additional dimensionality reduction methods (UMAP, PCA)
- More robust metadata and provenance capture

## 14. Risks & Mitigations
- **Risk:** Large embedding files cause memory pressure.
  - Mitigation: Stream parquet reads and chunked processing.
- **Risk:** Inconsistent dataset schemas across sensors.
  - Mitigation: Explicit schema validation in dataset loader.
- **Risk:** t-SNE runtime too slow for large datasets.
  - Mitigation: Sampling and alternative DR methods.

## 15. Appendix
- Repository structure follows Cookiecutter Data Science conventions.
- Dependencies are defined in pyproject.toml.
- Documentation in docs/ (MkDocs).

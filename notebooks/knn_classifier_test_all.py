"""Cross-validated KNN classifier runner that logs per-run metrics and aggregates summaries."""
# imports
from gelos.embedding_extraction import extract_embeddings
from gelos.embedding_generation import perturb_args_to_string
import geopandas as gpd
import yaml
from gelos.config import PROJ_ROOT, PROCESSED_DATA_DIR, DATA_VERSION, RAW_DATA_DIR
from gelos.config import REPORTS_DIR, FIGURES_DIR
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import typer
from loguru import logger
app = typer.Typer()
from typing import Optional
import json
import pandas as pd

n_timesteps = 4
config_yaml_names = [
    # Default configs evaluated by this script
    "prithvieov2300_noperturb.yaml",
    "prithvieov2600_noperturb.yaml",
    "terramindv1base_noperturb.yaml",
    # add more configs here
]
# extraction_strategy = "All Steps of Middle Patch"
max_workers = None  # set to an int to cap parallelism
n_neighbors = 5
pca_components = 0.95
LOG_DIR = PROCESSED_DATA_DIR / DATA_VERSION / "logs"
RUN_RESULTS_FILENAME = "knn_run_results.csv"

def _format_result_row(yaml_name: str, extraction_strategy: str, model_title: str, overall_acc: float, per_class: dict) -> str:
    """Return a single-line summary string for logging experiment results."""
    per_class_summary = "; ".join(f"{cls}:{acc:.3f}" for cls, acc in sorted(per_class.items()))
    return f"{yaml_name:<32} | {extraction_strategy:<24} | {model_title:<24} | {overall_acc:>7.3f} | {per_class_summary}"

def collect_run_results(base_dir: Path) -> pd.DataFrame:
    """
    Collect all per-run CSVs (named RUN_RESULTS_FILENAME) under base_dir into one DataFrame.

    Args:
        base_dir: Root directory to search for run result CSVs.

    Returns:
        Concatenated DataFrame of all discovered run result CSVs with a source path column.
    """
    csv_paths = list(base_dir.rglob(RUN_RESULTS_FILENAME))
    if not csv_paths:
        logger.warning(f"No run result CSVs found under {base_dir}")
        return pd.DataFrame()
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["source_csv"] = csv_path.as_posix()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def run_config(yaml_name: str, extraction_strategy: str, chip_gdf, n_neighbors: int = 5, pca_components: float = 0.95):
    """Run stratified 5-fold CV KNN for a given config/strategy, log metrics, and save per-run results.

    Args:
        yaml_name: Config filename to load model and extraction settings.
        extraction_strategy: Key into embedding_extraction_strategies for slicing embeddings.
        chip_gdf: GeoDataFrame containing chip metadata and labels.
        n_neighbors: KNN neighbors.
        pca_components: PCA n_components (float variance threshold or int).

    Returns:
        Tuple of (yaml_name, extraction_strategy, model_title, overall_accuracy, per_class_dict).
    """    

    extraction_strategy_lower = extraction_strategy.replace(" ","").lower()
    log_file = LOG_DIR / f"{Path(yaml_name).stem}_{extraction_strategy_lower}.log"
    logger.remove()
    logger.add(log_file, level="INFO")
    logger.info(f"Starting experiment config={yaml_name}, strategy={extraction_strategy}")
    yaml_path = PROJ_ROOT / "gelos" / "configs" / yaml_name
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    data_root = RAW_DATA_DIR / DATA_VERSION
    figures_dir = FIGURES_DIR / DATA_VERSION
    figures_dir.mkdir(exist_ok=True, parents=True)
    model_name = yaml_config["model"]["init_args"]["model"]
    model_title = yaml_config["model"]["title"]
    embedding_extraction_strategies = yaml_config["embedding_extraction_strategies"]
    perturb = yaml_config["data"]["init_args"].get("perturb_bands", None)
    perturb_string = perturb_args_to_string(perturb)  # encodes perturb settings into path-friendly string
    output_dir = PROCESSED_DATA_DIR / DATA_VERSION / model_name / perturb_string
    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]
    if not embeddings_directories:
        raise FileNotFoundError(f"No embeddings directories found for {yaml_name}")
    embeddings_directory = embeddings_directories[0]
    slice_args = embedding_extraction_strategies[extraction_strategy]  # select the slice policy for embeddings
    embeddings, chip_indices = extract_embeddings(embeddings_directory, slice_args=slice_args)
    label_col = "category"  # <-- set to the column in chip_gdf containing class labels
    labels = chip_gdf.iloc[chip_indices][label_col].to_numpy()
    X = embeddings
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.empty_like(labels)
    for i, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, labels), total=cv.get_n_splits(), desc=f"{yaml_name} folds")):
        # Fit PCA on train fold, transform train/test, then fit/predict KNN
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X[train_idx])
        X_test = pca.transform(X[test_idx])
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        logger.info("PCA complete, starting knn")
        knn.fit(X_train, labels[train_idx])
        y_pred[test_idx] = knn.predict(X_test)
    overall_acc = accuracy_score(labels, y_pred)
    cm = confusion_matrix(labels, y_pred, labels=np.unique(labels))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class = dict(zip(np.unique(labels), per_class_acc))
    logger.info(_format_result_row(yaml_name, extraction_strategy, model_title, overall_acc, per_class))

    run_dir = output_dir / Path(yaml_name).stem / extraction_strategy_lower  # per-run output folder
    run_dir.mkdir(parents=True, exist_ok=True)
    run_results_path = run_dir / RUN_RESULTS_FILENAME  # per-run CSV path
    pd.DataFrame(
        [{
            "config": yaml_name,
            "strategy": extraction_strategy,
            "model": model_title,
            "accuracy": overall_acc,
            "per_class": json.dumps(per_class),
        }]
    ).to_csv(run_results_path, index=False)
    logger.info(f"Saved run results to {run_results_path}")

    return yaml_name, extraction_strategy, model_title, overall_acc, per_class

@app.command()
def main(
    yaml_path: Optional[Path] = typer.Option(
        None, "--yaml-path", "-y", help="Path to a single yaml config to process."
    )
):
    """
    Generate embeddings from a model and data specified in a yaml config.

    If --yaml-path is provided, only that yaml will be processed.
    Otherwise, all yamls in the default config directory will be processed.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(LOG_DIR / "knn_main.log", level="INFO", enqueue=True)
    
    data_root = RAW_DATA_DIR / DATA_VERSION
    chip_gdf = gpd.read_file(data_root / 'gelos_chip_tracker.geojson')
    figures_dir = FIGURES_DIR / DATA_VERSION
    figures_dir.mkdir(exist_ok=True, parents=True)

    if yaml_path:
        yaml_paths = [Path(yaml_path)]
    else:
        yaml_config_directory = PROJ_ROOT / "gelos" / "configs"
        yaml_paths = list(yaml_config_directory.glob("*noperturb*.yaml*")) # only do tsne transforms for non-perturbed
    logger.info(f"yamls to process: {yaml_paths}")
    tasks = []
    for yaml_name in config_yaml_names:
        # Build task list: every (config, extraction_strategy) combination
        yaml_path = PROJ_ROOT / "gelos" / "configs" / yaml_name
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        for extraction_strategy in yaml_config["embedding_extraction_strategies"].keys():
            tasks.append((yaml_name, extraction_strategy))
    workers = max_workers or min(len(tasks), os.cpu_count() or 1)
    header = f"{'Config':<32} | {'Strategy':<24} | {'Model':<24} | {'Acc':>7} | Per-class"
    logger.info("Experiment results (each row logs as jobs finish):")
    logger.info(header)
    logger.info("-" * len(header))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Submit all tasks to process pool
        futures = {
            ex.submit(run_config, yaml_name, extraction_strategy, chip_gdf, n_neighbors): (yaml_name, extraction_strategy)
            for yaml_name, extraction_strategy in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Configs x Strategies"):
            result = fut.result()
            logger.info(_format_result_row(*result))

    # After runs finish, aggregate per-run CSVs into a summary
    summary_df = collect_run_results(summary_root)
    if summary_df.empty:
        logger.warning("No per-run results available for summary.")
    else:
        summary_csv_path = summary_root / "knn_results_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved summary KNN results to {summary_csv_path}")
    


if __name__ == "__main__":
    app()
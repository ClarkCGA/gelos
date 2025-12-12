from gelos.config import RAW_DATA_DIR, PROJ_ROOT, PROCESSED_DATA_DIR, DATA_VERSION, REPORTS_DIR, FIGURES_DIR
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict, Counter
import yaml
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import typer
from loguru import logger
from typing import Optional
from joblib import Parallel, delayed
app = typer.Typer()
from gelos.embedding_generation import perturb_args_to_string


def random_forest_on_tsne(yaml_path: str, chip_gdf, figures_dir):

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    logger.info(f"processing {yaml_path}")

    model_name = yaml_config['model']['init_args']['model']
    model_title = yaml_config['model']['title']
    embedding_extraction_strategies = yaml_config['embedding_extraction_strategies']
    perturb = yaml_config['data']['init_args'].get('perturb_bands', None)
    perturb_string = perturb_args_to_string(perturb)
    output_dir = PROCESSED_DATA_DIR / DATA_VERSION / model_name / perturb_string

    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]

    for embeddings_directory in embeddings_directories:

        embedding_layer = embeddings_directory.stem

        for extraction_strategy, slice_args in embedding_extraction_strategies.items():


            model_title_lower = model_title.replace(" ", "").lower()
            extraction_strategy_lower = extraction_strategy.replace(" ", "").lower()
            embedding_layer_lower = embedding_layer.replace("_", "").lower()

            csv_path = output_dir / f"{model_title_lower}_{extraction_strategy_lower}_{embedding_layer_lower}_tsne.csv"
            if csv_path.exists():
                logger.info(f"{str(csv_path)} already exists, loading embeddings from file")
            embed_df = gpd.read_file(csv_path).drop("id", axis=1)
            chip_gdf = chip_gdf.merge(embed_df, left_index=True, right_index=True, how="left")
            X = chip_gdf[[
                f"{model_title_lower}_{extraction_strategy_lower}_tsne_x",
                f"{model_title_lower}_{extraction_strategy_lower}_tsne_y"
                ]]
            y = chip_gdf["category"]
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            cv_fold_metrics = []

            def run_fold(fold_idx, train_idx, test_idx):
                cv_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                cv_classifier.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_true_fold = y.iloc[test_idx]
                y_pred_fold = cv_classifier.predict(X.iloc[test_idx])
                fold_accuracy = accuracy_score(y_true_fold, y_pred_fold)
                return {"fold": fold_idx, "accuracy": fold_accuracy}

            cv_fold_metrics = Parallel(n_jobs=-1)(
                delayed(run_fold)(fold_idx, train_idx, test_idx)
                for fold_idx, (train_idx, test_idx) in enumerate(
                    tqdm(rskf.split(X, y), total=rskf.get_n_splits(X, y), desc="Cross-validation folds"),
                    start=1
                )
            )

            cv_metrics_df = pd.DataFrame(cv_fold_metrics)
            summary_rows = pd.DataFrame([
                {"fold": "mean", "accuracy": cv_metrics_df["accuracy"].mean()},
                {"fold": "std", "accuracy": cv_metrics_df["accuracy"].std()},
            ])
            cv_metrics_with_summary_df = pd.concat([cv_metrics_df, summary_rows], ignore_index=True)
            metrics_csv_path = output_dir / f"{model_title_lower}_{extraction_strategy_lower}_{embedding_layer_lower}_rfmetrics.csv"
            cv_metrics_with_summary_df.to_csv(metrics_csv_path, index=False)
        
def aggregate_rf_metrics(output_path: Optional[Path] = None):
    output_dir = PROCESSED_DATA_DIR / DATA_VERSION
    metrics_files = list(output_dir.rglob("*_rfmetrics.csv"))
    if not metrics_files:
        logger.warning(f"No rfmetrics files found under {output_dir}")
        return
    rows = []
    for metrics_file in metrics_files:
        perturbation = metrics_file.parent.name
        model_name = metrics_file.parent.parent.name
        base_name = metrics_file.stem.removesuffix("_rfmetrics")
        try:
            model_title_lower, extraction_strategy, embedding_layer = base_name.rsplit("_", 2)
        except ValueError:
            logger.warning(f"Could not parse metadata from {metrics_file.name}, skipping.")
            continue
        df = pd.read_csv(metrics_file)
        summary_df = df[df["fold"].astype(str).isin(["mean", "std"])]
        mean_val = summary_df.loc[summary_df["fold"].astype(str) == "mean", "accuracy"].squeeze()
        std_val = summary_df.loc[summary_df["fold"].astype(str) == "std", "accuracy"].squeeze()
        if pd.isna(mean_val) or pd.isna(std_val):
            logger.warning(f"Missing mean/std in {metrics_file}, skipping.")
            continue
        rows.append(
            {
                "model_name": model_name,
                "extraction_strategy": extraction_strategy,
                "embedding_layer": embedding_layer,
                "accuracy_mean": mean_val,
                "accuracy_std": std_val,
                "accuracy_formatted": f"{mean_val:.4f} +- {std_val:.4f}",
                "source_file": metrics_file.relative_to(output_dir),
            }
        )
    if not rows:
        logger.warning("No summary rows (mean/std) collected.")
        return
    result_df = pd.DataFrame(rows)
    output_csv = output_dir / output_path
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Aggregated RF metrics written to {output_csv}")




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
    # for yaml_path in yaml_paths:
    #     random_forest_on_tsne(yaml_path, chip_gdf, figures_dir)
    aggregate_rf_metrics("rf_scores_all_models.csv")


if __name__ == "__main__":
    app()
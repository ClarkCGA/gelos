from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-class accuracy from true and predicted labels."""
    classes = np.unique(y_true)
    per_class = {}
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            per_class[str(c)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return per_class


def _save_results_csv(
    results: dict,
    output_dir: Path,
    run_name: str,
    model_type: str,
) -> Path:
    """Save classification results to a CSV file."""
    csv_path = output_dir / f"{run_name}_{model_type}_results.csv"
    per_class = results["per_class"]
    rows = [{"class": k, "accuracy": v} for k, v in per_class.items()]
    rows.append({"class": "overall", "accuracy": results["accuracy"]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"saved {model_type} results to {csv_path}")
    return csv_path


def run_knn_cv(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    run_name: str,
    n_neighbors: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Run KNN classification with stratified k-fold cross-validation.

    Args:
        embeddings: Input features of shape (N, D).
        labels: Class labels of shape (N,).
        output_dir: Directory to save result CSV.
        run_name: Prefix for output file naming.
        n_neighbors: Number of neighbors for KNN.
        n_splits: Number of CV folds.
        random_state: Random state for reproducibility.

    Returns:
        Dict with ``accuracy``, ``per_class``, and ``predictions`` keys.
    """
    logger.info(f"running KNN CV: n_neighbors={n_neighbors}, n_splits={n_splits}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_preds = np.zeros_like(labels)

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train = labels[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        all_preds[test_idx] = clf.predict(X_test)

    overall_acc = float(accuracy_score(labels, all_preds))
    per_class = _per_class_accuracy(labels, all_preds)
    logger.info(f"KNN overall accuracy: {overall_acc:.4f}")

    results = {"accuracy": overall_acc, "per_class": per_class, "predictions": all_preds}
    _save_results_csv(results, output_dir, run_name, "knn")
    return results


def run_linear_probe_cv(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    run_name: str,
    max_iter: int = 1000,
    n_splits: int = 5,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Run logistic regression (linear probe) with stratified k-fold CV.

    Args:
        embeddings: Input features of shape (N, D).
        labels: Class labels of shape (N,).
        output_dir: Directory to save result CSV.
        run_name: Prefix for output file naming.
        max_iter: Maximum iterations for logistic regression.
        n_splits: Number of CV folds.
        random_state: Random state for reproducibility.

    Returns:
        Dict with ``accuracy``, ``per_class``, and ``predictions`` keys.
    """
    logger.info(f"running linear probe CV: max_iter={max_iter}, n_splits={n_splits}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_preds = np.zeros_like(labels)

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train = labels[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
        clf.fit(X_train, y_train)
        all_preds[test_idx] = clf.predict(X_test)

    overall_acc = float(accuracy_score(labels, all_preds))
    per_class = _per_class_accuracy(labels, all_preds)
    logger.info(f"linear probe overall accuracy: {overall_acc:.4f}")

    results = {"accuracy": overall_acc, "per_class": per_class, "predictions": all_preds}
    _save_results_csv(results, output_dir, run_name, "linear_probe")
    return results


def run_random_forest_cv(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    run_name: str,
    n_estimators: int = 100,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Run random forest with repeated stratified k-fold CV.

    Args:
        embeddings: Input features of shape (N, D).
        labels: Class labels of shape (N,).
        output_dir: Directory to save result CSV.
        run_name: Prefix for output file naming.
        n_estimators: Number of trees in the forest.
        n_splits: Number of CV folds.
        n_repeats: Number of CV repeats.
        random_state: Random state for reproducibility.

    Returns:
        Dict with ``accuracy``, ``per_class``, and ``predictions`` keys.
    """
    logger.info(
        f"running random forest CV: n_estimators={n_estimators}, "
        f"n_splits={n_splits}, n_repeats={n_repeats}"
    )
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    fold_accuracies = []
    # Use the last repeat's predictions for per-class metrics
    last_preds = np.zeros_like(labels)

    for fold, (train_idx, test_idx) in enumerate(rskf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        fold_accuracies.append(float(accuracy_score(y_test, preds)))
        last_preds[test_idx] = preds

    overall_acc = float(np.mean(fold_accuracies))
    per_class = _per_class_accuracy(labels, last_preds)
    logger.info(f"random forest mean accuracy: {overall_acc:.4f}")

    results = {
        "accuracy": overall_acc,
        "per_class": per_class,
        "predictions": last_preds,
    }
    _save_results_csv(results, output_dir, run_name, "random_forest")
    return results


MODELS: dict[str, callable] = {
    "knn": run_knn_cv,
    "linear_probe": run_linear_probe_cv,
    "random_forest": run_random_forest_cv,
}

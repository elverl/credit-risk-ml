"""Evaluation utilities for binary credit-risk classifiers."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


CHAMPION_METRIC_GROUPS = {
    "Discrimination Score": {
        "higher": ["AUC", "Gini", "KS", "PR-AUC"],
        "lower": [],
    },
    "Calibration Score": {
        "higher": [],
        "lower": ["Brier Score", "Log Loss"],
    },
    "Classification Score": {
        "higher": [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "MCC",
        ],
        "lower": ["FPR", "FNR"],
    },
}


def ks_statistic(y_true, y_score) -> float:
    """
    Calculate the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    y_true : array-like
        Binary target where 1 = default and 0 = non-default.
    y_score : array-like
        Predicted probability of default.

    Returns
    -------
    float
        KS statistic.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    score_default = y_score[y_true == 1]
    score_non_default = y_score[y_true == 0]

    if len(score_default) == 0 or len(score_non_default) == 0:
        return np.nan

    return stats.ks_2samp(
        score_default,
        score_non_default,
    ).statistic


def out_of_fold_probabilities(
    estimator,
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate out-of-fold predicted probabilities using stratified CV.

    Each observation receives a probability from a model that was not
    trained using that observation. These probabilities can then be used
    to select a classification threshold without using the final test set.

    Parameters
    ----------
    estimator
        Scikit-learn compatible binary classifier or pipeline implementing
        fit() and predict_proba().

    X : array-like or pandas.DataFrame
        Training features.

    y : array-like or pandas.Series
        Binary training target.

    n_splits : int, default=5
        Number of stratified cross-validation folds.

    random_state : int, default=42
        Random seed used for fold generation.

    Returns
    -------
    numpy.ndarray
        Out-of-fold probabilities for the positive class.
    """
    y_array = np.asarray(y)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probabilities = np.zeros(len(y_array), dtype=float)

    for train_idx, valid_idx in cv.split(X, y_array):
        model = clone(estimator)

        if hasattr(X, "iloc"):
            X_fold_train = X.iloc[train_idx]
            X_fold_valid = X.iloc[valid_idx]
        else:
            X_fold_train = X[train_idx]
            X_fold_valid = X[valid_idx]

        if hasattr(y, "iloc"):
            y_fold_train = y.iloc[train_idx]
        else:
            y_fold_train = y_array[train_idx]

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        oof_probabilities[valid_idx] = model.predict_proba(
            X_fold_valid
        )[:, 1]

    return oof_probabilities


def evaluate_binary_classifier(
    y_true,
    y_score,
    dataset_label: str,
    model_label: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate a binary credit-risk classifier.

    Parameters
    ----------
    y_true : array-like
        True binary target:
        0 = non-default
        1 = default.

    y_score : array-like
        Predicted probability of default.

    dataset_label : str
        Dataset name, e.g. Train OOF or Test.

    model_label : str
        Name of the evaluated model.

    threshold : float, default=0.5
        Probability threshold used to transform probabilities
        into binary predictions.

    Returns
    -------
    dict
        Discrimination, calibration and classification metrics.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            "y_true and y_score must contain the same number of observations."
        )

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "Model": model_label,
        "Dataset": dataset_label,
        "Threshold": threshold,

        # Discrimination
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks_statistic(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),

        # Calibration
        "Log Loss": log_loss(y_true, y_score),
        "Brier Score": brier_score_loss(y_true, y_score),

        # Threshold-dependent metrics
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            y_true,
            y_pred,
        ),

        # Error rates
        "FPR": fpr,
        "FNR": fnr,

        # Confusion matrix
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def rank_models_by_global_score(
    metrics_table: pd.DataFrame,
) -> pd.DataFrame:
    """Rank benchmark models using the three equally weighted V2 blocks."""
    if "Model" not in metrics_table.columns:
        raise ValueError("metrics_table must include a Model column.")
    if len(metrics_table) < 2:
        raise ValueError("At least two models are required for ranking.")

    required_metrics = {
        metric
        for directions in CHAMPION_METRIC_GROUPS.values()
        for direction in directions.values()
        for metric in direction
    }
    missing = sorted(required_metrics.difference(metrics_table.columns))
    if missing:
        raise ValueError(f"Missing metrics required for champion ranking: {missing}")

    ranking = metrics_table.set_index("Model")
    category_scores = pd.DataFrame(index=ranking.index)
    model_count = len(ranking)

    for category_name, directions in CHAMPION_METRIC_GROUPS.items():
        metric_scores = []
        for metric in directions["higher"]:
            rank = ranking[metric].rank(ascending=False, method="average")
            metric_scores.append((model_count - rank) / (model_count - 1))
        for metric in directions["lower"]:
            rank = ranking[metric].rank(ascending=True, method="average")
            metric_scores.append((model_count - rank) / (model_count - 1))
        category_scores[category_name] = pd.concat(
            metric_scores,
            axis=1,
        ).mean(axis=1)

    category_scores["Global Score"] = category_scores.mean(axis=1)
    return category_scores.sort_values("Global Score", ascending=False)

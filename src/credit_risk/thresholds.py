"""Threshold selection utilities for binary classifiers."""

import numpy as np
from sklearn.metrics import roc_curve


def threshold_youden(y_true, y_score) -> dict:
    """
    Select the optimal classification threshold using Youden's J statistic.

    Youden J = TPR - FPR

    Parameters
    ----------
    y_true : array-like
        True binary target.
    y_score : array-like
        Predicted probability of the positive class.

    Returns
    -------
    dict
        Optimal threshold and associated ROC statistics.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)

    return {
        "method": "Youden",
        "threshold": float(thresholds[best_idx]),
        "TPR": float(tpr[best_idx]),
        "FPR": float(fpr[best_idx]),
        "Youden_J": float(youden_j[best_idx]),
    }
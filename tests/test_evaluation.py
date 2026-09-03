"""Small deterministic checks for thresholds and binary evaluation."""

import pytest
import pandas as pd

from credit_risk.evaluation import (
    evaluate_binary_classifier,
    rank_models_by_global_score,
)
from credit_risk.thresholds import threshold_youden


def test_threshold_youden_returns_train_score_cutoff():
    result = threshold_youden(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )
    assert result["method"] == "Youden"
    assert result["threshold"] == pytest.approx(0.8)
    assert result["Youden_J"] == pytest.approx(1.0)


def test_evaluate_binary_classifier_basic_metrics():
    metrics = evaluate_binary_classifier(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        dataset_label="Test",
        model_label="Toy",
        threshold=0.5,
    )
    assert metrics["AUC"] == pytest.approx(1.0)
    assert metrics["Accuracy"] == pytest.approx(1.0)
    assert metrics["TP"] == 2
    assert metrics["TN"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0


def test_global_ranking_uses_all_three_metric_blocks():
    rows = []
    for model, high, low in (("Better", 0.9, 0.1), ("Worse", 0.6, 0.4)):
        rows.append(
            {
                "Model": model,
                **{
                    metric: high
                    for metric in [
                        "AUC", "Gini", "KS", "PR-AUC", "Accuracy",
                        "Balanced Accuracy", "Precision", "Recall",
                        "F1-score", "MCC",
                    ]
                },
                **{
                    metric: low
                    for metric in ["Brier Score", "Log Loss", "FPR", "FNR"]
                },
            }
        )
    ranking = rank_models_by_global_score(pd.DataFrame(rows))
    assert ranking.index[0] == "Better"
    assert ranking.loc["Better", "Global Score"] == pytest.approx(1.0)

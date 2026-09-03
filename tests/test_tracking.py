"""Pure checks for MLflow payload and backend selection; no server required."""

import json

import pytest

from credit_risk.models import build_logistic_regression
from credit_risk.tracking import (
    DAGSHUB_TRACKING_URI,
    build_run_params,
    build_tracking_manifest,
    mlflow_metrics,
    resolve_tracking_settings,
    rag_run_payloads,
)


def test_logistic_run_params_are_scalar_and_describe_scaler():
    params = build_run_params(
        "Logistic Regression",
        build_logistic_regression(),
        random_state=42,
        train_rows=80,
        test_rows=20,
        n_features=15,
    )
    assert params["has_standard_scaler"] is True
    assert params["random_state"] == 42
    assert params["estimator__random_state"] == 123
    assert params["cv_n_splits"] == 5
    assert params["threshold_selection_dataset"] == "train_oof"
    assert all(isinstance(value, (str, int, float, bool)) for value in params.values())


def test_mlflow_metrics_reuse_evaluation_values():
    sources = {
        "AUC": 0.71,
        "Gini": 0.42,
        "KS": 0.30,
        "Log Loss": 0.49,
        "Brier Score": 0.16,
        "PR-AUC": 0.43,
        "Accuracy": 0.65,
        "Balanced Accuracy": 0.64,
        "Precision": 0.36,
        "Recall": 0.63,
        "FPR": 0.34,
        "FNR": 0.37,
        "F1-score": 0.46,
        "MCC": 0.25,
    }
    payload = mlflow_metrics(sources, threshold=0.24)
    assert payload["auc"] == sources["AUC"]
    assert payload["brier_score"] == sources["Brier Score"]
    assert payload["threshold_youden"] == 0.24


def test_tracking_defaults_to_local_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_MODE", raising=False)
    settings = resolve_tracking_settings(tmp_path / "mlflow")
    assert settings.mode == "local"
    assert settings.tracking_uri.startswith("sqlite:///")
    assert settings.manifest_uri == "sqlite:///artifacts/mlflow/mlflow.db"


def test_dagshub_mode_requires_token(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_MODE", "dagshub")
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="DAGSHUB_TOKEN is required"):
        resolve_tracking_settings(tmp_path / "mlflow")


def test_dagshub_manifest_contains_no_secret(tmp_path, monkeypatch):
    secret = "test-token-that-must-not-be-persisted"
    monkeypatch.setenv("MLFLOW_TRACKING_MODE", "dagshub")
    monkeypatch.setenv("DAGSHUB_TOKEN", secret)
    settings = resolve_tracking_settings(tmp_path / "mlflow")
    manifest = build_tracking_manifest(
        settings,
        "experiment-id",
        {
            "xgboost": {
                "run_id": "run-id",
                "model_name": "XGBoost",
                "benchmark_role": "champion",
            }
        },
        {
            "model": "XGBoost",
            "model_slug": "xgboost",
            "run_id": "run-id",
            "global_score": 0.9,
            "criterion": "three_equal_blocks",
        },
    )
    serialized = json.dumps(manifest)
    assert settings.tracking_uri == DAGSHUB_TRACKING_URI
    assert secret not in serialized
    assert "MLFLOW_TRACKING_PASSWORD" not in serialized


def test_rag_payloads_reuse_frozen_summary_values():
    summary = {
        "cases": 24,
        "global": {
            "baseline": {"factual_correctness": 0.31, "groundedness": 0.73,
                         "answer_relevance": 0.21, "abstention_accuracy": 0.46,
                         "integrated_prediction_consistency": 0.75},
            "rag": {"factual_correctness": 0.62, "groundedness": 0.64,
                    "answer_relevance": 0.51, "abstention_accuracy": 0.96,
                    "integrated_prediction_consistency": 0.82},
        },
        "operational": {
            "baseline": {"average_latency_seconds": 0.8,
                         "token_totals": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            "rag": {"average_latency_seconds": 0.95,
                    "token_totals": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}},
        },
    }
    payloads = rag_run_payloads(summary)
    assert payloads["baseline"]["metrics"]["factual_correctness"] == 0.31
    assert payloads["rag"]["metrics"]["total_tokens"] == 28.0
    assert payloads["rag"]["params"]["retriever"] == "local-lsa-tfidf-svd-v1"
    assert "retriever" not in payloads["baseline"]["params"]
    assert payloads["baseline"]["params"]["n_cases"] == 24

"""Local or DagsHub MLflow tracking helpers for the V2 benchmark."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_NAME = "credit-risk-model-benchmark"
RAG_EXPERIMENT_NAME = "credit-risk-rag-evaluation"
DAGSHUB_TRACKING_URI = "https://dagshub.com/elverl/credit-risk-ml.mlflow"
DAGSHUB_USERNAME = "elverl"
TRACKING_MODES = {"local", "dagshub"}

MLFLOW_METRIC_MAP = {
    "AUC": "auc",
    "Gini": "gini",
    "KS": "ks",
    "Log Loss": "log_loss",
    "Brier Score": "brier_score",
    "PR-AUC": "pr_auc",
    "Accuracy": "accuracy",
    "Balanced Accuracy": "balanced_accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "FPR": "fpr",
    "FNR": "fnr",
    "F1-score": "f1_score",
    "MCC": "mcc",
}


@dataclass(frozen=True)
class TrackingSettings:
    """Non-secret MLflow settings safe to include in manifests."""

    mode: str
    tracking_uri: str
    manifest_uri: str


def resolve_tracking_settings(
    tracking_dir: Path,
    mode: str | None = None,
) -> TrackingSettings:
    """Resolve local or DagsHub settings and validate remote credentials."""
    tracking_mode = (
        mode or os.getenv("MLFLOW_TRACKING_MODE", "local")
    ).strip().lower()
    if tracking_mode not in TRACKING_MODES:
        raise ValueError(
            "MLFLOW_TRACKING_MODE must be 'local' or 'dagshub'."
        )

    if tracking_mode == "dagshub":
        token = os.getenv("DAGSHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "DAGSHUB_TOKEN is required when "
                "MLFLOW_TRACKING_MODE=dagshub."
            )
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        return TrackingSettings(
            mode="dagshub",
            tracking_uri=DAGSHUB_TRACKING_URI,
            manifest_uri=DAGSHUB_TRACKING_URI,
        )

    tracking_dir.mkdir(parents=True, exist_ok=True)
    database_path = (tracking_dir / "mlflow.db").resolve()
    return TrackingSettings(
        mode="local",
        tracking_uri=f"sqlite:///{database_path.as_posix()}",
        manifest_uri="sqlite:///artifacts/mlflow/mlflow.db",
    )


def configure_tracking(
    tracking_dir: Path,
    mode: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[TrackingSettings, str]:
    """Configure the selected backend and return safe settings and experiment ID."""
    settings = resolve_tracking_settings(tracking_dir, mode=mode)
    mlflow.set_tracking_uri(settings.tracking_uri)

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        if settings.mode == "local":
            experiment_id = client.create_experiment(
                experiment_name,
                artifact_location=(
                    tracking_dir / "artifacts"
                ).resolve().as_uri(),
            )
        else:
            experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return settings, experiment_id


def rag_run_payloads(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map the frozen comparison summary to two comparable MLflow payloads."""
    common_params = {
        "provider": "groq",
        "llm_model": "openai/gpt-oss-20b",
        "evalset_version": "rag_evalset_v1",
        "n_cases": int(summary["cases"]),
        "model_champion": "xgboost",
        "decision_threshold": 0.2399873,
        "kb_version": 1,
    }
    quality_keys = (
        "factual_correctness",
        "groundedness",
        "answer_relevance",
        "abstention_accuracy",
        "integrated_prediction_consistency",
    )
    payloads = {}
    for role in ("baseline", "rag"):
        operational = summary["operational"][role]
        tokens = operational["token_totals"]
        metrics = {key: float(summary["global"][role][key]) for key in quality_keys}
        metrics.update({
            "avg_latency_seconds": float(operational["average_latency_seconds"]),
            "total_tokens": float(tokens["total_tokens"]),
            "prompt_tokens": float(tokens["prompt_tokens"]),
            "completion_tokens": float(tokens["completion_tokens"]),
        })
        params = {**common_params, "evaluation_variant": role}
        if role == "rag":
            params.update({
                "retriever": "local-lsa-tfidf-svd-v1",
                "top_k": 3,
                "n_chunks": 56,
                "hit_rate_at_3": 0.75,
                "mrr_at_3": 0.597222,
            })
        payloads[role] = {"params": params, "metrics": metrics}
    return payloads


def build_tracking_manifest(
    settings: TrackingSettings,
    experiment_id: str,
    runs: dict[str, dict[str, str]],
    champion: dict[str, str | float],
) -> dict[str, Any]:
    """Build a credential-free manifest for the latest benchmark execution."""
    return {
        "tracking_mode": settings.mode,
        "tracking_uri": settings.manifest_uri,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "runs": runs,
        "champion": champion,
    }


def _serializable_param(value: Any) -> str | int | float | bool:
    """Convert estimator parameters to MLflow-safe scalar values."""
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_run_params(
    model_name: str,
    model,
    *,
    random_state: int,
    train_rows: int,
    test_rows: int,
    n_features: int,
) -> dict[str, str | int | float | bool]:
    """Build the stable parameter payload for one benchmark candidate."""
    estimator = model
    params: dict[str, str | int | float | bool] = {
        "model_name": model_name,
        "random_state": random_state,
        "cv_n_splits": 5,
        "cv_shuffle": True,
        "cv_random_state": random_state,
        "threshold_method": "youden",
        "threshold_selection_dataset": "train_oof",
        "positive_class": 1,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "n_features": n_features,
    }

    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
        params["has_standard_scaler"] = isinstance(
            model.named_steps.get("scaler"),
            StandardScaler,
        )

    for key, value in estimator.get_params(deep=False).items():
        params[f"estimator__{key}"] = _serializable_param(value)
    return params


def mlflow_metrics(metrics: dict[str, Any], threshold: float) -> dict[str, float]:
    """Map the already-computed V2 metrics to stable MLflow keys."""
    payload = {
        target: float(metrics[source])
        for source, target in MLFLOW_METRIC_MAP.items()
    }
    payload["threshold_youden"] = float(threshold)
    return payload


def log_fitted_model(model_slug: str, model) -> None:
    """Log a fitted model with the native MLflow flavor for its family."""
    if model_slug in {"logistic_regression", "random_forest"}:
        mlflow.sklearn.log_model(model, artifact_path="model")
    elif model_slug == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model")
    elif model_slug == "lightgbm":
        mlflow.lightgbm.log_model(model, artifact_path="model")
    else:
        raise ValueError(f"Unsupported model slug for MLflow: {model_slug}")

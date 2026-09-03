"""Train and evaluate the four fixed V2 credit-risk models."""

import json
from pathlib import Path
import pickle
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_risk.config import FEATURES, RANDOM_STATE
from credit_risk.evaluation import (
    evaluate_binary_classifier,
    out_of_fold_probabilities,
    rank_models_by_global_score,
)
from credit_risk.models import (
    build_lightgbm,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)
from credit_risk.thresholds import threshold_youden
from credit_risk.tracking import (
    EXPERIMENT_NAME,
    build_run_params,
    build_tracking_manifest,
    configure_tracking,
    log_fitted_model,
    mlflow_metrics,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
MLFLOW_DIR = PROJECT_ROOT / "artifacts" / "mlflow"

MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "Random Forest": build_random_forest,
    "XGBoost": build_xgboost,
    "LightGBM": build_lightgbm,
}

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}

METRIC_COLUMNS = [
    "Model",
    "Threshold",
    "AUC",
    "Gini",
    "KS",
    "PR-AUC",
    "Log Loss",
    "Brier Score",
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "MCC",
    "FPR",
    "FNR",
]


def load_processed_data():
    """Load and validate the four datasets created by preprocess.py."""
    required_paths = {
        "X_train": PROCESSED_DIR / "X_train.csv",
        "X_test": PROCESSED_DIR / "X_test.csv",
        "y_train": PROCESSED_DIR / "y_train.csv",
        "y_test": PROCESSED_DIR / "y_test.csv",
    }
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required_paths.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Processed datasets are missing. Run scripts/preprocess.py first: "
            f"{missing}"
        )

    X_train = pd.read_csv(required_paths["X_train"])
    X_test = pd.read_csv(required_paths["X_test"])
    y_train = pd.read_csv(required_paths["y_train"]).squeeze("columns")
    y_test = pd.read_csv(required_paths["y_test"]).squeeze("columns")

    if X_train.columns.tolist() != FEATURES:
        raise ValueError("X_train columns or order do not match FEATURES.")
    if X_test.columns.tolist() != FEATURES:
        raise ValueError("X_test columns or order do not match FEATURES.")

    return X_train, X_test, y_train, y_test


def main() -> None:
    """Execute the fixed four-model OOF + Youden benchmark."""
    X_train, X_test, y_train, y_test = load_processed_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tracking_settings, experiment_id = configure_tracking(MLFLOW_DIR)

    metrics_rows = []
    run_ids = {}
    thresholds = {
        "selection_dataset": "Train OOF",
        "positive_class": 1,
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "models": {},
    }

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    for model_name, builder in MODEL_BUILDERS.items():
        print(f"Training {model_name}...")

        oof_probability = out_of_fold_probabilities(
            builder(),
            X_train,
            y_train,
            n_splits=5,
            random_state=RANDOM_STATE,
        )
        threshold_info = threshold_youden(y_train, oof_probability)

        final_model = builder()
        final_model.fit(X_train, y_train)
        test_probability = final_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary_classifier(
            y_test,
            test_probability,
            dataset_label="Test",
            model_label=model_name,
            threshold=threshold_info["threshold"],
        )

        model_slug = MODEL_SLUGS[model_name]
        with open(MODELS_DIR / f"{model_slug}.pkl", "wb") as file:
            pickle.dump(final_model, file)

        tracked_metrics = mlflow_metrics(
            metrics,
            threshold_info["threshold"],
        )
        run_params = build_run_params(
            model_name,
            final_model,
            random_state=RANDOM_STATE,
            train_rows=len(X_train),
            test_rows=len(X_test),
            n_features=X_train.shape[1],
        )
        threshold_metadata = {
            "model": model_name,
            "model_slug": model_slug,
            "method": "youden",
            "selection_dataset": "train_oof",
            "positive_class": 1,
            "threshold": float(threshold_info["threshold"]),
            "oof_tpr": float(threshold_info["TPR"]),
            "oof_fpr": float(threshold_info["FPR"]),
            "oof_youden_j": float(threshold_info["Youden_J"]),
        }
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=model_slug,
            tags={
                "model_slug": model_slug,
                "benchmark_role": "candidate",
                "methodology": "v2_oof_youden",
            },
        ) as run:
            mlflow.log_params(run_params)
            mlflow.log_metrics(tracked_metrics)
            mlflow.log_dict(tracked_metrics, "metadata/test_metrics.json")
            mlflow.log_dict(
                threshold_metadata,
                "metadata/threshold.json",
            )
            log_fitted_model(model_slug, final_model)
            run_ids[model_slug] = run.info.run_id

        metrics_rows.append(metrics)
        thresholds["models"][model_slug] = {
            "model": model_name,
            "method": threshold_info["method"],
            "threshold": threshold_info["threshold"],
            "oof_tpr": threshold_info["TPR"],
            "oof_fpr": threshold_info["FPR"],
            "oof_youden_j": threshold_info["Youden_J"],
        }
        print(f"  OOF Youden threshold: {threshold_info['threshold']:.6f}")

    metrics_table = pd.DataFrame(metrics_rows)[METRIC_COLUMNS]
    metrics_path = METRICS_DIR / "four_model_benchmark_metrics.csv"
    thresholds_path = METRICS_DIR / "model_thresholds.json"
    metrics_table.to_csv(metrics_path, index=False)
    thresholds_path.write_text(
        json.dumps(thresholds, indent=2),
        encoding="utf-8",
    )

    champion_ranking = rank_models_by_global_score(metrics_table)
    champion_name = champion_ranking.index[0]
    champion_slug = MODEL_SLUGS[champion_name]
    champion_score = float(
        champion_ranking.loc[champion_name, "Global Score"]
    )
    ranking_path = METRICS_DIR / "champion_ranking.csv"
    champion_ranking.to_csv(ranking_path)

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    champion_run_id = run_ids[champion_slug]
    client.set_tag(champion_run_id, "benchmark_role", "champion")
    client.set_tag(
        champion_run_id,
        "champion_criterion",
        "equal_weight_discrimination_calibration_classification",
    )
    client.log_metric(
        champion_run_id,
        "champion_global_score",
        champion_score,
    )

    manifest_runs = {
        model_slug: {
            "run_id": run_id,
            "model_name": model_name,
            "benchmark_role": (
                "champion" if model_slug == champion_slug else "candidate"
            ),
        }
        for model_name, model_slug in MODEL_SLUGS.items()
        for run_id in [run_ids[model_slug]]
    }
    run_manifest = build_tracking_manifest(
        tracking_settings,
        experiment_id,
        manifest_runs,
        {
            "model": champion_name,
            "model_slug": champion_slug,
            "run_id": champion_run_id,
            "global_score": champion_score,
            "criterion": (
                "equal_weight_discrimination_calibration_classification"
            ),
        },
    )
    manifest_path = METRICS_DIR / "mlflow_run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print("\nTest metrics")
    print(metrics_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nModels: {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Thresholds: {thresholds_path.relative_to(PROJECT_ROOT)}")
    print(
        f"Champion: {champion_name} | Global score: {champion_score:.6f}"
    )
    print(f"MLflow experiment: {EXPERIMENT_NAME} ({experiment_id})")
    print(f"MLflow tracking mode: {tracking_settings.mode}")
    print(f"MLflow runs: {run_ids}")


if __name__ == "__main__":
    main()
